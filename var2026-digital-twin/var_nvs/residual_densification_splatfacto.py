"""D1b Splatfacto with residual/edge-aware Gaussian densification."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from var_nvs.perceptual_splatfacto import (
    PerceptualSplatfactoModel,
    PerceptualSplatfactoModelConfig,
)


@dataclass
class ResidualDensificationSplatfactoModelConfig(PerceptualSplatfactoModelConfig):
    """Configuration for :class:`ResidualDensificationSplatfactoModel`."""

    _target: type = field(default_factory=lambda: ResidualDensificationSplatfactoModel)
    residual_densify_blend: float = 0.0
    """Blend from stock AbsGrad (0) to fully priority-weighted AbsGrad (1)."""
    residual_densify_start_step: int = 6000
    """First step at which residual-aware gradient accumulation is active."""
    residual_densify_end_step: int = -1
    """First inactive step; negative follows Splatfacto's stop_split_at."""
    residual_densify_edge_weight: float = 1.0
    """Strength of ground-truth edge magnitude in the residual priority map."""
    residual_densify_smoothing_kernel: int = 5
    """Odd average-pooling kernel applied to the image-space priority map."""
    residual_densify_min_multiplier: float = 0.25
    """Lower clamp for a Gaussian's normalized priority multiplier."""
    residual_densify_max_multiplier: float = 3.0
    """Upper clamp for a Gaussian's normalized priority multiplier."""


class ResidualDensificationSplatfactoModel(PerceptualSplatfactoModel):
    """Redistributes Splatfacto's existing AbsGrad using image residuals.

    Nerfstudio 1.1.4 already requests ``absgrad=True`` from gsplat and uses
    ``self.xys.absgrad`` for refinement. This model therefore does not add
    AbsGS again. Instead, it samples a detached image-space priority map at
    each visible Gaussian center before accumulating its AbsGrad statistic.

    The sampled multipliers are normalized to mean one. The experiment changes
    *where* the fixed densification criterion is met instead of globally
    lowering ``densify_grad_thresh`` or extending ``stop_split_at``.
    """

    config: ResidualDensificationSplatfactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        if not 0.0 <= self.config.residual_densify_blend <= 1.0:
            raise ValueError("residual_densify_blend must be between zero and one")
        if self.config.residual_densify_start_step < 0:
            raise ValueError("residual_densify_start_step must be non-negative")
        if (
            self.config.residual_densify_end_step >= 0
            and self.config.residual_densify_end_step
            <= self.config.residual_densify_start_step
        ):
            raise ValueError(
                "residual_densify_end_step must be negative or greater than "
                "residual_densify_start_step"
            )
        if self.config.residual_densify_edge_weight < 0.0:
            raise ValueError("residual_densify_edge_weight must be non-negative")
        kernel = self.config.residual_densify_smoothing_kernel
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError("residual_densify_smoothing_kernel must be a positive odd integer")
        minimum = self.config.residual_densify_min_multiplier
        maximum = self.config.residual_densify_max_multiplier
        if minimum <= 0.0 or maximum < minimum:
            raise ValueError("Residual densification multiplier bounds are invalid")

        sobel = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            ],
            dtype=torch.float32,
        ) / 4.0
        self.register_buffer(
            "_residual_densify_sobel",
            sobel[:, None, :, :],
            persistent=False,
        )
        self._residual_densify_priority: torch.Tensor | None = None

    def _residual_densification_is_active(self) -> bool:
        if not self.training or self.config.residual_densify_blend == 0.0:
            return False
        if self.step < self.config.residual_densify_start_step:
            return False
        if self.step >= self.config.stop_split_at:
            return False
        return (
            self.config.residual_densify_end_step < 0
            or self.step < self.config.residual_densify_end_step
        )

    def _build_priority_map(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return a detached HxW residual/edge priority map."""
        prediction_rgb = prediction[..., :3].detach()
        target_rgb = target[..., :3].detach()
        residual = torch.mean(torch.abs(prediction_rgb - target_rgb), dim=-1)

        target_nchw = target_rgb.permute(2, 0, 1).unsqueeze(0)
        luma_weights = target_nchw.new_tensor([0.2126, 0.7152, 0.0722]).view(
            1, 3, 1, 1
        )
        luma = torch.sum(target_nchw * luma_weights, dim=1, keepdim=True)
        kernels = self._residual_densify_sobel.to(
            device=luma.device,
            dtype=luma.dtype,
        )
        padded = F.pad(luma, (1, 1, 1, 1), mode="replicate")
        gradients = F.conv2d(padded, kernels)
        edge_magnitude = torch.sqrt(torch.sum(gradients.square(), dim=1) + 1e-12)[0]
        edge_scale = torch.clamp(edge_magnitude.mean(), min=1e-6)
        normalized_edge = torch.clamp(edge_magnitude / edge_scale, 0.0, 4.0)

        priority = residual * (
            1.0 + self.config.residual_densify_edge_weight * normalized_edge
        )
        if mask is not None:
            mask_map = mask[..., 0] if mask.ndim == 3 else mask
            priority = priority * mask_map.detach()

        kernel = self.config.residual_densify_smoothing_kernel
        if kernel > 1:
            priority = F.avg_pool2d(
                priority[None, None],
                kernel_size=kernel,
                stride=1,
                padding=kernel // 2,
            )[0, 0]
        return priority

    def get_loss_dict(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        metrics_dict: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        self._residual_densify_priority = None
        if not self._residual_densification_is_active():
            return loss_dict

        target = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        prediction = outputs["rgb"]
        mask = None
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            if mask.shape[:2] != target.shape[:2] or target.shape[:2] != prediction.shape[:2]:
                raise ValueError("Mask, ground-truth, and prediction shapes do not match")

        self._residual_densify_priority = self._build_priority_map(
            prediction,
            target,
            mask,
        )
        return loss_dict

    def _visible_priority_multipliers(
        self,
        visible_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Sample and normalize image priority at visible Gaussian centers."""
        priority = self._residual_densify_priority
        if priority is None:
            return torch.ones(
                int(visible_mask.sum().item()),
                device=self.device,
                dtype=torch.float32,
            )

        centers = self.xys.detach()[0][visible_mask]
        if centers.shape[0] == 0:
            return torch.empty(0, device=self.device, dtype=torch.float32)
        height, width = priority.shape
        finite = torch.isfinite(centers).all(dim=-1)
        inside = (
            finite
            & (centers[:, 0] >= 0)
            & (centers[:, 0] <= width - 1)
            & (centers[:, 1] >= 0)
            & (centers[:, 1] <= height - 1)
        )
        neutral_priority = torch.nan_to_num(
            priority.mean().to(torch.float32),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        neutral_priority = torch.clamp(neutral_priority, min=1e-8)
        sampled = torch.ones(
            centers.shape[0],
            device=centers.device,
            dtype=torch.float32,
        ) * neutral_priority
        x = torch.round(centers[inside, 0]).long().clamp(0, width - 1)
        y = torch.round(centers[inside, 1]).long().clamp(0, height - 1)
        sampled[inside] = priority[y, x].to(torch.float32)

        sampled_mean = sampled.mean()
        normalized = sampled / torch.clamp(sampled_mean, min=1e-8)
        normalized = torch.nan_to_num(
            normalized,
            nan=1.0,
            posinf=self.config.residual_densify_max_multiplier,
            neginf=self.config.residual_densify_min_multiplier,
        )
        normalized = torch.clamp(
            normalized,
            min=self.config.residual_densify_min_multiplier,
            max=self.config.residual_densify_max_multiplier,
        )
        blend = self.config.residual_densify_blend
        multipliers = 1.0 + blend * (normalized - 1.0)
        # Preserve the mean gradient scale so the ablation changes allocation,
        # rather than acting like a hidden densify_grad_thresh reduction.
        return multipliers / torch.clamp(multipliers.mean(), min=1e-8)

    def after_train(self, step: int) -> None:
        """Accumulate priority-weighted AbsGrad while preserving stock control."""
        if (
            not self._residual_densification_is_active()
            or self._residual_densify_priority is None
        ):
            super().after_train(step)
            return

        assert step == self.step
        with torch.no_grad():
            visible_mask = (self.radii > 0).flatten()
            grads = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            multipliers = self._visible_priority_multipliers(visible_mask)
            grads = grads * multipliers.to(device=grads.device, dtype=grads.dtype)

            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(
                    self.num_points,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.vis_counts = torch.ones(
                    self.num_points,
                    device=self.device,
                    dtype=torch.float32,
                )
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads

            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            new_radii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                new_radii / float(max(self.last_size[0], self.last_size[1])),
            )
        self._residual_densify_priority = None
