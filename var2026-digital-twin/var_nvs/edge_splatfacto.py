"""Splatfacto with an optional Sobel-gradient reconstruction loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch
import torch.nn.functional as F

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class EdgeSplatfactoModelConfig(SplatfactoModelConfig):
    """Configuration for :class:`EdgeSplatfactoModel`."""

    _target: Type = field(default_factory=lambda: EdgeSplatfactoModel)
    edge_loss_weight: float = 0.0
    """Weight applied to the Sobel-gradient L1 loss. Zero reproduces Splatfacto."""
    edge_loss_start_step: int = 0
    """First training step at which edge loss is active."""
    edge_loss_end_step: int = -1
    """First step at which edge loss is disabled; negative means never disable it."""


class EdgeSplatfactoModel(SplatfactoModel):
    """Adds a normalized RGB Sobel loss while preserving Splatfacto behavior."""

    config: EdgeSplatfactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        if self.config.edge_loss_weight < 0.0:
            raise ValueError("edge_loss_weight must be non-negative")
        if self.config.edge_loss_start_step < 0:
            raise ValueError("edge_loss_start_step must be non-negative")
        if (
            self.config.edge_loss_end_step >= 0
            and self.config.edge_loss_end_step <= self.config.edge_loss_start_step
        ):
            raise ValueError("edge_loss_end_step must be negative or greater than edge_loss_start_step")

        # Divide standard Sobel kernels by four so a unit step has a response
        # near one. This keeps edge-loss weights interpretable relative to L1.
        kernels = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            ],
            dtype=torch.float32,
        ) / 4.0
        self.register_buffer("_edge_kernels", kernels[:, None, :, :], persistent=False)

    def _sobel_gradients(self, image: torch.Tensor) -> torch.Tensor:
        """Return X/Y gradients as ``[1, C, 2, H, W]`` for an HWC image."""
        if image.ndim != 3 or image.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC RGB(A) image, got shape {tuple(image.shape)}")
        image_nchw = image[..., :3].permute(2, 0, 1).unsqueeze(0)
        channels = image_nchw.shape[1]
        kernels = self._edge_kernels.to(
            device=image_nchw.device,
            dtype=image_nchw.dtype,
        ).repeat(channels, 1, 1, 1)
        padded = F.pad(image_nchw, (1, 1, 1, 1), mode="replicate")
        gradients = F.conv2d(padded, kernels, groups=channels)
        return gradients.reshape(1, channels, 2, image.shape[0], image.shape[1])

    def _edge_loss_is_active(self) -> bool:
        if self.config.edge_loss_weight == 0.0:
            return False
        if self.step < self.config.edge_loss_start_step:
            return False
        return self.config.edge_loss_end_step < 0 or self.step < self.config.edge_loss_end_step

    def get_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        metrics_dict: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self._edge_loss_is_active():
            return loss_dict

        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Match the mask handling in Splatfacto's original photometric loss.
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            if mask.shape[:2] != gt_img.shape[:2] or gt_img.shape[:2] != pred_img.shape[:2]:
                raise ValueError("Mask, ground-truth, and prediction shapes do not match")
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        target_gradients = self._sobel_gradients(gt_img)
        predicted_gradients = self._sobel_gradients(pred_img)
        raw_edge_loss = torch.mean(torch.abs(predicted_gradients - target_gradients))
        loss_dict["edge_loss"] = self.config.edge_loss_weight * raw_edge_loss
        return loss_dict
