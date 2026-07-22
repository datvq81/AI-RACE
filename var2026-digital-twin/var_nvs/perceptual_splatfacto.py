"""Splatfacto with an optional full-image LPIPS training loss."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig


@dataclass
class PerceptualSplatfactoModelConfig(SplatfactoModelConfig):
    """Configuration for :class:`PerceptualSplatfactoModel`."""

    _target: Type = field(default_factory=lambda: PerceptualSplatfactoModel)
    lpips_loss_weight: float = 0.0
    """Weight applied to full-image Alex-LPIPS. Zero reproduces Splatfacto."""
    lpips_loss_start_step: int = 6000
    """First training step at which LPIPS loss is active."""
    lpips_loss_end_step: int = -1
    """First step at which LPIPS loss is disabled; negative means never disable it."""


class PerceptualSplatfactoModel(SplatfactoModel):
    """Adds differentiable full-image LPIPS to Splatfacto's training objective.

    Nerfstudio's Splatfacto model already owns an Alex-LPIPS metric for image
    evaluation. Reusing that frozen network avoids loading a duplicate AlexNet.
    Calling a TorchMetrics metric through ``forward`` remains differentiable;
    its accumulated state is reset after every training use.
    """

    config: PerceptualSplatfactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        if self.config.lpips_loss_weight < 0.0:
            raise ValueError("lpips_loss_weight must be non-negative")
        if self.config.lpips_loss_start_step < 0:
            raise ValueError("lpips_loss_start_step must be non-negative")
        if (
            self.config.lpips_loss_end_step >= 0
            and self.config.lpips_loss_end_step <= self.config.lpips_loss_start_step
        ):
            raise ValueError(
                "lpips_loss_end_step must be negative or greater than lpips_loss_start_step"
            )

        # LPIPS is a fixed perceptual feature extractor, never a trainable part
        # of the reconstruction model.
        self.lpips.requires_grad_(False)
        self.lpips.eval()
        self.lpips.reset()

    def train(self, mode: bool = True) -> "PerceptualSplatfactoModel":
        """Keep the LPIPS backbone in eval mode when the renderer trains."""
        super().train(mode)
        if hasattr(self, "lpips"):
            self.lpips.eval()
        return self

    def _lpips_loss_is_active(self) -> bool:
        if self.config.lpips_loss_weight == 0.0:
            return False
        if self.step < self.config.lpips_loss_start_step:
            return False
        return self.config.lpips_loss_end_step < 0 or self.step < self.config.lpips_loss_end_step

    def get_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        metrics_dict: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self._lpips_loss_is_active():
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

        # Splatfacto configures this TorchMetrics object with normalize=True,
        # hence it expects [0, 1] NCHW input. Detaching the target prevents an
        # unnecessary target-side autograd graph while preserving prediction
        # gradients through the frozen AlexNet.
        pred_nchw = pred_img[..., :3].permute(2, 0, 1).unsqueeze(0)
        target_nchw = gt_img[..., :3].permute(2, 0, 1).unsqueeze(0).detach()
        raw_lpips_loss = self.lpips(pred_nchw, target_nchw)
        self.lpips.reset()

        loss_dict["lpips_loss"] = self.config.lpips_loss_weight * raw_lpips_loss
        return loss_dict
