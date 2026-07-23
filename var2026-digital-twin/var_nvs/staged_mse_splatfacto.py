"""Splatfacto-perceptual with a late-stage MSE photometric objective."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Type

import torch

from var_nvs.perceptual_splatfacto import (
    PerceptualSplatfactoModel,
    PerceptualSplatfactoModelConfig,
)


@dataclass
class StagedMSESplatfactoModelConfig(PerceptualSplatfactoModelConfig):
    """Configuration for :class:`StagedMSESplatfactoModel`."""

    _target: Type = field(default_factory=lambda: StagedMSESplatfactoModel)
    staged_mse_weight: float = 0.0
    """Late-stage MSE weight taken from the original L1 coefficient."""
    staged_mse_start_step: int = 15000
    """First training step at which the L1-to-MSE weight transfer is active."""
    staged_mse_end_step: int = -1
    """First inactive step; a negative value keeps MSE active until training ends."""


class StagedMSESplatfactoModel(PerceptualSplatfactoModel):
    """Transfers part of Splatfacto's L1 weight to MSE late in training.

    With Splatfacto's default ``ssim_lambda=0.2`` and the prepared F1 weight
    ``0.35``, the main photometric objective changes from

    ``0.80 L1 + 0.20 DSSIM``

    to

    ``0.45 L1 + 0.35 MSE + 0.20 DSSIM``.

    LPIPS and all regularizers inherited from :class:`PerceptualSplatfactoModel`
    remain unchanged. A zero MSE weight exactly reproduces that parent model.
    """

    config: StagedMSESplatfactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        max_mse_weight = 1.0 - self.config.ssim_lambda
        if self.config.staged_mse_weight < 0.0:
            raise ValueError("staged_mse_weight must be non-negative")
        if self.config.staged_mse_weight > max_mse_weight:
            raise ValueError(
                "staged_mse_weight cannot exceed Splatfacto's L1 weight "
                f"(1 - ssim_lambda = {max_mse_weight})"
            )
        if self.config.staged_mse_start_step < 0:
            raise ValueError("staged_mse_start_step must be non-negative")
        if (
            self.config.staged_mse_end_step >= 0
            and self.config.staged_mse_end_step <= self.config.staged_mse_start_step
        ):
            raise ValueError(
                "staged_mse_end_step must be negative or greater than staged_mse_start_step"
            )

    def _staged_mse_is_active(self) -> bool:
        if self.config.staged_mse_weight == 0.0:
            return False
        if self.step < self.config.staged_mse_start_step:
            return False
        return (
            self.config.staged_mse_end_step < 0
            or self.step < self.config.staged_mse_end_step
        )

    def get_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        metrics_dict: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if not self._staged_mse_is_active():
            return loss_dict

        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        # Keep mask semantics identical to Splatfacto and the LPIPS extension.
        if "mask" in batch:
            mask = self._downscale_if_required(batch["mask"]).to(self.device)
            if mask.shape[:2] != gt_img.shape[:2] or gt_img.shape[:2] != pred_img.shape[:2]:
                raise ValueError("Mask, ground-truth, and prediction shapes do not match")
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        residual = gt_img - pred_img
        l1_loss = residual.abs().mean()
        mse_loss = residual.square().mean()

        # Parent main_loss already contains:
        #   (1 - ssim_lambda) * L1 + ssim_lambda * DSSIM.
        # Transfer `weight` from L1 to MSE without recomputing DSSIM.
        weight = self.config.staged_mse_weight
        loss_dict["main_loss"] = loss_dict["main_loss"] + weight * (
            mse_loss - l1_loss
        )
        return loss_dict
