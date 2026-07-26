"""Splatfacto-perceptual with pose-conditioned global exposure correction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

from nerfstudio.cameras.cameras import Cameras

from var_nvs.perceptual_splatfacto import (
    PerceptualSplatfactoModel,
    PerceptualSplatfactoModelConfig,
)


@dataclass
class PoseExposureSplatfactoModelConfig(PerceptualSplatfactoModelConfig):
    """Configuration for :class:`PoseExposureSplatfactoModel`."""

    _target: Type = field(default_factory=lambda: PoseExposureSplatfactoModel)
    use_pose_exposure: bool = False
    """Enable pose-conditioned RGB gain and bias correction."""
    exposure_start_step: int = 6000
    """First training step at which the exposure network is active."""
    exposure_hidden_dim: int = 32
    """Hidden width of the two-layer pose MLP."""
    exposure_position_scale: float = 1.0
    """Scene-space position divisor before the bounded pose encoding."""
    exposure_gain_limit: float = 0.25
    """Maximum absolute log-gain; 0.25 corresponds to roughly +/-28 percent."""
    exposure_bias_limit: float = 0.10
    """Maximum absolute additive RGB bias."""
    exposure_regularization_weight: float = 0.001
    """L2 regularization weight for the predicted log-gain and bias."""


class PoseExposureSplatfactoModel(PerceptualSplatfactoModel):
    """Adds a continuous, novel-view-safe exposure model to D1b.

    A small MLP maps the camera's normalized world position and forward
    direction to six image-global values: RGB log-gain and RGB bias. Unlike a
    free per-image embedding, this mapping is defined for every novel camera
    pose used by the submission renderer.
    """

    config: PoseExposureSplatfactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        if self.config.exposure_start_step < 0:
            raise ValueError("exposure_start_step must be non-negative")
        if self.config.exposure_hidden_dim <= 0:
            raise ValueError("exposure_hidden_dim must be positive")
        if self.config.exposure_position_scale <= 0.0:
            raise ValueError("exposure_position_scale must be positive")
        if self.config.exposure_gain_limit < 0.0:
            raise ValueError("exposure_gain_limit must be non-negative")
        if self.config.exposure_bias_limit < 0.0:
            raise ValueError("exposure_bias_limit must be non-negative")
        if self.config.exposure_regularization_weight < 0.0:
            raise ValueError("exposure_regularization_weight must be non-negative")

        hidden_dim = self.config.exposure_hidden_dim
        self.pose_exposure = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

        # Identity initialization: the renderer is exactly unchanged before the
        # exposure head learns a non-zero final layer.
        final_layer = self.pose_exposure[-1]
        if not isinstance(final_layer, nn.Linear):
            raise TypeError("pose_exposure final layer must be linear")
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        groups["pose_exposure"] = list(self.pose_exposure.parameters())
        return groups

    def _camera_pose_features(self, camera: Cameras) -> torch.Tensor:
        if self.training:
            camera_to_world = self.camera_optimizer.apply_to_camera(camera).detach()
        else:
            camera_to_world = camera.camera_to_worlds.detach()

        camera_to_world = camera_to_world.reshape(-1, 3, 4).to(
            self.device, torch.float32
        )
        center = camera_to_world[:, :3, 3]
        center = torch.tanh(center / self.config.exposure_position_scale)

        # Nerfstudio cameras look along local -Z.
        forward = -camera_to_world[:, :3, 2]
        forward = F.normalize(forward, dim=-1)
        return torch.cat((center, forward), dim=-1)

    def _exposure_parameters(
        self,
        camera: Cameras,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not self.config.use_pose_exposure
            or (self.training and self.step < self.config.exposure_start_step)
        ):
            neutral = torch.zeros(3, device=self.device, dtype=torch.float32)
            return neutral, neutral

        pose_features = self._camera_pose_features(camera)
        if pose_features.shape[0] != 1:
            raise ValueError("PoseExposureSplatfactoModel supports one camera per render")

        raw_parameters = self.pose_exposure(pose_features)[0]
        log_gain = (
            torch.tanh(raw_parameters[:3]) * self.config.exposure_gain_limit
        )
        bias = (
            torch.tanh(raw_parameters[3:]) * self.config.exposure_bias_limit
        )
        return log_gain, bias

    def get_outputs(
        self,
        camera: Cameras,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = super().get_outputs(camera)
        rgb = outputs.get("rgb")
        if not isinstance(rgb, torch.Tensor):
            return outputs

        log_gain, bias = self._exposure_parameters(camera)
        if self.config.use_pose_exposure and (
            not self.training or self.step >= self.config.exposure_start_step
        ):
            outputs["rgb"] = torch.clamp(
                rgb * torch.exp(log_gain) + bias,
                0.0,
                1.0,
            )

        # Expose the correction for diagnostics and regularization. Nerfstudio's
        # standard image metrics ignore unknown output keys.
        outputs["exposure_log_gain"] = log_gain
        outputs["exposure_bias"] = bias
        return outputs

    def get_loss_dict(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
        metrics_dict: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if (
            not self.config.use_pose_exposure
            or self.step < self.config.exposure_start_step
            or self.config.exposure_regularization_weight == 0.0
        ):
            return loss_dict

        log_gain = outputs.get("exposure_log_gain")
        bias = outputs.get("exposure_bias")
        if not isinstance(log_gain, torch.Tensor) or not isinstance(bias, torch.Tensor):
            raise ValueError("Exposure outputs are missing during exposure training")

        regularization = log_gain.square().mean() + bias.square().mean()
        loss_dict["pose_exposure_reg"] = (
            self.config.exposure_regularization_weight * regularization
        )
        return loss_dict
