"""D1b Splatfacto with Pixel-GS-style density statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch

from nerfstudio.cameras.cameras import Cameras

from var_nvs.perceptual_splatfacto import (
    PerceptualSplatfactoModel,
    PerceptualSplatfactoModelConfig,
)
from var_nvs.radical.pixel_gradient import (
    clipped_footprint_coverage,
    depth_gradient_scale,
)


@dataclass
class PixelGradientSplatfactoModelConfig(PerceptualSplatfactoModelConfig):
    """Configuration for :class:`PixelGradientSplatfactoModel`."""

    _target: type = field(default_factory=lambda: PixelGradientSplatfactoModel)
    use_pixel_aware_gradient: bool = False
    """Use projected coverage instead of an equal per-view gradient average."""
    pixel_grad_use_distance_scaling: bool = True
    """Apply Pixel-GS camera-depth scaling to suppress near-camera floaters."""
    pixel_grad_gamma_depth: float = 0.37
    """Depth scale from the Pixel-GS paper."""
    pixel_grad_scene_radius: float = 1.0
    """Scene radius in Nerfstudio coordinates used by depth scaling."""
    pixel_grad_min_coverage: float = 1e-8
    """Numerical floor for a visible Gaussian's normalized footprint."""


class PixelGradientSplatfactoModel(PerceptualSplatfactoModel):
    """Accumulate coverage-weighted AbsGrad without changing stock refinement.

    Nerfstudio 1.1.4 already obtains absolute screen-space gradients from
    gsplat. P1 only changes the cross-view statistic accumulated for the stock
    split/clone decision. P0 delegates to the parent implementation exactly.
    """

    config: PixelGradientSplatfactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        if self.config.pixel_grad_gamma_depth <= 0.0:
            raise ValueError("pixel_grad_gamma_depth must be positive")
        if self.config.pixel_grad_scene_radius <= 0.0:
            raise ValueError("pixel_grad_scene_radius must be positive")
        if self.config.pixel_grad_min_coverage <= 0.0:
            raise ValueError("pixel_grad_min_coverage must be positive")
        self._pixel_grad_camera_depth: torch.Tensor | None = None

    def _pixel_gradient_is_active(self) -> bool:
        return (
            self.training
            and self.config.use_pixel_aware_gradient
            and self.step < self.config.stop_split_at
        )

    def get_outputs(
        self,
        camera: Cameras,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = super().get_outputs(camera)
        self._pixel_grad_camera_depth = None
        if not self._pixel_gradient_is_active():
            return outputs

        camera_to_world = self.camera_optimizer.apply_to_camera(camera).detach()
        camera_to_world = camera_to_world.reshape(-1, 3, 4).to(
            self.device, torch.float32
        )
        if camera_to_world.shape[0] != 1:
            raise ValueError("PixelGradientSplatfactoModel supports one training camera")
        center = camera_to_world[0, :3, 3]
        forward = -camera_to_world[0, :3, 2]
        forward = torch.nn.functional.normalize(forward, dim=-1)
        offsets = self.means.detach().to(torch.float32) - center
        self._pixel_grad_camera_depth = torch.sum(offsets * forward, dim=-1)
        return outputs

    def after_train(self, step: int) -> None:
        """Accumulate Pixel-GS numerator and denominator for refinement."""
        if not self._pixel_gradient_is_active():
            super().after_train(step)
            return

        assert step == self.step
        with torch.no_grad():
            visible_mask = (self.radii > 0).flatten()
            gradients = self.xys.absgrad[0][visible_mask].norm(dim=-1)  # type: ignore
            centers = self.xys.detach()[0][visible_mask]
            radii = self.radii.detach()[visible_mask]
            height, width = self.last_size
            coverage = clipped_footprint_coverage(
                centers,
                radii,
                image_height=height,
                image_width=width,
            )
            coverage = torch.where(
                coverage > 0.0,
                torch.clamp(
                    coverage,
                    min=self.config.pixel_grad_min_coverage,
                ),
                coverage,
            )

            if self.config.pixel_grad_use_distance_scaling:
                if self._pixel_grad_camera_depth is None:
                    raise RuntimeError("Camera depth was not captured for Pixel-GS")
                scales = depth_gradient_scale(
                    self._pixel_grad_camera_depth[visible_mask],
                    scene_radius=self.config.pixel_grad_scene_radius,
                    gamma_depth=self.config.pixel_grad_gamma_depth,
                )
                gradients = gradients * scales.to(
                    device=gradients.device,
                    dtype=gradients.dtype,
                )

            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(
                    self.num_points,
                    device=self.device,
                    dtype=torch.float32,
                )
                self.vis_counts = torch.zeros(
                    self.num_points,
                    device=self.device,
                    dtype=torch.float32,
                )
            assert self.vis_counts is not None
            weights = coverage.to(device=self.device, dtype=torch.float32)
            self.xys_grad_norm[visible_mask] += gradients.to(torch.float32) * weights
            self.vis_counts[visible_mask] += weights

            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                radii / float(max(height, width)),
            )
        self._pixel_grad_camera_depth = None
