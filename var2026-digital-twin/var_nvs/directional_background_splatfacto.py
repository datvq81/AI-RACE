"""Splatfacto with LPIPS training loss and a directional SH background."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch
from torch.nn import Parameter

from nerfstudio.cameras.cameras import Cameras

from var_nvs.perceptual_splatfacto import (
    PerceptualSplatfactoModel,
    PerceptualSplatfactoModelConfig,
)


_SH_C0 = 0.28209479177387814


def _real_sh_basis(directions: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate real spherical-harmonic bases through degree three.

    Args:
        directions: Unit directions with shape ``[..., 3]``.
        degree: Maximum SH degree in ``[0, 3]``.

    Returns:
        A tensor of shape ``[..., (degree + 1) ** 2]``.
    """
    x, y, z = directions.unbind(dim=-1)
    basis = [torch.full_like(x, _SH_C0)]

    if degree >= 1:
        basis.extend(
            (
                -0.4886025119029199 * y,
                0.4886025119029199 * z,
                -0.4886025119029199 * x,
            )
        )
    if degree >= 2:
        basis.extend(
            (
                1.0925484305920792 * x * y,
                -1.0925484305920792 * y * z,
                0.31539156525252005 * (3.0 * z.square() - 1.0),
                -1.0925484305920792 * x * z,
                0.5462742152960396 * (x.square() - y.square()),
            )
        )
    if degree >= 3:
        basis.extend(
            (
                -0.5900435899266435 * y * (3.0 * x.square() - y.square()),
                2.890611442640554 * x * y * z,
                -0.4570457994644658 * y * (5.0 * z.square() - 1.0),
                0.3731763325901154 * z * (5.0 * z.square() - 3.0),
                -0.4570457994644658 * x * (5.0 * z.square() - 1.0),
                1.445305721320277 * z * (x.square() - y.square()),
                -0.5900435899266435 * x * (x.square() - 3.0 * y.square()),
            )
        )
    return torch.stack(basis, dim=-1)


@dataclass
class DirectionalBackgroundSplatfactoModelConfig(PerceptualSplatfactoModelConfig):
    """Configuration for :class:`DirectionalBackgroundSplatfactoModel`."""

    _target: Type = field(default_factory=lambda: DirectionalBackgroundSplatfactoModel)
    use_directional_background: bool = True
    """Replace the constant/random background by a learned view-direction field."""
    background_sh_degree: int = 3
    """Maximum SH degree for the directional background; supported range is 0--3."""
    background_start_step: int = 1000
    """First step at which the background SH coefficients receive gradients."""
    background_init_color: tuple[float, float, float] = (0.5, 0.5, 0.5)
    """Initial RGB background color before directional coefficients are learned."""


class DirectionalBackgroundSplatfactoModel(PerceptualSplatfactoModel):
    """Composites Gaussian foreground over a small learned directional sky model.

    The foreground rasterizer is unchanged. It renders against black, then the
    residual transmittance ``1 - accumulation`` is filled by an RGB function of
    the world-space camera-ray direction. Only 48 scalar coefficients are added
    for degree three, so the VRAM impact is negligible.
    """

    config: DirectionalBackgroundSplatfactoModelConfig

    def populate_modules(self) -> None:
        super().populate_modules()
        if not 0 <= self.config.background_sh_degree <= 3:
            raise ValueError("background_sh_degree must be between 0 and 3")
        if self.config.background_start_step < 0:
            raise ValueError("background_start_step must be non-negative")
        if len(self.config.background_init_color) != 3:
            raise ValueError("background_init_color must contain exactly three values")

        init_color = torch.tensor(self.config.background_init_color, dtype=torch.float32)
        if torch.any((init_color <= 0.0) | (init_color >= 1.0)):
            raise ValueError("background_init_color values must be strictly between 0 and 1")

        num_coefficients = (self.config.background_sh_degree + 1) ** 2
        coefficients = torch.zeros((num_coefficients, 3), dtype=torch.float32)
        coefficients[0] = torch.logit(init_color) / _SH_C0
        self.background_shs = Parameter(coefficients)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        groups = super().get_param_groups()
        groups["directional_background"] = [self.background_shs]
        return groups

    def _get_background_color(self) -> torch.Tensor:
        if self.config.use_directional_background:
            # Render the Gaussian foreground against black. The directional
            # background is composited exactly once after rasterization.
            return torch.zeros(3, device=self.device)
        return super()._get_background_color()

    def _camera_directions(
        self,
        camera: Cameras,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Return one normalized world-space direction for every output pixel."""
        if self.training:
            camera_to_world = self.camera_optimizer.apply_to_camera(camera).detach()
        else:
            camera_to_world = camera.camera_to_worlds.detach()

        source_width = camera.width.reshape(-1)[0].to(self.device, torch.float32)
        source_height = camera.height.reshape(-1)[0].to(self.device, torch.float32)
        scale_x = width / source_width
        scale_y = height / source_height

        fx = camera.fx.reshape(-1)[0].to(self.device, torch.float32) * scale_x
        fy = camera.fy.reshape(-1)[0].to(self.device, torch.float32) * scale_y
        cx = camera.cx.reshape(-1)[0].to(self.device, torch.float32) * scale_x
        cy = camera.cy.reshape(-1)[0].to(self.device, torch.float32) * scale_y

        pixel_y, pixel_x = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32) + 0.5,
            torch.arange(width, device=self.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        camera_directions = torch.stack(
            (
                (pixel_x - cx) / fx,
                -(pixel_y - cy) / fy,
                -torch.ones_like(pixel_x),
            ),
            dim=-1,
        )
        camera_directions = torch.nn.functional.normalize(camera_directions, dim=-1)

        rotation = camera_to_world.reshape(-1, 3, 4)[0, :3, :3].to(
            self.device, torch.float32
        )
        world_directions = camera_directions @ rotation.transpose(0, 1)
        return torch.nn.functional.normalize(world_directions, dim=-1)

    def _directional_background(
        self,
        camera: Cameras,
        height: int,
        width: int,
    ) -> torch.Tensor:
        directions = self._camera_directions(camera, height, width)
        basis = _real_sh_basis(directions, self.config.background_sh_degree)
        coefficients = self.background_shs
        if self.step < self.config.background_start_step:
            coefficients = coefficients.detach()
        return torch.sigmoid(basis @ coefficients)

    def get_outputs(
        self,
        camera: Cameras,
    ) -> Dict[str, Union[torch.Tensor, List]]:
        outputs = super().get_outputs(camera)
        if not self.config.use_directional_background or "rgb" not in outputs:
            return outputs

        rgb = outputs["rgb"]
        accumulation = outputs["accumulation"]
        if not isinstance(rgb, torch.Tensor) or not isinstance(accumulation, torch.Tensor):
            return outputs

        height, width = rgb.shape[:2]
        background = self._directional_background(camera, height, width)
        outputs["rgb"] = torch.clamp(rgb + (1.0 - accumulation) * background, 0.0, 1.0)
        outputs["background"] = background
        return outputs
