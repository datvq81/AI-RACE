"""Pure tensor helpers for Pixel-GS density statistics."""

from __future__ import annotations

import math

import torch


def clipped_footprint_coverage(
    centers: torch.Tensor,
    radii: torch.Tensor,
    image_height: int,
    image_width: int,
) -> torch.Tensor:
    """Approximate the in-frame projected footprint as an image-area ratio.

    gsplat 1.0.0 does not expose the exact per-Gaussian contributing-pixel
    count used by the official Pixel-GS rasterizer. Nerfstudio does expose the
    projected center and radius, so this computes a clipped circular footprint.
    The ``pi / 4`` factor makes a fully visible bounding box equal ``pi*r^2``.
    """
    if centers.ndim != 2 or centers.shape[-1] != 2:
        raise ValueError("centers must have shape [N, 2]")
    if radii.ndim != 1 or radii.shape[0] != centers.shape[0]:
        raise ValueError("radii must have shape [N]")
    if image_height <= 0 or image_width <= 0:
        raise ValueError("image dimensions must be positive")

    centers = centers.to(dtype=torch.float32)
    radii = radii.to(device=centers.device, dtype=torch.float32)
    finite = torch.isfinite(centers).all(dim=-1) & torch.isfinite(radii)
    centers = torch.nan_to_num(centers)
    radii = torch.clamp(radii, min=0.0)
    radii = torch.nan_to_num(radii, nan=0.0, posinf=0.0, neginf=0.0)
    x, y = centers.unbind(dim=-1)
    span_x = torch.clamp(x + radii, min=0.0, max=float(image_width)) - torch.clamp(
        x - radii, min=0.0, max=float(image_width)
    )
    span_y = torch.clamp(y + radii, min=0.0, max=float(image_height)) - torch.clamp(
        y - radii, min=0.0, max=float(image_height)
    )
    footprint_pixels = (math.pi / 4.0) * torch.clamp(span_x, min=0.0) * torch.clamp(
        span_y, min=0.0
    )
    coverage = footprint_pixels / float(image_height * image_width)
    return torch.where(finite, coverage, torch.zeros_like(coverage))


def depth_gradient_scale(
    camera_depth: torch.Tensor,
    scene_radius: float,
    gamma_depth: float = 0.37,
) -> torch.Tensor:
    """Return Pixel-GS's clipped squared camera-depth multiplier."""
    if scene_radius <= 0.0:
        raise ValueError("scene_radius must be positive")
    if gamma_depth <= 0.0:
        raise ValueError("gamma_depth must be positive")
    normalized = torch.clamp(camera_depth.to(dtype=torch.float32), min=0.0)
    normalized = normalized / float(scene_radius * gamma_depth)
    return torch.clamp(normalized.square(), min=0.0, max=1.0)


def pixel_weighted_gradient_average(
    gradients: torch.Tensor,
    coverage: torch.Tensor,
    distance_scale: torch.Tensor | None = None,
    *,
    dim: int = 0,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Average screen-space gradients using projected coverage as weights."""
    if gradients.shape != coverage.shape:
        raise ValueError("gradients and coverage must have identical shapes")
    if distance_scale is not None and distance_scale.shape != gradients.shape:
        raise ValueError("distance_scale must match gradients")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if torch.any(coverage < 0):
        raise ValueError("coverage must be non-negative")

    gradients = torch.nan_to_num(gradients.to(dtype=torch.float32))
    weights = torch.nan_to_num(coverage.to(device=gradients.device, dtype=torch.float32))
    numerator_values = gradients
    if distance_scale is not None:
        numerator_values = numerator_values * torch.nan_to_num(
            distance_scale.to(device=gradients.device, dtype=torch.float32)
        )
    numerator = torch.sum(weights * numerator_values, dim=dim)
    denominator = torch.sum(weights, dim=dim)
    return torch.where(
        denominator > epsilon,
        numerator / torch.clamp(denominator, min=epsilon),
        torch.zeros_like(numerator),
    )
