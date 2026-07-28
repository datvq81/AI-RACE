"""Synthetic tests for Pixel-GS weighting and distance scaling."""

from __future__ import annotations

import math
import unittest

import torch

from var_nvs.radical.pixel_gradient import (
    clipped_footprint_coverage,
    depth_gradient_scale,
    pixel_weighted_gradient_average,
)


class PixelGradientTest(unittest.TestCase):
    def test_weighted_average_favors_large_coverage_view(self) -> None:
        gradients = torch.tensor([[1.0], [3.0]])
        coverage = torch.tensor([[1.0], [3.0]])
        result = pixel_weighted_gradient_average(gradients, coverage)
        torch.testing.assert_close(result, torch.tensor([2.5]))

    def test_uniform_coverage_matches_stock_average(self) -> None:
        gradients = torch.tensor([[1.0, 4.0], [3.0, 2.0]])
        coverage = torch.ones_like(gradients)
        result = pixel_weighted_gradient_average(gradients, coverage)
        torch.testing.assert_close(result, gradients.mean(dim=0))

    def test_distance_scale_suppresses_near_camera_gradient(self) -> None:
        depths = torch.tensor([0.185, 0.37, 0.74])
        result = depth_gradient_scale(depths, scene_radius=1.0, gamma_depth=0.37)
        torch.testing.assert_close(result, torch.tensor([0.25, 1.0, 1.0]))

    def test_distance_scale_only_changes_numerator(self) -> None:
        gradients = torch.tensor([[4.0], [4.0]])
        coverage = torch.ones_like(gradients)
        distance = torch.tensor([[0.25], [1.0]])
        result = pixel_weighted_gradient_average(
            gradients,
            coverage,
            distance_scale=distance,
        )
        torch.testing.assert_close(result, torch.tensor([2.5]))

    def test_footprint_is_area_ratio_and_clips_at_frame(self) -> None:
        centers = torch.tensor([[5.0, 5.0], [0.0, 5.0]])
        radii = torch.tensor([2.0, 2.0])
        result = clipped_footprint_coverage(centers, radii, 10, 10)
        self.assertAlmostEqual(float(result[0]), math.pi * 4.0 / 100.0, places=6)
        self.assertAlmostEqual(float(result[1]), math.pi * 2.0 / 100.0, places=6)

    def test_nonfinite_footprint_has_zero_weight(self) -> None:
        centers = torch.tensor([[float("nan"), 5.0], [5.0, 5.0]])
        radii = torch.tensor([2.0, float("inf")])
        result = clipped_footprint_coverage(centers, radii, 10, 10)
        torch.testing.assert_close(result, torch.zeros(2))


if __name__ == "__main__":
    unittest.main()
