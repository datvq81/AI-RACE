"""Diagnose what limits a rendered local-validation experiment.

This script deliberately does not change the leaderboard-style evaluator. It
uses ground truth to compute several *oracles* that are valid for diagnosis
only:

* small integer image shifts (camera/intrinsics alignment),
* per-image affine RGB correction (exposure/white balance),
* light Gaussian blur (aliasing/over-sharp rendering), and
* error concentration on strong ground-truth edges (thin structures).

Oracle-corrected images must never be used for a competition submission because
their corrections are fitted using local-validation ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

try:
    from torchmetrics.functional.image import structural_similarity_index_measure
except ImportError:  # Compatibility with older torchmetrics bundled by Nerfstudio.
    from torchmetrics.functional import structural_similarity_index_measure

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
except ImportError:
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "data" / "local_validation"
DEFAULT_PREDICTION_ROOT = PROJECT_ROOT / "data" / "local_validation_predictions"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "outputs" / "local_validation_reports"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _safe_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{label} may contain only letters, digits, dot, underscore, and hyphen")
    return value


def _image_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Image directory not found: {directory}")
    files = {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if not files:
        raise ValueError(f"No images found in {directory}")
    return files


def _load_rgb(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32).copy() / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _score(psnr: float, ssim: float, lpips: float, psnr_max: float) -> float:
    return 100.0 * (
        0.4 * (1.0 - lpips)
        + 0.3 * ssim
        + 0.3 * min(max(psnr / psnr_max, 0.0), 1.0)
    )


def _metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    lpips_metric: LearnedPerceptualImagePatchSimilarity,
    psnr_max: float,
) -> dict[str, float]:
    mse = torch.mean((prediction - target) ** 2)
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
    ssim = structural_similarity_index_measure(prediction, target, data_range=1.0)
    # Reset makes the per-image behavior explicit across torchmetrics versions.
    lpips_metric.reset()
    lpips = lpips_metric(prediction, target)
    lpips_metric.reset()
    values = {
        "psnr": float(psnr.item()),
        "ssim": float(ssim.item()),
        "lpips": float(lpips.item()),
    }
    values["score"] = _score(values["psnr"], values["ssim"], values["lpips"], psnr_max)
    return values


def _aggregate(rows: list[dict[str, float]], psnr_max: float) -> dict[str, float]:
    if not rows:
        raise ValueError("Cannot aggregate an empty metric list")
    result = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in ("psnr", "ssim", "lpips")
    }
    result["score"] = _score(result["psnr"], result["ssim"], result["lpips"], psnr_max)
    return result


def _metric_delta(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("psnr", "ssim", "lpips", "score")
    }


def _fixed_shift_crop(
    prediction: torch.Tensor,
    target: torch.Tensor,
    dx: int,
    dy: int,
    margin: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    height, width = prediction.shape[-2:]
    if target.shape[-2:] != (height, width):
        raise ValueError(
            f"Prediction/ground-truth size mismatch: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if margin < 0 or abs(dx) > margin or abs(dy) > margin:
        raise ValueError("Shift must be inside the fixed crop margin")
    if height <= 2 * margin or width <= 2 * margin:
        raise ValueError(
            f"Image {width}x{height} is too small for an alignment margin of {margin}"
        )
    target_crop = target[:, :, margin : height - margin, margin : width - margin]
    prediction_crop = prediction[
        :,
        :,
        margin + dy : height - margin + dy,
        margin + dx : width - margin + dx,
    ]
    return prediction_crop, target_crop


def _luma(image: torch.Tensor) -> torch.Tensor:
    weights = image.new_tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
    return torch.sum(image * weights, dim=1, keepdim=True)


def _search_integer_shift(
    prediction: torch.Tensor,
    target: torch.Tensor,
    max_shift: int,
) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
    losses: dict[tuple[int, int], float] = {}
    prediction_luma = _luma(prediction)
    target_luma = _luma(target)
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            shifted, target_crop = _fixed_shift_crop(
                prediction_luma, target_luma, dx, dy, max_shift
            )
            losses[(dx, dy)] = float(torch.mean((shifted - target_crop) ** 2).item())
    best_shift = min(losses, key=losses.__getitem__)
    return best_shift, losses


def _fit_affine_rgb(
    prediction: torch.Tensor,
    target: torch.Tensor,
    gain_min: float,
    gain_max: float,
    bias_limit: float,
) -> tuple[torch.Tensor, list[float], list[float]]:
    x = prediction.permute(0, 2, 3, 1).reshape(-1, 3)
    y = target.permute(0, 2, 3, 1).reshape(-1, 3)
    x_mean = torch.mean(x, dim=0)
    y_mean = torch.mean(y, dim=0)
    variance = torch.mean((x - x_mean) ** 2, dim=0)
    covariance = torch.mean((x - x_mean) * (y - y_mean), dim=0)
    gain = covariance / torch.clamp(variance, min=1e-8)
    gain = torch.clamp(gain, min=gain_min, max=gain_max)
    bias = torch.clamp(y_mean - gain * x_mean, min=-bias_limit, max=bias_limit)
    corrected = torch.clamp(
        prediction * gain.view(1, 3, 1, 1) + bias.view(1, 3, 1, 1),
        0.0,
        1.0,
    )
    return (
        corrected,
        [float(value) for value in gain.detach().cpu().tolist()],
        [float(value) for value in bias.detach().cpu().tolist()],
    )


def _gaussian_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return image
    radius = max(1, math.ceil(3.0 * sigma))
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / torch.sum(kernel_1d)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    channels = image.shape[1]
    weight = kernel_2d.view(1, 1, *kernel_2d.shape).repeat(channels, 1, 1, 1)
    padded = F.pad(image, (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, weight, groups=channels)


def _edge_statistics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    edge_fraction: float,
) -> dict[str, float]:
    target_luma = _luma(target)
    sobel_x = target.new_tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    padded = F.pad(target_luma, (1, 1, 1, 1), mode="reflect")
    gradient_x = F.conv2d(padded, sobel_x)
    gradient_y = F.conv2d(padded, sobel_y)
    magnitude = torch.sqrt(gradient_x**2 + gradient_y**2 + 1e-12)
    threshold = torch.quantile(magnitude.flatten(), 1.0 - edge_fraction)
    edge_mask = magnitude >= threshold
    pixel_mse = torch.mean((prediction - target) ** 2, dim=1, keepdim=True)
    pixel_mae = torch.mean(torch.abs(prediction - target), dim=1, keepdim=True)
    flat_mask = ~edge_mask

    edge_count = int(torch.sum(edge_mask).item())
    flat_count = int(torch.sum(flat_mask).item())
    edge_sse = float(torch.sum(pixel_mse[edge_mask]).item())
    flat_sse = float(torch.sum(pixel_mse[flat_mask]).item())
    edge_sae = float(torch.sum(pixel_mae[edge_mask]).item())
    flat_sae = float(torch.sum(pixel_mae[flat_mask]).item())
    total_sse = edge_sse + flat_sse
    actual_fraction = edge_count / max(edge_count + flat_count, 1)
    error_share = edge_sse / max(total_sse, 1e-12)
    return {
        "edge_count": float(edge_count),
        "flat_count": float(flat_count),
        "edge_sse": edge_sse,
        "flat_sse": flat_sse,
        "edge_sae": edge_sae,
        "flat_sae": flat_sae,
        "edge_fraction": actual_fraction,
        "edge_mse": edge_sse / max(edge_count, 1),
        "flat_mse": flat_sse / max(flat_count, 1),
        "edge_mae": edge_sae / max(edge_count, 1),
        "flat_mae": flat_sae / max(flat_count, 1),
        "edge_error_share": error_share,
        "edge_error_concentration": error_share / max(actual_fraction, 1e-12),
    }


def _merge_edge_statistics(rows: list[dict[str, float]]) -> dict[str, float]:
    totals = {
        key: sum(row[key] for row in rows)
        for key in ("edge_count", "flat_count", "edge_sse", "flat_sse", "edge_sae", "flat_sae")
    }
    edge_count = totals["edge_count"]
    flat_count = totals["flat_count"]
    total_count = edge_count + flat_count
    total_sse = totals["edge_sse"] + totals["flat_sse"]
    edge_fraction = edge_count / max(total_count, 1.0)
    error_share = totals["edge_sse"] / max(total_sse, 1e-12)
    return {
        "edge_fraction": edge_fraction,
        "edge_mse": totals["edge_sse"] / max(edge_count, 1.0),
        "flat_mse": totals["flat_sse"] / max(flat_count, 1.0),
        "edge_mae": totals["edge_sae"] / max(edge_count, 1.0),
        "flat_mae": totals["flat_sae"] / max(flat_count, 1.0),
        "edge_error_share": error_share,
        "edge_error_concentration": error_share / max(edge_fraction, 1e-12),
    }


def _save_error_panel(
    prediction_path: Path,
    target_path: Path,
    output_path: Path,
) -> None:
    with Image.open(prediction_path) as image:
        prediction = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    with Image.open(target_path) as image:
        target = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    if prediction.shape != target.shape:
        raise ValueError(
            f"Image size mismatch for error map: {prediction.shape} vs {target.shape}"
        )
    error = np.mean(np.abs(prediction - target), axis=2)
    scale = float(np.quantile(error, 0.99))
    normalized = np.clip(error / max(scale, 1e-6), 0.0, 1.0)
    heatmap = np.stack(
        [
            normalized,
            np.sqrt(normalized) * 0.55,
            np.zeros_like(normalized),
        ],
        axis=2,
    )
    separator = np.ones((prediction.shape[0], 4, 3), dtype=np.float32)
    panel = np.concatenate([target, separator, prediction, separator, heatmap], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(panel * 255.0).astype(np.uint8)).save(output_path)


def _recommendations(
    alignment: dict[str, Any],
    color: dict[str, Any],
    blur: dict[str, Any],
    regions: dict[str, float],
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    global_shift_gain = float(alignment["global_shift"]["delta"]["score"])
    per_image_shift_gain = float(alignment["per_image_oracle"]["delta"]["score"])
    color_gain = float(color["delta"]["score"])
    blur_gain = float(blur["best_global"]["delta"]["score"])
    blur_oracle_gain = float(blur["per_image_oracle"]["delta"]["score"])
    best_shift = alignment["global_shift"]["offset"]

    if global_shift_gain >= 0.5 and (best_shift["dx"] != 0 or best_shift["dy"] != 0):
        recommendations.append(
            {
                "priority": "camera_alignment",
                "evidence": (
                    f"One global integer shift gains {global_shift_gain:.3f} score "
                    f"at dx={best_shift['dx']}, dy={best_shift['dy']}."
                ),
                "next_step": "Audit COLMAP-to-transforms intrinsics, pixel centers, resizing and distortion.",
            }
        )
    elif per_image_shift_gain >= 0.5:
        recommendations.append(
            {
                "priority": "camera_pose",
                "evidence": (
                    f"Per-image shift oracle gains {per_image_shift_gain:.3f}, while the "
                    f"best global shift gains only {global_shift_gain:.3f}."
                ),
                "next_step": "Inspect pose residuals and test a conservative camera optimizer.",
            }
        )

    if color_gain >= 0.3:
        recommendations.append(
            {
                "priority": "appearance_exposure",
                "evidence": f"Per-image affine RGB oracle gains {color_gain:.3f} score.",
                "next_step": (
                    "Inspect fitted gains/biases; only then revisit a novel-view-safe appearance model."
                ),
            }
        )

    if blur_gain >= 0.2 or blur_oracle_gain >= 0.3:
        recommendations.append(
            {
                "priority": "aliasing",
                "evidence": (
                    f"Best global blur gains {blur_gain:.3f}; per-image blur oracle gains "
                    f"{blur_oracle_gain:.3f}."
                ),
                "next_step": "Test Mip-Splatting-style 3D smoothing and 2D filtering.",
            }
        )

    edge_concentration = float(regions["edge_error_concentration"])
    if edge_concentration >= 1.25:
        recommendations.append(
            {
                "priority": "thin_structure_densification",
                "evidence": (
                    f"Strong edges contain {100.0 * regions['edge_error_share']:.1f}% of "
                    f"squared error in {100.0 * regions['edge_fraction']:.1f}% of pixels "
                    f"(concentration {edge_concentration:.2f}x)."
                ),
                "next_step": "Implement residual/edge-aware AbsGrad densification against a D0/F0 control.",
            }
        )

    if not recommendations:
        recommendations.append(
            {
                "priority": "geometry_densification",
                "evidence": "No simple alignment, color or blur oracle produced a material score gain.",
                "next_step": "Audit existing absgrad use, then test residual-aware densification.",
            }
        )
    return recommendations


def diagnose(
    predictions_dir: Path,
    ground_truth_dir: Path,
    output_path: Path,
    per_image_path: Path,
    error_map_dir: Path,
    save_worst: int,
    psnr_max: float,
    device_name: str,
    max_shift: int,
    blur_sigmas: list[float],
    edge_fraction: float,
    gain_min: float,
    gain_max: float,
    bias_limit: float,
) -> dict[str, Any]:
    if not math.isfinite(psnr_max) or psnr_max <= 0:
        raise ValueError("--psnr-max must be a positive finite number")
    if max_shift < 0:
        raise ValueError("--max-shift must be non-negative")
    if not 0.0 < edge_fraction < 1.0:
        raise ValueError("--edge-fraction must be between zero and one")
    if gain_min <= 0.0 or gain_max < gain_min:
        raise ValueError("Color gain bounds are invalid")
    if bias_limit < 0.0:
        raise ValueError("--color-bias-limit must be non-negative")
    if any(not math.isfinite(sigma) or sigma <= 0.0 for sigma in blur_sigmas):
        raise ValueError("--blur-sigmas values must be positive and finite")
    blur_sigmas = sorted(set(blur_sigmas))

    predictions = _image_files(predictions_dir)
    ground_truth = _image_files(ground_truth_dir)
    missing = sorted(set(ground_truth) - set(predictions))
    unexpected = sorted(set(predictions) - set(ground_truth))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing predictions={missing[:8]}")
        if unexpected:
            details.append(f"unexpected predictions={unexpected[:8]}")
        raise ValueError("Prediction/ground-truth filenames do not match: " + "; ".join(details))

    device = _resolve_device(device_name)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True, reduction="mean"
    ).to(device)
    lpips_metric.eval()

    names = sorted(ground_truth)
    baseline_rows: list[dict[str, float]] = []
    central_rows: list[dict[str, float]] = []
    per_image_shift_rows: list[dict[str, float]] = []
    color_rows: list[dict[str, float]] = []
    edge_rows: list[dict[str, float]] = []
    blur_rows: dict[float, list[dict[str, float]]] = {0.0: baseline_rows}
    blur_rows.update({sigma: [] for sigma in blur_sigmas})
    shift_loss_sums = {
        (dx, dy): 0.0
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
    }
    per_image: list[dict[str, Any]] = []

    print(f"[*] Device       : {device}")
    print(f"[*] Images       : {len(names)}")
    print(f"[*] Shift search : +/-{max_shift} px")
    print(f"[*] Blur sigmas  : {blur_sigmas}")
    print("[*] Running diagnostic oracles...", flush=True)

    with torch.inference_mode():
        for index, name in enumerate(names, start=1):
            prediction = _load_rgb(predictions[name], device)
            target = _load_rgb(ground_truth[name], device)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Image size mismatch for {name}: prediction={tuple(prediction.shape)}, "
                    f"ground_truth={tuple(target.shape)}"
                )

            baseline = _metrics(prediction, target, lpips_metric, psnr_max)
            baseline_rows.append(baseline)

            best_shift, shift_losses = _search_integer_shift(
                prediction, target, max_shift
            )
            for shift, loss in shift_losses.items():
                shift_loss_sums[shift] += loss
            central_prediction, central_target = _fixed_shift_crop(
                prediction, target, 0, 0, max_shift
            )
            central = _metrics(central_prediction, central_target, lpips_metric, psnr_max)
            central_rows.append(central)
            shifted_prediction, shifted_target = _fixed_shift_crop(
                prediction, target, best_shift[0], best_shift[1], max_shift
            )
            if best_shift == (0, 0):
                shifted_metrics = central
            else:
                shifted_metrics = _metrics(
                    shifted_prediction, shifted_target, lpips_metric, psnr_max
                )
            per_image_shift_rows.append(shifted_metrics)

            corrected, gains, biases = _fit_affine_rgb(
                prediction, target, gain_min, gain_max, bias_limit
            )
            color_metrics = _metrics(corrected, target, lpips_metric, psnr_max)
            color_rows.append(color_metrics)

            image_blur_metrics: dict[float, dict[str, float]] = {0.0: baseline}
            for sigma in blur_sigmas:
                blurred = _gaussian_blur(prediction, sigma)
                candidate = _metrics(blurred, target, lpips_metric, psnr_max)
                blur_rows[sigma].append(candidate)
                image_blur_metrics[sigma] = candidate
            best_image_sigma = max(
                image_blur_metrics, key=lambda sigma: image_blur_metrics[sigma]["score"]
            )

            edge = _edge_statistics(prediction, target, edge_fraction)
            edge_rows.append(edge)
            per_image.append(
                {
                    "image_name": name,
                    "baseline": baseline,
                    "shift": {
                        "dx": best_shift[0],
                        "dy": best_shift[1],
                        "search_mse": shift_losses[best_shift],
                        "metrics": shifted_metrics,
                    },
                    "color": {
                        "gain": gains,
                        "bias": biases,
                        "metrics": color_metrics,
                    },
                    "blur": {
                        "best_sigma": best_image_sigma,
                        "metrics": image_blur_metrics[best_image_sigma],
                    },
                    "edge": {
                        key: edge[key]
                        for key in (
                            "edge_fraction",
                            "edge_mse",
                            "flat_mse",
                            "edge_mae",
                            "flat_mae",
                            "edge_error_share",
                            "edge_error_concentration",
                        )
                    },
                }
            )
            print(f"    [{index:>3}/{len(names)}] {name}", flush=True)

    baseline_summary = _aggregate(baseline_rows, psnr_max)
    central_summary = _aggregate(central_rows, psnr_max)
    best_global_shift = min(shift_loss_sums, key=shift_loss_sums.__getitem__)
    global_shift_rows: list[dict[str, float]] = []
    if best_global_shift == (0, 0):
        global_shift_rows = central_rows
    else:
        with torch.inference_mode():
            for name in names:
                prediction = _load_rgb(predictions[name], device)
                target = _load_rgb(ground_truth[name], device)
                shifted_prediction, shifted_target = _fixed_shift_crop(
                    prediction,
                    target,
                    best_global_shift[0],
                    best_global_shift[1],
                    max_shift,
                )
                global_shift_rows.append(
                    _metrics(shifted_prediction, shifted_target, lpips_metric, psnr_max)
                )
    global_shift_summary = _aggregate(global_shift_rows, psnr_max)
    per_image_shift_summary = _aggregate(per_image_shift_rows, psnr_max)
    color_summary = _aggregate(color_rows, psnr_max)
    edge_summary = _merge_edge_statistics(edge_rows)

    blur_summaries = {
        sigma: _aggregate(rows, psnr_max)
        for sigma, rows in blur_rows.items()
    }
    best_global_sigma = max(
        blur_summaries, key=lambda sigma: blur_summaries[sigma]["score"]
    )
    per_image_blur_rows = [
        item["blur"]["metrics"]
        for item in per_image
    ]
    per_image_blur_summary = _aggregate(per_image_blur_rows, psnr_max)

    alignment_report = {
        "fixed_crop_margin_pixels": max_shift,
        "selection_metric": "luma_mse",
        "central_crop_baseline": central_summary,
        "global_shift": {
            "offset": {"dx": best_global_shift[0], "dy": best_global_shift[1]},
            "metrics": global_shift_summary,
            "delta": _metric_delta(global_shift_summary, central_summary),
        },
        "per_image_oracle": {
            "metrics": per_image_shift_summary,
            "delta": _metric_delta(per_image_shift_summary, central_summary),
        },
    }
    color_report = {
        "fit": "per-image, per-channel target = gain * prediction + bias",
        "gain_bounds": [gain_min, gain_max],
        "bias_bounds": [-bias_limit, bias_limit],
        "metrics": color_summary,
        "delta": _metric_delta(color_summary, baseline_summary),
    }
    blur_report = {
        "candidates_sigma": [0.0, *blur_sigmas],
        "global_candidates": {
            str(sigma): {
                "metrics": metrics,
                "delta": _metric_delta(metrics, baseline_summary),
            }
            for sigma, metrics in blur_summaries.items()
        },
        "best_global": {
            "sigma": best_global_sigma,
            "metrics": blur_summaries[best_global_sigma],
            "delta": _metric_delta(blur_summaries[best_global_sigma], baseline_summary),
        },
        "per_image_oracle": {
            "metrics": per_image_blur_summary,
            "delta": _metric_delta(per_image_blur_summary, baseline_summary),
        },
    }
    recommendations = _recommendations(
        alignment_report, color_report, blur_report, edge_summary
    )
    worst = sorted(per_image, key=lambda item: item["baseline"]["score"])[:save_worst]

    report: dict[str, Any] = {
        "format_version": 1,
        "warning": (
            "All corrected results are ground-truth oracles for diagnosis only; "
            "do not use corrected images in a submission."
        ),
        "prediction_dir": str(predictions_dir.resolve()),
        "ground_truth_dir": str(ground_truth_dir.resolve()),
        "device": str(device),
        "num_images": len(names),
        "psnr_max": psnr_max,
        "baseline": baseline_summary,
        "alignment": alignment_report,
        "color_affine_oracle": color_report,
        "blur_oracle": blur_report,
        "edge_analysis": edge_summary,
        "recommendations": recommendations,
        "worst_images": [
            {
                "image_name": item["image_name"],
                "baseline_score": item["baseline"]["score"],
            }
            for item in worst
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    per_image_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_name",
        "baseline_psnr",
        "baseline_ssim",
        "baseline_lpips",
        "baseline_score",
        "shift_dx",
        "shift_dy",
        "shift_score",
        "color_gain_r",
        "color_gain_g",
        "color_gain_b",
        "color_bias_r",
        "color_bias_g",
        "color_bias_b",
        "color_score",
        "best_blur_sigma",
        "blur_score",
        "edge_fraction",
        "edge_mse",
        "flat_mse",
        "edge_error_share",
        "edge_error_concentration",
    ]
    with per_image_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in per_image:
            writer.writerow(
                {
                    "image_name": item["image_name"],
                    "baseline_psnr": item["baseline"]["psnr"],
                    "baseline_ssim": item["baseline"]["ssim"],
                    "baseline_lpips": item["baseline"]["lpips"],
                    "baseline_score": item["baseline"]["score"],
                    "shift_dx": item["shift"]["dx"],
                    "shift_dy": item["shift"]["dy"],
                    "shift_score": item["shift"]["metrics"]["score"],
                    "color_gain_r": item["color"]["gain"][0],
                    "color_gain_g": item["color"]["gain"][1],
                    "color_gain_b": item["color"]["gain"][2],
                    "color_bias_r": item["color"]["bias"][0],
                    "color_bias_g": item["color"]["bias"][1],
                    "color_bias_b": item["color"]["bias"][2],
                    "color_score": item["color"]["metrics"]["score"],
                    "best_blur_sigma": item["blur"]["best_sigma"],
                    "blur_score": item["blur"]["metrics"]["score"],
                    "edge_fraction": item["edge"]["edge_fraction"],
                    "edge_mse": item["edge"]["edge_mse"],
                    "flat_mse": item["edge"]["flat_mse"],
                    "edge_error_share": item["edge"]["edge_error_share"],
                    "edge_error_concentration": item["edge"]["edge_error_concentration"],
                }
            )

    if save_worst > 0:
        error_map_dir.mkdir(parents=True, exist_ok=True)
        for rank, item in enumerate(worst, start=1):
            name = item["image_name"]
            output_name = f"{rank:02d}_{Path(name).stem}_gt_pred_error.png"
            _save_error_panel(
                predictions[name],
                ground_truth[name],
                error_map_dir / output_name,
            )

    print("\nLOCAL VALIDATION DIAGNOSTIC")
    print("=" * 52)
    print(f"Baseline score       : {baseline_summary['score']:.5f}")
    print(
        "Global shift         : "
        f"dx={best_global_shift[0]:+d}, dy={best_global_shift[1]:+d}, "
        f"gain={alignment_report['global_shift']['delta']['score']:+.5f}"
    )
    print(
        "Per-image shift      : "
        f"gain={alignment_report['per_image_oracle']['delta']['score']:+.5f}"
    )
    print(f"Affine RGB oracle    : gain={color_report['delta']['score']:+.5f}")
    print(
        "Best global blur     : "
        f"sigma={best_global_sigma:g}, "
        f"gain={blur_report['best_global']['delta']['score']:+.5f}"
    )
    print(
        "Edge error           : "
        f"{100.0 * edge_summary['edge_error_share']:.2f}% error / "
        f"{100.0 * edge_summary['edge_fraction']:.2f}% pixels "
        f"({edge_summary['edge_error_concentration']:.2f}x)"
    )
    print(f"Priority             : {recommendations[0]['priority']}")
    print(f"Report               : {output_path}")
    print(f"Per-image CSV        : {per_image_path}")
    if save_worst > 0:
        print(f"Error panels         : {error_map_dir}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run alignment, exposure, blur and edge diagnostics on local validation."
    )
    parser.add_argument("--scene", required=True, help="Scene name, for example HCM0421")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument(
        "--tag",
        help="Experiment tag without the localval_<scene>_ prefix",
    )
    identity.add_argument(
        "--experiment",
        help="Complete experiment directory name, for example localval_HCM0421_F0_...",
    )
    parser.add_argument("--predictions", type=Path, help="Override prediction directory")
    parser.add_argument("--ground-truth", type=Path, help="Override ground-truth directory")
    parser.add_argument("--output", type=Path, help="Override diagnostics JSON path")
    parser.add_argument("--per-image", type=Path, help="Override per-image CSV path")
    parser.add_argument("--error-map-dir", type=Path, help="Override error-panel directory")
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--psnr-max", type=float, default=50.0)
    parser.add_argument("--max-shift", type=int, default=3)
    parser.add_argument("--blur-sigmas", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=0.20,
        help="Fraction of strongest ground-truth Sobel pixels treated as edges",
    )
    parser.add_argument("--color-gain-min", type=float, default=0.5)
    parser.add_argument("--color-gain-max", type=float, default=1.5)
    parser.add_argument("--color-bias-limit", type=float, default=0.25)
    parser.add_argument(
        "--save-worst",
        type=int,
        default=10,
        help="Number of lowest-score GT/prediction/error panels; zero disables them",
    )
    arguments = parser.parse_args()

    try:
        scene = _safe_name(arguments.scene, "scene")
        if arguments.experiment:
            experiment = _safe_name(arguments.experiment, "experiment")
        else:
            tag = _safe_name(arguments.tag, "tag")
            experiment = f"localval_{scene}_{tag}"
        predictions = (
            arguments.predictions.resolve()
            if arguments.predictions
            else arguments.prediction_root.resolve() / experiment
        )
        ground_truth = (
            arguments.ground_truth.resolve()
            if arguments.ground_truth
            else arguments.validation_root.resolve() / scene / "local_gt" / "images"
        )
        report_dir = arguments.report_root.resolve() / experiment
        output = arguments.output.resolve() if arguments.output else report_dir / "diagnostics.json"
        per_image = (
            arguments.per_image.resolve()
            if arguments.per_image
            else report_dir / "diagnostics_per_image.csv"
        )
        error_map_dir = (
            arguments.error_map_dir.resolve()
            if arguments.error_map_dir
            else report_dir / "error_maps"
        )
        if arguments.save_worst < 0:
            raise ValueError("--save-worst must be non-negative")
        diagnose(
            predictions_dir=predictions,
            ground_truth_dir=ground_truth,
            output_path=output,
            per_image_path=per_image,
            error_map_dir=error_map_dir,
            save_worst=arguments.save_worst,
            psnr_max=arguments.psnr_max,
            device_name=arguments.device,
            max_shift=arguments.max_shift,
            blur_sigmas=arguments.blur_sigmas,
            edge_fraction=arguments.edge_fraction,
            gain_min=arguments.color_gain_min,
            gain_max=arguments.color_gain_max,
            bias_limit=arguments.color_bias_limit,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
