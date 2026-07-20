"""Score rendered local-validation images with leaderboard-style metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


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


def evaluate(
    predictions_dir: Path,
    ground_truth_dir: Path,
    output_path: Path,
    per_image_path: Path | None,
    psnr_max: float,
    device_name: str,
) -> dict:
    if not math.isfinite(psnr_max) or psnr_max <= 0:
        raise ValueError("--psnr-max must be a positive finite number")

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

    rows: list[dict[str, float | str]] = []
    with torch.inference_mode():
        for name in sorted(ground_truth):
            prediction = _load_rgb(predictions[name], device)
            target = _load_rgb(ground_truth[name], device)
            if prediction.shape != target.shape:
                raise ValueError(
                    f"Image size mismatch for {name}: prediction={tuple(prediction.shape)}, "
                    f"ground_truth={tuple(target.shape)}"
                )

            mse = torch.mean((prediction - target) ** 2)
            psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
            ssim = structural_similarity_index_measure(prediction, target, data_range=1.0)
            lpips = lpips_metric(prediction, target)
            psnr_value = float(psnr.item())
            ssim_value = float(ssim.item())
            lpips_value = float(lpips.item())
            image_score = 100.0 * (
                0.4 * (1.0 - lpips_value)
                + 0.3 * ssim_value
                + 0.3 * min(max(psnr_value / psnr_max, 0.0), 1.0)
            )
            rows.append(
                {
                    "image_name": name,
                    "psnr": psnr_value,
                    "ssim": ssim_value,
                    "lpips": lpips_value,
                    "score": image_score,
                }
            )

    mean_psnr = sum(float(row["psnr"]) for row in rows) / len(rows)
    mean_ssim = sum(float(row["ssim"]) for row in rows) / len(rows)
    mean_lpips = sum(float(row["lpips"]) for row in rows) / len(rows)
    score = 100.0 * (
        0.4 * (1.0 - mean_lpips)
        + 0.3 * mean_ssim
        + 0.3 * min(max(mean_psnr / psnr_max, 0.0), 1.0)
    )
    report = {
        "format_version": 1,
        "prediction_dir": str(predictions_dir.resolve()),
        "ground_truth_dir": str(ground_truth_dir.resolve()),
        "device": str(device),
        "num_images": len(rows),
        "psnr": mean_psnr,
        "ssim": mean_ssim,
        "lpips": mean_lpips,
        "psnr_max": psnr_max,
        "score": score,
        "leaderboard_display": {
            "PSNR": mean_psnr,
            "SSIM": 100.0 * mean_ssim,
            "LPIPS": 100.0 * mean_lpips,
            "Score": score,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    if per_image_path is not None:
        per_image_path.parent.mkdir(parents=True, exist_ok=True)
        with per_image_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["image_name", "psnr", "ssim", "lpips", "score"])
            writer.writeheader()
            writer.writerows(rows)

    print("\nLOCAL VALIDATION RESULT")
    print("=" * 42)
    print(f"Images : {len(rows)}")
    print(f"PSNR   : {mean_psnr:.6f}")
    print(f"SSIM   : {mean_ssim:.6f}  (display: {100.0 * mean_ssim:.4f})")
    print(f"LPIPS  : {mean_lpips:.6f}  (display: {100.0 * mean_lpips:.4f})")
    print(f"Score  : {score:.5f}")
    print(f"Report : {output_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a rendered local-validation scene.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "local_validation_result.json")
    parser.add_argument("--per-image", type=Path, help="Optional CSV with metrics for every image")
    parser.add_argument("--psnr-max", type=float, default=50.0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    arguments = parser.parse_args()
    try:
        evaluate(
            arguments.predictions,
            arguments.ground_truth,
            arguments.output,
            arguments.per_image,
            arguments.psnr_max,
            arguments.device,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
