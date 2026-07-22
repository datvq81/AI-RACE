"""Tự động convert, train và render toàn bộ scene."""

import argparse
import csv
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent
# private_test1 is the authoritative leaderboard scene set. Every directory
# under it is included, including bonsai/chair when they are present there.
DATA_DIRS = [PROJECT_ROOT / "data/private_test1"]
PRED_DIR = PROJECT_ROOT / "data/predictions"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
RENDER_SCRIPT = PROJECT_ROOT / "scripts/render_test_pose.py"
CONVERT_SCRIPT = PROJECT_ROOT / "scripts/convert_var_to_nerfstudio.py"

RENDERER_VERSION = "2.0-redistort"
MANIFEST_FILENAME = ".render_manifest.json"


@dataclass(frozen=True)
class PipelinePreset:
    """A reproducible training configuration for every submission scene."""

    method: str
    iterations: int
    train_args: Tuple[str, ...]


PRESETS = {
    "a2": PipelinePreset(
        method="splatfacto-big",
        iterations=30000,
        train_args=(
            "--pipeline.model.sh-degree", "3",
            "--pipeline.model.use-scale-regularization", "False",
            "--pipeline.model.rasterize-mode", "classic",
        ),
    ),
    "d1b": PipelinePreset(
        method="splatfacto-perceptual",
        iterations=20000,
        train_args=(
            "--pipeline.model.sh-degree", "3",
            "--pipeline.model.use-scale-regularization", "False",
            "--pipeline.model.rasterize-mode", "classic",
            "--pipeline.model.lpips-loss-start-step", "6000",
            "--pipeline.model.lpips-loss-weight", "0.03",
        ),
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_all_scenes() -> Dict[str, Path]:
    """Lấy đường dẫn của toàn bộ scene."""
    scenes: Dict[str, Path] = {}
    for data_dir in DATA_DIRS:
        if not data_dir.is_dir():
            continue
        for scene_path in data_dir.iterdir():
            if scene_path.is_dir() and not scene_path.name.startswith("."):
                scenes[scene_path.name] = scene_path
    return scenes


def _find_pose_csv(scene_path: Path) -> Optional[Path]:
    for filename in ("test_poses.csv", "test_pose.csv"):
        csv_path = scene_path / "test" / filename
        if csv_path.is_file():
            return csv_path.resolve()
    return None


def _read_expected_image_names(csv_path: Path) -> List[str]:
    names: List[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if "image_name" not in (reader.fieldnames or []):
            raise ValueError(f"CSV thiếu cột image_name: {csv_path}")
        for row in reader:
            image_name = (row.get("image_name") or "").strip().replace("\\", "/")
            if not image_name:
                raise ValueError(f"CSV có image_name trống tại dòng {reader.line_num}: {csv_path}")
            names.append(PurePosixPath(image_name).as_posix())
    if not names:
        raise ValueError(f"CSV không chứa test pose nào: {csv_path}")
    return names


def _find_camera_metadata(csv_path: Path) -> Optional[Path]:
    for candidate in (
        csv_path.parent.parent / "transforms.json",
        csv_path.parent / "transforms.json",
    ):
        if candidate.is_file():
            return candidate.resolve()
    return None


def _render_output_is_current(
    prediction_dir: Path,
    csv_path: Path,
    config_path: Path,
) -> Tuple[bool, str]:
    """Chỉ resume khi manifest và toàn bộ ảnh khớp đúng đầu vào hiện tại."""
    manifest_path = prediction_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        return False, "chưa có render manifest"

    try:
        with manifest_path.open("r", encoding="utf-8-sig") as file:
            manifest = json.load(file)
        image_names = _read_expected_image_names(csv_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return False, f"manifest/CSV không hợp lệ: {error}"

    checks = (
        (manifest.get("schema_version") == 1, "schema manifest đã cũ"),
        (
            manifest.get("renderer_version") == RENDERER_VERSION,
            "phiên bản renderer đã thay đổi",
        ),
        (
            manifest.get("renderer_sha256") == _sha256_file(RENDER_SCRIPT),
            "mã render_test_pose.py đã thay đổi",
        ),
        (
            manifest.get("redistortion_enabled") is True,
            "lần render trước chưa bật redistortion",
        ),
        (
            manifest.get("config_path") == str(config_path.resolve()),
            "config render đã thay đổi",
        ),
        (
            manifest.get("config_sha256") == _sha256_file(config_path),
            "nội dung config đã thay đổi",
        ),
        (
            manifest.get("csv_path") == str(csv_path.resolve()),
            "đường dẫn CSV đã thay đổi",
        ),
        (
            manifest.get("csv_sha256") == _sha256_file(csv_path),
            "nội dung CSV đã thay đổi",
        ),
        (
            manifest.get("image_count") == len(image_names),
            "số lượng ảnh không khớp CSV",
        ),
        (
            manifest.get("image_names") == image_names,
            "danh sách ảnh không khớp CSV",
        ),
    )
    for passed, reason in checks:
        if not passed:
            return False, reason

    camera_metadata = _find_camera_metadata(csv_path)
    if camera_metadata is None:
        return False, "không tìm thấy transforms.json chứa distortion"
    if manifest.get("camera_metadata_path") != str(camera_metadata):
        return False, "nguồn thông số camera đã thay đổi"
    if manifest.get("camera_metadata_sha256") != _sha256_file(camera_metadata):
        return False, "thông số camera/distortion đã thay đổi"

    for image_name in image_names:
        image_path = prediction_dir.joinpath(*PurePosixPath(image_name).parts)
        try:
            if not image_path.is_file() or image_path.stat().st_size == 0:
                return False, f"ảnh thiếu hoặc rỗng: {image_name}"
        except OSError as error:
            return False, f"không đọc được ảnh {image_name}: {error}"

    return True, "manifest và toàn bộ ảnh đều hợp lệ"


def _find_configs(experiment_name: str) -> List[Path]:
    pattern = str(OUTPUTS_DIR / experiment_name / "*" / "*" / "config.yml")
    configs = [Path(path).resolve() for path in glob.glob(pattern)]
    return sorted(configs, key=lambda path: path.stat().st_mtime, reverse=True)


def _has_checkpoint(config_path: Path) -> bool:
    model_dir = config_path.parent / "nerfstudio_models"
    return model_dir.is_dir() and any(model_dir.glob("*.ckpt"))


def _run_checked(command: List[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _budget_label(iterations: int) -> str:
    return f"{iterations // 1000}k" if iterations % 1000 == 0 else f"{iterations}it"


def _safe_run_tag(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError("run tag may contain only letters, digits, dot, underscore, and hyphen")
    return value


def run_pipeline(
    preset_name: str = "d1b",
    iterations_override: Optional[int] = None,
    seed: int = 42,
    run_tag_override: Optional[str] = None,
) -> int:
    preset = PRESETS[preset_name]
    iterations = iterations_override or preset.iterations
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    run_tag = _safe_run_tag(
        run_tag_override or f"{preset_name}_{_budget_label(iterations)}_s{seed}"
    )

    print(f"[*] Preset     : {preset_name}")
    print(f"[*] Method     : {preset.method}")
    print(f"[*] Iterations : {iterations}")
    print(f"[*] Seed       : {seed}")
    print(f"[*] Run tag    : {run_tag}")

    print("\n" + "=" * 60)
    print(" KHỞI ĐỘNG AUTO-PIPELINE HUẤN LUYỆN 3D")
    print("=" * 60)

    print("[*] Đang rà soát dữ liệu COLMAP và sinh transforms.json...")
    try:
        _run_checked([sys.executable, str(CONVERT_SCRIPT)])
    except (OSError, subprocess.CalledProcessError) as error:
        print(f"[!] Không thể chạy bước convert tự động: {error}")

    scenes = get_all_scenes()
    if not scenes:
        print(f"[X] No scenes found under {DATA_DIRS[0]}")
        return 1

    failed_scenes: List[str] = []
    for scene, scene_path in sorted(scenes.items()):
        print(f"\n[>] ĐANG XỬ LÝ SCENE: {scene}")
        experiment_name = f"submit_{run_tag}_{scene}"
        prediction_dir = PRED_DIR / scene
        csv_path = _find_pose_csv(scene_path)
        if csv_path is None:
            failed_scenes.append(scene)
            print(f"    [X] Không tìm thấy test_poses.csv/test_pose.csv của {scene}.")
            continue

        # Search only this preset/budget/seed namespace. Otherwise an older A2
        # checkpoint can silently shadow the requested D1b run.
        configs = _find_configs(experiment_name)

        # Một output có manifest hợp lệ vẫn dùng được ngay cả khi checkpoint đã được dọn.
        current_config = None
        for config in configs:
            is_current, _ = _render_output_is_current(prediction_dir, csv_path, config)
            if is_current:
                current_config = config
                break
        if current_config is not None:
            print(f"    [SKIP] Prediction đã đủ, đúng phiên bản và đã redistort: {scene}")
            continue

        latest_config = next((config for config in configs if _has_checkpoint(config)), None)
        if latest_config is not None:
            print("    [SKIP] Đã tìm thấy checkpoint hoàn chỉnh; bỏ qua bước train.")
        else:
            if configs:
                print("    [!] Config cũ không có checkpoint hoàn chỉnh; sẽ train lại.")
            print(f"    [...] Đang huấn luyện {iterations:,} iterations với {preset.method}...")
            train_command = [
                sys.executable,
                "-m",
                "nerfstudio.scripts.train",
                preset.method,
                "--experiment-name",
                experiment_name,
                "--machine.seed",
                str(seed),
                "--max-num-iterations",
                str(iterations),
                "--viewer.quit-on-train-completion",
                "True",
            ]
            train_command.extend(preset.train_args)
            train_command.extend(["--data", str(scene_path)])
            try:
                _run_checked(train_command)
            except (OSError, subprocess.CalledProcessError) as error:
                print(f"    [X] Lỗi huấn luyện {scene}: {error}")
                failed_scenes.append(scene)
                continue

            configs = _find_configs(experiment_name)
            latest_config = next(
                (config for config in configs if _has_checkpoint(config)),
                None,
            )
            if latest_config is None:
                print(f"    [X] Không tìm thấy checkpoint của {scene} sau khi train.")
                failed_scenes.append(scene)
                continue

        is_current, stale_reason = _render_output_is_current(
            prediction_dir,
            csv_path,
            latest_config,
        )
        if is_current:
            print(f"    [SKIP] Prediction hiện tại đã hợp lệ: {scene}")
            continue
        if prediction_dir.exists():
            print(f"    [*] Sẽ render lại vì output cũ/thiếu: {stale_reason}")

        print("    [CAMERA] Đang render pinhole và redistort về camera gốc...")
        render_command = [
            sys.executable,
            str(RENDER_SCRIPT),
            "--config",
            str(latest_config),
            "--csv",
            str(csv_path),
            "--out",
            str(prediction_dir),
            "--clean-output",
        ]
        try:
            _run_checked(render_command)
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"    [X] Lỗi render ảnh cho {scene}: {error}")
            failed_scenes.append(scene)
            continue

        is_current, validation_reason = _render_output_is_current(
            prediction_dir,
            csv_path,
            latest_config,
        )
        if is_current:
            print(f"    [OK] Đã render và kiểm tra xong {scene}.")
        else:
            print(f"    [X] Render kết thúc nhưng output chưa hợp lệ: {validation_reason}")
            failed_scenes.append(scene)

    print("\n" + "=" * 60)
    if failed_scenes:
        print(f" AUTO-PIPELINE THẤT BẠI: {', '.join(sorted(set(failed_scenes)))}")
        print(" Không tạo submission cho đến khi tất cả scene đều thành công.")
        print("=" * 60)
        return 1
    print(f" AUTO-PIPELINE ĐÃ HOÀN TẤT {len(scenes)} SCENE VỚI PRESET {preset_name}")
    print("=" * 60)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train and render every official submission scene with a reproducible preset."
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="d1b",
        help="Training preset. d1b is the current local-validation winner.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Override the preset iteration budget; changes the checkpoint namespace.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--run-tag",
        help="Optional output namespace. Do not reuse it for different effective settings.",
    )
    arguments = parser.parse_args()
    try:
        return run_pipeline(
            preset_name=arguments.preset,
            iterations_override=arguments.iterations,
            seed=arguments.seed,
            run_tag_override=arguments.run_tag,
        )
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
