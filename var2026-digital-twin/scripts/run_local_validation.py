"""Train and/or evaluate one local-validation scene."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "private_test1"
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "data" / "local_validation"
DEFAULT_PREDICTION_ROOT = PROJECT_ROOT / "data" / "local_validation_predictions"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "outputs" / "local_validation_reports"


def _run(command: list[str]) -> None:
    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _safe_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{label} may contain only letters, digits, dot, underscore, and hyphen")
    return value


def _complete_configs(experiment_name: str) -> list[Path]:
    experiment_dir = PROJECT_ROOT / "outputs" / experiment_name
    configs = sorted(
        experiment_dir.glob("**/config.yml") if experiment_dir.is_dir() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [
        config
        for config in configs
        if (config.parent / "nerfstudio_models").is_dir()
        and any((config.parent / "nerfstudio_models").glob("*.ckpt"))
    ]


def _contains_option(arguments: list[str], option: str) -> bool:
    return any(value == option or value.startswith(option + "=") for value in arguments)


def _uses_stock_splatfacto_defaults(method: str) -> bool:
    """Return whether this method accepts Nerfstudio's stock Splatfacto flags."""
    return method == "splatfacto" or method.startswith("splatfacto-")


def run_experiment(arguments: argparse.Namespace) -> Path:
    scene = _safe_name(arguments.scene, "scene")
    tag = _safe_name(arguments.tag, "tag")
    experiment_name = f"localval_{scene}_{tag}"
    validation_scene = arguments.validation_root.resolve() / scene
    manifest = validation_scene / ".local_validation.json"

    if arguments.rebuild_split or not manifest.is_file():
        split_command = [
            sys.executable,
            str(SCRIPTS_DIR / "make_val_split.py"),
            "--scene", scene,
            "--data-root", str(arguments.source_root.resolve()),
            "--output-root", str(arguments.validation_root.resolve()),
        ]
        if arguments.val_count is not None:
            split_command.extend(["--count", str(arguments.val_count)])
        elif arguments.val_ratio is not None:
            split_command.extend(["--ratio", str(arguments.val_ratio)])
        if arguments.rebuild_split:
            split_command.append("--overwrite")
        _run(split_command)
    else:
        print(f"[SKIP] Reusing local split: {validation_scene}")

    configs = _complete_configs(experiment_name)
    if arguments.eval_only and arguments.new_run:
        raise ValueError("--eval-only cannot be combined with --new-run")
    if configs and arguments.new_run:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_name = f"{experiment_name}_{timestamp}"
        configs = []

    if configs:
        config = configs[0]
        print(f"[SKIP] Reusing completed checkpoint: {config}")
    elif arguments.eval_only:
        raise FileNotFoundError(
            f"--eval-only requested but no complete checkpoint was found for {experiment_name}"
        )
    else:
        extra_train_args = list(arguments.train_args)
        if extra_train_args and extra_train_args[0] == "--":
            extra_train_args = extra_train_args[1:]
        train_command = [
            sys.executable,
            "-m", "nerfstudio.scripts.train",
            arguments.method,
            "--experiment-name", experiment_name,
        ]
        if _uses_stock_splatfacto_defaults(arguments.method):
            if not _contains_option(extra_train_args, "--pipeline.model.sh-degree"):
                train_command.extend(["--pipeline.model.sh-degree", "3"])
            if not _contains_option(extra_train_args, "--pipeline.model.use-scale-regularization"):
                train_command.extend(["--pipeline.model.use-scale-regularization", "True"])
        if not _contains_option(extra_train_args, "--max-num-iterations"):
            train_command.extend(["--max-num-iterations", str(arguments.iterations)])
        if not _contains_option(extra_train_args, "--viewer.quit-on-train-completion"):
            train_command.extend(["--viewer.quit-on-train-completion", "True"])
        train_command.extend(extra_train_args)
        train_command.extend(["--data", str(validation_scene)])
        _run(train_command)
        configs = _complete_configs(experiment_name)
        if not configs:
            raise RuntimeError(f"Training ended but no complete checkpoint was found for {experiment_name}")
        config = configs[0]

    if arguments.train_only:
        print(f"\n[OK] Training stage complete: {experiment_name}")
        print(f"     config: {config}")
        print(f"     checkpoint_dir: {config.parent / 'nerfstudio_models'}")
        return config

    prediction_dir = arguments.prediction_root.resolve() / experiment_name
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "render_test_pose.py"),
            "--config", str(config),
            "--csv", str(validation_scene / "test" / "test_poses.csv"),
            "--camera-meta", str(validation_scene / "transforms.json"),
            "--out", str(prediction_dir),
            "--clean-output",
        ]
    )

    report_dir = arguments.report_root.resolve() / experiment_name
    report_path = report_dir / "metrics.json"
    _run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "evaluate_local_validation.py"),
            "--predictions", str(prediction_dir),
            "--ground-truth", str(validation_scene / "local_gt" / "images"),
            "--output", str(report_path),
            "--per-image", str(report_dir / "per_image.csv"),
            "--psnr-max", str(arguments.psnr_max),
            "--device", arguments.metric_device,
        ]
    )
    print(f"\n[OK] Experiment complete: {experiment_name}")
    print(f"     report: {report_path}")
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create/reuse a holdout, then train and/or evaluate one scene."
    )
    parser.add_argument("--scene", required=True, help="Scene name, for example HCM0421")
    parser.add_argument("--tag", required=True, help="Short experiment name, for example baseline10k")
    parser.add_argument("--method", default="splatfacto-big")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--val-ratio", type=float)
    split_group.add_argument("--val-count", type=int)
    parser.add_argument("--rebuild-split", action="store_true")
    stage_group = parser.add_mutually_exclusive_group()
    stage_group.add_argument(
        "--train-only",
        action="store_true",
        help="Stop after a complete checkpoint is available; do not load/render/score it",
    )
    stage_group.add_argument(
        "--eval-only",
        action="store_true",
        help="Require an existing checkpoint and only render/score it",
    )
    parser.add_argument(
        "--new-run",
        action="store_true",
        help="Train a timestamped run instead of reusing a checkpoint with the same tag",
    )
    parser.add_argument("--psnr-max", type=float, default=50.0)
    parser.add_argument("--metric-device", default="auto")
    parser.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Additional ns-train arguments after '--'",
    )
    arguments = parser.parse_args()
    if arguments.iterations <= 0:
        parser.error("--iterations must be positive")
    try:
        run_experiment(arguments)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
