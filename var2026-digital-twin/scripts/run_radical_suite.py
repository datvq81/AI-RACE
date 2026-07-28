"""Run a radical experiment suite with an auditable manifest.

The actual train/render/score path remains ``run_experiment_suite.py`` and
``run_local_validation.py``. This adapter only constrains suites to the radical
namespace and records the environment, commands, logs, artifacts, GPU memory,
and declared gate result.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RADICAL_CONFIG_ROOT = PROJECT_ROOT / "configs" / "experiments" / "radical"
SUITE_RUNNER = PROJECT_ROOT / "scripts" / "run_experiment_suite.py"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "outputs" / "radical_runs"
DEFAULT_REPORT_ROOT = PROJECT_ROOT / "outputs" / "local_validation_reports"
DEFAULT_PREDICTION_ROOT = PROJECT_ROOT / "data" / "local_validation_predictions"
DEFAULT_VALIDATION_ROOT = PROJECT_ROOT / "data" / "local_validation"
RUN_HEADER = re.compile(r"^\[\d+/\d+\]\s+([A-Za-z0-9_.-]+)\s+\|")
PACKAGES = (
    "nerfstudio",
    "gsplat",
    "torch",
    "torchmetrics",
    "lpips",
    "opencv-python",
    "var2026-digital-twin-methods",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _git_state() -> dict[str, Any]:
    commit = _command_output(["git", "rev-parse", "HEAD"])
    branch = _command_output(["git", "branch", "--show-current"])
    status = _command_output(["git", "status", "--short"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def _environment() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in PACKAGES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    gpu_inventory = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "gpu_inventory": gpu_inventory.splitlines() if gpu_inventory else [],
    }


def _gpu_memory_samples() -> dict[str, dict[str, Any]]:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    samples: dict[str, dict[str, Any]] = {}
    if not output:
        return samples
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            continue
        try:
            memory_mib = int(parts[2])
        except ValueError:
            continue
        samples[parts[0]] = {"name": parts[1], "memory_used_mib": memory_mib}
    return samples


def _run_with_log(
    command: list[str],
    log_path: Path,
) -> tuple[int, dict[str, dict[str, dict[str, Any]]]]:
    """Tee output and sample device memory against the active suite run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    active = {"id": None}
    peak_by_run: dict[str, dict[str, dict[str, Any]]] = {}

    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None

        def read_output() -> None:
            for line in process.stdout:
                match = RUN_HEADER.match(line.strip())
                if match:
                    active["id"] = match.group(1)
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        while process.poll() is None:
            experiment_id = active["id"]
            if experiment_id is not None:
                experiment_peaks = peak_by_run.setdefault(experiment_id, {})
                for gpu_id, sample in _gpu_memory_samples().items():
                    previous = experiment_peaks.get(gpu_id)
                    if (
                        previous is None
                        or sample["memory_used_mib"] > previous["memory_used_mib"]
                    ):
                        experiment_peaks[gpu_id] = sample
            time.sleep(1.0)
        reader.join()
        return process.returncode, peak_by_run


def _latest_config(experiment_name: str) -> Path | None:
    experiment_dir = PROJECT_ROOT / "outputs" / experiment_name
    if not experiment_dir.is_dir():
        return None
    configs = sorted(
        experiment_dir.glob("**/config.yml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return configs[0] if configs else None


def _gaussian_count(checkpoint: Path | None) -> dict[str, Any]:
    if checkpoint is None:
        return {"value": None, "status": "checkpoint-unavailable"}
    try:
        import torch

        try:
            payload = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        state = payload.get("pipeline", payload) if isinstance(payload, dict) else {}
        if not isinstance(state, dict):
            raise ValueError("checkpoint does not contain a state dictionary")
        candidate_keys = [
            key
            for key in state
            if key.endswith("gauss_params.means") or key.endswith(".means")
        ]
        if not candidate_keys:
            raise KeyError("Gaussian means tensor not found")
        means = state[candidate_keys[0]]
        value = int(means.shape[0])
        del payload, state, means
        gc.collect()
        return {
            "value": value,
            "status": "recorded",
            "checkpoint": str(checkpoint),
            "state_key": candidate_keys[0],
        }
    except (ImportError, OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        return {
            "value": None,
            "status": "unavailable",
            "checkpoint": str(checkpoint),
            "reason": str(error),
        }


def _collect_artifacts(
    suite_summary: dict[str, Any],
    validation_root: Path,
    prediction_root: Path,
    report_root: Path,
    peak_by_run: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    artifacts: dict[str, dict[str, Any]] = {}
    for run in suite_summary.get("runs", []):
        if not isinstance(run, dict):
            continue
        experiment_id = str(run.get("id"))
        scene = str(run.get("scene"))
        tag = str(run.get("tag"))
        experiment_name = f"localval_{scene}_{tag}"
        config = _latest_config(experiment_name)
        checkpoint = None
        if config is not None:
            checkpoints = sorted(
                (config.parent / "nerfstudio_models").glob("*.ckpt"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            checkpoint = checkpoints[0] if checkpoints else None

        report_path = report_root / experiment_name / "metrics.json"
        render_manifest_path = prediction_root / experiment_name / ".render_manifest.json"
        split_manifest_path = validation_root / scene / ".local_validation.json"
        metrics = _load_json(report_path) if report_path.is_file() else None
        render_manifest = (
            _load_json(render_manifest_path) if render_manifest_path.is_file() else None
        )
        split_manifest = (
            _load_json(split_manifest_path) if split_manifest_path.is_file() else None
        )
        artifacts[experiment_id] = {
            "scene": scene,
            "seed": run.get("seed"),
            "iterations": run.get("iterations"),
            "command": run.get("command"),
            "status": run.get("status"),
            "config": str(config) if config is not None else None,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "metrics_path": str(report_path) if report_path.is_file() else None,
            "metrics": metrics,
            "render_manifest_path": (
                str(render_manifest_path) if render_manifest_path.is_file() else None
            ),
            "render_image_count": (
                render_manifest.get("image_count") if render_manifest else None
            ),
            "split_validation_count": (
                split_manifest.get("validation_count") if split_manifest else None
            ),
            "gaussian_count": _gaussian_count(checkpoint),
            "peak_device_memory": peak_by_run.get(experiment_id, {}),
        }
    return artifacts


def _score(artifacts: dict[str, dict[str, Any]], experiment_id: str) -> float | None:
    metrics = artifacts.get(experiment_id, {}).get("metrics")
    if not isinstance(metrics, dict):
        return None
    value = metrics.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def _evaluate_gate(
    suite: dict[str, Any],
    artifacts: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gate = suite.get("radical_gate")
    if not isinstance(gate, dict):
        return {"status": "not-declared", "requirements": []}
    requirements = gate.get("requirements")
    if not isinstance(requirements, list):
        return {"status": "invalid", "requirements": []}

    results: list[dict[str, Any]] = []
    for requirement in requirements:
        if not isinstance(requirement, dict):
            results.append({"status": "invalid", "reason": "requirement is not an object"})
            continue
        kind = requirement.get("kind")
        result = dict(requirement)
        if kind == "render_count_matches":
            run_ids = requirement.get("runs", [])
            counts = []
            for experiment_id in run_ids:
                artifact = artifacts.get(str(experiment_id), {})
                counts.append(
                    (
                        artifact.get("render_image_count"),
                        artifact.get("split_validation_count"),
                        (artifact.get("metrics") or {}).get("num_images")
                        if isinstance(artifact.get("metrics"), dict)
                        else None,
                    )
                )
            available = bool(counts) and all(None not in values for values in counts)
            result["observed"] = counts
            result["status"] = (
                "passed"
                if available and all(a == b == c for a, b, c in counts)
                else ("unavailable" if not available else "failed")
            )
        elif kind in {"score_abs_delta_max", "score_delta_min"}:
            baseline = _score(artifacts, str(requirement.get("baseline")))
            candidate = _score(artifacts, str(requirement.get("candidate")))
            threshold = requirement.get("value")
            if (
                baseline is None
                or candidate is None
                or not isinstance(threshold, (int, float))
            ):
                result["status"] = "unavailable"
            else:
                delta = candidate - baseline
                result["observed_delta"] = delta
                passed = (
                    abs(delta) <= float(threshold)
                    if kind == "score_abs_delta_max"
                    else delta >= float(threshold)
                )
                result["status"] = "passed" if passed else "failed"
        else:
            result["status"] = "invalid"
            result["reason"] = f"unknown gate kind: {kind}"
        results.append(result)

    statuses = {result["status"] for result in results}
    if "failed" in statuses or "invalid" in statuses:
        status = "failed"
    elif "unavailable" in statuses:
        status = "unavailable"
    else:
        status = "passed"
    return {"status": status, "requirements": results}


def _radical_suite_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(RADICAL_CONFIG_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Radical suites must live under {RADICAL_CONFIG_ROOT}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(f"Suite not found: {resolved}")
    return resolved


def run(arguments: argparse.Namespace) -> int:
    suite_path = _radical_suite_path(arguments.suite)
    suite = _load_json(suite_path)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir = arguments.run_root.resolve() / f"{timestamp}_{suite_path.stem}_{arguments.stage}"
    run_dir.mkdir(parents=True, exist_ok=False)
    suite_summary_path = run_dir / "suite_summary.json"
    log_path = run_dir / "run.log"
    manifest_path = run_dir / "manifest.json"

    command = [
        sys.executable,
        str(SUITE_RUNNER),
        "--suite",
        str(suite_path),
        "--stage",
        arguments.stage,
        "--summary",
        str(suite_summary_path),
        "--validation-root",
        str(arguments.validation_root.resolve()),
        "--prediction-root",
        str(arguments.prediction_root.resolve()),
        "--report-root",
        str(arguments.report_root.resolve()),
    ]
    for option, value in (
        ("--scene", arguments.scene),
        ("--iterations", arguments.iterations),
        ("--seed", arguments.seed),
        ("--source-root", arguments.source_root),
        ("--metric-device", arguments.metric_device),
    ):
        if value is not None:
            resolved_value = value.resolve() if isinstance(value, Path) else value
            command.extend([option, str(resolved_value)])
    for option in ("only", "skip"):
        for value in getattr(arguments, option) or []:
            command.extend([f"--{option}", value])
    for flag in (
        "dry_run",
        "list",
        "continue_on_error",
        "new_run",
        "rebuild_split",
    ):
        if getattr(arguments, flag):
            command.append("--" + flag.replace("_", "-"))

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "suite": str(suite_path),
        "stage": arguments.stage,
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None,
        "status": "running",
        "git": _git_state(),
        "environment": _environment(),
        "adapter_command": command,
        "log_path": str(log_path),
        "suite_summary_path": str(suite_summary_path),
        "runs": {},
        "gate": {"status": "pending", "requirements": []},
    }
    _atomic_json(manifest_path, manifest)
    print("$ " + subprocess.list2cmdline(command), flush=True)
    return_code, peak_by_run = _run_with_log(command, log_path)
    suite_summary = (
        _load_json(suite_summary_path) if suite_summary_path.is_file() else {"runs": []}
    )
    artifacts = _collect_artifacts(
        suite_summary,
        arguments.validation_root.resolve(),
        arguments.prediction_root.resolve(),
        arguments.report_root.resolve(),
        peak_by_run,
    )
    manifest["runs"] = artifacts
    manifest["gate"] = _evaluate_gate(suite, artifacts)
    manifest["finished_at"] = datetime.now().astimezone().isoformat()
    manifest["return_code"] = return_code
    suite_status = suite_summary.get("status")
    manifest["status"] = (
        str(suite_status)
        if return_code == 0 and isinstance(suite_status, str)
        else ("completed" if return_code == 0 else "failed")
    )
    _atomic_json(manifest_path, manifest)
    print(f"\nRadical manifest: {manifest_path}")
    print(f"Gate status     : {manifest['gate']['status']}")
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a radical suite through the locked local-validation path."
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=RADICAL_CONFIG_ROOT / "r0_control.json",
    )
    parser.add_argument("--stage", choices=("train", "eval", "full"), default="full")
    parser.add_argument("--scene")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--only", action="append")
    parser.add_argument("--skip", action="append")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--metric-device")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--validation-root", type=Path, default=DEFAULT_VALIDATION_ROOT)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    arguments = parser.parse_args()
    if arguments.iterations is not None and arguments.iterations <= 0:
        parser.error("--iterations must be positive")
    if arguments.seed is not None and arguments.seed < 0:
        parser.error("--seed must be non-negative")
    try:
        return run(arguments)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
