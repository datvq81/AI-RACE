"""Run a configurable suite of local-validation experiments sequentially.

Experiment definitions live in JSON files under ``configs/experiments``.  A
suite may change ordinary Splatfacto options or select an entirely different
Nerfstudio method/plugin through its per-experiment ``method`` field.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(__file__).resolve().parent / "run_local_validation.py"
DEFAULT_SUITE = PROJECT_ROOT / "configs" / "experiments" / "a_baseline.json"
DEFAULT_SUMMARY_ROOT = PROJECT_ROOT / "outputs" / "experiment_suites"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
RESERVED_TRAIN_OPTIONS = {
    "--data",
    "--experiment-name",
    "--max-num-iterations",
    "--machine.seed",
    "--output-dir",
    "--viewer.quit-on-train-completion",
}
RESERVED_RUNNER_OPTIONS = {
    "--scene",
    "--tag",
    "--method",
    "--iterations",
    "--train-only",
    "--eval-only",
    "--new-run",
    "--rebuild-split",
    "--psnr-max",
    "--metric-device",
    "--source-root",
    "--validation-root",
    "--prediction-root",
    "--report-root",
}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("Suite root must be a JSON object")
    return payload


def _string_list(value: Any, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a JSON array of strings")
    return list(value)


def _option_name(argument: str) -> str | None:
    if not argument.startswith("--"):
        return None
    return argument.split("=", 1)[0]


def _validate_options(
    arguments: list[str],
    experiment_id: str,
    field: str,
    reserved: set[str],
) -> None:
    for argument in arguments:
        option = _option_name(argument)
        if option in reserved:
            raise ValueError(
                f"Experiment {experiment_id}: {option} is controlled by the suite runner; "
                f"do not put it in {field}"
            )


def _name(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{field} may contain only letters, digits, dot, underscore, and hyphen")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _non_negative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _budget_label(iterations: int) -> str:
    return f"{iterations // 1000}k" if iterations % 1000 == 0 else f"{iterations}it"


def _selected_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        result.update(item.strip() for item in value.split(",") if item.strip())
    return result


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


def _suite_experiments(
    suite: dict[str, Any],
    cli: argparse.Namespace,
) -> tuple[str, list[dict[str, Any]]]:
    if suite.get("schema_version") != 1:
        raise ValueError("Only experiment suite schema_version=1 is supported")
    suite_name = _name(suite.get("name"), "suite.name")
    defaults = suite.get("defaults", {})
    if not isinstance(defaults, dict):
        raise ValueError("suite.defaults must be a JSON object")
    raw_experiments = suite.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise ValueError("suite.experiments must be a non-empty JSON array")

    only = _selected_ids(cli.only)
    skip = _selected_ids(cli.skip) or set()
    common_train_args = _string_list(defaults.get("train_args"), "defaults.train_args")
    common_runner_args = _string_list(defaults.get("runner_args"), "defaults.runner_args")
    experiments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_tags: set[str] = set()

    for index, raw in enumerate(raw_experiments):
        if not isinstance(raw, dict):
            raise ValueError(f"experiments[{index}] must be a JSON object")
        experiment_id = _name(raw.get("id"), f"experiments[{index}].id")
        if experiment_id in seen_ids:
            raise ValueError(f"Duplicate experiment id: {experiment_id}")
        seen_ids.add(experiment_id)
        if raw.get("enabled", True) is not True:
            continue
        if only is not None and experiment_id not in only:
            continue
        if experiment_id in skip:
            continue

        scene = _name(
            cli.scene or raw.get("scene") or defaults.get("scene"),
            f"{experiment_id}.scene",
        )
        method = _name(
            raw.get("method") or defaults.get("method", "splatfacto-big"),
            f"{experiment_id}.method",
        )
        iterations = _positive_integer(
            cli.iterations or raw.get("iterations") or defaults.get("iterations", 10000),
            f"{experiment_id}.iterations",
        )
        seed_value = cli.seed if cli.seed is not None else raw.get("seed", defaults.get("seed", 42))
        seed = _non_negative_integer(seed_value, f"{experiment_id}.seed")
        template = raw.get("tag", f"{experiment_id}_{{budget}}_s{{seed}}")
        if not isinstance(template, str):
            raise ValueError(f"{experiment_id}.tag must be a string")
        try:
            tag = template.format(
                id=experiment_id,
                scene=scene,
                iterations=iterations,
                budget=_budget_label(iterations),
                seed=seed,
                seed_suffix="" if seed == 42 else f"_s{seed}",
            )
        except KeyError as error:
            raise ValueError(f"{experiment_id}.tag uses unknown placeholder: {error}") from error
        tag = _name(tag, f"{experiment_id}.rendered_tag")
        unique_tag = f"{scene}/{tag}"
        if unique_tag in seen_tags:
            raise ValueError(f"Duplicate rendered scene/tag: {unique_tag}")
        seen_tags.add(unique_tag)

        train_args = common_train_args + _string_list(raw.get("train_args"), f"{experiment_id}.train_args")
        runner_args = common_runner_args + _string_list(raw.get("runner_args"), f"{experiment_id}.runner_args")
        _validate_options(train_args, experiment_id, "train_args", RESERVED_TRAIN_OPTIONS)
        _validate_options(runner_args, experiment_id, "runner_args", RESERVED_RUNNER_OPTIONS)
        experiments.append(
            {
                "id": experiment_id,
                "description": str(raw.get("description", "")),
                "scene": scene,
                "method": method,
                "iterations": iterations,
                "seed": seed,
                "tag": tag,
                "train_args": train_args,
                "runner_args": runner_args,
            }
        )

    unknown_only = (only or set()) - seen_ids
    unknown_skip = skip - seen_ids
    if unknown_only:
        raise ValueError(f"Unknown --only experiment ids: {sorted(unknown_only)}")
    if unknown_skip:
        raise ValueError(f"Unknown --skip experiment ids: {sorted(unknown_skip)}")
    if not experiments:
        raise ValueError("No enabled experiments remain after --only/--skip filtering")
    return suite_name, experiments


def _command(
    experiment: dict[str, Any],
    cli: argparse.Namespace,
    rebuild_split: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--scene", experiment["scene"],
        "--tag", experiment["tag"],
        "--method", experiment["method"],
        "--iterations", str(experiment["iterations"]),
        "--psnr-max", str(cli.psnr_max),
        "--metric-device", cli.metric_device,
    ]
    stage_option = {"train": "--train-only", "eval": "--eval-only", "full": None}[cli.stage]
    if stage_option is not None:
        command.append(stage_option)
    if cli.new_run:
        command.append("--new-run")
    if rebuild_split:
        command.append("--rebuild-split")
    for option, value in (
        ("--source-root", cli.source_root),
        ("--validation-root", cli.validation_root),
        ("--prediction-root", cli.prediction_root),
        ("--report-root", cli.report_root),
    ):
        if value is not None:
            command.extend([option, str(value.resolve())])
    command.extend(experiment["runner_args"])
    if cli.stage != "eval":
        command.extend(["--", "--machine.seed", str(experiment["seed"])])
        command.extend(experiment["train_args"])
    return command


def run_suite(cli: argparse.Namespace) -> int:
    suite_path = cli.suite.resolve()
    suite = _load_json(suite_path)
    suite_name, experiments = _suite_experiments(suite, cli)
    if cli.list:
        for experiment in experiments:
            print(
                f"{experiment['id']:8s} scene={experiment['scene']:10s} "
                f"method={experiment['method']:20s} iterations={experiment['iterations']:6d} "
                f"tag={experiment['tag']}"
            )
        return 0

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    summary_path = (
        cli.summary.resolve()
        if cli.summary is not None
        else DEFAULT_SUMMARY_ROOT / suite_name / f"{timestamp}_{cli.stage}.json"
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "suite": suite_name,
        "suite_file": str(suite_path),
        "stage": cli.stage,
        "started_at": datetime.now().astimezone().isoformat(),
        "finished_at": None,
        "status": "running",
        "runs": [],
    }
    _atomic_json(summary_path, summary)

    failed = False
    first = True
    for position, experiment in enumerate(experiments, start=1):
        command = _command(experiment, cli, rebuild_split=cli.rebuild_split and first)
        first = False
        print("\n" + "=" * 72)
        print(
            f"[{position}/{len(experiments)}] {experiment['id']} | "
            f"{experiment['scene']} | {experiment['method']} | {experiment['iterations']} iterations"
        )
        if experiment["description"]:
            print(experiment["description"])
        print("$ " + shlex.join(command), flush=True)
        run_record: dict[str, Any] = {
            **experiment,
            "command": command,
            "started_at": datetime.now().astimezone().isoformat(),
            "finished_at": None,
            "duration_seconds": None,
            "return_code": None,
            "status": "dry-run" if cli.dry_run else "running",
        }
        summary["runs"].append(run_record)
        _atomic_json(summary_path, summary)
        if cli.dry_run:
            continue

        start = time.monotonic()
        try:
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            return_code = result.returncode
        except KeyboardInterrupt:
            run_record["status"] = "interrupted"
            run_record["return_code"] = 130
            run_record["duration_seconds"] = round(time.monotonic() - start, 3)
            run_record["finished_at"] = datetime.now().astimezone().isoformat()
            summary["status"] = "interrupted"
            summary["finished_at"] = datetime.now().astimezone().isoformat()
            _atomic_json(summary_path, summary)
            print(f"\n[INTERRUPTED] Summary: {summary_path}")
            return 130

        run_record["return_code"] = return_code
        run_record["duration_seconds"] = round(time.monotonic() - start, 3)
        run_record["finished_at"] = datetime.now().astimezone().isoformat()
        run_record["status"] = "completed" if return_code == 0 else "failed"
        _atomic_json(summary_path, summary)
        if return_code != 0:
            failed = True
            if not cli.continue_on_error:
                print(f"[STOP] {experiment['id']} failed with exit code {return_code}")
                break

    summary["finished_at"] = datetime.now().astimezone().isoformat()
    summary["status"] = "failed" if failed else ("dry-run" if cli.dry_run else "completed")
    _atomic_json(summary_path, summary)
    print("\n" + "=" * 72)
    print(f"Suite status : {summary['status']}")
    print(f"Summary      : {summary_path}")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a JSON-defined experiment suite sequentially.")
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--stage", choices=("train", "eval", "full"), default="train")
    parser.add_argument("--scene", help="Override scene for every selected experiment")
    parser.add_argument("--iterations", type=int, help="Override budget for every selected experiment")
    parser.add_argument("--seed", type=int, help="Override seed for every selected experiment")
    parser.add_argument("--only", action="append", help="Comma-separated experiment IDs to run")
    parser.add_argument("--skip", action="append", help="Comma-separated experiment IDs to skip")
    parser.add_argument("--list", action="store_true", help="List resolved experiments and exit")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--new-run", action="store_true")
    parser.add_argument("--rebuild-split", action="store_true")
    parser.add_argument("--psnr-max", type=float, default=50.0)
    parser.add_argument("--metric-device", default="auto")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--prediction-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--summary", type=Path)
    cli = parser.parse_args()
    if cli.iterations is not None and cli.iterations <= 0:
        parser.error("--iterations must be positive")
    if cli.seed is not None and cli.seed < 0:
        parser.error("--seed must be non-negative")
    if cli.psnr_max <= 0:
        parser.error("--psnr-max must be positive")
    if cli.stage == "eval" and cli.new_run:
        parser.error("--stage eval cannot be combined with --new-run")
    try:
        return run_suite(cli)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
