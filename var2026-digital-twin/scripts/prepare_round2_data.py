"""Validate and import VAI_NVS_DATA_ROUND2 into the repository data layout.

Expected source layout::

    VAI_NVS_DATA_ROUND2/
      <scene>/
        train/images/*
        train/sparse/0/{cameras.bin,images.bin,points3D.bin}
        test/test_poses.csv

The pipeline consumes the following destination layout::

    var2026-digital-twin/data/private_test1/<scene>/...

The import is staged first. If ``private_test1`` already exists, it is renamed
to a timestamped backup before the staged Round 2 dataset is activated. This
prevents old and new competition scenes from being mixed accidentally.

Usage from ``var2026-digital-twin``::

    python scripts/prepare_round2_data.py --dry-run
    python scripts/prepare_round2_data.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = PROJECT_ROOT.parent / "VAI_NVS_DATA_ROUND2"
DEFAULT_DESTINATION = PROJECT_ROOT / "data" / "private_test1"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
REQUIRED_SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin")
REQUIRED_CSV_COLUMNS = {
    "image_name",
    "qw",
    "qx",
    "qy",
    "qz",
    "tx",
    "ty",
    "tz",
    "fx",
    "fy",
    "cx",
    "cy",
    "width",
    "height",
}


@dataclass(frozen=True)
class SceneSummary:
    name: str
    train_images: int
    test_poses: int
    files: int
    bytes: int


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_image_name(raw_name: str, scene_name: str, line_number: int) -> str:
    image_name = (raw_name or "").strip().replace("\\", "/")
    posix_path = PurePosixPath(image_name)
    windows_path = PureWindowsPath(image_name)
    if (
        not image_name
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in posix_path.parts)
    ):
        raise ValueError(
            f"Scene {scene_name}: image_name không an toàn tại dòng "
            f"{line_number}: {raw_name!r}"
        )
    return posix_path.as_posix()


def _finite_float(row: dict[str, str], key: str, scene_name: str, line_number: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Scene {scene_name}: cột {key!r} không hợp lệ tại dòng {line_number}"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"Scene {scene_name}: cột {key!r} phải hữu hạn tại dòng {line_number}"
        )
    return value


def _validate_pose_csv(csv_path: Path, scene_name: str) -> int:
    seen_names: set[str] = set()
    row_count = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_CSV_COLUMNS - fieldnames)
        if missing:
            raise ValueError(
                f"Scene {scene_name}: {csv_path.name} thiếu các cột {missing}"
            )

        for row in reader:
            line_number = reader.line_num
            image_name = _safe_image_name(
                row.get("image_name", ""), scene_name, line_number
            )
            name_key = image_name.casefold()
            if name_key in seen_names:
                raise ValueError(
                    f"Scene {scene_name}: image_name trùng tại dòng "
                    f"{line_number}: {image_name}"
                )
            seen_names.add(name_key)

            quaternion = [
                _finite_float(row, key, scene_name, line_number)
                for key in ("qw", "qx", "qy", "qz")
            ]
            if math.sqrt(sum(value * value for value in quaternion)) < 1e-12:
                raise ValueError(
                    f"Scene {scene_name}: quaternion bằng 0 tại dòng {line_number}"
                )

            for key in ("tx", "ty", "tz", "cx", "cy"):
                _finite_float(row, key, scene_name, line_number)

            for key in ("fx", "fy"):
                if _finite_float(row, key, scene_name, line_number) <= 0:
                    raise ValueError(
                        f"Scene {scene_name}: {key} phải lớn hơn 0 tại dòng "
                        f"{line_number}"
                    )

            for key in ("width", "height"):
                value = _finite_float(row, key, scene_name, line_number)
                if value <= 0 or not value.is_integer():
                    raise ValueError(
                        f"Scene {scene_name}: {key} phải là số nguyên dương "
                        f"tại dòng {line_number}"
                    )

            row_count += 1

    if row_count == 0:
        raise ValueError(f"Scene {scene_name}: {csv_path} không chứa test pose")
    return row_count


def _iter_scene_files(scene_path: Path) -> Iterable[Path]:
    for path in scene_path.rglob("*"):
        if path.is_symlink():
            raise ValueError(
                f"Scene {scene_path.name}: không chấp nhận symbolic link: {path}"
            )
        if path.is_file():
            yield path


def validate_scene(scene_path: Path) -> SceneSummary:
    scene_name = scene_path.name
    train_images_dir = scene_path / "train" / "images"
    sparse_dir = scene_path / "train" / "sparse" / "0"
    pose_csv = scene_path / "test" / "test_poses.csv"

    if not train_images_dir.is_dir():
        raise ValueError(f"Scene {scene_name}: thiếu thư mục train/images")
    if not sparse_dir.is_dir():
        raise ValueError(f"Scene {scene_name}: thiếu thư mục train/sparse/0")
    if not pose_csv.is_file():
        raise ValueError(f"Scene {scene_name}: thiếu test/test_poses.csv")

    for filename in REQUIRED_SPARSE_FILES:
        sparse_file = sparse_dir / filename
        if not sparse_file.is_file() or sparse_file.stat().st_size == 0:
            raise ValueError(
                f"Scene {scene_name}: thiếu hoặc rỗng train/sparse/0/{filename}"
            )

    image_names: set[str] = set()
    train_image_count = 0
    for image_path in train_images_dir.iterdir():
        if not image_path.is_file() or image_path.suffix.casefold() not in IMAGE_EXTENSIONS:
            continue
        name_key = image_path.name.casefold()
        if name_key in image_names:
            raise ValueError(
                f"Scene {scene_name}: tên ảnh train trùng khi không phân biệt hoa/thường: "
                f"{image_path.name}"
            )
        image_names.add(name_key)
        train_image_count += 1

    if train_image_count == 0:
        raise ValueError(f"Scene {scene_name}: train/images không có ảnh hợp lệ")

    test_pose_count = _validate_pose_csv(pose_csv, scene_name)
    files = list(_iter_scene_files(scene_path))
    return SceneSummary(
        name=scene_name,
        train_images=train_image_count,
        test_poses=test_pose_count,
        files=len(files),
        bytes=sum(path.stat().st_size for path in files),
    )


def discover_scenes(source: Path) -> list[Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục nguồn: {source}")

    scenes = sorted(
        (
            path
            for path in source.iterdir()
            if path.is_dir()
            and not path.name.startswith(".")
            and path.name.casefold() != "__macosx"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not scenes:
        raise ValueError(f"Không tìm thấy scene nào trong {source}")

    names: set[str] = set()
    for scene in scenes:
        name_key = scene.name.casefold()
        if name_key in names:
            raise ValueError(
                "Tên scene bị trùng khi không phân biệt hoa/thường: " f"{scene.name}"
            )
        names.add(name_key)
    return scenes


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def _unique_backup_path(destination: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    base = destination.with_name(f"{destination.name}_backup_{timestamp}")
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{suffix}")
        suffix += 1
    return candidate


def _write_manifest(
    staging: Path,
    source: Path,
    destination: Path,
    summaries: Sequence[SceneSummary],
) -> None:
    manifest = {
        "schema_version": 1,
        "imported_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(destination),
        "scenes": [asdict(summary) for summary in summaries],
    }
    manifest_path = staging / ".round2_import.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
        file.write("\n")


def import_round2(source: Path, destination: Path, dry_run: bool) -> None:
    source = source.resolve()
    destination = destination.resolve()

    if source == destination or _is_relative_to(source, destination) or _is_relative_to(
        destination, source
    ):
        raise ValueError(
            "Nguồn và đích không được trùng nhau hoặc nằm lồng trong nhau: "
            f"source={source}, destination={destination}"
        )
    if destination.exists() and not destination.is_dir():
        raise NotADirectoryError(f"Đích tồn tại nhưng không phải thư mục: {destination}")

    scene_paths = discover_scenes(source)
    summaries: list[SceneSummary] = []
    print(f"[*] Đang kiểm tra {len(scene_paths)} scene tại {source}")
    for scene_path in scene_paths:
        summary = validate_scene(scene_path)
        summaries.append(summary)
        print(
            f"    [OK] {summary.name}: {summary.train_images} train, "
            f"{summary.test_poses} test poses, {_human_bytes(summary.bytes)}"
        )

    total_size = sum(summary.bytes for summary in summaries)
    print(f"[*] Tổng dung lượng cần sao chép: {_human_bytes(total_size)}")
    print(f"[*] Đích pipeline: {destination}")
    if destination.exists():
        print("[*] Dữ liệu private_test1 hiện tại sẽ được đổi tên thành backup.")

    if dry_run:
        print("[DRY-RUN] Dữ liệu hợp lệ; chưa sao chép hoặc thay đổi thư mục nào.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}-round2-staging-", dir=destination.parent
        )
    )
    backup: Path | None = None
    activated = False
    try:
        for index, scene_path in enumerate(scene_paths, start=1):
            print(f"[{index}/{len(scene_paths)}] Đang sao chép {scene_path.name}...")
            shutil.copytree(scene_path, staging / scene_path.name, copy_function=shutil.copy2)

        _write_manifest(staging, source, destination, summaries)

        print("[*] Đang kiểm tra lại bản sao trước khi kích hoạt...")
        staged_summaries = [
            validate_scene(staging / summary.name) for summary in summaries
        ]
        if staged_summaries != summaries:
            raise RuntimeError("Thông tin bản sao không khớp dữ liệu nguồn")

        if destination.exists():
            backup = _unique_backup_path(destination)
            destination.rename(backup)
            print(f"[*] Đã backup dữ liệu cũ tại: {backup}")

        try:
            staging.rename(destination)
            activated = True
        except Exception:
            if backup is not None and backup.exists() and not destination.exists():
                backup.rename(destination)
                backup = None
            raise

    finally:
        if not activated and staging.exists():
            # Chỉ dọn đúng thư mục tạm do tempfile tạo trong destination.parent.
            staging_parent = staging.parent.resolve()
            if (
                staging_parent == destination.parent.resolve()
                and staging.name.startswith(f".{destination.name}-round2-staging-")
            ):
                shutil.rmtree(staging)

    print(
        f"[+] Hoàn tất: {len(summaries)} scene Round 2 đã sẵn sàng tại "
        f"{destination}"
    )
    if backup is not None:
        print(f"[+] Có thể khôi phục dữ liệu cũ từ: {backup}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Kiểm tra và đưa VAI_NVS_DATA_ROUND2 vào "
            "var2026-digital-twin/data/private_test1."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Thư mục Round 2 (mặc định: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=f"Thư mục private dataset của repo (mặc định: {DEFAULT_DESTINATION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ kiểm tra và in kế hoạch, không sao chép dữ liệu.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import_round2(args.source, args.destination, args.dry_run)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"[X] Import Round 2 thất bại: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
