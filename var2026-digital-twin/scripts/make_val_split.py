"""Create a deterministic local-validation scene without modifying source data.

The generated scene contains only the selected training images in
``train/images``. Held-out images are linked into ``local_gt/images`` and their
COLMAP poses are exported to ``test/test_poses.csv`` so the existing
``render_test_pose.py`` can render them exactly like competition test poses.

This is intentionally a fast validation split: the original COLMAP sparse
model and point cloud are reused. Consequently, validation camera geometry has
indirectly contributed to the initializer, but validation RGB images never
contribute to the training loss.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "private_test1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "local_validation"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

# COLMAP camera model id -> (name, number of parameters).
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]


@dataclass(frozen=True)
class ImagePose:
    image_id: int
    name: str
    camera_id: int
    qvec: tuple[float, float, float, float]
    tvec: tuple[float, float, float]


def _read_exact(file, size: int) -> bytes:
    data = file.read(size)
    if len(data) != size:
        raise ValueError("COLMAP binary file is truncated")
    return data


def read_cameras(path: Path) -> dict[int, Camera]:
    cameras: dict[int, Camera] = {}
    with path.open("rb") as file:
        (count,) = struct.unpack("<Q", _read_exact(file, 8))
        for _ in range(count):
            camera_id, model_id, width, height = struct.unpack(
                "<iiQQ", _read_exact(file, 24)
            )
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            model, parameter_count = CAMERA_MODELS[model_id]
            params = struct.unpack(
                "<" + "d" * parameter_count,
                _read_exact(file, 8 * parameter_count),
            )
            cameras[camera_id] = Camera(
                camera_id=camera_id,
                model=model,
                width=int(width),
                height=int(height),
                params=tuple(params),
            )
    if not cameras:
        raise ValueError(f"No camera found in {path}")
    return cameras


def read_image_poses(path: Path) -> dict[str, ImagePose]:
    poses: dict[str, ImagePose] = {}
    with path.open("rb") as file:
        (count,) = struct.unpack("<Q", _read_exact(file, 8))
        for _ in range(count):
            values = struct.unpack("<idddddddi", _read_exact(file, 64))
            image_id = int(values[0])
            qvec = tuple(float(value) for value in values[1:5])
            tvec = tuple(float(value) for value in values[5:8])
            camera_id = int(values[8])

            name_bytes = bytearray()
            while True:
                character = _read_exact(file, 1)
                if character == b"\0":
                    break
                name_bytes.extend(character)
            name = name_bytes.decode("utf-8")

            (point_count,) = struct.unpack("<Q", _read_exact(file, 8))
            file.seek(point_count * 24, os.SEEK_CUR)
            poses[name] = ImagePose(image_id, name, camera_id, qvec, tvec)
    if not poses:
        raise ValueError(f"No registered image found in {path}")
    return poses


def camera_intrinsics(camera: Camera) -> tuple[float, float, float, float]:
    if camera.model in {
        "SIMPLE_PINHOLE",
        "SIMPLE_RADIAL",
        "RADIAL",
        "SIMPLE_RADIAL_FISHEYE",
        "RADIAL_FISHEYE",
    }:
        focal, cx, cy = camera.params[:3]
        return focal, focal, cx, cy
    fx, fy, cx, cy = camera.params[:4]
    return fx, fy, cx, cy


def distortion_parameters(camera: Camera) -> tuple[float, float, float, float]:
    if camera.model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        return camera.params[3], 0.0, 0.0, 0.0
    if camera.model in {"RADIAL", "RADIAL_FISHEYE"}:
        return camera.params[3], camera.params[4], 0.0, 0.0
    if camera.model in {"OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
        return camera.params[4], camera.params[5], camera.params[6], camera.params[7]
    return 0.0, 0.0, 0.0, 0.0


def _rotation_matrix(qvec: Sequence[float]) -> list[list[float]]:
    w, x, y, z = qvec
    return [
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ]


def colmap_to_nerfstudio_transform(pose: ImagePose) -> list[list[float]]:
    rotation_w2c = _rotation_matrix(pose.qvec)
    rotation_c2w = [[rotation_w2c[column][row] for column in range(3)] for row in range(3)]
    translation_c2w = [
        -sum(rotation_c2w[row][column] * pose.tvec[column] for column in range(3))
        for row in range(3)
    ]
    transform = [
        rotation_c2w[row] + [translation_c2w[row]] for row in range(3)
    ] + [[0.0, 0.0, 0.0, 1.0]]
    # COLMAP/OpenCV camera coordinates -> Nerfstudio/OpenGL coordinates.
    for row in range(3):
        transform[row][1] *= -1
        transform[row][2] *= -1
    return transform


def _natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def _uniform_indices(total: int, count: int) -> list[int]:
    if not 0 < count < total:
        raise ValueError(f"Validation count must be between 1 and {total - 1}, got {count}")
    # Select the centre of equal-width temporal bins. This avoids adjacent-video
    # leakage caused by a random split and is deterministic across experiments.
    indices = [min(total - 1, int((index + 0.5) * total / count)) for index in range(count)]
    if len(set(indices)) != count:
        raise RuntimeError("Could not create unique uniform validation indices")
    return indices


def _count_official_test_poses(scene_path: Path) -> int | None:
    for name in ("test_poses.csv", "test_pose.csv"):
        path = scene_path / "test" / name
        if path.is_file():
            with path.open("r", encoding="utf-8-sig", newline="") as file:
                return sum(1 for _ in csv.DictReader(file))
    return None


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_point_cloud(points_path: Path, output_path: Path) -> int:
    vertices: list[tuple[float, float, float, int, int, int]] = []
    with points_path.open("rb") as file:
        (count,) = struct.unpack("<Q", _read_exact(file, 8))
        for _ in range(count):
            values = struct.unpack("<QdddBBBd", _read_exact(file, 43))
            vertices.append((values[1], values[2], values[3], values[4], values[5], values[6]))
            (track_length,) = struct.unpack("<Q", _read_exact(file, 8))
            file.seek(track_length * 8, os.SEEK_CUR)
    with output_path.open("wb") as file:
        header = (
            "ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(vertices)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        file.write(header.encode("ascii"))
        for vertex in vertices:
            file.write(struct.pack("<fffBBB", *vertex))
    return len(vertices)


def _write_pose_csv(
    path: Path,
    validation_images: Iterable[Path],
    poses: dict[str, ImagePose],
    cameras: dict[int, Camera],
) -> None:
    columns = [
        "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
        "fx", "fy", "cx", "cy", "width", "height",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for image_path in validation_images:
            pose = poses[image_path.name]
            camera = cameras[pose.camera_id]
            fx, fy, cx, cy = camera_intrinsics(camera)
            writer.writerow(
                {
                    "image_name": pose.name,
                    "qw": pose.qvec[0], "qx": pose.qvec[1],
                    "qy": pose.qvec[2], "qz": pose.qvec[3],
                    "tx": pose.tvec[0], "ty": pose.tvec[1], "tz": pose.tvec[2],
                    "fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "width": camera.width, "height": camera.height,
                }
            )


def _write_transforms(
    path: Path,
    training_images: Iterable[Path],
    poses: dict[str, ImagePose],
    cameras: dict[int, Camera],
) -> None:
    training_images = list(training_images)
    used_camera_ids = {poses[image.name].camera_id for image in training_images}
    if len(used_camera_ids) != 1:
        raise ValueError("Local validation currently requires exactly one camera per scene")
    camera = cameras[next(iter(used_camera_ids))]
    fx, fy, cx, cy = camera_intrinsics(camera)
    k1, k2, p1, p2 = distortion_parameters(camera)
    payload = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": camera.width,
        "h": camera.height,
        "k1": k1,
        "k2": k2,
        "p1": p1,
        "p2": p2,
        "ply_file_path": "sparse_pc.ply",
        # Explicit split lists prevent Nerfstudio's default 90/10 fractional
        # split from withholding a second subset. Internal eval deliberately
        # reuses train images; the real local holdout stays entirely external
        # and is scored only after rendering.
        "train_filenames": [f"train/images/{image.name}" for image in training_images],
        "val_filenames": [f"train/images/{image.name}" for image in training_images],
        "test_filenames": [f"train/images/{image.name}" for image in training_images],
        "frames": [
            {
                "file_path": f"train/images/{image.name}",
                "transform_matrix": colmap_to_nerfstudio_transform(poses[image.name]),
            }
            for image in training_images
        ],
    }
    with path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")


def create_split(
    scene: str,
    data_root: Path,
    output_root: Path,
    ratio: float | None,
    count: int | None,
    overwrite: bool,
) -> Path:
    source = (data_root / scene).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Scene not found: {source}")
    image_dir = source / "train" / "images"
    sparse_dir = source / "train" / "sparse" / "0"
    for required in (image_dir, sparse_dir / "cameras.bin", sparse_dir / "images.bin", sparse_dir / "points3D.bin"):
        if not required.exists():
            raise FileNotFoundError(f"Required input not found: {required}")

    images = sorted(
        (path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_key,
    )
    cameras = read_cameras(sparse_dir / "cameras.bin")
    poses = read_image_poses(sparse_dir / "images.bin")
    missing_poses = [image.name for image in images if image.name not in poses]
    if missing_poses:
        raise ValueError(f"Images missing from COLMAP model: {missing_poses[:5]}")

    if count is None:
        if ratio is None:
            official_count = _count_official_test_poses(source)
            ratio = official_count / (len(images) + official_count) if official_count else 0.2
        if not 0.0 < ratio < 1.0:
            raise ValueError("--ratio must be in the open interval (0, 1)")
        count = max(1, round(len(images) * ratio))

    validation_indices = set(_uniform_indices(len(images), count))
    validation_images = [image for index, image in enumerate(images) if index in validation_indices]
    training_images = [image for index, image in enumerate(images) if index not in validation_indices]

    output_root = output_root.resolve()
    destination = output_root / scene
    output_root.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}. Pass --overwrite to replace it.")

    staging = Path(tempfile.mkdtemp(prefix=f".{scene}.staging-", dir=output_root))
    try:
        (staging / "train" / "images").mkdir(parents=True)
        (staging / "train" / "sparse" / "0").mkdir(parents=True)
        (staging / "test").mkdir(parents=True)
        (staging / "local_gt" / "images").mkdir(parents=True)

        link_modes: set[str] = set()
        for image in training_images:
            link_modes.add(_link_or_copy(image, staging / "train" / "images" / image.name))
        for image in validation_images:
            link_modes.add(_link_or_copy(image, staging / "local_gt" / "images" / image.name))
        for sparse_file in sparse_dir.iterdir():
            if sparse_file.is_file():
                link_modes.add(_link_or_copy(sparse_file, staging / "train" / "sparse" / "0" / sparse_file.name))

        source_ply = source / "sparse_pc.ply"
        if source_ply.is_file():
            link_modes.add(_link_or_copy(source_ply, staging / "sparse_pc.ply"))
            point_count = None
        else:
            point_count = _write_point_cloud(sparse_dir / "points3D.bin", staging / "sparse_pc.ply")

        _write_pose_csv(staging / "test" / "test_poses.csv", validation_images, poses, cameras)
        _write_transforms(
            staging / "transforms.json",
            training_images,
            poses,
            cameras,
        )
        manifest = {
            "format_version": 1,
            "validation_type": "fast-val",
            "scene": scene,
            "source_scene": str(source),
            "selection": "uniform-temporal",
            "train_count": len(training_images),
            "validation_count": len(validation_images),
            "validation_ratio": len(validation_images) / len(images),
            "validation_images": [image.name for image in validation_images],
            "file_materialization": sorted(link_modes),
            "point_count": point_count,
            "warning": "The original COLMAP sparse model is reused; compare configs on the same split.",
        }
        with (staging / ".local_validation.json").open("w", encoding="utf-8", newline="\n") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")

        if destination.exists():
            shutil.rmtree(destination)
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    print(f"[OK] Local validation scene: {destination}")
    print(f"     train={len(training_images)}, validation={len(validation_images)}")
    print("     mode=fast-val (RGB holdout, shared COLMAP initializer)")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a safe local-validation split for one scene.")
    parser.add_argument("--scene", required=True, help="Scene name, for example HCM0421")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ratio", type=float, help="Holdout ratio; default mirrors official test ratio")
    group.add_argument("--count", type=int, help="Exact number of validation images")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing derived split")
    arguments = parser.parse_args()
    try:
        create_split(
            arguments.scene,
            arguments.data_root,
            arguments.output_root,
            arguments.ratio,
            arguments.count,
            arguments.overwrite,
        )
    except (OSError, ValueError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
