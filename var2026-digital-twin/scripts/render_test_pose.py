"""Render các test pose và đưa ảnh pinhole về lưới ảnh méo của camera gốc."""

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.serialization

from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.utils.eval_utils import eval_setup


RENDERER_VERSION = "2.0-redistort"
MANIFEST_FILENAME = ".render_manifest.json"
REDISTORTION_MARGIN_PIXELS = 2
MAX_RENDER_PIXEL_MULTIPLIER = 16

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
class PoseRow:
    image_name: str
    qvec: np.ndarray
    tvec: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class DistortionInfo:
    coefficients: np.ndarray
    camera_model: str
    source_path: Path


@dataclass(frozen=True)
class RenderGeometry:
    target_width: int
    target_height: int
    render_width: int
    render_height: int
    render_cx: float
    render_cy: float
    map_x: Optional[np.ndarray]
    map_y: Optional[np.ndarray]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _load_pipeline(config_path: Path):
    """Nạp checkpoint tin cậy, tương thích cả PyTorch 2.1 và 2.6+."""
    add_safe_globals = getattr(torch.serialization, "add_safe_globals", None)
    if callable(add_safe_globals):
        # API này chưa có trong PyTorch 2.1.2 của Dockerfile.
        add_safe_globals([np.core.multiarray.scalar])

    original_load = torch.load

    def trusted_checkpoint_load(*args, **kwargs):
        # Checkpoint là sản phẩm do chính pipeline huấn luyện, không phải file lạ.
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = trusted_checkpoint_load
    try:
        return eval_setup(config_path)
    finally:
        # Không để monkey-patch ảnh hưởng các phần còn lại của tiến trình.
        torch.load = original_load


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Chuyển quaternion Hamilton (w, x, y, z) thành ma trận xoay W2C."""
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ],
        dtype=np.float64,
    )


def load_dataparser_transforms(transform_path: Path) -> Tuple[float, np.ndarray]:
    """Đọc phép biến đổi toàn cục do Nerfstudio lưu sau khi train."""
    with transform_path.open("r", encoding="utf-8-sig") as file:
        metadata = json.load(file)

    try:
        scale = float(metadata["scale"])
        transform = np.asarray(metadata["transform"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Dataparser transform không hợp lệ: {transform_path}") from error

    if transform.shape != (3, 4):
        raise ValueError(
            f"'transform' phải có kích thước 3x4, nhận được {transform.shape}: {transform_path}"
        )
    if not math.isfinite(scale) or scale <= 0 or not np.isfinite(transform).all():
        raise ValueError(f"Scale/transform chứa giá trị không hợp lệ: {transform_path}")

    transform_matrix = np.eye(4, dtype=np.float64)
    transform_matrix[:3, :] = transform
    return scale, transform_matrix


def _finite_float(row: dict, key: str, line_number: int) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Dòng CSV {line_number}: '{key}' không phải số hợp lệ") from error
    if not math.isfinite(value):
        raise ValueError(f"Dòng CSV {line_number}: '{key}' phải là số hữu hạn")
    return value


def _positive_integer(row: dict, key: str, line_number: int) -> int:
    value = _finite_float(row, key, line_number)
    if value <= 0 or not value.is_integer():
        raise ValueError(f"Dòng CSV {line_number}: '{key}' phải là số nguyên dương")
    return int(value)


def _normalise_image_name(raw_name: str, line_number: int) -> str:
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
        raise ValueError(f"Dòng CSV {line_number}: image_name không an toàn: {raw_name!r}")
    return posix_path.as_posix()


def read_pose_rows(csv_path: Path) -> List[PoseRow]:
    rows: List[PoseRow] = []
    seen_names = set()
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_CSV_COLUMNS - fieldnames)
        if missing_columns:
            raise ValueError(
                f"CSV thiếu các cột bắt buộc {missing_columns}: {csv_path}"
            )

        for row in reader:
            line_number = reader.line_num
            image_name = _normalise_image_name(row.get("image_name", ""), line_number)
            name_key = image_name.casefold()
            if name_key in seen_names:
                raise ValueError(f"Dòng CSV {line_number}: image_name bị trùng: {image_name}")
            seen_names.add(name_key)

            qvec = np.array(
                [_finite_float(row, key, line_number) for key in ("qw", "qx", "qy", "qz")],
                dtype=np.float64,
            )
            qnorm = float(np.linalg.norm(qvec))
            if qnorm < 1e-12:
                raise ValueError(f"Dòng CSV {line_number}: quaternion có norm bằng 0")
            qvec /= qnorm

            tvec = np.array(
                [_finite_float(row, key, line_number) for key in ("tx", "ty", "tz")],
                dtype=np.float64,
            )
            fx = _finite_float(row, "fx", line_number)
            fy = _finite_float(row, "fy", line_number)
            if fx <= 0 or fy <= 0:
                raise ValueError(f"Dòng CSV {line_number}: fx và fy phải lớn hơn 0")

            rows.append(
                PoseRow(
                    image_name=image_name,
                    qvec=qvec,
                    tvec=tvec,
                    fx=fx,
                    fy=fy,
                    cx=_finite_float(row, "cx", line_number),
                    cy=_finite_float(row, "cy", line_number),
                    width=_positive_integer(row, "width", line_number),
                    height=_positive_integer(row, "height", line_number),
                )
            )

    if not rows:
        raise ValueError(f"CSV không chứa test pose nào: {csv_path}")
    return rows


def _find_camera_metadata(csv_path: Path, explicit_path: Optional[Path]) -> Path:
    candidates = [explicit_path] if explicit_path is not None else [
        csv_path.parent.parent / "transforms.json",
        csv_path.parent / "transforms.json",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Không tìm thấy transforms.json chứa thông số distortion. "
        f"Đã kiểm tra: {searched}. Có thể chỉ định bằng --camera-meta."
    )


def load_distortion_info(csv_path: Path, explicit_path: Optional[Path]) -> DistortionInfo:
    metadata_path = _find_camera_metadata(csv_path, explicit_path)
    with metadata_path.open("r", encoding="utf-8-sig") as file:
        metadata = json.load(file)

    camera_model = str(metadata.get("camera_model", "OPENCV")).upper()
    if "FISHEYE" in camera_model:
        raise ValueError(
            f"Camera model {camera_model} cần công thức fisheye riêng; script hiện dùng radial OpenCV."
        )

    # OpenCV dùng thứ tự [k1, k2, p1, p2, k3]. Nerfstudio đôi khi lưu
    # distortion_params theo thứ tự [k1, k2, k3, k4, p1, p2].
    if any(key in metadata for key in ("k1", "k2", "k3", "p1", "p2")):
        values = [
            metadata.get("k1", 0.0),
            metadata.get("k2", 0.0),
            metadata.get("p1", 0.0),
            metadata.get("p2", 0.0),
            metadata.get("k3", 0.0),
        ]
    else:
        nerfstudio_values = list(metadata.get("distortion_params", []))
        nerfstudio_values.extend([0.0] * (6 - len(nerfstudio_values)))
        k1, k2, k3, k4, p1, p2 = nerfstudio_values[:6]
        if float(k4) != 0.0:
            raise ValueError("OpenCV 5 hệ số trong script chưa hỗ trợ k4 khác 0")
        values = [k1, k2, p1, p2, k3]

    try:
        coefficients = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Thông số distortion không hợp lệ: {metadata_path}") from error
    if coefficients.shape != (5,) or not np.isfinite(coefficients).all():
        raise ValueError(f"Thông số distortion không hợp lệ: {metadata_path}")

    return DistortionInfo(
        coefficients=coefficients,
        camera_model=camera_model,
        source_path=metadata_path,
    )


def build_redistortion_geometry(
    pose: PoseRow,
    coefficients: np.ndarray,
) -> RenderGeometry:
    """Tạo inverse-map: pixel ảnh méo đích -> pixel pinhole nguồn."""
    if np.count_nonzero(coefficients) == 0:
        return RenderGeometry(
            target_width=pose.width,
            target_height=pose.height,
            render_width=pose.width,
            render_height=pose.height,
            render_cx=pose.cx,
            render_cy=pose.cy,
            map_x=None,
            map_y=None,
        )

    camera_matrix = np.array(
        [[pose.fx, 0.0, pose.cx], [0.0, pose.fy, pose.cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    grid_x, grid_y = np.meshgrid(
        np.arange(pose.width, dtype=np.float32),
        np.arange(pose.height, dtype=np.float32),
    )
    distorted_pixels = np.stack((grid_x, grid_y), axis=-1).reshape(-1, 1, 2)
    undistorted_pixels = cv2.undistortPoints(
        distorted_pixels,
        camera_matrix,
        coefficients,
        P=camera_matrix,
    ).reshape(pose.height, pose.width, 2)

    if not np.isfinite(undistorted_pixels).all():
        raise ValueError("Lưới redistortion chứa NaN/Inf; hãy kiểm tra hệ số camera")

    undistorted_x = undistorted_pixels[:, :, 0]
    undistorted_y = undistorted_pixels[:, :, 1]
    min_x = min(
        0,
        math.floor(float(undistorted_x.min())) - REDISTORTION_MARGIN_PIXELS,
    )
    min_y = min(
        0,
        math.floor(float(undistorted_y.min())) - REDISTORTION_MARGIN_PIXELS,
    )
    max_x = max(
        pose.width - 1,
        math.ceil(float(undistorted_x.max())) + REDISTORTION_MARGIN_PIXELS,
    )
    max_y = max(
        pose.height - 1,
        math.ceil(float(undistorted_y.max())) + REDISTORTION_MARGIN_PIXELS,
    )

    render_width = max_x - min_x + 1
    render_height = max_y - min_y + 1
    if (
        render_width * render_height
        > pose.width * pose.height * MAX_RENDER_PIXEL_MULTIPLIER
    ):
        raise ValueError(
            "Distortion yêu cầu canvas pinhole quá lớn "
            f"({render_width}x{render_height} cho ảnh {pose.width}x{pose.height})"
        )

    # Dịch lưới và principal point cùng một lượng để tia camera không đổi.
    map_x = np.ascontiguousarray(undistorted_x - min_x, dtype=np.float32)
    map_y = np.ascontiguousarray(undistorted_y - min_y, dtype=np.float32)

    return RenderGeometry(
        target_width=pose.width,
        target_height=pose.height,
        render_width=render_width,
        render_height=render_height,
        render_cx=pose.cx - min_x,
        render_cy=pose.cy - min_y,
        map_x=map_x,
        map_y=map_y,
    )


def _geometry_key(pose: PoseRow, coefficients: np.ndarray) -> Tuple[float, ...]:
    return (
        float(pose.width),
        float(pose.height),
        pose.fx,
        pose.fy,
        pose.cx,
        pose.cy,
        *coefficients.tolist(),
    )


def _pose_to_nerfstudio_c2w(
    pose: PoseRow,
    global_transform: np.ndarray,
    scale: float,
) -> np.ndarray:
    # CSV/COLMAP lưu world-to-camera (OpenCV).
    rotation_w2c = qvec2rotmat(pose.qvec)
    rotation_c2w = rotation_w2c.T
    translation_c2w = -rotation_c2w @ pose.tvec

    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation_c2w
    c2w[:3, 3] = translation_c2w

    # OpenCV (+Y xuống, +Z trước) -> OpenGL/Nerfstudio (+Y lên, -Z trước).
    c2w[:3, 1:3] *= -1

    # Giữ đúng thứ tự của Nerfstudio: transform toàn cục trước, scale translation sau.
    c2w = global_transform @ c2w
    c2w[:3, 3] *= scale
    return c2w


def _output_path(output_root: Path, image_name: str) -> Path:
    relative_path = Path(*PurePosixPath(image_name).parts)
    output_path = (output_root / relative_path).resolve()
    try:
        output_path.relative_to(output_root)
    except ValueError as error:
        raise ValueError(f"Đường dẫn ảnh vượt khỏi output_dir: {image_name}") from error
    return output_path


def _write_image_atomic(output_path: Path, bgr: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    try:
        success = cv2.imwrite(str(temp_path), bgr)
        if not success:
            raise IOError(f"Không thể ghi file ảnh tại: {output_path}")
        os.replace(temp_path, output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _clean_expected_outputs(output_root: Path, rows: List[PoseRow]) -> None:
    """Chỉ xóa các ảnh do CSV hiện tại quản lý, không rm -rf cả thư mục."""
    for pose in rows:
        output_path = _output_path(output_root, pose.image_name)
        if output_path.is_file() or output_path.is_symlink():
            output_path.unlink()
        elif output_path.exists():
            raise IsADirectoryError(f"Output ảnh lại là một thư mục: {output_path}")


def main(
    config_path: str,
    csv_path: str,
    output_dir: str,
    camera_meta_path: Optional[str] = None,
    disable_redistortion: bool = False,
    clean_output: bool = False,
) -> None:
    config = Path(config_path).resolve()
    pose_csv = Path(csv_path).resolve()
    output_root = Path(output_dir).resolve()
    explicit_camera_meta = (
        Path(camera_meta_path).resolve() if camera_meta_path is not None else None
    )

    if not config.is_file():
        raise FileNotFoundError(f"Không tìm thấy config: {config}")
    if not pose_csv.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV: {pose_csv}")

    transform_path = config.parent / "dataparser_transforms.json"
    if not transform_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy {transform_path}. Hãy kiểm tra thư mục train."
        )

    rows = read_pose_rows(pose_csv)
    scale, global_transform = load_dataparser_transforms(transform_path)

    if disable_redistortion:
        distortion_info = None
        coefficients = np.zeros(5, dtype=np.float64)
        print("[!] Đã tắt redistortion: ảnh đầu ra sẽ là pinhole.")
    else:
        distortion_info = load_distortion_info(pose_csv, explicit_camera_meta)
        coefficients = distortion_info.coefficients
        print(
            "[+] Distortion từ "
            f"{distortion_info.source_path}: model={distortion_info.camera_model}, "
            f"[k1, k2, p1, p2, k3]={coefficients.tolist()}"
        )

    geometry_cache: Dict[Tuple[float, ...], RenderGeometry] = {}
    for pose in rows:
        key = _geometry_key(pose, coefficients)
        if key not in geometry_cache:
            geometry = build_redistortion_geometry(pose, coefficients)
            geometry_cache[key] = geometry
            if geometry.map_x is not None:
                print(
                    "[+] Canvas pinhole "
                    f"{geometry.render_width}x{geometry.render_height} -> "
                    f"ảnh méo {geometry.target_width}x{geometry.target_height}"
                )

    print(f"[>] Đang nạp mô hình từ: {config}")
    _, pipeline, _, _ = _load_pipeline(config)
    pipeline.eval()
    print(f"[+] Global transform đã nạp; scale={scale}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest_path.unlink()
    if clean_output:
        _clean_expected_outputs(output_root, rows)

    print(f"[>] Có {len(rows)} target poses. Bắt đầu render...")
    with torch.inference_mode():
        for index, pose in enumerate(rows, start=1):
            geometry = geometry_cache[_geometry_key(pose, coefficients)]
            c2w = _pose_to_nerfstudio_c2w(pose, global_transform, scale)
            c2w_tensor = torch.tensor(c2w[:3, :4], dtype=torch.float32)

            camera = Cameras(
                camera_to_worlds=c2w_tensor.unsqueeze(0),
                fx=pose.fx,
                fy=pose.fy,
                cx=geometry.render_cx,
                cy=geometry.render_cy,
                width=geometry.render_width,
                height=geometry.render_height,
                camera_type=CameraType.PERSPECTIVE,
            ).to(pipeline.device)

            outputs = pipeline.model.get_outputs_for_camera(camera)
            if "rgb" not in outputs:
                raise KeyError("Model output không có khóa 'rgb'")
            rgb = outputs["rgb"].detach().cpu().numpy()
            expected_shape = (geometry.render_height, geometry.render_width, 3)
            if rgb.shape != expected_shape:
                raise ValueError(
                    f"Ảnh render {pose.image_name} có shape {rgb.shape}, "
                    f"mong đợi {expected_shape}"
                )

            bgr_pinhole = np.rint(np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)[
                :, :, ::-1
            ]
            if geometry.map_x is None:
                bgr_output = bgr_pinhole
            else:
                bgr_output = cv2.remap(
                    bgr_pinhole,
                    geometry.map_x,
                    geometry.map_y,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )

            expected_output_shape = (
                geometry.target_height,
                geometry.target_width,
                3,
            )
            if bgr_output.shape != expected_output_shape:
                raise ValueError(
                    f"Ảnh sau redistortion có shape {bgr_output.shape}, "
                    f"mong đợi {expected_output_shape}"
                )

            output_path = _output_path(output_root, pose.image_name)
            _write_image_atomic(output_path, bgr_output)

            if index % 10 == 0 or index == len(rows):
                print(f"    - Rendered {index}/{len(rows)}: {pose.image_name}")

    manifest = {
        "schema_version": 1,
        "renderer_version": RENDERER_VERSION,
        "renderer_sha256": _sha256_file(Path(__file__).resolve()),
        "config_path": str(config),
        "config_sha256": _sha256_file(config),
        "csv_path": str(pose_csv),
        "csv_sha256": _sha256_file(pose_csv),
        "camera_metadata_path": (
            str(distortion_info.source_path) if distortion_info is not None else None
        ),
        "camera_metadata_sha256": (
            _sha256_file(distortion_info.source_path)
            if distortion_info is not None
            else None
        ),
        "redistortion_enabled": not disable_redistortion,
        "distortion_coefficients_opencv": coefficients.tolist(),
        "image_count": len(rows),
        "image_names": [pose.image_name for pose in rows],
    }
    _atomic_write_json(manifest_path, manifest)
    print(f"[+] Hoàn tất: {len(rows)} ảnh đã được lưu tại {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render test poses và redistort về camera/lưới ảnh gốc."
    )
    parser.add_argument("--config", required=True, help="Đường dẫn config.yml")
    parser.add_argument("--csv", required=True, help="Đường dẫn test_poses.csv")
    parser.add_argument("--out", required=True, help="Thư mục lưu ảnh render")
    parser.add_argument(
        "--camera-meta",
        help="Đường dẫn transforms.json chứa k1/k2/p1/p2 (mặc định tự tìm từ CSV)",
    )
    parser.add_argument(
        "--disable-redistortion",
        action="store_true",
        help="Render pinhole thuần túy để đối chiếu/A-B test",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Xóa các ảnh đích cũ trong CSV trước khi render lại",
    )
    arguments = parser.parse_args()
    main(
        arguments.config,
        arguments.csv,
        arguments.out,
        camera_meta_path=arguments.camera_meta,
        disable_redistortion=arguments.disable_redistortion,
        clean_output=arguments.clean_output,
    )
