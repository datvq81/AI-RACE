# scripts/render_test_pose.py
import os
import csv
import json
import argparse
import numpy as np
import cv2
import torch
from pathlib import Path

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType

def qvec2rotmat(qvec):
    """Chuyển Quaternion (w, x, y, z) sang Ma trận xoay 3x3"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def load_dataparser_transforms(transform_path):
    """Đọc file biến đổi tọa độ toàn cục của Nerfstudio"""
    with open(transform_path, 'r') as f:
        meta = json.load(f)
    scale = meta["scale"]
    transform_matrix = np.eye(4)
    transform_matrix[:3, :] = np.array(meta["transform"])
    return scale, transform_matrix

def main(config_path, csv_path, output_dir):
    print(f"[>] Đang nạp mô hình từ: {config_path} ...")
    _, pipeline, _, _ = eval_setup(Path(config_path))
    pipeline.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Tự động tìm file dataparser_transforms.json nằm cùng thư mục với config.yml
    config_dir = os.path.dirname(config_path)
    transform_path = os.path.join(config_dir, "dataparser_transforms.json")
    if not os.path.exists(transform_path):
        raise FileNotFoundError(f"Không tìm thấy {transform_path}. Vui lòng kiểm tra lại thư mục train.")
    
    scale, global_transform = load_dataparser_transforms(transform_path)
    print(f"[+] Đã nạp Global Transform: Scale = {scale}")

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"[>] Tìm thấy {len(rows)} target poses. Bắt đầu render...")
    
    with torch.no_grad():
        for i, row in enumerate(rows):
            img_name = row['image_name']
            
            # 1. Đọc W2C từ COLMAP
            qw, qx, qy, qz = map(float, [row['qw'], row['qx'], row['qy'], row['qz']])
            tx, ty, tz = map(float, [row['tx'], row['ty'], row['tz']])
            
            # 2. Đổi W2C sang C2W
            R_w2c = qvec2rotmat([qw, qx, qy, qz])
            t_w2c = np.array([tx, ty, tz])
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            
            c2w = np.eye(4)
            c2w[:3, :3] = R_c2w
            c2w[:3, 3] = t_c2w
            
            # 3. Đổi trục OpenCV sang OpenGL
            c2w[0:3, 1:3] *= -1
            
            # 4. ÁP DỤNG ĐỒNG BỘ TỌA ĐỘ TRAIN/TEST (Fix lỗi gai nhọn)
            c2w[:3, 3] *= scale
            c2w = global_transform @ c2w
            
            c2w_tensor = torch.tensor(c2w[:3, :4], dtype=torch.float32)

            # 5. Render
            fx, fy = float(row['fx']), float(row['fy'])
            cx, cy = float(row['cx']), float(row['cy'])
            width, height = int(row['width']), int(row['height'])

            camera = Cameras(
                camera_to_worlds=c2w_tensor.unsqueeze(0),
                fx=fx, fy=fy, cx=cx, cy=cy,
                width=width, height=height,
                camera_type=CameraType.PERSPECTIVE
            ).to(pipeline.device)

            outputs = pipeline.model.get_outputs_for_camera(camera)
            rgb = outputs["rgb"].cpu().numpy()
            bgr = (rgb * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
            
            out_img_path = os.path.join(output_dir, img_name)
            cv2.imwrite(out_img_path, bgr)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(rows):
                print(f"    - Rendered {i + 1}/{len(rows)}: {img_name}")

    print(f"[+] HOÀN TẤT! Toàn bộ ảnh đã được lưu tại: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Đường dẫn đến file config.yml")
    parser.add_argument("--csv", type=str, required=True, help="Đường dẫn đến file test_poses.csv")
    parser.add_argument("--out", type=str, required=True, help="Thư mục chứa ảnh render")
    args = parser.parse_args()
    main(args.config, args.csv, args.out)