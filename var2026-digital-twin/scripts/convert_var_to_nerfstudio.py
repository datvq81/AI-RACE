# scripts/convert_var_to_nerfstudio.py
import os
import json
import struct
import numpy as np

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    fmt = endian_character + format_char_sequence
    bytes_to_read = struct.calcsize(fmt)
    data_bytes = fid.read(bytes_to_read)
    if len(data_bytes) < bytes_to_read: return None
    return struct.unpack(fmt, data_bytes)

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2, 2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3], 2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3], 1 - 2 * qvec[1]**2 - 2 * qvec[3]**2, 2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2], 2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1], 1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def convert_scene(scene_dir):
    print(f"\n[>] Đang xử lý convert cho scene: {scene_dir}")
    sparse_path = os.path.join(scene_dir, "train", "sparse", "0")
    img_dir = os.path.join(scene_dir, "train", "images")
    out_json = os.path.join(scene_dir, "transforms.json")

    # Đọc danh sách ảnh có thật trên ổ cứng
    if not os.path.exists(img_dir):
        print("[-] Không tìm thấy thư mục images.")
        return
    actual_images = set(os.listdir(img_dir))
    
    # 1. Khởi tạo JSON
    out_data = {
        "camera_model": "OPENCV",
        "frames": []
    }

    # 2. Đọc Camera Intrinsics
    with open(os.path.join(sparse_path, "cameras.bin"), "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        # Đọc camera đầu tiên
        _, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
        
        # SIMPLE_RADIAL (id=2) thường có 4 tham số: f, cx, cy, k1.
        # Xử lý linh hoạt số lượng tham số tùy chuẩn COLMAP
        params = []
        while True:
            try:
                param = read_next_bytes(fid, 8, "d")
                if param is None: break
                params.append(param[0])
            except:
                break
        
        f = params[0]
        cx, cy = params[1], params[2]
        
        out_data.update({
            "fl_x": f, "fl_y": f,
            "cx": cx, "cy": cy,
            "w": width, "h": height,
            "k1": params[3] if len(params) > 3 else 0.0,
            "k2": 0.0, "p1": 0.0, "p2": 0.0
        })

    # 3. Đọc Poses và Ánh xạ Hệ tọa độ
    with open(os.path.join(sparse_path, "images.bin"), "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        valid_count = 0
        
        for _ in range(num_reg_images):
            header = read_next_bytes(fid, 64, "idddddddi")
            if header is None: break
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = header
            
            img_name = b""
            while True:
                char = fid.read(1)
                if char == b"\0" or char == b"": break
                img_name += char
            img_name = img_name.decode("utf-8")
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * 24, os.SEEK_CUR)
            
            # BỘ LỌC QUAN TRỌNG: Bỏ qua ảnh ảo
            if img_name not in actual_images:
                continue
                
            # Đổi W2C thành C2W
            R_w2c = qvec2rotmat([qw, qx, qy, qz])
            t_w2c = np.array([tx, ty, tz])
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            
            c2w = np.eye(4)
            c2w[0:3, 0:3] = R_c2w
            c2w[0:3, 3] = t_c2w
            
            # ĐỔI TRỤC OPENCV SANG OPENGL (Nhân cột Y và Z với -1)
            c2w[0:3, 1:3] *= -1
            
            out_data["frames"].append({
                "file_path": f"train/images/{img_name}",
                "transform_matrix": c2w.tolist()
            })
            valid_count += 1

    # 4. Lưu kết quả
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=4)
    print(f"[+] Hoàn tất! Đã lưu {valid_count} ảnh hợp lệ vào {out_json}")

if __name__ == "__main__":
    # Thay đổi đường dẫn này để quét toàn bộ các scene sau này
    convert_scene("data/public/hcm0031")