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

# =========================================================================
# MODULE MỚI: Đọc file nhị phân points3D.bin và xuất ra định dạng PLY
# =========================================================================
def extract_point_cloud(sparse_path, out_ply_path):
    pts_file = os.path.join(sparse_path, "points3D.bin")
    if not os.path.exists(pts_file):
        print("[-] Không tìm thấy points3D.bin để tạo Point Cloud.")
        return False
        
    print(f"[*] Đang trích xuất hạt mầm Point Cloud từ {pts_file}...")
    with open(pts_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        vertices = []
        
        for _ in range(num_points):
            # Cấu trúc của COLMAP: 43 bytes = id(8) + xyz(24) + rgb(3) + error(8)
            data = fid.read(43)
            if not data: break
            _, x, y, z, r, g, b, _ = struct.unpack("<QdddBBBd", data)
            
            # Đọc độ dài track và bỏ qua track elements
            track_length = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(track_length * 8, os.SEEK_CUR)
            
            vertices.append((x, y, z, r, g, b))
            
    # Ghi dữ liệu ra định dạng chuẩn PLY (Binary Little Endian)
    with open(out_ply_path, "wb") as f:
        f.write(b"ply\n")
        f.write(b"format binary_little_endian 1.0\n")
        f.write(f"element vertex {len(vertices)}\n".encode('utf-8'))
        f.write(b"property float x\n")
        f.write(b"property float y\n")
        f.write(b"property float z\n")
        f.write(b"property uchar red\n")
        f.write(b"property uchar green\n")
        f.write(b"property uchar blue\n")
        f.write(b"end_header\n")
        
        for v in vertices:
            # Chuyển double (float64) về float32 để Nerfstudio đọc mượt mà
            f.write(struct.pack("<fffBBB", v[0], v[1], v[2], v[3], v[4], v[5]))
            
    print(f"[+] Đã tạo thành công Point Cloud: {out_ply_path} ({len(vertices):,} điểm)")
    return True

def convert_scene(scene_dir):
    print(f"\n[>] Đang xử lý convert cho scene: {scene_dir}")
    sparse_path = os.path.join(scene_dir, "train", "sparse", "0")
    img_dir = os.path.join(scene_dir, "train", "images")
    out_json = os.path.join(scene_dir, "transforms.json")
    out_ply = os.path.join(scene_dir, "sparse_pc.ply") # Đường dẫn file PLY đầu ra

    if not os.path.exists(img_dir):
        print("[-] Không tìm thấy thư mục images.")
        return
    actual_images = set(os.listdir(img_dir))
    
    # KÍCH HOẠT HÀM TRÍCH XUẤT PLY
    has_ply = extract_point_cloud(sparse_path, out_ply)
    
    # 1. Khởi tạo JSON (Tự động chèn ply_file_path nếu có file)
    out_data = {
        "camera_model": "OPENCV"
    }
    if has_ply:
        out_data["ply_file_path"] = "sparse_pc.ply"
        
    out_data["frames"] = []

    # 2. Đọc Camera Intrinsics
    with open(os.path.join(sparse_path, "cameras.bin"), "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        _, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
        
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
            
            if img_name not in actual_images:
                continue
                
            R_w2c = qvec2rotmat([qw, qx, qy, qz])
            t_w2c = np.array([tx, ty, tz])
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            
            c2w = np.eye(4)
            c2w[0:3, 0:3] = R_c2w
            c2w[0:3, 3] = t_c2w
            
            c2w[0:3, 1:3] *= -1
            
            out_data["frames"].append({
                "file_path": f"train/images/{img_name}",
                "transform_matrix": c2w.tolist()
            })
            valid_count += 1

    # 4. Lưu kết quả JSON
    with open(out_json, "w") as f:
        json.dump(out_data, f, indent=4)
    print(f"[+] Hoàn tất! Đã lưu {valid_count} ảnh hợp lệ vào {out_json}")

if __name__ == "__main__":
    convert_scene("data/public/hcm0031")