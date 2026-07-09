# scripts/inspect_scene.py
import os
import struct
import numpy as np

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Đọc dữ liệu nhị phân theo định dạng chỉ định."""
    fmt = endian_character + format_char_sequence
    bytes_to_read = struct.calcsize(fmt)
    data_bytes = fid.read(bytes_to_read)
    if len(data_bytes) < bytes_to_read:
        return None
    return struct.unpack(fmt, data_bytes)

def read_cameras_binary(path_to_model_file):
    """Đọc file cameras.bin của COLMAP."""
    cameras = {}
    # Định nghĩa ID các model camera của COLMAP
    CAMERA_MODEL_NAMES = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL", 3: "RADIAL", 4: "OPENCV"}
    
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            if camera_properties is None:
                break
            camera_id, model_id, width, height = camera_properties
            model_name = CAMERA_MODEL_NAMES.get(model_id, f"UNKNOWN_{model_id}")
            
            # Đọc các tham số tùy theo model
            num_params = 0
            if model_id in [0, 2]: num_params = 3  # f, cx, cy
            elif model_id in [1, 3]: num_params = 4 # fx, fy, cx, cy
            elif model_id == 4: num_params = 8      # fx, fy, cx, cy, k1, k2, p1, p2
            
            params = []
            for _ in range(num_params):
                param = read_next_bytes(fid, 8, "d")[0]
                params.append(param)
                
            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params
            }
    return cameras

def qvec2rotmat(qvec):
    """Chuyển đổi Quaternion (w, x, y, z) sang Ma trận xoay 3x3."""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def read_images_binary(path_to_model_file):
    """Đọc file images.bin của COLMAP để lấy Extrinsics (W2C)."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_header = read_next_bytes(fid, 64, "idddddddi")
            if binary_image_header is None:
                break
            image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = binary_image_header
            
            # Đọc tên file ảnh (chuỗi kết thúc bằng byte \0)
            image_name = b""
            while True:
                char = fid.read(1)
                if char == b"\0" or char == b"":
                    break
                image_name += char
            image_name = image_name.decode("utf-8")
            
            # Bỏ qua dữ liệu các điểm keypoint 2D trong file nhị phân
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(num_points2D * 24, os.SEEK_CUR)
            
            # Tính toán ma trận C2W từ cấu trúc W2C của COLMAP
            qvec = np.array([qw, qx, qy, qz])
            R_w2c = qvec2rotmat(qvec)
            t_w2c = np.array([tx, ty, tz])
            
            # Nghịch đảo ma trận để ra hệ Camera-to-World (C2W) dùng cho GS
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ t_w2c
            
            c2w = np.eye(4)
            c2w[0:3, 0:3] = R_c2w
            c2w[0:3, 3] = t_c2w
            
            images[image_id] = {
                "name": image_name,
                "camera_id": camera_id,
                "c2w_matrix": c2w
            }
    return images

def inspect_scene(scene_path):
    print(f"\n================ INSPECTING: {os.path.basename(scene_path)} ================")
    sparse_path = os.path.join(scene_path, "train", "sparse", "0")
    images_dir = os.path.join(scene_path, "train", "images")
    test_dir = os.path.join(scene_path, "test")
    
    if not os.path.exists(sparse_path):
        print(f"[-] Lỗi: Không tìm thấy thư mục cấu trúc sparse tại {sparse_path}")
        return
        
    # 1. Đọc Intrinsics
    cameras = read_cameras_binary(os.path.join(sparse_path, "cameras.bin"))
    print(f"[+] Tìm thấy {len(cameras)} camera profile(s):")
    for cam_id, cam in cameras.items():
        print(f"   - Camera ID {cam_id}: Model={cam['model']}, Size={cam['width']}x{cam['height']}")
        print(f"     Params (fx, fy, cx, cy...): {cam['params']}")

    # 2. Đọc Extrinsics
    images = read_images_binary(os.path.join(sparse_path, "images.bin"))
    print(f"[+] Tìm thấy {len(images)} ảnh đã được định vị (registered) trong file sparse.")
    
    # 3. Kiểm tra tệp ảnh thực tế
    actual_images = os.listdir(images_dir) if os.path.exists(images_dir) else []
    print(f"[+] Số ảnh thực tế trong thư mục train/images: {len(actual_images)}")
    
    # Kiểm tra thử hướng nhìn của camera đầu tiên để xem trục tọa độ
    if images:
        first_id = list(images.keys())[0]
        sample_c2w = images[first_id]["c2w_matrix"]
        print(f"[+] Ma trận vị trí C2W mẫu (Ảnh: {images[first_id]['name']}):")
        print(sample_c2w)
        # Vector hướng nhìn của camera trong không gian thế giới (Trục Z của hệ OpenGL/OpenCV)
        look_vector = sample_c2w[0:3, 2]
        print(f"   - Hướng vector nhìn của camera: {look_vector}")

    # 4. Kiểm tra cấu trúc thư mục Test mục tiêu
    if os.path.exists(test_dir):
        test_contents = os.listdir(test_dir)
        print(f"[+] Thư mục 'test' chứa: {test_contents}")
        # Kiểm tra xem có thư mục con images hoặc file csv/json nào không
        for item in test_contents:
            item_path = os.path.join(test_dir, item)
            if os.path.isdir(item_path):
                print(f"   - Thư mục con: test/{item}/ chứa {len(os.listdir(item_path))} items")
    else:
        print("[-] Cảnh báo: Không tìm thấy thư mục 'test'")

if __name__ == "__main__":
    # Chạy thử nghiệm kiểm tra trên scene public đầu tiên
    inspect_scene("data/public/hcm0031")