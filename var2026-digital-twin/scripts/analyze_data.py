import os
import struct
import cv2
import numpy as np

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    fmt = endian_character + format_char_sequence
    bytes_to_read = struct.calcsize(fmt)
    data_bytes = fid.read(bytes_to_read)
    if len(data_bytes) < bytes_to_read: return None
    return struct.unpack(fmt, data_bytes)

def analyze_colmap_data(sparse_dir):
    print("\n" + "="*50)
    print(" 📊 PHÂN TÍCH HÌNH HỌC (COLMAP SPARSE DATA)")
    print("="*50)
    
    # 1. Phân tích Cameras
    cam_file = os.path.join(sparse_dir, "cameras.bin")
    if os.path.exists(cam_file):
        with open(cam_file, "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            print(f"[+] Số lượng Camera Models: {num_cameras}")
    else:
        print("[-] Không tìm thấy cameras.bin")

    # 2. Phân tích Poses (Images)
    img_bin_file = os.path.join(sparse_dir, "images.bin")
    num_reg_images = 0
    if os.path.exists(img_bin_file):
        with open(img_bin_file, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            print(f"[+] Số lượng ảnh đã được map tọa độ 3D (Registered Poses): {num_reg_images} ảnh")
    else:
        print("[-] Không tìm thấy images.bin")

    # 3. Phân tích Point Cloud
    pts_file = os.path.join(sparse_dir, "points3D.bin")
    if os.path.exists(pts_file):
        with open(pts_file, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            print(f"[+] Tổng số điểm 3D (Hạt mầm Gaussian): {num_points:,} điểm")
            if num_points < 10000:
                print("    ⚠️ CẢNH BÁO: Mật độ điểm quá thưa thớt, mô hình sẽ khó hội tụ!")
    else:
        print("[-] Không tìm thấy points3D.bin")
        
    return num_reg_images

def analyze_images(image_dir):
    print("\n" + "="*50)
    print(" 🖼️ PHÂN TÍCH CHẤT LƯỢNG HÌNH ẢNH")
    print("="*50)
    
    if not os.path.exists(image_dir):
        print(f"[-] Không tìm thấy thư mục ảnh: {image_dir}")
        return

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"[+] Tổng số ảnh thô trên ổ cứng: {total_images} ảnh")
    
    resolutions = set()
    blur_scores = []
    corrupted_images = []
    
    print("[*] Đang quét chất lượng từng ảnh (Đo lường độ mờ)...")
    for i, img_name in enumerate(image_files):
        img_path = os.path.join(image_dir, img_name)
        
        # Đọc ảnh dưới dạng Grayscale để tính toán cho nhanh
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            corrupted_images.append(img_name)
            continue
            
        h, w = img.shape
        resolutions.add(f"{w}x{h}")
        
        # Tính điểm độ sắc nét (Laplacian Variance)
        # Điểm càng thấp => Ảnh càng mờ (Motion blur / Out of focus)
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
        blur_scores.append((img_name, blur_score))
        
        if (i + 1) % 50 == 0:
            print(f"    ... Đã quét {i + 1}/{total_images} ảnh")

    # Báo cáo kết quả phân giải
    print(f"\n[+] Độ phân giải phát hiện được: {', '.join(resolutions)}")
    if len(resolutions) > 1:
        print("    ⚠️ CẢNH BÁO: Dữ liệu có nhiều độ phân giải khác nhau! Cần Crop/Resize để đồng nhất.")
        
    # Báo cáo ảnh hỏng
    if corrupted_images:
        print(f"    ❌ PHÁT HIỆN {len(corrupted_images)} ẢNH HỎNG (Corrupted):")
        for bad_img in corrupted_images[:5]: print(f"       - {bad_img}")

    # Thống kê độ mờ
    if blur_scores:
        blur_scores.sort(key=lambda x: x[1]) # Sắp xếp từ mờ nhất đến sắc nét nhất
        avg_blur = sum([x[1] for x in blur_scores]) / len(blur_scores)
        print(f"\n[+] Thống kê độ sắc nét (Blur Score - Variance of Laplacian):")
        print(f"    - Điểm trung bình toàn scene: {avg_blur:.2f}")
        
        # Ngưỡng mờ thông thường là dưới 100.
        blurry_threshold = 100.0 
        blurry_images = [x for x in blur_scores if x[1] < blurry_threshold]
        
        if blurry_images:
            print(f"    ⚠️ CẢNH BÁO: Phát hiện {len(blurry_images)} ảnh có nguy cơ bị mờ nhòe (Score < 100).")
            print("    Top 5 ảnh mờ nhất (Nên cân nhắc loại bỏ để tránh hỏng Gaussian):")
            for name, score in blurry_images[:5]:
                print(f"       - {name} (Score: {score:.2f})")
        else:
            print("    ✅ Toàn bộ ảnh đều sắc nét (Không có ảnh nào dưới ngưỡng 100).")

if __name__ == "__main__":
    # Thay đổi đường dẫn tới Scene bạn muốn phân tích
    TARGET_SCENE = "data/public/hcm0031"
    
    sparse_dir = os.path.join(TARGET_SCENE, "train", "sparse", "0")
    image_dir = os.path.join(TARGET_SCENE, "train", "images")
    
    num_registered = analyze_colmap_data(sparse_dir)
    analyze_images(image_dir)