import os
import cv2

def downscale_images(scene_dir, factor=4):
    # Đường dẫn ảnh gốc của Ban tổ chức
    src_dir = os.path.join(scene_dir, "train", "images")
    
    # Nerfstudio mặc định tìm thư mục images_4 ở ngay cấp ngoài cùng của scene
    dst_dir = os.path.join(scene_dir, f"images_{factor}")
    os.makedirs(dst_dir, exist_ok=True)
    
    images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
    print(f"[>] Đang giảm độ phân giải {len(images)} ảnh xuống {factor} lần...")
    print(f"[>] Quá trình này sẽ mất khoảng 15-30 giây tùy tốc độ ổ cứng...")

    for img_name in images:
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        
        # Bỏ qua nếu ảnh đã được thu nhỏ từ trước
        if os.path.exists(dst_path):
            continue
            
        img = cv2.imread(src_path)
        if img is not None:
            h, w = img.shape[:2]
            # Thu nhỏ bằng thuật toán INTER_AREA để giữ nét tốt nhất khi downscale
            resized = cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_path, resized)

    print(f"[+] Hoàn tất! Đã lưu ảnh vào: {dst_dir}")

if __name__ == "__main__":
    downscale_images("data/public/hcm0031", factor=4)