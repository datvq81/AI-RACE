import os
import csv
import zipfile
from PIL import Image

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN THEO CẤU TRÚC THỰC TẾ
# ==========================================
PRED_DIR = "data/predictions"
SEARCH_DIRS = ["data/public", "data/private_test1"] 
OUTPUT_ZIP = "submission.zip"

def find_csv(scene_name):
    """Tìm file test_poses.csv bên trong thư mục test/ của mỗi scene"""
    for d in SEARCH_DIRS:
        # Đường dẫn dựa trên cấu trúc thực tế bạn cung cấp: <root>/<scene>/test/test_poses.csv
        csv_path = os.path.join(d, scene_name, "test", "test_poses.csv")
        if os.path.exists(csv_path):
            return csv_path
    return None

def create_submission():
    print(f"\n{'='*55}\n 📦 ĐÓNG GÓI SUBMISSION CHUẨN LEADERBOARD\n{'='*55}")
    
    if not os.path.exists(PRED_DIR):
        print(f"[-] Lỗi: Không tìm thấy thư mục {PRED_DIR}")
        return

    scenes = [s for s in os.listdir(PRED_DIR) if os.path.isdir(os.path.join(PRED_DIR, s))]
    if not scenes:
        print(f"[-] Lỗi: Thư mục {PRED_DIR} trống. Hãy chạy render trước!")
        return
        
    total_missing = 0
    
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for scene in sorted(scenes):
            scene_pred_dir = os.path.join(PRED_DIR, scene)
            csv_path = find_csv(scene)
            
            if not csv_path:
                print(f"[-] Bỏ qua {scene}: Không tìm thấy file CSV trong thư mục test/")
                continue
                
            print(f"[>] Đang xử lý Scene: {scene}")
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    img_name = row.get('image_name')
                    if not img_name: continue
                    
                    # Lấy kích thước chuẩn từ CSV
                    target_w = int(row.get('width', row.get('w', 0)))
                    target_h = int(row.get('height', row.get('h', 0)))
                    
                    pred_img_path = os.path.join(scene_pred_dir, img_name)
                    
                    if not os.path.exists(pred_img_path):
                        print(f"    ❌ THIẾU ẢNH: {img_name}")
                        total_missing += 1
                        continue
                        
                    # Tự động Resize nếu cần thiết
                    with Image.open(pred_img_path) as img:
                        if (target_w > 0 and target_h > 0) and (img.size != (target_w, target_h)):
                            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                            img.save(pred_img_path)

                    # Ghi vào ZIP: <scene_name>/<image_name>
                    zipf.write(pred_img_path, os.path.join(scene, img_name))
            
    print(f"\n{'='*55}\n✅ Đóng gói hoàn tất. File: {OUTPUT_ZIP}")
    if total_missing > 0: print(f"⚠️ Cảnh báo: Có {total_missing} ảnh bị thiếu!")

if __name__ == "__main__":
    create_submission()