import os
import csv
import zipfile
from PIL import Image
import io

# ==========================================
# CẤU HÌNH THƯ MỤC KIỂM TRA CHÉO
# ==========================================
SUBMISSION_ZIP = "submission.zip"
DATA_DIRS = ["data/public", "data/private_test1"]

def get_all_expected_scenes():
    """Lấy toàn bộ danh sách scene thực tế đang có trong thư mục data"""
    expected_scenes = set()
    for d in DATA_DIRS:
        if os.path.exists(d):
            for s in os.listdir(d):
                if os.path.isdir(os.path.join(d, s)):
                    expected_scenes.add(s)
    return expected_scenes

def find_csv_path(scene_name):
    """Tìm file CSV tọa độ gốc của BTC"""
    for d in DATA_DIRS:
        csv_path = os.path.join(d, scene_name, "test", "test_poses.csv")
        if os.path.exists(csv_path):
            return csv_path
    return None

def check_submission():
    print("\n" + "="*60)
    print(" 🛡️  HỆ THỐNG KIỂM TRA SUBMISSION TỰ ĐỘNG (PRE-FLIGHT CHECK)")
    print("="*60)

    if not os.path.exists(SUBMISSION_ZIP):
        print(f"❌ THẤT BẠI: Không tìm thấy file '{SUBMISSION_ZIP}' tại thư mục gốc!")
        return

    # 1. Đọc cấu trúc bên trong file ZIP
    try:
        with zipfile.ZipFile(SUBMISSION_ZIP, 'r') as zipf:
            namelist = zipf.namelist()
    except Exception as e:
        print(f"❌ THẤT BẠI: File ZIP bị lỗi cấu trúc hoặc không thể giải nén: {e}")
        return

    # Lấy danh sách các scene có trong file ZIP (dựa trên tên thư mục cha)
    submitted_scenes = set()
    for name in namelist:
        parts = name.strip("/").split("/")
        if parts:
            submitted_scenes.add(parts[0])

    expected_scenes = get_all_expected_scenes()
    
    print(f"[*] Tìm thấy {len(submitted_scenes)} scenes trong file ZIP.")
    print(f"[*] Hệ thống yêu cầu kiểm tra chéo với {len(expected_scenes)} scenes gốc.")

    # 2. KIỂM TRA LUẬT 1: Đúng số lượng và tên scene (Không thừa, không thiếu)
    missing_scenes = expected_scenes - submitted_scenes
    extra_scenes = submitted_scenes - expected_scenes
    
    has_error = False

    if missing_scenes:
        print(f"❌ LỖI CHÍ MẠNG: Bạn bị THIẾU các scene sau trong file nộp: {sorted(list(missing_scenes))}")
        has_error = True
    if extra_scenes:
        print(f"❌ LỖI CHÍ MẠNG: Bạn bị THỪA các scene không hợp lệ: {sorted(list(extra_scenes))}")
        has_error = True

    # 3. KIỂM TRA LUẬT 2: Chi tiết từng file ảnh bên trong các scene hợp lệ
    for scene in sorted(submitted_scenes):
        if scene not in expected_scenes:
            continue
            
        csv_path = find_csv_path(scene)
        if not csv_path:
            print(f"⚠️ Cảnh báo: Không tìm thấy file cấu hình test_poses.csv cho scene {scene} để đối chiếu.")
            continue

        print(f"\n[>] Đang quét kỹ thuật Scene: {scene}")
        
        # Đọc danh sách ảnh bắt buộc từ file CSV của BTC
        expected_images = {}
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row.get('image_name')
                if img_name:
                    expected_images[img_name] = {
                        'w': int(row.get('width', row.get('w', 0))),
                        'h': int(row.get('height', row.get('h', 0)))
                    }

        # Kiểm tra từng ảnh bắt buộc
        scene_missing_count = 0
        scene_size_error_count = 0
        
        with zipfile.ZipFile(SUBMISSION_ZIP, 'r') as zipf:
            for img_name, target_size in expected_images.items():
                archive_path = f"{scene}/{img_name}"
                
                # Kiểm tra sự tồn tại của ảnh
                if archive_path not in namelist:
                    print(f"   ❌ Thiếu ảnh bắt buộc: {archive_path}")
                    scene_missing_count += 1
                    has_error = True
                    continue
                
                # Kiểm tra kích thước hình ảnh (Width, Height)
                try:
                    img_data = zipf.read(archive_path)
                    with Image.open(io.BytesIO(img_data)) as img:
                        w, h = img.size
                        if target_size['w'] > 0 and target_size['h'] > 0:
                            if w != target_size['w'] or h != target_size['h']:
                                print(f"   ❌ Sai kích thước ở {img_name}: Hiện tại ({w}x{h}) - Yêu cầu ({target_size['w']}x{target_size['h']})")
                                scene_size_error_count += 1
                                has_error = True
                except Exception as e:
                    print(f"   ❌ Ảnh {img_name} bị lỗi định dạng hoặc hỏng file: {e}")
                    has_error = True

        # Báo cáo nhanh cho scene
        if scene_missing_count == 0 and scene_size_error_count == 0:
            print(f"   📊 Định dạng & Số lượng: HỢP LỆ ({len(expected_images)}/{len(expected_images)} ảnh)")
        else:
            print(f"   📊 Trạng thái lỗi: Thiếu {scene_missing_count} ảnh | Sai kích thước {scene_size_error_count} ảnh.")

    print("\n" + "="*60)
    if has_error:
        print("🚨 KẾT LUẬN: FILE SUBMISSION KHÔNG HỢP LỆ!")
        print("👉 Vui lòng không nộp file này lên hệ thống. Hãy kiểm tra các lỗi '❌' ở trên.")
    else:
        print("🏆 KẾT LUẬN: FILE SUBMISSION HOÀN HẢO 100%!")
        print("👉 Đầy đủ scene, đúng số lượng ảnh, khớp tuyệt đối độ phân giải hình ảnh.")
        print("👉 Bạn có thể tự tin tải file 'submission.zip' lên Leaderboard ngay bây giờ.")
    print("="*60 + "\n")

if __name__ == "__main__":
    check_submission()