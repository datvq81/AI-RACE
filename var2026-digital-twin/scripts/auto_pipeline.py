import os
import subprocess
import glob
import sys

# Các thư mục chứa dữ liệu
DATA_DIRS = ["data/public", "data/private_test1"]
PRED_DIR = "data/predictions"

def get_all_scenes():
    """Lấy đường dẫn của toàn bộ 13 scenes"""
    scenes = {}
    for d in DATA_DIRS:
        if os.path.exists(d):
            for s in os.listdir(d):
                scene_path = os.path.join(d, s)
                if os.path.isdir(scene_path) and not s.startswith('.'):
                    scenes[s] = scene_path
    return scenes

def run_pipeline():
    print(f"\n{'='*60}\n 🚀 KHỞI ĐỘNG AUTO-PIPELINE HUẤN LUYỆN 3D\n{'='*60}")
    
    scenes = get_all_scenes()
    
    for scene, path in sorted(scenes.items()):
        scene_pred_dir = os.path.join(PRED_DIR, scene)
        if os.path.exists(scene_pred_dir) and len(os.listdir(scene_pred_dir)) > 0:
            print(f"⏭️  Bỏ qua {scene}: Đã có dữ liệu render.")
            continue
            
        print(f"\n[>] ĐANG XỬ LÝ SCENE: {scene}...")
        
        # 1. KHAI HOẢ HUẤN LUYỆN (Train) - FIX LỖI PATH VỚI PYTHON MODULE
        print(f"    ⏳ Đang huấn luyện 30.000 iterations (sẽ mất khoảng 30-45 phút)...")
        # Sử dụng sys.executable để lấy đúng môi trường Python hiện tại và gọi module
        train_cmd = f"{sys.executable} -m nerfstudio.scripts.train splatfacto --data {path} --pipeline.model.sh-degree 3 --max-num-iterations 30000"
        try:
            subprocess.run(train_cmd, shell=True, check=True)
        except subprocess.CalledProcessError:
            print(f"    ❌ Lỗi trong quá trình huấn luyện {scene}. Chuyển sang scene tiếp theo.")
            continue

        # 2. TÌM FILE CONFIG MỚI NHẤT
        config_search = f"outputs/{scene}/splatfacto/*/config.yml"
        configs = glob.glob(config_search)
        if not configs:
            print(f"    ❌ Không tìm thấy mô hình (config.yml) của {scene} sau khi train.")
            continue
        latest_config = max(configs, key=os.path.getmtime)
        
        # 3. KẾT XUẤT ẢNH DỰ ĐOÁN (Render)
        print(f"    📸 Đang render ảnh từ các góc camera test...")
        csv_path = f"{path}/test/test_poses.csv"
        render_cmd = f"{sys.executable} scripts/render_test_pose.py --config {latest_config} --csv {csv_path} --out {scene_pred_dir}"
        try:
            subprocess.run(render_cmd, shell=True, check=True)
            print(f"    ✅ Đã render xong {scene}!")
        except subprocess.CalledProcessError:
            print(f"    ❌ Lỗi khi render ảnh cho {scene}.")

    print(f"\n{'='*60}\n 🎉 AUTO-PIPELINE ĐÃ HOÀN TẤT TOÀN BỘ CÁC TRẠM!\n{'='*60}")

if __name__ == "__main__":
    run_pipeline()