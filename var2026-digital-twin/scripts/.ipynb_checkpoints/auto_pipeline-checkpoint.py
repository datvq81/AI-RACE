import os
import subprocess
import glob
import sys

# Các thư mục chứa dữ liệu
DATA_DIRS = ["data/private_test1", "data/public"]
PRED_DIR = "data/predictions"

def get_all_scenes():
    """Lấy đường dẫn của toàn bộ scenes"""
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
    
    print(f"[*] Đang rà soát và vá lỗi dữ liệu COLMAP (Tự động sinh transforms.json)...")
    try:
        subprocess.run(f"{sys.executable} scripts/convert_var_to_nerfstudio.py", shell=True)
    except Exception as e:
        print(f"⚠️ Không thể chạy kịch bản convert tự động: {e}")

    scenes = get_all_scenes()
    
    for scene, path in sorted(scenes.items()):
        scene_pred_dir = os.path.join(PRED_DIR, scene)
        if os.path.exists(scene_pred_dir) and len(os.listdir(scene_pred_dir)) > 0:
            print(f"⏭️  Bỏ qua {scene}: Đã có dữ liệu render.")
            continue
            
        print(f"\n[>] ĐANG XỬ LÝ SCENE: {scene}...")
        
        # TÌM TẤT CẢ CÁC CONFIG CỦA SCENE NÀY (Tìm cả splatfacto và splatfacto-big)
        config_search = f"outputs/{scene}/*/*/config.yml"
        configs = glob.glob(config_search)
        
        is_trained = False
        latest_config = None
        
        # KIỂM TRA CHỐNG LỖI "VỎ RỖNG"
        if configs:
            latest_config = max(configs, key=os.path.getmtime)
            model_dir = os.path.join(os.path.dirname(latest_config), "nerfstudio_models")
            
            # Bắt buộc phải có file checkpoint (.ckpt) thì mới tính là đã train xong
            if os.path.exists(model_dir) and any(f.endswith('.ckpt') for f in os.listdir(model_dir)):
                is_trained = True
                print(f"    ⏭️ Đã tìm thấy mô hình huấn luyện hoàn chỉnh. Bỏ qua bước Train!")
            else:
                print(f"    ⚠️ Phát hiện mô hình lỗi/chưa train xong ở {scene}. Bắt đầu train lại!")

        if not is_trained:
            # 1. KHAI HOẢ HUẤN LUYỆN (Train) - Bản nâng cấp xịn nhất
            print(f"    ⏳ Đang huấn luyện 30.000 iterations...")
            
            base_args = f"--experiment-name {scene} --pipeline.model.sh-degree 3 --pipeline.model.use-scale-regularization True --max-num-iterations 30000 --viewer.quit-on-train-completion True"
            
            # Gọi splatfacto-big để chống rác và chi tiết thanh thép tốt hơn
            train_cmd = f"{sys.executable} -m nerfstudio.scripts.train splatfacto-big {base_args} --data {path}"

            try:
                subprocess.run(train_cmd, shell=True, check=True)
            except subprocess.CalledProcessError:
                print(f"    ❌ Lỗi trong quá trình huấn luyện {scene}. Chuyển sang scene tiếp theo.")
                continue
                
            # Cập nhật lại config mới nhất sau khi quá trình train thành công
            configs = glob.glob(config_search)
            if configs:
                latest_config = max(configs, key=os.path.getmtime)
            else:
                print(f"    ❌ Không tìm thấy mô hình của {scene} sau khi train.")
                continue

        # 3. KẾT XUẤT ẢNH DỰ ĐOÁN (Render)
        print(f"    📸 Đang render ảnh từ các góc camera test...")
        csv_path = os.path.join(path, "test", "test_poses.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(path, "test", "test_pose.csv")

        render_cmd = f"{sys.executable} scripts/render_test_pose.py --config {latest_config} --csv {csv_path} --out {scene_pred_dir}"
        try:
            subprocess.run(render_cmd, shell=True, check=True)
            print(f"    ✅ Đã render xong {scene}!")
        except subprocess.CalledProcessError:
            print(f"    ❌ Lỗi khi render ảnh cho {scene}.")

    print(f"\n{'='*60}\n 🎉 AUTO-PIPELINE ĐĐA HOÀN TẤT TOÀN BỘ CÁC TRẠM!\n{'='*60}")

if __name__ == "__main__":
    run_pipeline()