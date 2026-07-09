import os

# Định nghĩa các cặp thư mục cần so sánh (Nguồn của BTC -> Đích của bạn)
FOLDERS_TO_CHECK = [
    ("VAI_NVS_DATA/phase1/public_set", "var2026-digital-twin/data/public"),
    ("VAI_NVS_DATA/phase1/private_set1", "var2026-digital-twin/data/private_test1")
]

def compare_directories(src_dir, dst_dir):
    print(f"\n=======================================================")
    print(f"Đang so sánh:\n Nguồn: {src_dir}\n Đích : {dst_dir}")
    print(f"=======================================================")
    
    if not os.path.exists(src_dir):
        print(f"[-] LỖI: Thư mục nguồn '{src_dir}' không tồn tại.")
        return
    if not os.path.exists(dst_dir):
        print(f"[-] LỖI: Thư mục đích '{dst_dir}' không tồn tại.")
        return

    missing_files = []
    size_mismatch = []
    total_files_checked = 0

    # Duyệt qua toàn bộ file trong thư mục nguồn
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            total_files_checked += 1
            src_file_path = os.path.join(root, file)
            
            # Tính toán đường dẫn tương đối để map sang thư mục đích
            rel_dir = os.path.relpath(root, src_dir)
            if rel_dir == ".":
                dst_file_path = os.path.join(dst_dir, file)
            else:
                dst_file_path = os.path.join(dst_dir, rel_dir, file)

            # Kiểm tra file có tồn tại ở đích không
            if not os.path.exists(dst_file_path):
                missing_files.append(dst_file_path)
            # Kiểm tra dung lượng file có khớp nhau không
            elif os.path.getsize(src_file_path) != os.path.getsize(dst_file_path):
                size_mismatch.append((src_file_path, dst_file_path))

    # In báo cáo
    print(f"[i] Đã quét tổng cộng {total_files_checked} file.")
    
    if not missing_files and not size_mismatch:
        print("[+] TUYỆT VỜI! Dữ liệu đã được copy qua đầy đủ và khớp 100%.")
    else:
        if missing_files:
            print(f"\n[-] PHÁT HIỆN THIẾU {len(missing_files)} file trong thư mục đích:")
            for f in missing_files[:10]:  # Chỉ in tối đa 10 file để tránh làm rối màn hình
                print(f"    - {f}")
            if len(missing_files) > 10:
                print(f"    ... và {len(missing_files) - 10} file khác.")

        if size_mismatch:
            print(f"\n[-] PHÁT HIỆN SAI LỆCH DUNG LƯỢNG ở {len(size_mismatch)} file (Copy bị lỗi):")
            for src, dst in size_mismatch[:10]:
                print(f"    - {src}")
            if len(size_mismatch) > 10:
                print(f"    ... và {len(size_mismatch) - 10} file khác.")

if __name__ == "__main__":
    for source, destination in FOLDERS_TO_CHECK:
        compare_directories(source, destination)