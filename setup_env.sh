#!/bin/bash
echo "=================================================="
echo " 🚀 BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG VAR 2026..."
echo "=================================================="

# 1. Cài đặt các thư viện cốt lõi
echo "[1/3] Đang tải và cài đặt thư viện qua pip..."
pip install -q --ignore-installed blinker nerfstudio==1.1.4 gsplat==1.0.0 torchmetrics lpips opencv-python

# 2. Vá lỗi xung đột PyTorch (Duplicate Template Name)
echo "[2/3] Đang tiêm bản vá lỗi hệ tọa độ và PyTorch 2.6..."
sed -i 's/@torch_compile()/# @torch_compile()/g' /usr/local/lib/python3.12/dist-packages/nerfstudio/models/splatfacto.py

echo "[3/3] Môi trường đã sẵn sàng 100%!"
echo "=================================================="
# chmod +x setup_env.sh
