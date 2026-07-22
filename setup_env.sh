#!/usr/bin/env bash

# Cài môi trường VAR 2026 trên RunPod.
# - Hiện đầy đủ tiến trình pip thay vì chạy ở chế độ quiet.
# - Dừng ngay khi một bước thất bại.
# - Tự tìm đúng vị trí cài Nerfstudio, không hard-code Python 3.12.
# - Giữ pip cache để lần chạy lại không phải tải lại mọi thứ.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SETUP_LOG_FILE:-${SCRIPT_DIR}/setup_env.log}"
PYTHON_BIN="${PYTHON_BIN:-python}"
START_SECONDS=$SECONDS
CURRENT_STEP="khởi tạo"

# Vừa hiển thị trên terminal, vừa lưu log để chẩn đoán nếu pod bị ngắt.
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

on_error() {
    local exit_code=$?
    echo
    echo "=================================================="
    echo "❌ CÀI ĐẶT THẤT BẠI tại bước: ${CURRENT_STEP}"
    echo "   Exit code: ${exit_code}"
    echo "   Xem log: ${LOG_FILE}"
    echo "=================================================="
    exit "$exit_code"
}

on_interrupt() {
    echo
    echo "⚠️  Đã nhận Ctrl+C. Quá trình cài đặt bị hủy và CHƯA hoàn tất."
    echo "   Có thể chạy lại cùng lệnh; pip sẽ tận dụng cache đã tải."
    echo "   Xem log: ${LOG_FILE}"
    exit 130
}

trap on_error ERR
trap on_interrupt INT TERM

print_step() {
    local number=$1
    shift
    CURRENT_STEP="$*"
    echo
    echo "--------------------------------------------------"
    echo "[${number}/6] ${CURRENT_STEP}"
    echo "--------------------------------------------------"
}

run_pip_install() {
    echo "+ ${PYTHON_BIN} -m pip install --progress-bar on $*"
    PIP_PROGRESS_BAR=on "$PYTHON_BIN" -m pip install \
        --progress-bar on \
        --no-input \
        --retries 5 \
        --timeout 120 \
        --upgrade-strategy only-if-needed \
        "$@"
}

echo "=================================================="
echo "🚀 BẮT ĐẦU CÀI ĐẶT MÔI TRƯỜNG VAR 2026"
echo "=================================================="
echo "Thời gian : $(date --iso-8601=seconds 2>/dev/null || date)"
echo "Thư mục   : ${SCRIPT_DIR}"
echo "Log       : ${LOG_FILE}"

print_step 1 "Kiểm tra Python, pip và GPU"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "Không tìm thấy Python executable: ${PYTHON_BIN}" >&2
    exit 1
fi

"$PYTHON_BIN" --version
"$PYTHON_BIN" -m pip --version

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total \
        --format=csv,noheader
else
    echo "⚠️  Không tìm thấy nvidia-smi. Tiếp tục cài package nhưng cần kiểm tra lại GPU."
fi

print_step 2 "Cập nhật công cụ cài package"

run_pip_install --upgrade pip setuptools wheel

print_step 3 "Cài blinker riêng để tránh xung đột package hệ thống"

# Chỉ ignore bản blinker do hệ thống quản lý; không ép cài lại toàn bộ
# dependency graph như script cũ.
run_pip_install --ignore-installed blinker

print_step 4 "Cài Nerfstudio, gsplat và metric/image dependencies"

run_pip_install --upgrade \
    "nerfstudio==1.1.4" \
    "gsplat==1.0.0" \
    torchmetrics \
    lpips \
    opencv-python

print_step 5 "Tự tìm và áp dụng bản vá torch_compile nếu cần"

SPLATFACTO_FILE="$($PYTHON_BIN - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("nerfstudio")
if spec is None or not spec.submodule_search_locations:
    raise SystemExit("Không tìm thấy package nerfstudio sau khi cài")

package_root = Path(next(iter(spec.submodule_search_locations)))
print(package_root / "models" / "splatfacto.py")
PY
)"

echo "Splatfacto: ${SPLATFACTO_FILE}"

if [[ ! -f "$SPLATFACTO_FILE" ]]; then
    echo "Không tìm thấy splatfacto.py tại đường dẫn đã phát hiện" >&2
    exit 1
fi

if grep -Eq '^[[:space:]]*@torch_compile\(\)' "$SPLATFACTO_FILE"; then
    sed -i \
        's/^\([[:space:]]*\)@torch_compile()/\1# @torch_compile()/g' \
        "$SPLATFACTO_FILE"
    echo "[OK] Đã vô hiệu hóa decorator @torch_compile() gây xung đột."
elif grep -Eq '^[[:space:]]*#[[:space:]]*@torch_compile\(\)' "$SPLATFACTO_FILE"; then
    echo "[SKIP] Bản vá @torch_compile() đã tồn tại."
else
    echo "[SKIP] Phiên bản này không có decorator @torch_compile() cần vá."
fi

print_step 6 "Kiểm tra import và phiên bản đã cài"

CUSTOM_METHOD_ROOT="${SCRIPT_DIR}/var2026-digital-twin"
if [[ ! -f "${CUSTOM_METHOD_ROOT}/pyproject.toml" ]]; then
    echo "Custom-method package not found: ${CUSTOM_METHOD_ROOT}/pyproject.toml" >&2
    exit 1
fi

echo "[+] Registering repo-owned Nerfstudio methods from: ${CUSTOM_METHOD_ROOT}"
run_pip_install --no-deps --editable "$CUSTOM_METHOD_ROOT"

"$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

import cv2
import gsplat
import lpips
import nerfstudio
import torch
import torchmetrics
from nerfstudio.models.splatfacto import SplatfactoModel
from var_nvs.edge_splatfacto import EdgeSplatfactoModel

packages = (
    "nerfstudio",
    "gsplat",
    "torch",
    "torchmetrics",
    "lpips",
    "opencv-python",
    "var2026-digital-twin-methods",
)

print("\nPhiên bản package:")
for package in packages:
    print(f"  - {package:16s}: {version(package)}")

print(f"  - CUDA available   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  - CUDA runtime     : {torch.version.cuda}")
    print(f"  - GPU              : {torch.cuda.get_device_name(0)}")

print("\n[OK] Import SplatfactoModel thành công.")
PY

if command -v ns-train >/dev/null 2>&1; then
    ns-train --help >/dev/null
    ns-train splatfacto-edge --help >/dev/null
    echo "[OK] Custom method splatfacto-edge is registered."
    echo "[OK] Lệnh ns-train hoạt động."
else
    echo "Không tìm thấy lệnh ns-train sau khi cài Nerfstudio" >&2
    exit 1
fi

ELAPSED_SECONDS=$((SECONDS - START_SECONDS))

echo
echo "=================================================="
echo "✅ MÔI TRƯỜNG ĐÃ CÀI ĐẶT VÀ KIỂM TRA THÀNH CÔNG"
printf '   Thời gian: %dm %02ds\n' \
    "$((ELAPSED_SECONDS / 60))" \
    "$((ELAPSED_SECONDS % 60))"
echo "   Log: ${LOG_FILE}"
echo "=================================================="
