# Sử dụng image PyTorch có sẵn CUDA devel để build nhân C++/CUDA cho gsplat
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# Thiết lập biến môi trường tránh tương tác trực tiếp khi cài đặt gói
ENV DEBIAN_FRONTEND=noninteractive
ENV FORCE_CUDA=1

# Cài đặt các thư viện hệ thống cần thiết cho COLMAP và OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    vulkan-tools \
    libvulkan-dev \
    && rm -rf /var/lib/apt/lists/*

# Nâng cấp pip và cài đặt các thư viện bổ trợ cơ bản
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir opencv-python plyfile jaxtyping tyro

# Cài đặt gsplat (Khóa phiên bản ổn định cho 2D/3DGS)
RUN pip install gsplat==1.0.0 --no-cache-dir

# Cài đặt Nerfstudio từ nguồn để dễ dàng tùy biến loss sau này
RUN git clone https://github.com/nerfstudio-project/nerfstudio.git /opt/nerfstudio \
    && cd /opt/nerfstudio \
    && git checkout v1.1.0 \
    && pip install -e . --no-cache-dir

WORKDIR /workspace