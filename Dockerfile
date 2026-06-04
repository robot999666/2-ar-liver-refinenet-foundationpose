# 在线 Docker 环境：Ubuntu 22.04、系统 Python 3.10、CUDA 11.8。
# archives/wheelhouse.zip 是独立的 Linux/Python 3.12 云端依赖包，
# 本镜像不会使用它。
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV PYTHONUNBUFFERED=1
ENV OPENCV_IO_ENABLE_OPENEXR=1
ENV AR_ROOT=/workspace/AR
ENV AR_PROJECT_ROOT=/workspace/AR
ENV DA2_PROJECT_DIR=/workspace/AR/src
ENV DA2_CHECKPOINT_PATH=/workspace/AR/weights/depth_anything_v2_vitl.pth
ENV PYTHONPATH=/workspace/AR/src:/workspace/AR/scripts
ARG PIP_MIRROR=https://mirrors.aliyun.com/pypi/simple/

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN if [ -f /etc/apt/sources.list ]; then \
        sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list; \
    fi \
    && if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i 's|http://archive.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g; s|http://security.ubuntu.com/ubuntu/|https://mirrors.aliyun.com/ubuntu/|g' /etc/apt/sources.list.d/ubuntu.sources; \
    fi \
    && apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -f /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.11.2 ]; then \
        ln -sf /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so.11.2 /usr/local/cuda/targets/x86_64-linux/lib/libnvrtc.so; \
        echo /usr/local/cuda/targets/x86_64-linux/lib > /etc/ld.so.conf.d/cuda-targets.conf; \
        ldconfig; \
    fi

WORKDIR /workspace

RUN python3 -m pip install -i ${PIP_MIRROR} --trusted-host mirrors.aliyun.com --upgrade pip setuptools wheel

RUN pip install -i ${PIP_MIRROR} --trusted-host mirrors.aliyun.com --retries 10 --timeout 120 \
    filelock \
    typing-extensions \
    sympy \
    networkx \
    jinja2 \
    triton==2.0.0 \
    numpy \
    requests \
    pillow

RUN pip install --retries 10 --timeout 120 --no-deps \
    torch==2.0.0+cu118 \
    torchvision==0.15.1+cu118 \
    torchaudio==2.0.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/requirements.txt
RUN pip install -i ${PIP_MIRROR} --trusted-host mirrors.aliyun.com --retries 10 --timeout 300 -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

WORKDIR /workspace/AR

CMD ["/bin/bash"]
