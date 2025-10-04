FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Cài đặt các gói hệ thống cần thiết + STUB DRIVER
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils curl ca-certificates \
    build-essential git cmake ninja-build ffmpeg sox libsndfile1 \
    nvidia-cuda-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure system linker to recognize the new stub libraries
RUN ldconfig

# Tạo và kích hoạt môi trường ảo
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Cài đặt Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - 
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Sao chép các file của Poetry và cài đặt dependency
# Tối ưu hóa Docker cache: chỉ khi các file này thay đổi thì bước cài đặt mới chạy lại
COPY pyproject.toml poetry.lock ./

# Cấu hình Poetry để sử dụng môi trường ảo đã tạo, không tạo môi trường mới
RUN poetry config virtualenvs.create false

# Cài đặt tất cả dependency từ file lock. 
# Lệnh này cũng sẽ build llama-cpp-python từ source với cờ CUDA đã được thiết lập.
RUN CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=70" FORCE_CMAKE=1 \
    poetry install --no-root --no-interaction --no-ansi

# Sao chép phần còn lại của mã nguồn ứng dụng
COPY . .

HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=10 \
  CMD python -m tools.healthcheck_app

CMD ["python", "-m", "app.main"]