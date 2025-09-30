FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3.11-distutils curl ca-certificates \
    build-essential git cmake ninja-build ffmpeg sox libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

WORKDIR /app
COPY requirements-app.txt .

RUN python3.11 -m venv /opt/venv && . /opt/venv/bin/activate && \
    pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements-app.txt

ENV PATH="/opt/venv/bin:$PATH"

# Llama.cpp CUDA build cache-friendly (optional, nếu cần custom build)
# RUN pip install --no-binary=:llama-cpp-python: "llama-cpp-python==0.2.90" \
#     --config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=70

COPY . .
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=10 \
  CMD python -m tools.healthcheck_app

CMD ["python", "-m", "app.main"]
