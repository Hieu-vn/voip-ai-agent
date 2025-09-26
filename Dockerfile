# Use an official NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies including Python, git, and git-lfs
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    sox \
    python3.11-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up a working directory
WORKDIR /app

# Create and activate a virtual environment
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Install base python packages for building other wheels
RUN pip install --upgrade pip wheel setuptools
RUN pip install numpy typing_extensions Cython

# --- Staged Installation of AI/ML Stack ---

# Stage 1: Install the core PyTorch stack
RUN pip install \
    torch==2.3.1 \
    torchaudio==2.3.1 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Stage 2: Install Unsloth and its specific dependencies
# This depends on torch being installed first.
RUN pip install \
    "xformers<0.0.27" \
    bitsandbytes \
    psutil \
    unsloth \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Stage 3: Install NeMo and its specific dependencies
# This also depends on torch and needs a specific huggingface-hub version.
RUN pip install \
    "huggingface-hub<0.20.0" \
    nemo_toolkit[tts]==1.23.0

# Copy the cleaned requirements file
COPY requirements.txt .

# Install remaining application-level dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8001

# Define the command to run the application
# The server will be started by docker-compose, which can override this command.
CMD ["python", "tts_server/server.py"]