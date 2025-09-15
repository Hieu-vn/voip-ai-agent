#!/usr/bin/env bash
set -euo pipefail # Exit on error, unset variables, pipefail

# Tìm thư mục gốc của dự án, nơi script này đang nằm
DIR=$(cd "$(dirname "$0")/.." && pwd)

# Thêm thư mục 'src' vào PYTHONPATH để Python có thể tìm thấy các module
export PYTHONPATH="$DIR/src"

# Di chuyển vào thư mục gốc của dự án
cd "$DIR"

# Kích hoạt virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source "venv/bin/activate"
else
    echo "Error: Virtual environment 'venv' not found. Please run 'scripts/setup_env.sh' first."
    exit 1 # Exit if venv is not found
fi

# Chạy ứng dụng chính
echo "Starting AI Agent application..."
python3 "src/main.py"
