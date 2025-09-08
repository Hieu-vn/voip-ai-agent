#!/bin/bash
# Script để chạy ứng dụng AI Agent

# Tìm thư mục gốc của dự án, nơi script này đang nằm
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Thêm thư mục 'src' vào PYTHONPATH để Python có thể tìm thấy các module
export PYTHONPATH="$DIR/src"

# Di chuyển vào thư mục gốc của dự án
cd "$DIR"

# Kích hoạt virtual environment
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source "venv/bin/activate"
else
    echo "Warning: venv not found."
fi

# Chạy ứng dụng chính
echo "Starting AI Agent application..."
python3 "src/main.py"
