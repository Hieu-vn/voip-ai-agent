#!/bin/bash
# Script tự động cài đặt Docker và Docker Compose trên Debian 12
# Dựa trên hướng dẫn chính thức của Docker.

set -e

echo "--- 1. Cập nhật hệ thống ---"
sudo apt-get update && sudo apt-get upgrade -y

echo "--- 2. Cài đặt các gói cần thiết ---"
sudo apt-get install -y ca-certificates curl gnupg lsb-release

echo "--- 3. Thêm GPG key chính thức của Docker ---"
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo "--- 4. Thêm Docker repository ---"
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "--- 5. Cài đặt Docker Engine và Docker Compose plugin ---"
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "--- 6. (Tùy chọn) Thêm user hiện tại vào group docker để chạy không cần sudo ---"
sudo usermod -aG docker $USER

echo "--- CÀI ĐẶT HOÀN TẤT! ---"
echo "QUAN TRỌNG: Bạn cần ĐĂNG XUẤT và ĐĂNG NHẬP LẠI (hoặc khởi động lại terminal) để có thể chạy lệnh 'docker' không cần 'sudo'."
echo "Sau khi đăng nhập lại, bạn có thể kiểm tra bằng lệnh: docker run hello-world"
