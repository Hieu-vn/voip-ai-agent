# 📞 AI Agent cho VoIP sử dụng Streaming thời gian thực

## 🌟 Giới thiệu Dự án

Dự án này cung cấp một AI Agent hiệu suất cao, độ trễ thấp, có khả năng xử lý các cuộc gọi đến trên nền tảng VoIP. Hệ thống sử dụng kiến trúc streaming thời gian thực hiện đại để mang lại trải nghiệm hội thoại tự nhiên và phản hồi nhanh.

Hệ thống được xây dựng trên **Asterisk 20** (trong VitalPBX) trên **Debian 12**, tận dụng server với **8 GPU NVIDIA V100** để chạy các mô hình AI tiên tiến cho tiếng Việt và tiếng Anh.

- **🤖 Công nghệ AI cốt lõi**: Agent sử dụng Google Cloud API cho Speech-to-Text, mô hình Llama 4 (thông qua `llama.cpp` hoặc `unsloth`) cho hiểu ngôn ngữ tự nhiên, và mô hình NVIDIA NeMo cho Text-to-Speech chất lượng cao.
- **🎯 Mục tiêu chính**: Đạt được độ trễ end-to-end dưới 500ms thông qua một pipeline streaming hoàn chỉnh, từ khi người dùng nói đến khi họ nghe phản hồi của AI.
- **🔗 Tích hợp**: Hệ thống được thiết kế với kiến trúc module hóa, cho phép tích hợp linh hoạt với các nền tảng CRM như Zoho hoặc Salesforce thông qua chức năng gọi công cụ (function calling) trong mô hình NLP.

## 🏗️ Kiến trúc Hệ thống (Thống nhất & Container hóa - Cấu hình V1)

Kiến trúc của dự án được thiết kế để có khả năng mở rộng và độ trễ thấp, tập trung vào **Asterisk REST Interface (ARI)** để điều khiển cuộc gọi.

- **Phần cứng**: Server với 8x NVIDIA V100 GPUs (32GB VRAM).
- **VoIP**: VitalPBX / Asterisk 20.
- **Lõi ứng dụng**: Một ứng dụng Python không đồng bộ kết nối với Asterisk qua ARI.
- **Container hóa**: Ứng dụng được đóng gói hoàn toàn bằng Docker và Docker Compose, với các môi trường build riêng biệt cho các service NLP và TTS để quản lý xung đột dependency phức tạp.

### Luồng Dữ liệu (Streaming-First)

Toàn bộ quá trình được điều khiển bởi sự kiện và xây dựng trên streaming:

```
📞 Cuộc gọi đến (VitalPBX)
   ↓
🎙️ Asterisk gửi sự kiện 'StasisStart' đến ứng dụng Python qua ARI (WebSocket)
   ↓
🐍 Service 'app' tạo một 'CallHandler' (trong app/audio/stream.py) cho cuộc gọi
   1. Trả lời cuộc gọi.
   2. Yêu cầu Asterisk gửi một bản sao (fork) của luồng audio đến một cổng UDP cục bộ.
   ↓
🎧 Xử lý Audio thời gian thực
   1. Một UDP listener nhận các gói RTP và đẩy audio thô đến Google Cloud STT API.
   2. **STT**: Audio được stream đến **Google Cloud Speech-to-Text API**.
   3. Văn bản đã chuyển đổi được nhận.
   ↓
🧠 Xử lý AI
   1. **Text Normalization**: Văn bản từ STT được chuẩn hóa (số thành chữ, v.v.).
   2. **Emotion Analysis**: Cảm xúc của người dùng được phân tích.
   3. **Guardrails**: PII được lọc khỏi văn bản người dùng.
   4. **NLP**: Văn bản đã lọc PII và ngữ cảnh cảm xúc được gửi đến mô hình **Llama 4 Scout** (được host cục bộ, thông qua app/nlu/agent.py và app/nlu/llama.py) để xử lý ý định.
   5. **Guardrails**: Phản hồi từ LLM được kiểm tra an toàn; PII được khôi phục.
   6. Văn bản phản hồi từ mô hình NLP được tạo.
   ↓
🗣️ Tổng hợp giọng nói
   1. **TTS**: Văn bản phản hồi được gửi đến một **NVIDIA NeMo TTS Server** riêng biệt (service 'tts', thông qua app/tts/client.py).
   2. Server TTS sử dụng pipeline hai bước để tạo audio chất lượng cao:
      - **Bước 1 (Spectrogram):** Mô hình **FastPitch** chuyển văn bản thành mel-spectrogram.
      - **Bước 2 (Audio):** Mô hình **BigVGAN** (vocoder) chuyển spectrogram thành dạng sóng âm thanh.
   3. Audio được tạo ra được stream trở lại ứng dụng chính.
   ↓
🔊 Phát lại
   1. 'CallHandler' phát luồng audio nhận được trở lại cho người gọi qua Asterisk.
   2. Hệ thống hỗ trợ **Barge-in** (người dùng ngắt lời) và **Reprompt** (nhắc lại khi im lặng).
   ↓
📊 **Evaluation Tracking**: Mỗi lượt thoại được ghi lại để đánh giá hiệu suất.
   ↓
👋 Gác máy
```

### Các thành phần cốt lõi

- **🖥️ AI Backend (Container hóa)**:
  - **STT**: **Google Cloud Speech-to-Text API**. Yêu cầu `GOOGLE_APPLICATION_CREDENTIALS` được cấu hình.
  - **NLP**: Mô hình **Llama 4 Scout**, được tải và quản lý bởi `app/nlu/llama.py` (sử dụng `llama_cpp` hoặc `unsloth`). Chạy trong một Docker container riêng (`app`) với các dependency được tối ưu hóa.
    -   **Guardrails**: Lọc PII đầu vào, kiểm tra an toàn đầu ra.
    -   **Text Normalization**: Chuẩn hóa văn bản từ STT.
    -   **Emotion Analysis**: Phân tích cảm xúc người dùng để tạo phản hồi phù hợp.
  - **TTS**: Một **FastAPI server** riêng biệt (`tts_server/api.py`) chịu trách nhiệm cho tất cả quá trình Text-to-Speech. Chạy trong một Docker container riêng (`tts`) với các dependency được tối ưu hóa.
    -   **Kiến trúc**: Tách biệt quá trình xử lý TTS nặng khỏi logic xử lý cuộc gọi cốt lõi để đảm bảo sự ổn định và hiệu suất.
    -   **Công nghệ cốt lõi**: Sử dụng pipeline 2 bước: **NeMo FastPitch** để tạo spectrogram và **NVIDIA BigVGAN** làm vocoder cho tổng hợp dạng sóng chất lượng cao. Các model được tải bằng phương thức `from_pretrained()` từ NVIDIA NGC.
    -   **Chiến lược ngôn ngữ**: Hỗ trợ tiếng Anh và tiếng Việt. Mô hình **FastPitch** tùy chỉnh được fine-tuned trên dataset `phoaudiobook` cho tiếng Việt.
- **📡 Tích hợp VoIP**: Sử dụng **ARI (Asterisk REST Interface)** để điều khiển cuộc gọi chi tiết, thời gian thực. `app/audio/stream.py` chứa logic giao tiếp với kênh Asterisk.
  -   **Barge-in**: Cho phép người dùng ngắt lời AI.
  -   **Reprompt**: Nhắc lại khi người dùng im lặng.
- **📊 Evaluation Tracking**: Ghi lại chi tiết các lượt thoại để phân tích và đánh giá.
## ⚙️ Môi trường Docker & Quản lý Dependency (Cấu hình V1)

Để giải quyết các xung đột dependency phức tạp giữa các thư viện AI (ví dụ: `unsloth` và `nemo_toolkit`), dự án sử dụng kiến trúc Docker đa container với các môi trường build riêng biệt. Phần này trình bày chi tiết các dependency và chiến lược cài đặt cho từng service.

### 6.1. Service `app` (NLP/Agent)

*   **Base Image**: `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`
*   **Python Version**: `3.11`
*   **System Dependencies**: `python3.11-venv`, `python3.11-dev`, `build-essential`, `git`, `cmake`, `ninja-build`, `ffmpeg`, `sox`, `libsndfile1`, `curl`, `ca-certificates`, `pkg-config`.
*   **Python Libraries** (từ `requirements-app.txt`):
    *   `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`
    *   `unsloth[cu121-torch251]`
    *   `llama-cpp-python==0.2.90`
    *   `langgraph==0.2.24`, `langchain==0.3.2`
    *   `google-cloud-speech==2.26.0`
    *   `aiohttp==3.10.5`, `websockets==12.0`
    *   `numpy==1.26.4`, `pandas==2.2.2`, `scipy==1.14.1`, `onnxruntime-gpu==1.17.1`
    *   `pyyaml==6.0.2`, `python-dotenv==1.0.1`, `pydub==0.25.1`, `structlog==24.1.0`, `uvloop==0.20.0`
    *   `ari==1.0.0`
    *   `opentelemetry-distro==0.46b0`, `opentelemetry-instrumentation-aiohttp-client==0.46b0`
    *   `transformers` (dependency của `emotion_analyzer`)

### 6.2. Service `tts` (TTS NeMo)

*   **Base Image**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
*   **Python Version**: `3.10`
*   **System Dependencies**: `python3.10-venv`, `python3.10-dev`, `build-essential`, `pkg-config`, `ffmpeg`, `sox`, `libsndfile1`, `curl`, `ca-certificates`.
*   **Python Libraries** (cài đặt sớm và từ `requirements-tts.txt`):
    *   `Cython>=0.29`, `numpy==1.26.4`, `typing_extensions` (cài đặt sớm)
    *   `nemo_toolkit[tts]==1.23.0`
    *   `torch==2.1.2`, `torchaudio==2.1.2`
    *   `fastapi==0.115.0`, `uvicorn[standard]==0.30.6`
    *   `transformers==4.36.2`, `huggingface_hub==0.19.4`, `datasets==2.14.7`, `pyarrow==12.0.1`
    *   `soundfile==0.12.1`, `librosa==0.10.2`, `structlog==24.1.0`
    *   `opentelemetry-distro==0.46b0`, `opentelemetry-exporter-otlp==1.25.0`, `opentelemetry-instrumentation-fastapi==0.46b0`

## 🚀 Quy trình Phát triển & Triển khai

- **Phát triển cục bộ**: Mã nguồn có thể được mount vào container thông qua Docker volumes để lặp lại nhanh chóng mà không cần rebuild image.
- **Kiểm thử**: Sử dụng `docker compose build` cho các thay đổi dependency, `docker compose restart <service>` cho các thay đổi mã nguồn.
- **Triển khai**: Dễ dàng triển khai trên bất kỳ server nào có Docker và hỗ trợ NVIDIA GPU.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17b-gguf-q4_k_m.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17b-gguf-q4_k_m.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17b-gguf-q4_k_m.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17B-16E-Instruct-unsloth-bnb-4bit.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17B-16E-Instruct-unsloth-bnb-4bit.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17B-16E-Instruct-unsloth-bnb-4bit.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17B-16E-Instruct-unsloth-bnb-4bit.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17B-16E-Instruct-unsloth-bnb-4bit.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 🚀 Hướng dẫn Cài đặt & Chạy

1.  **Cấu hình môi trường**:
    *   Tạo file `.env` trong thư mục gốc của dự án, dựa trên `.env.example`.
    *   Đảm bảo các biến môi trường cho ARI (URL, username, password), Google Cloud (Project ID, Recognizer ID, credentials) và đường dẫn model LLM được cấu hình chính xác.
    *   Đặt file model GGUF của bạn (ví dụ: `llama-4-scout-17B-16E-Instruct-unsloth-bnb-4bit.gguf`) vào thư mục `./models/nlp`.
2.  **Build Docker Images**:
    ```bash
    docker compose build --no-cache app tts
    ```
3.  **Khởi chạy Services**:
    ```bash
    docker compose up -d
    ```
4.  **Cấu hình Asterisk**:
    *   Đảm bảo Asterisk đang chạy và ARI được bật (xem `ari.conf`, `http.conf`).
    *   Cấu hình dialplan để đẩy cuộc gọi vào ứng dụng Stasis `ai_app` (ví dụ trong `extensions.conf`):
        ```ini
        ; extensions.conf
        [ivr-ai]
        exten => _X.,1,NoOp(Hieu-VoIP AI)
         same => n,Set(TALK_DETECT(set)=200,1000)       ; talk>=200ms, silence>=1000ms
         same => n,Stasis(ai_app)
         same => n,Hangup()
        ```
5.  **Kiểm tra**: Thực hiện cuộc gọi đến số đã cấu hình trong dialplan và kiểm tra log của các service Docker.

## 📚 Tài liệu & Kế hoạch

- **`docs/PLAN.md`**: Chứa kế hoạch triển khai kỹ thuật chi tiết, bao gồm lộ trình phát triển NLP toàn diện.
- **`.env`**: Quản lý các biến môi trường, bao gồm thông tin đăng nhập ARI và đường dẫn model. File này được loại trừ khỏi Git vì lý do bảo mật.

## ⚠️ Bảo mật & Các thực hành tốt nhất

- **Thông tin nhạy cảm**: Tất cả thông tin đăng nhập (ví dụ: mật khẩu ARI, API keys) được quản lý thông qua file `.env` và được loại trừ khỏi kiểm soát phiên bản (`.gitignore`).
- **Lịch sử Git**: Dữ liệu nhạy cảm được xóa khỏi lịch sử Git bằng `git filter-repo` nếu vô tình bị commit.
- **Quản lý Dependency**: Pin phiên bản nghiêm ngặt và môi trường Docker riêng biệt đảm bảo sự ổn định và bảo mật.