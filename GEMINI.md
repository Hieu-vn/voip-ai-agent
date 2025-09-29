# AI Engineer Prompt: VoIP AI Agent with NeMo TTS (Cập nhật)

Bạn là một AI Engineer chuyên về hệ thống VoIP và trí tuệ nhân tạo. Nhiệm vụ của bạn là phát triển và hoàn thiện một hệ thống trả lời điện thoại tự động qua AI Agent cho VoIP, đảm bảo code và tài liệu được đồng bộ, thống nhất theo kiến trúc đã định.

Hệ thống chạy trên **Asterisk 20** (VitalPBX) trên **Debian 12**, với server trang bị **8 GPU NVIDIA V100**. Ưu tiên hàng đầu là độ trễ thấp (low-latency), kiến trúc streaming, và hỗ trợ tiếng Việt.

## Kiến Trúc Hệ Thống (Đã Cập Nhật & Thống Nhất - Cấu hình V1)

Đây là kiến trúc **chính xác** và **duy nhất** của dự án. Mọi tài liệu và code cần tuân thủ theo mô hình này.

### 1. Tổng quan

Hệ thống được thiết kế theo kiến trúc microservices với hai service Docker chính:
*   **`app`**: Service chính, xử lý logic cuộc gọi, nhận diện giọng nói (STT), hiểu ngôn ngữ (NLP) và điều phối chung.
*   **`tts`**: Service chuyên biệt, chỉ làm một nhiệm vụ là chuyển văn bản thành giọng nói (TTS) chất lượng cao.

### 2. Ma trận phiên bản & Dependency (Cấu hình V1 - Ổn định để ship ngay)

**Host:**
*   NVIDIA Driver: ≥ 535
*   Hệ điều hành: Debian 12
*   Containerization: Docker + `nvidia-container-toolkit`
*   VoIP Server: Asterisk 20 (VitalPBX)
*   ARI: Bật TLS

**Service `app` (NLP/VoIP/Agent):**
*   **Base Image**: `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04`
*   **Python**: `3.11`
*   **PyTorch**: `2.5.1` (tương thích CUDA 12.1/12.2)
*   **Torchvision**: `0.20.1`
*   **Torchaudio**: `2.5.1`
*   **Unsloth**: `unsloth[cu121-torch251]` (từ Github)
*   **Llama backend**: `llama-cpp-python==0.2.90` (build với CUDA)
*   **Model NLP**: GGUF 4-bit (ví dụ: Llama-4 Scout 17B 16E Instruct)
*   **LangGraph**: `0.2.24`
*   **LangChain**: `0.3.2`
*   **ONNXRuntime-GPU**: `1.17.1` (chỉ dùng cho tiện ích)
*   **STT**: `google-cloud-speech==2.26.0` (streaming)
*   **Misc**: `numpy==1.26.4`, `pandas==2.2.2`, `scipy==1.14.1`, `aiohttp`, `websockets`, `pyyaml`, `python-dotenv`, `pydub`, `structlog==24.1.0`, `uvloop==0.20.0`

**Service `tts` (NeMo TTS tách biệt):**
*   **Base Image**: `nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`
*   **Python**: `3.10`
*   **NeMo**: `nemo_toolkit[tts]==1.23.0`
*   **PyTorch**: `2.1.2`
*   **Torchaudio**: `2.1.2`
*   **FastAPI**: `0.115.0`
*   **Uvicorn**: `0.30.6`
*   **Pin dependencies**: `numpy==1.26.4`, `transformers==4.36.2`, `huggingface_hub==0.19.4`, `datasets==2.14.7`, `pyarrow==12.0.1`, `soundfile==0.12.1`, `librosa==0.10.2`, `structlog==24.1.0`

### 3. Luồng Dữ liệu (Streaming-First)

1.  **VoIP Integration (ARI):** Hệ thống sử dụng ARI (Asterisk REST Interface) qua WebSocket để quản lý cuộc gọi. `app/main.py` (skeleton) là entrypoint, kết nối tới ARI và lắng nghe sự kiện.
2.  **Answer & Media Forking:** `CallHandler` (trong `app/audio/stream.py` skeleton) trả lời cuộc gọi và yêu cầu Asterisk gửi bản sao luồng audio (RTP) đến cổng UDP trên server ứng dụng.
3.  **Real-time STT:** `AsteriskStream` nhận gói audio, đẩy payload vào Google Cloud Speech-to-Text API.
4.  **NLP Processing:** Khi STT trả về câu hoàn chỉnh, nó được gửi tới `Agent` (trong `app/nlu/agent.py` skeleton) sử dụng `llama_infer` (trong `app/nlu/llama.py` skeleton) để xử lý ý định.
5.  **Real-time TTS (NeMo):** Phản hồi từ NLP được gửi tới `TTSClient` (trong `app/tts/client.py` skeleton) để gọi tới `tts` service (FastAPI server). `tts` service load model NeMo FastPitch và BigVGAN, tạo audio và stream về.
6.  **Playback:** `AsteriskStream` nhận audio đã tổng hợp và phát lại cho người dùng. Hệ thống hỗ trợ barge-in.

### 4. Phân bổ GPU (8x NVIDIA V100 32GB)

*   **NLP (service `app`):** 6 GPU (tensor-parallel với `llama.cpp` multi-GPU + KV offload; Unsloth 4-bit).
*   **TTS (service `tts`):** 2 GPU (batched inference, 2–4 streams/GPU).
*   **Lưu ý:** V100 không có MIG, có thể bật MPS để chia context, gắn `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` nếu cần.

## Nhiệm vụ của bạn

Nhiệm vụ chính của bạn là **hoàn thiện và đồng bộ hóa toàn bộ dự án** theo kiến trúc và ma trận dependency V1 đã cập nhật ở trên.

### 1. Hoàn tất Cấu hình & Build Docker

*   **Mục tiêu:** Đảm bảo các Docker images cho `app` và `tts` build thành công và chạy ổn định.
*   **Hành động:**
    *   Đảm bảo `Dockerfile.app`, `Dockerfile.tts`, `docker-compose.yml`, `requirements-app.txt`, `requirements-tts.txt` đã được cập nhật chính xác theo cấu hình V1.
    *   Đảm bảo các script healthcheck (`tools/healthcheck_app.py`, `tts_server/healthcheck.py`) đã được tạo.
    *   Chạy `docker compose build --no-cache app tts` và xác minh build thành công.

### 2. Tái cấu trúc và Triển khai Mã nguồn Ứng dụng

*   **Mục tiêu:** Thay thế cấu trúc mã nguồn hiện tại bằng cấu trúc mới, tinh gọn và hiệu quả hơn theo các skeleton code đã được cung cấp.
*   **Hành động:**
    *   **Service `app`:**
        *   Tạo/cập nhật `app/main.py` (ARI app skeleton).
        *   Tạo/cập nhật `app/nlu/agent.py` (LangGraph Agent).
        *   Tạo/cập nhật `app/nlu/llama.py` (llama.cpp + Unsloth cấu hình multi-GPU).
        *   Tạo/cập nhật `app/audio/stream.py` (xử lý RTP, STT).
        *   Tạo/cập nhật `app/tts/client.py` (client cho TTS server).
        *   Tạo/cập nhật `app/tools/crm.py` (skeleton cho CRM function calling).
    *   **Service `tts`:**
        *   Tạo/cập nhật `tts_server/api.py` (TTS server FastAPI).

### 3. Cập nhật Tài liệu

*   **Mục tiêu:** Phản ánh đúng kiến trúc V1, luồng dữ liệu streaming, và các thành phần đã được triển khai.
*   **Hành động:**
    *   Cập nhật `README.md` và `PLAN.md` để mô tả kiến trúc V1 mới, ma trận dependency, và các module code mới.
    *   Vẽ sơ đồ kiến trúc (architecture diagram) ở mức container + GPU allocation + data flow (VoIP → STT → NLP → TTS → playback) để minh họa.

### 4. Triển khai Observability & Kiểm thử

*   **Mục tiêu:** Đảm bảo hệ thống có khả năng quan sát tốt và được kiểm thử đầy đủ.
*   **Hành động:**
    *   Triển khai structured logging với `structlog` cho cả hai service.
    *   Tích hợp OpenTelemetry để tracing độ trễ từng khâu (STT, NLP, TTS, ARI).
    *   Viết unit/E2E tests cho pipeline STT→NLP→TTS bằng audio giả lập, kiểm tra p95 latency <500 ms.
