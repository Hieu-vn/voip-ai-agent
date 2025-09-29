# Kế hoạch Triển khai Kỹ thuật: AI Agent cho VoIP

**Mục tiêu:** Đồng bộ hóa mã nguồn và tài liệu của dự án với kiến trúc thống nhất, hiện đại, streaming-first sử dụng Asterisk ARI và NVIDIA NeMo cho TTS.

## 1. Kiến trúc Hệ thống (Nguồn thông tin duy nhất)

- **Giao diện VoIP:** Asterisk REST Interface (ARI). Hoạt động dựa trên sự kiện và không đồng bộ.
- **Lõi ứng dụng:** Một ứng dụng Python không đồng bộ kết nối với Asterisk qua ARI.
- **STT:** Google Cloud Speech-to-Text API (streaming).
- **NLP:** Llama 4 Scout (chạy cục bộ).
- **TTS:** Một NeMo server riêng biệt, chuyên dụng (FastPitch + BigVGAN) mà ứng dụng chính giao tiếp qua API REST.
- **Container hóa:** Ứng dụng được đóng gói hoàn toàn bằng Docker và Docker Compose, với các môi trường build riêng biệt cho các service NLP và TTS để quản lý xung đột dependency phức tạp.

### Tóm tắt Luồng Dữ liệu

```
Cuộc gọi -> Asterisk (Sự kiện ARI) -> app/audio/stream.py (fork RTP Audio) -> Google Cloud STT -> Nhận Văn bản -> 
app/nlu/agent.py (Llama 4 Scout) -> Nhận Văn bản phản hồi -> app/tts/client.py -> NeMo TTS Server (FastPitch + BigVGAN) -> Nhận Luồng Audio -> app/audio/stream.py (Phát lại) -> Người dùng
```

## 2. Hợp đồng Dữ liệu & Đảm bảo Chất lượng

Phần này định nghĩa các quy tắc và tiêu chuẩn cho tất cả các tập dữ liệu được sử dụng trong huấn luyện để đảm bảo chất lượng, tính nhất quán và khả năng tái tạo.

- **Chuẩn hóa Văn bản:**
    - **Nguồn:** Tất cả các quy tắc chuẩn hóa (ví dụ: chuyển số thành chữ, xử lý từ viết tắt, ký tự đặc biệt) phải được định nghĩa trong một file cấu hình riêng (ví dụ: `config/normalization_rules.yaml`).
    - **Lý do:** Tách rời các quy tắc khỏi mã nguồn, cho phép dễ dàng bảo trì và quản lý phiên bản các lược đồ chuẩn hóa khác nhau.
    - **Triển khai:** Các script phải tải và áp dụng các quy tắc bên ngoài này.

- **Lọc Dữ liệu Audio:**
    - **Thời lượng:** Các đoạn audio phải có thời lượng từ **1.5 đến 20 giây**.
    - **Ứng dụng:** Bộ lọc này được áp dụng *trước khi* ghi các manifest huấn luyện cuối cùng.
    - **Lý do:** Ngăn ngừa các vấn đề căn chỉnh mô hình do các đoạn clip quá ngắn hoặc quá dài.

- **Nguồn gốc và Quản lý phiên bản Dữ liệu:**
    - **Checksum:** Mỗi file WAV đã xử lý phải có một checksum (ví dụ: MD5) được ghi trong manifest.
    - **Phiên bản Dataset:** Mỗi bộ manifest được tạo (train/validation/test) sẽ được gán một phiên bản (ví dụ: `v1.0`, `v1.1`).
    - **Metadata:** Một "Dataset Card" sẽ đi kèm với mỗi phiên bản, chi tiết:
        - Dữ liệu nguồn.
        - Các quy tắc chuẩn hóa đã áp dụng.
        - Tổng thời lượng, số lượng người nói.
        - Các số liệu thống kê chính từ kiểm tra QA.
        - Git commit hash của mã nguồn được sử dụng để tạo ra nó.

- **Chia tập dữ liệu:**
    - **Phương pháp:** Một lần chia **phân tầng (stratified split)** duy nhất, toàn cục dựa trên ID người nói phải được thực hiện trên *toàn bộ* tập dữ liệu.
    - **Tỷ lệ:** Hiện tại, **90% cho huấn luyện, 5% cho validation và 5% cho kiểm thử.**
    - **Lý do:** Đảm bảo đánh giá khách quan bằng cách ngăn chặn dữ liệu người nói bị rò rỉ giữa các tập huấn luyện và validation.

- **Các chỉ số Đảm bảo Chất lượng (QA):**
    - **Định lượng:** Trước khi hoàn thiện một phiên bản dataset, các số liệu thống kê sau phải được báo cáo:
        - Phân bố thời lượng (histogram).
        - Tỷ lệ phần trăm cắt audio (audio clipping).
        - Phân bố độ lớn (LUFS).
        - Tỷ lệ từ chối (các file không đạt bộ lọc).
    - **Định tính:** Một mẫu nhỏ, ngẫu nhiên các audio từ nhiều người nói sẽ được xem xét thủ công để kiểm tra độ rõ ràng, nhiễu và tính chính xác của bản ghi.

- **Tính nhất quán của Tham số Mô hình:**
    - **Mel-Spectrogram:** Các tham số (n_mels, hop_length, fmax, v.v.) phải giống hệt nhau giữa pipeline fine-tuning (FastPitch) và vocoder của server suy luận (BigVGAN). Điều này rất quan trọng để tránh suy giảm chất lượng audio.

## 3. Môi trường Docker & Quản lý Dependency (Cấu hình V1)

Để giải quyết các xung đột dependency phức tạp giữa các thư viện AI (ví dụ: `unsloth` và `nemo_toolkit`), dự án sử dụng kiến trúc Docker đa container với các môi trường build riêng biệt. Phần này trình bày chi tiết các dependency và chiến lược cài đặt cho từng service theo **Cấu hình V1 hiện tại**.

### 3.1. Service `app` (NLP/Agent)

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

### 3.2. Service `tts` (TTS NeMo)

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

## 4. Kế hoạch Phát triển Module NLP (Production-Ready)

*Đây là kế hoạch phát triển toàn diện để nâng cấp module NLP từ một bản mẫu chức năng thành một agent hội thoại chuyên nghiệp, sẵn sàng cho production, dựa trên các phân tích và đề xuất chuyên sâu.* 

### Giai đoạn 0: Ổn định Nền tảng (Đã hoàn thành và Xác minh)

*   **Mục tiêu:** Khởi động thành công hệ thống hiện tại trên Docker.
*   **Kết quả:** Đã hoàn tất thành công `docker compose build --no-cache app tts` và xác minh cả hai service `app` và `tts` đều khởi động và chạy ổn định mà không gặp lỗi runtime. Các vấn đề về dependency, build và tải model đã được giải quyết triệt để.

### Giai đoạn 1: Lõi Đối thoại & Tương tác (Core Dialogue & Interaction)

*   **Mục tiêu:** Xây dựng nền tảng cho việc hiểu sâu và quản lý hội thoại một cách có cấu trúc.
*   **Hạng mục:**
    1.  **Chuẩn hóa Đầu vào (Input Processing):**
        *   **Intent/Slot Schema:** Định nghĩa cấu trúc JSON chuẩn cho các tác vụ thoại (vd: `tra_cuu_don_hang`).
        *   **Vietnamese Normalization:** Xây dựng module chuẩn hóa tiếng Việt (số, ngày, tiền tệ, không dấu).
        *   **Post-ASR Normalization:** Phục hồi dấu câu, chuẩn hóa chính tả sau khi STT để NLP và TTS hoạt động tốt hơn.
    2.  **Quản lý Trạng thái Hội thoại (Dialogue State Management):**
        *   **State Manager:** Tích hợp `DialogueStateManager` vào LangGraph để quản lý bộ nhớ phiên và chính sách hội thoại (khi nào hỏi lại, khi khi nào chuyển máy).
        *   **Barge-in/Interrupt Handling:** Thiết kế cơ chế để agent có thể xử lý việc người dùng ngắt lời.
    3.  **Ràng buộc Đầu ra & Function Calling An toàn:**
        *   **Structured Output:** Sử dụng GBNF grammar (với `llama.cpp`) hoặc JSON-guided decoding (với `transformers`) để ép LLM trả về đúng định dạng JSON khi gọi tool.
        *   **Input Sanitization:** Làm sạch và kiểm tra đầu vào trước khi truyền cho các tool gọi API của CRM để chống prompt injection.

### Giai đoạn 2: Tối ưu hóa Độ trễ & Tri thức (Latency & Knowledge Optimization)

*   **Mục tiêu:** Giảm độ trễ đầu-cuối xuống dưới 500ms và tăng cường "trí thông minh" cho agent.
*   **Hạng mục:**
    1.  **Tối ưu Streaming & Latency:**
        *   **Endpointing:** Tích hợp VAD để phát hiện người dùng ngưng nói và xử lý audio sớm.
        *   **Speculative Decoding:** Nghiên cứu áp dụng các kỹ thuật như speculative decoding để tăng tốc độ tạo token của LLM.
        *   **KV-Cache Reuse:** Tối ưu việc tái sử dụng KV-cache giữa các lượt nói.
    2.  **Tích hợp Vector Database:**
        *   **Nâng cấp `KnowledgeService`:** Thay thế file JSON bằng Vector DB (ChromaDB, Weaviate) để tìm kiếm ngữ nghĩa, giúp câu trả lời ngữ cảnh chính xác hơn.
        *   **Data Pipeline:** Xây dựng pipeline để cập nhật tri thức cho Vector DB một cách định kỳ.

### Giai đoạn 3: An toàn, Giám sát & Hoàn thiện (Security, Observability & Polish)

*   **Mục tiêu:** Đảm bảo hệ thống an toàn, dễ dàng theo dõi, và có thể đánh giá được hiệu quả.
*   **Hạng mục:**
    1.  **An toàn & Tuân thủ Mở rộng:**
        *   **Advanced PII:** Tích hợp Presidio để dò và che thông tin nhạy cảm trong cả log và prompt.
        *   **Guardrails:** Xây dựng các rào chắn để ngăn agent trả lời các chủ đề không mong muốn và tự động chuyển máy khi phát hiện rủi ro.
    2.  **Quan trắc & Đánh giá Chuyên sâu:**
        *   **Metrics:** Mở rộng hệ thống metric để đo chi tiết latency từng bước (ASR, NLU, TTS) và các chỉ số nghiệp vụ (tỷ lệ thành công, tỷ lệ chuyển máy).
        *   **Bộ Test Tiếng Việt:** Xây dựng bộ test chuyên biệt cho tiếng Việt với các kịch bản thực tế (tiếng ồn, nói không dấu, tên riêng phức tạp).
    3.  **Quản lý Cấu hình & Triển khai:**
        *   **Centralized Config:** Sử dụng Hydra hoặc Pydantic-settings để quản lý tất cả cấu hình một cách tập trung.
        *   **Secret Management:** Tích hợp giải pháp quản lý secret (như HashiCorp Vault hoặc Doppler) cho các khóa API.