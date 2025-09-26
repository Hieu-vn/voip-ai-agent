# Technical Implementation Plan: AI Agent for VoIP

**Objective:** To align the project's codebase and documentation with a unified, modern, streaming-first architecture using Asterisk ARI and NVIDIA NeMo for TTS.

## 1. System Architecture (The Single Source of Truth)

- **VoIP Interface:** Asterisk REST Interface (ARI). Event-driven and asynchronous.
- **Application Core:** A main Python process (`src/main.py`) that spawns a `CallHandler` (`src/core/call_handler.py`) for each incoming call.
- **STT:** Google Cloud Speech-to-Text API (streaming).
- **NLP:** Llama 4 Scout (local).
- **TTS:** A separate, dedicated NVIDIA NeMo server (FastPitch + BigVGAN) that the main application communicates with via a REST API.

### Data Flow Summary

```
Call -> Asterisk (ARI Event) -> CallHandler (forks RTP Audio) -> RTPAudioForwarder -> STTModule (Google Cloud STT) -> Get Text -> 
NLPModule (Llama 4 Scout) -> Get Response Text -> TTSModule (Client) -> NeMo TTS Server (FastPitch + BigVGAN) -> Get Audio Stream -> CallHandler (Playback) -> User
```

## 2. Data Contract & Quality Assurance

This section defines the rules and standards for all datasets used in training to ensure quality, consistency, and reproducibility.

- **Text Normalization:**
    - **Source:** All normalization rules (e.g., converting numbers to text, handling abbreviations, special characters) must be defined in a separate configuration file (e.g., `config/normalization_rules.yaml`).
    - **Rationale:** Decouples rules from code, allowing for easier maintenance and versioning of different normalization schemes.
    - **Implementation:** Scripts must load and apply these external rules.

- **Audio Data Filtering:**
    - **Duration:** Audio clips must have a duration between **1.5 and 20 seconds**.
    - **Application:** This filter is applied *before* writing the final training manifests.
    - **Rationale:** Prevents model alignment issues caused by clips that are too short or too long.

- **Data Provenance and Versioning:**
    - **Checksum:** Each processed WAV file should have a checksum (e.g., MD5) recorded in the manifest.
    - **Dataset Version:** Each generated manifest set (train/validation/test) will be assigned a version (e.g., `v1.0`, `v1.1`).
    - **Metadata:** A "Dataset Card" will accompany each version, detailing:
        - Source data.
        - Normalization rules applied.
        - Total duration, number of speakers.
        - Key statistics from QA checks.
        - Git commit hash of the code used to generate it.

- **Dataset Splitting:**
    - **Method:** A single, global **stratified split** based on speaker ID must be performed on the *entire* dataset.
    - **Ratios:** Currently, **90% for training, 5% for validation, and 5% for testing.**
    - **Rationale:** Ensures objective evaluation by preventing speaker data from leaking between training and validation sets.

- **Quality Assurance (QA) Metrics:**
    - **Quantitative:** Before finalizing a dataset version, the following stats must be reported:
        - Duration distribution (histogram).
        - Percentage of audio clipping.
        - Loudness distribution (LUFS).
        - Rejection rate (files failing filters).
    - **Qualitative:** A small, random sample of audio from various speakers will be manually reviewed to check for clarity, noise, and correctness of transcription.

- **Model Parameter Consistency:**
    - **Mel-Spectrogram:** Parameters (n_mels, hop_length, fmax, etc.) must be identical between the fine-tuning pipeline (FastPitch) and the inference server's vocoder (BigVGAN). This is critical to avoid audio quality degradation.

## 3. Task Breakdown & Implementation Plan

This plan is broken down into sequential stages for a structured and verifiable rollout of the TTS system.

### Stage 1: Foundation & Model Setup (Current Stage)

*   **Task 1.1: Code Cleanup**
    *   **Action:** Delete `app.py` and `src/tts/generate_audio.py`.
    *   **Status:** Done.

*   **Task 1.2: Update Dependencies**
    *   **Action:** Add `nemo_toolkit[all]`, `fastapi`, `uvicorn` to `requirements.txt`.
    *   **Status:** Done.

*   **Task 1.3: Download Pre-trained Models**
    *   **Action:** Manually download the following models from NVIDIA NGC:
        -   `nvidia/tts_en_fastpitch`
        -   `nvidia/bigvgan_v2_22khz_80band_256x`
    *   **Action:** Place the downloaded models into the `models/tts/en` (FastPitch) and `models/tts/vocoder` (BigVGAN) directories.
    *   **Status:** Done.

*   **Task 1.4: Install & Configure Docker Environment**
    *   **Action:** Install Docker Engine and Docker Compose plugin on Debian 12 (using `scripts/setup/install_docker.sh`).
    *   **Action:** Create `.dockerignore` to optimize build context.
    *   **Action:** Configure Docker to use `/data` partition for its data root.
    *   **Action:** Optimize `Dockerfile` and `requirements.txt` for robust build (including system dependencies like `build-essential`, `sox`, `python3.11-dev`, and Python build-time dependencies like `numpy`, `typing_extensions`, `Cython`, `wheel`).
    *   **Status:** Done.

*   **Task 1.5: Update Environment Configuration**
    *   **Action:** Update `.env` file with correct paths to downloaded models.
    *   **Status:** Done.

### Stage 2: NeMo TTS Server Implementation (English)

*   **Task 2.1: Implement the FastAPI Server**
    *   **Action:** Develop the server logic in `tts_server/server.py`.
    *   **Details:**
        -   The server will load the FastPitch and BigVGAN models based on paths specified in the `.env` file.
        -   It must expose a `/synthesize` endpoint that takes `text` and `language` as input.
        -   For now, it will only handle the `en` language code.

*   **Task 2.2: Client-Side Integration**
    *   **Action:** Modify `src/core/tts_module.py` to make a POST request to the TTS server.
    *   **Action:** Ensure the `TTS_SERVER_URL` in the `.env` file is correctly configured.

*   **Task 2.3: Initial Launch & Testing**
    *   **Action:** Update `.env` with the paths to the English models.
    *   **Status:** Done (Configuration is ready, actual launch pending successful Docker build).
    *   **Action:** Launch both the main application and the TTS server.
    *   **Verification:** Place a test call and verify that English TTS is working end-to-end.

### Stage 3: Vietnamese Model Fine-Tuning

*   **Task 3.1: Prepare Dataset**
    *   **Action:** Run the data preparation scripts in `scripts/preparation/` (`1_prepare_audio_parquet.py`, `2_create_manifest.py`) on the `phoaudiobook` dataset.
    *   **Goal:** To create the manifest files required for NeMo training (train, validation, test).
    *   **Status:** Done.

*   **Task 3.2: Run Fine-Tuning Script**
    *   **Action:** Execute `scripts/training/run_finetune.py` using the configuration from `config/training/config_finetune_vi.yaml`.
    *   **Action:** Ensure `config/training/config_finetune_vi.yaml` points to the correct manifest files.
    *   **Status:** Configuration updated.
    *   **Goal:** To generate a fine-tuned FastPitch model for Vietnamese.

### Stage 4: Final Integration & Completion

*   **Task 4.1: Update Configuration**
    *   **Action:** Modify the `.env` file to include the path to the newly trained Vietnamese model.

*   **Task 4.2: Final Verification**
    *   **Action:** Restart the TTS server.
    *   **Verification:** Place test calls and confirm that the system can now synthesize speech in both English and Vietnamese.

---
## 4. Kế hoạch Phát triển Module NLP (Production-Ready)

*Đây là kế hoạch phát triển toàn diện để nâng cấp module NLP từ một bản mẫu chức năng thành một agent hội thoại chuyên nghiệp, sẵn sàng cho production, dựa trên các phân tích và đề xuất chuyên sâu.*

### Giai đoạn 0: Ổn định Nền tảng (Ưu tiên tức thì)
*   **Mục tiêu:** Khởi động thành công hệ thống hiện tại trên Docker. Đây là bước bắt buộc để có một nền tảng ổn định trước khi thực hiện các nâng cấp.
*   **Hành động:**
    1.  Hoàn tất thành công `docker compose build --no-cache`.
    2.  Chạy `docker compose up` và xác minh cả hai service `app` và `tts_server` đều khởi động mà không gặp lỗi.

### Giai đoạn 1: Lõi Đối thoại & Tương tác (Core Dialogue & Interaction)
*   **Mục tiêu:** Xây dựng nền tảng cho việc hiểu sâu và quản lý hội thoại một cách có cấu trúc.
*   **Hạng mục:**
    1.  **Chuẩn hóa Đầu vào (Input Processing):**
        *   **Intent/Slot Schema:** Định nghĩa cấu trúc JSON chuẩn cho các tác vụ thoại (vd: `tra_cuu_don_hang`).
        *   **Vietnamese Normalization:** Xây dựng module chuẩn hóa tiếng Việt (số, ngày, tiền tệ, không dấu).
        *   **Post-ASR Normalization:** Phục hồi dấu câu, chuẩn hóa chính tả sau khi STT để NLP và TTS hoạt động tốt hơn.
    2.  **Quản lý Trạng thái Hội thoại (Dialogue State Management):**
        *   **State Manager:** Tích hợp `DialogueStateManager` vào LangGraph để quản lý bộ nhớ phiên và chính sách hội thoại (khi nào hỏi lại, khi nào chuyển máy).
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