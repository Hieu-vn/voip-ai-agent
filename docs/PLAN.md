## Kế hoạch triển khai hệ thống trả lời điện thoại tự động AI Agent trên VoIP

**Mục tiêu:** Triển khai hệ thống trả lời điện thoại tự động AI Agent trên Asterisk/VitalPBX, tích hợp STT, NLP và TTS, đạt độ trễ thấp (dưới 500ms), tự chủ (agentic AI), và tối ưu cho tiếng Việt.

### 1. Thiết kế kiến trúc kỹ thuật

#### 1.1. Luồng dữ liệu tổng quan (Text-based/ASCII Art)

```
Cuộc gọi VoIP (Inbound)
      |
      V
Asterisk (VitalPBX)
      | (AGI: voip_ai_agent.sh)
      V
Python AGI Handler (agi_handler.py)
      |
      +--- Ghi âm giọng nói người dùng (file WAV)
      |
      +--- STT (transcribe.py - Google Cloud STT API)
      |      (Chuyển WAV -> Text)
      V
Văn bản người dùng
      |
      +--- NLP (process_nlp -> Llama 4 Scout API Server)
      |      (Xử lý ý định, tạo phản hồi, MCP/Function Calling, Emotion Detection)
      V
Văn bản phản hồi AI
      |
      +--- TTS (generate_audio.py -> TTS Model)
      |      (Chuyển Text -> WAV)
      V
Audio phản hồi AI (lưu vào Asterisk sounds/vi/custom/)
      |
      V
Asterisk (stream_file)
      |
      V
Playback cho người dùng
      |
      V
Kết thúc cuộc gọi hoặc tiếp tục vòng lặp hội thoại
```

#### 1.2. Tích hợp Asterisk (AGI)

*   **Cốt lõi:** AGI (Asterisk Gateway Interface) cho phép Asterisk tạm dừng cuộc gọi và trao quyền điều khiển cho script Python bên ngoài.
*   **Luồng:** Dialplan của Asterisk sẽ cấu hình để thực thi `AGI(voip_ai_agent.sh)`. Script shell này kích hoạt `agi_handler.py` (logic Python chính) để giao tiếp hai chiều với Asterisk.

#### 1.3. Tích hợp CRM (MCP) và AI Agent (LangGraph)

*   **MCP (Model Context Protocol):** Sử dụng function calling trong mô hình NLP (Llama 4 Scout) để AI Agent tra cứu dữ liệu CRM (như Zoho, Salesforce) hoặc cơ sở dữ liệu khác (ví dụ: trạng thái đơn hàng).
*   **AI Agent (LangGraph):** Xây dựng agentic AI với LangGraph (hoặc framework tương tự) để điều phối các tác vụ phức tạp (STT, tra cứu, phản hồi, emotion detection).
*   **Emotion Detection:** Tích hợp để điều chỉnh phản hồi của AI Agent.

### 2. Cài đặt môi trường

1.  **Cài đặt cơ bản:**
    *   Debian 12, VitalPBX/Asterisk 20.
    *   NVIDIA drivers/CUDA 12.x cho 8 GPU NVIDIA V100.
    *   `ffmpeg` (yêu cầu bởi `pydub`).

2.  **Cấu hình môi trường Python:**
    *   Môi trường ảo Python (`venv`) tại `/data/voip-ai-agent/venv/`.
    *   Cài đặt các thư viện Python từ `requirements.txt` (bao gồm Unsloth, Hugging Face Transformers, llama.cpp, google-cloud-speech, google-cloud-texttospeech, loguru, pydub, asterisk.agi).

3.  **Cấu hình thư mục âm thanh Asterisk:**
    *   Tạo các thư mục `/var/lib/asterisk/sounds/vi/` và `/var/lib/asterisk/sounds/vi/custom/`. (Sử dụng `mkdir -p`).
    *   Tạo các tệp âm thanh giả (silent WAV) cho các lời nhắc của Asterisk (`welcome`, `beep`, `please_repeat`, `error_stt`). (Sử dụng một script Python để tạo).
    *   Đặt quyền và quyền sở hữu thích hợp cho các tệp và thư mục âm thanh Asterisk.

### 3. Triển khai mã nguồn AI Agent

1.  **`agi_handler.py`:**
    *   **Chức năng:** Là trung tâm điều khiển luồng cuộc gọi, giao tiếp hai chiều với Asterisk thông qua AGI. Nó điều phối các tác vụ STT, NLP và TTS.
    *   **Thành phần chính:** Xử lý cuộc gọi, ghi âm, gọi STT, NLP, TTS, phát lại âm thanh, xử lý vòng lặp hội thoại.

2.  **`transcribe.py`:**
    *   **Chức năng:** Chuyển đổi giọng nói thành văn bản (STT) sử dụng Google Cloud Speech-to-Text API (đồng bộ, file-based).
    *   **Thành phần chính:** Nhận tệp âm thanh, gửi đến Google Cloud STT, trả về văn bản phiên âm.

3.  **`generate_audio.py`:**
    *   **Chức năng:** Chuyển đổi văn bản thành giọng nói (TTS) sử dụng mô hình TTS (ví dụ: `facebook/mms-tts-vie`).
    *   **Thành phần chính:** Tải và sử dụng mô hình TTS, tổng hợp giọng nói, lưu tệp âm thanh.

4.  **`voip_ai_agent.sh`:**
    *   **Chức năng:** Script shell trung gian được Asterisk gọi, thiết lập môi trường Python và khởi chạy `agi_handler.py`.
    *   **Thành phần chính:** Thiết lập biến môi trường (`PYTHONPATH`, `GOOGLE_APPLICATION_CREDENTIALS`), thực thi `agi_handler.py`, chuyển hướng lỗi Python.

### 4. Cấu hình Asterisk/VitalPBX

1.  **Xác nhận Dialplan:**
    *   Đảm bảo `vitalpbx/extensions__99-ai-agent.conf` chứa context `[custom-ai-agent]` và gọi `AGI(voip_ai_agent.sh)`.
    *   Đảm bảo `vitalpbx/extensions__50-1-dialplan.conf` định tuyến cuộc gọi đến `custom-contexts,cc-1,1` cho DID mong muốn.

2.  **Cấu hình Inbound Route trong VitalPBX:**
    *   Tạo Inbound Route mới trong giao diện web VitalPBX, với DID là số điện thoại của bạn.
    *   Destination là Custom Destination, với giá trị `custom-contexts,cc-1,1`.
    *   Lưu và áp dụng thay đổi.

### 5. Khởi chạy và kiểm tra

1.  **Khởi chạy máy chủ NLP (Llama 4 Scout):**
    *   **Mục đích:** Cung cấp API cho `agi_handler.py` để xử lý văn bản bằng mô hình Llama 4 Scout.
    *   **Thành phần chính:**
        *   Mô hình Llama 4 Scout (tối ưu hóa bằng Unsloth, có thể chuyển đổi sang GGUF cho `llama.cpp`).
        *   API Server (ví dụ: FastAPI/Uvicorn) để expose mô hình qua HTTP.
        *   Sử dụng `device_map='auto'` hoặc `model.parallelize()` để tận dụng 8 GPU V100 cho multi-GPU inference.
        *   API phải tương thích với OpenAI Chat Completions API tại `http://localhost:8000/v1/chat/completions`.

2.  **Theo dõi nhật ký:**
    *   Mở các cửa sổ terminal riêng biệt để theo dõi nhật ký hoạt động và lỗi của các thành phần.

3.  **Thực hiện cuộc gọi kiểm tra:**
    *   Gọi đến số DID đã cấu hình.
    *   Nói rõ ràng vào điện thoại.
    *   Quan sát nhật ký để xem quá trình STT, NLP và TTS.

### 6. Tối ưu hóa GPU và hiệu năng

*   **Unsloth (4-bit quantization):** Sử dụng để giảm VRAM và latency cho Llama 4 Scout.
*   **llama.cpp (GGUF):** Tùy chọn để chạy Llama 4 Scout trên 8 V100 với hiệu suất cao hơn.
*   **Đo latency:** Đặt mục tiêu dưới 500ms. Tối ưu hóa bằng cách điều chỉnh batch size, multi-GPU.

### 7. Dài hạn (Technical Roadmap)

*   **Scalability:** Xử lý 1000+ cuộc gọi đồng thời (multi-GPU, Kubernetes).
*   **Maintainability:** Mã nguồn modular (classes/functions), cấu trúc Git repo rõ ràng.
*   **Cập nhật công nghệ:** Kế hoạch tích hợp Llama 4 updates, federated learning cho privacy, multi-modal (SMS/image nếu mở rộng).
*   **Privacy:** Đề xuất encryption cho audio/data, tuân thủ các quy định bảo mật (GDPR-like).

### 8. Yêu cầu code

*   Code Python phải clean, modular, có comment chi tiết, error handling, và logging.
*   Liệt kê đầy đủ thư viện (trong `requirements.txt`).
*   Đảm bảo code chạy trên Debian 12 với quyền root.

**Lưu ý:** Mã nguồn đầy đủ cho các tệp `agi_handler.py`, `transcribe.py`, `generate_audio.py` được quản lý trong các tệp tương ứng trong dự án. Vui lòng tham khảo trực tiếp các tệp đó để xem chi tiết triển khai.