# 📞 Hệ Thống Trả Lời Điện Thoại Tự Động Qua AI Agent Cho VoIP 🚀

## 🌟 Giới Thiệu Dự Án

Chào mừng bạn đến với dự án **Hệ Thống Trả Lời Điện Thoại Tự Động Qua AI Agent**! Đây là một giải pháp thực tế sử dụng trí tuệ nhân tạo (AI) để xử lý các cuộc gọi inbound trên nền tảng VoIP, giảm tải cho đội ngũ CSKH và mang đến trải nghiệm tự nhiên cho khách hàng. Hệ thống được xây dựng trên **Asterisk 20** (đóng gói bởi **VitalPBX**) chạy trên **Debian 12**, tận dụng sức mạnh của **8 GPU NVIDIA V100 (32GB VRAM)** để đảm bảo hiệu suất ổn định và khả năng mở rộng.

- **🤖 Công nghệ AI cốt lõi**: Sử dụng model NLP hiệu quả (hiện tại là Llama 4 Scout), ưu tiên hỗ trợ **tiếng Việt** và **tiếng Anh**, với khả năng mở rộng đến 10 ngôn ngữ khác (Arabic, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai) sau khi tối ưu hóa.
- **🎯 Mục tiêu**: Độ trễ end-to-end dưới 800ms, độ chính xác cao (WER < 10% cho STT, BLEU/ROUGE > 0.7 cho NLP), khả năng mở rộng cho hàng trăm cuộc gọi đồng thời ban đầu.
- **🔗 Tích hợp mở**: Module CRM linh hoạt với placeholder API, sẵn sàng tích hợp với Zoho, Salesforce hoặc bất kỳ nền tảng nào sau khi quyết định.

Hệ thống kết hợp **agentic AI**, **emotion detection**, và **federated learning** (làm optional dài hạn) để tạo ra một giải pháp thông minh, an toàn và dễ mở rộng. Dự án tập trung vào tính khả thi thực tế, bắt đầu với prototype đơn giản để kiểm tra latency và accuracy trước khi scale, sử dụng các công nghệ tiên tiến được cộng đồng ưa chuộng như vLLM cho inference tối ưu và Pipecat cho voice agent stack.

## 🎯 Mục Tiêu Và Phạm Vi

### Mục Tiêu
- **📈 Tự động hóa CSKH**: Xử lý 60-70% cuộc gọi cơ bản (hỏi đáp, tra cứu thông tin), giảm chi phí vận hành lên đến 40%.
- **🌍 Đa ngôn ngữ**: Ưu tiên tiếng Việt/Anh, hỗ trợ mở rộng đến 10 ngôn ngữ khác cho khách hàng quốc tế sau prototype.
- **😊 Trải nghiệm thông minh**: Phát hiện cảm xúc để điều chỉnh phản hồi, nâng cao trải nghiệm người dùng.
- **🔒 Bảo mật**: Tuân thủ GDPR-like với encryption và federated learning (triển khai dần).
- **🚀 Mở rộng**: Thiết kế modular để dễ scale và tích hợp tính năng mới.

### Phạm Vi
- **✅ Bao gồm**:
  - AI components: STT (Google Cloud Speech-to-Text API), NLP (Llama 4 Scout), TTS (Orpheus-TTS cho giọng tự nhiên và emotion-adjusted).
  - Tích hợp VoIP qua AGI/ARI.
  - Module CRM mở với placeholder API để tra cứu thông tin.
  - Dashboard cơ bản (Streamlit hoặc Flask) cho monitoring latency và accuracy.
- **❌ Không bao gồm**:
  - Tính năng ngoài AI/VoIP (thanh toán, marketing).
- **📋 Yêu Cầu**:
  - Dataset CSKH (lịch sử chat/cuộc gọi) với ít nhất 50.000 mẫu sẵn có hoặc thu thập bổ sung.
  - Tích hợp CRM cụ thể sẽ thực hiện sau khi chọn nền tảng.

## 🏗 Kiến Trúc Hệ Thống (Đã cập nhật)

Kiến trúc được đơn giản hóa để phản ánh tình trạng hiện tại của dự án và các quyết định mới.

- **Hardware**: Server với 8 GPU NVIDIA V100 (32GB VRAM).
- **VoIP**: VitalPBX / Asterisk 20.
- **AGI Script**: `src/agi/agi_handler.py` là trái tim của hệ thống, điều phối toàn bộ luồng xử lý.

Luồng dữ liệu thực tế như sau:

```
📞 Cuộc gọi Inbound (VitalPBX)
   ↓
🎙 Asterisk Dialplan thực thi AGI script
   ↓
🐍 AGI Script (`agi_handler.py`)
   1. Ghi âm giọng nói người dùng.
   2. Gửi âm thanh đến Google STT API.
   3. Nhận lại văn bản.
   4. Gửi văn bản đến NLP Server (API tại http://localhost:8000).
   5. Nhận lại văn bản trả lời.
   6. Gọi API của TTS để chuyển thành giọng nói.
   7. Phát lại âm thanh cho người dùng.
   ↓
👋 Kết thúc cuộc gọi
```

### Giải thích chi tiết về luồng hoạt động của AGI

Về cốt lõi, **AGI (Asterisk Gateway Interface)** là một "cây cầu" cho phép Asterisk tạm dừng xử lý một cuộc gọi và **trao quyền điều khiển cuộc gọi đó cho một script bên ngoài**. Luồng hoạt động của nó trong dự án này như sau:

1.  **Gọi đến Dialplan**: Một cuộc gọi đi vào hệ thống VitalPBX/Asterisk.
2.  **Thực thi lệnh `AGI()`**: Trong dialplan (được cấu hình qua giao diện web VitalPBX), Asterisk được lệnh thực thi ứng dụng `AGI()` và trỏ đến script của chúng ta.
3.  **Chạy Script Wrapper**: Asterisk tìm và chạy script `voip_ai_agent.sh` trong thư mục `/var/lib/asterisk/agi-bin/`.
4.  **Kích hoạt môi trường Python**: Script wrapper này thiết lập `PYTHONPATH` và thực thi script logic chính là `src/agi/agi_handler.py`.
5.  **Giao tiếp hai chiều**: Script Python bắt đầu một vòng lặp "nói chuyện" với Asterisk. Nó gửi các lệnh như `ANSWER`, `RECORD FILE`, `STREAM FILE` tới Asterisk và nhận lại kết quả. Trong quá trình này, nó cũng gọi tới các API của Google STT và Llama.
6.  **Trả lại quyền điều khiển**: Khi script Python kết thúc, quyền điều khiển được trả lại cho Asterisk để kết thúc cuộc gọi (`Hangup()`).

### Các Thành Phần Chính
- **🖥️ AI Backend**:
  - **STT**: **Google Cloud Speech-to-Text API**. Yêu cầu file credentials được cấu hình qua biến môi trường `GOOGLE_APPLICATION_CREDENTIALS`.
  - **NLP**: Model **`unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit`**, được phục vụ qua một API tương thích OpenAI (ví dụ: vLLM) tại `http://localhost:8000`.
  - **TTS**: Một service Text-to-Speech (ví dụ: Google TTS, Piper, v.v.). Code hiện tại trong `src/tts/generate_audio.py` cần được kiểm tra và hoàn thiện.
- **📡 Tích hợp VoIP**:
  - Sử dụng **AGI (Asterisk Gateway Interface)**. Script `src/agi/agi_handler.py` đã bao gồm logic giao tiếp với Asterisk.

## 🚀 Kế Hoạch Hành Động (Đã cập nhật)

Dựa trên code đã có, kế hoạch được điều chỉnh để tập trung vào việc "lắp ráp" và đưa hệ thống vào hoạt động.

### Giai đoạn 1: Cài đặt và Cấu hình Môi trường (Hành động ngay)

1.  **Cài đặt Dependencies**: Chạy `pip install -r requirements.txt` để cài đặt tất cả các thư viện Python cần thiết.
2.  **Cấu hình Google Cloud**:
    *   Tạo một Service Account trên Google Cloud Platform với quyền truy cập Speech-to-Text API.
    *   Tải file credentials JSON về server.
    *   Đặt biến môi trường: `export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"`.
3.  **Chuẩn bị file âm thanh**: Tạo các file âm thanh (`welcome.wav`, `stt_error.wav`, `nlp_error.wav`, `tts_error.wav`) và đặt chúng vào thư mục sounds của Asterisk (ví dụ: `/var/lib/asterisk/sounds/en`).

### Giai đoạn 2: Lựa chọn và Triển khai NLP Server

1.  **Đánh giá và Chọn Model NLP**: Dựa trên yêu cầu về ngôn ngữ (tiếng Việt), tốc độ và độ chính xác, hãy chọn một model từ Hugging Face (ví dụ: một model từ VinAI, FPT AI, hoặc một model Llama đã được fine-tune).
2.  **Chạy NLP Server**: Sử dụng `vLLM` hoặc một công cụ tương tự để triển khai model đã chọn thành một API service tại `http://localhost:8000`. AGI script đã sẵn sàng để gọi đến endpoint này.

### Giai đoạn 3: Tích hợp và Kiểm thử End-to-End

1.  **Triển khai AGI Script**:
    *   Copy toàn bộ thư mục dự án (`voip-ai-agent`) vào một vị trí hợp lý trên server (ví dụ: `/opt/voip-ai-agent`).
    *   Tạo một "entrypoint" script trong thư mục `agi-bin` của Asterisk (`/var/lib/asterisk/agi-bin/`) để gọi đến `src/agi/agi_handler.py` và đảm bảo `PYTHONPATH` được thiết lập đúng.
    *   Cấp quyền thực thi (`chmod +x`) cho script đó.
2.  **Cấu hình Asterisk Dialplan**: Chỉnh sửa dialplan trong VitalPBX để khi có cuộc gọi đến, nó sẽ thực thi AGI script của bạn.
3.  **Kiểm thử**: Thực hiện cuộc gọi đến hệ thống và kiểm tra toàn bộ luồng hoạt động. Theo dõi file log `agi_handler.log` để gỡ lỗi.

### Giai đoạn 4: Tối ưu và Mở rộng (Tương lai)

-   Fine-tune model NLP với dữ liệu CSKH của riêng bạn.
-   Tối ưu hóa độ trễ của từng thành phần (STT, NLP, TTS).
-   Đóng gói ứng dụng bằng Docker/Kubernetes để dễ dàng quản lý và scale.
-   Tích hợp module CRM như kế hoạch ban đầu.

## 💻 Tài Nguyên Cần Thiết

- **🖥 Hardware**:
  - Server: 8 GPU NVIDIA V100 (32GB VRAM mỗi cái) hoặc tương đương trên cloud (AWS/GCP) để fallback.
  - Lưu trữ: SSD 2TB+ cho dataset/models.
- **🛠 Software/Công Cụ**:
  - Hệ điều hành: **Debian 12**.
  - VoIP: **VitalPBX/Asterisk 20**.
  - AI: **Unsloth**, **Hugging Face Transformers**, **LangGraph**, **llama.cpp**, **vLLM**, **TensorRT-LLM**, **Pipecat** (mã nguồn mở).
- **📚 Dataset**:
  - ~50.000 mẫu CSKH (chat logs, ghi âm tiếng Việt/Anh).
  - Nếu chưa đủ: Thu thập từ lịch sử nội bộ, augmentation tools, hoặc synthetic data chất lượng cao.

## ⚠ Rủi Ro Và Giải Pháp

- **🔥 Latency cao**:
  - **Giải pháp**: Quantization 4-bit, multi-GPU với vLLM/TensorRT-LLM, streaming STT/TTS; fallback mô hình nhỏ hơn hoặc cloud offload.
- **📉 Dataset hạn chế**:
  - **Giải pháp**: Bổ sung dữ liệu thực tế (record calls với consent), sử dụng augmentation như NeMo cho tiếng Việt.
- **🔌 Tích hợp CRM chưa rõ**:
  - **Giải pháp**: Module mở với placeholder API, custom code cho từng CRM khi cần.
- **🔐 Privacy**:
  - **Giải pháp**: Encryption, federated learning optional; auditing logs đầy đủ.
- **📈 Tải cao**:
  - **Giải pháp**: Kubernetes cho load balancing, scale ngang; bắt đầu với hàng trăm calls, monitor để expand.

## 🌈 Roadmap Dài Hạn

- **📅 Ngắn Hạn**:
  - Cập nhật các phiên bản Llama 4 Scout từ Unsloth.
  - Tích hợp federated learning cho privacy nếu cần.
- **📆 Trung Hạn**:
  - Kích hoạt module CRM (Zoho/Salesforce).
  - Hỗ trợ multi-modal (SMS/image).
  - Mở rộng ngôn ngữ hỗ trợ đầy đủ.
- **🚀 Dài Hạn**:
  - Tích hợp mô hình mới (như Llama 5 nếu ra mắt).
  - Duy trì CI/CD với GitHub Actions.

## 📢 Liên Hệ

Để thảo luận hoặc điều chỉnh kế hoạch, liên hệ đội ngũ kỹ thuật qua email hoặc kênh nội bộ. Cùng xây dựng một hệ thống VoIP thông minh và khả thi! 🚀