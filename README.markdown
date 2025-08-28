# 📞 Hệ Thống Trả Lời Điện Thoại Tự Động Qua AI Agent Cho VoIP 🚀

## 🌟 Giới Thiệu Dự Án

Chào mừng bạn đến với dự án **Hệ Thống Trả Lời Điện Thoại Tự Động Qua AI Agent**! Đây là một giải pháp thực tế sử dụng trí tuệ nhân tạo (AI) để xử lý các cuộc gọi inbound trên nền tảng VoIP, giảm tải cho đội ngũ CSKH và mang đến trải nghiệm tự nhiên cho khách hàng. Hệ thống được xây dựng trên **Asterisk 20** (đóng gói bởi **VitalPBX**) chạy trên **Debian 12**, tận dụng sức mạnh của **8 GPU NVIDIA V100 (32GB VRAM)** để đảm bảo hiệu suất ổn định và khả năng mở rộng.

- **🤖 Công nghệ AI cốt lõi**: Sử dụng **Llama 4 Maverick** (`unsloth/Llama-4-Maverick-17B-128E-Instruct-unsloth-bnb-4bit`) từ Unsloth trên Hugging Face, ưu tiên hỗ trợ **tiếng Việt** và **tiếng Anh**, với khả năng mở rộng đến 10 ngôn ngữ khác (Arabic, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai) sau khi tối ưu hóa.
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
  - AI components: STT (Faster-Whisper dựa trên Whisper-large-v3), NLP (Llama 4 Maverick, bắt đầu prototype với Llama 3 8B để test), TTS (Orpheus-TTS cho giọng tự nhiên và emotion-adjusted).
  - Tích hợp VoIP qua AGI/ARI.
  - Module CRM mở với placeholder API để tra cứu thông tin.
  - Dashboard cơ bản (Streamlit hoặc Flask) cho monitoring latency và accuracy.
- **❌ Không bao gồm**:
  - Tính năng ngoài AI/VoIP (thanh toán, marketing).
- **📋 Yêu Cầu**:
  - Dataset CSKH (lịch sử chat/cuộc gọi) với ít nhất 50.000 mẫu sẵn có hoặc thu thập bổ sung.
  - Tích hợp CRM cụ thể sẽ thực hiện sau khi chọn nền tảng.

## 🏗 Kiến Trúc Hệ Thống

Hệ thống được thiết kế **modular** để đảm bảo tính linh hoạt, dễ bảo trì và mở rộng. Dưới đây là luồng dữ liệu chính (ASCII art):

```
📞 Cuộc gọi Inbound
   ↓
🎙 Asterisk ghi âm (5-30s, streaming realtime)
   ↓
🔊 STT (Faster-Whisper) → Text (tiếng Việt/Anh)
   ↓
🧠 NLP (Llama 4 Maverick hoặc Llama 3 8B) + MCP (CRM mở) + Emotion Detection
   ↓
📝 Phản hồi thông minh
   ↓
🎵 TTS (Orpheus-TTS) → Audio (emotion-adjusted)
   ↓
🔊 Playback qua Asterisk
   ↓
🔄 Nếu phức tạp: Chuyển nhân viên/SMS/Email (handover seamless)
```

### Các Thành Phần Chính
- **🖥 AI Backend**:
  - Chạy trong **Docker container** trên 8 GPU V100 (multi-GPU, device_map='auto').
  - STT: **Faster-Whisper** (optimized version of Whisper-large-v3 cho realtime, hỗ trợ tiếng Việt/Anh, multilingual).
  - NLP: **Llama 4 Maverick 4-bit** (Unsloth, ~20-25GB VRAM, fine-tune cho CSKH; prototype với Llama 3 8B để giảm latency).
  - TTS: **Orpheus-TTS** (giọng tự nhiên, human-like emotion, zero-shot multilingual cloning).
- **📡 VoIP Integration**:
  - AGI script Python kết nối Asterisk với AI, sử dụng streaming để giảm latency.
  - ARI (WebSocket) cho low-latency, tích hợp Pipecat cho voice agent orchestration.
- **🔗 CRM Module Mở**:
  - Class `CRMIntegrator` với config file/env vars, placeholder API để tra cứu.
  - Dễ switch giữa Zoho, Salesforce hoặc nền tảng khác.
- **🛠 Công nghệ hỗ trợ**:
  - **LangGraph**: Orchestration agentic AI (multi-agent flow).
  - **Unsloth/llama.cpp**: Quantization 4-bit, tối ưu GPU.
  - **vLLM/TensorRT-LLM**: Optimized inference cho LLM trên GPU V100.
  - **Pipecat**: Full-stack framework cho voice agents (STT/TTS integration).
  - **Federated learning**: Optional cho bảo mật dữ liệu dài hạn.

### Metrics Kỹ Thuật
- **🔊 STT**: Word Error Rate (WER) < 10%.
- **🧠 NLP**: BLEU/ROUGE > 0.7.
- **⏱ Latency**: End-to-end < 800ms (target thực tế dựa trên benchmarks 2025).

## 🚀 Các Giai Đoạn Phát Triển

Dự án chia thành 5 giai đoạn để đảm bảo triển khai có hệ thống:

1. **📝 Nghiên Cứu Và Thiết Kế**:
   - Thu thập yêu cầu kỹ thuật (dataset, ngôn ngữ).
   - Thiết kế kiến trúc AI, module CRM mở (placeholder API).
   - So sánh Llama 4 Maverick với các variant khác, chọn Orpheus-TTS cho TTS và Faster-Whisper cho STT.
   - **Output**: Tài liệu thiết kế, sơ đồ kiến trúc.

2. **🛠 Cài Đặt Và Prototype**:
   - Cài đặt Debian 12, VitalPBX/Asterisk, CUDA, Unsloth, Hugging Face, vLLM, Pipecat.
   - Xây prototype STT/NLP/TTS với Llama 3 8B, tích hợp AGI script.
   - Test module CRM mở với mock API, thêm dashboard cơ bản cho monitoring.
   - **Output**: Prototype hoạt động, báo cáo latency thực tế.

3. **🤖 Phát Triển Core AI**:
   - STT realtime với Faster-Whisper, NLP fine-tune Llama 4 Maverick qua vLLM, TTS emotion-adjusted với Orpheus-TTS.
   - Tích hợp LangGraph, MCP, emotion detection, Pipecat cho voice flow.
   - **Output**: Hệ thống đầy đủ, sẵn sàng test.

4. **⚙ Test Và Tối Ưu Hóa**:
   - Unit/end-to-end tests (WER, BLEU/ROUGE, latency).
   - Tối ưu GPU (4-bit quantization, multi-GPU với TensorRT-LLM, fallback cloud nếu cần).
   - Đảm bảo hỗ trợ tiếng Việt/Anh trước, mở rộng đa ngôn ngữ sau.
   - **Output**: Hệ thống tối ưu, báo cáo test.

5. **🌐 Triển Khai Và Bảo Trì**:
   - Deploy với Docker/Kubernetes.
   - Thiết lập monitoring (latency, accuracy).
   - Hỗ trợ tích hợp CRM cụ thể nếu được chọn.
   - **Output**: Hệ thống sản xuất, kế hoạch bảo trì.

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
  - Cập nhật Llama 4 Maverick từ Unsloth.
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