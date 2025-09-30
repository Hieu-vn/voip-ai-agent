# Ke hoach trien khai ky thuat: AI Agent cho VoIP

**Muc tieu:** Hoan thien nen tang agent streaming-first tren Asterisk ARI, giu do tre <500ms va toan bo quan ly bang Docker/observability.

## 1. Kien truc he thong (single source of truth)
- **VoIP interface:** Asterisk REST Interface (ARI) dieu khien qua WebSocket su kien.
- **Ung dung chinh:** Service Python `app` dung asyncio + ari-py, tao `CallHandler` moi cho moi cuoc goi.
- **Speech-to-Text:** Google Cloud Speech-to-Text API v2 streaming (telephony, VAD).
- **NLU/LLM:** Llama 4 Scout dang GGUF chay bang `llama-cpp-python`, LangGraph dieu phoi, GBNF grammar ep JSON.
- **Text-to-Speech:** Service `tts` rieng su dung NVIDIA NeMo (FastPitch + HiFiGAN) phat PCM 16-bit 8kHz.
- **Observability:** structlog JSON logging + OpenTelemetry tracing cho ca hai service.

### So do luong du lieu
```
Cuoc goi -> ARI Stasis -> app/main.py -> CallHandler (app/audio/stream.py)
  -> Google STT (`app/stt/client.py`) -> TextNormalizer -> LangGraph Agent (`app/nlu/agent.py`)
  -> Guardrails/Emotion -> TTS client (`app/tts/client.py`) -> NeMo TTS (`tts_server/api.py`)
  -> RTP outbound ve Asterisk -> nguoi dung
```

## 2. Chat luong du lieu & chuan hoa
- **Chuan hoa van ban:** Quy tac trong `config/normalization_rules.yaml`, bo sung logic number-to-words trong `TextNormalizer`.
- **Luu vet hoi thoai:** `app/evaluation/tracker.py` ghi JSONL (unicode, timestamp, metadata) cho moi turn.
- **Emotion & Guardrails:** `EmotionAnalyzer` (Hugging Face) va `Guardrails` (PII + keyword) duoc chen vao pipeline LangGraph.
- **Barge-in/Reprompt:** CallHandler quan ly silence timeout, VAD tu Google STT, se bo sung logics double-talk trong giai doan toi.

## 3. Nen tang Docker & phan chia GPU
- **Service `app`:** `Dockerfile.app` (CUDA 12.2, Python 3.11), mount models GGUF tai `/models/nlp`. GPU 0-5, `llama-cpp-python` su dung `LLAMA_TENSOR_SPLIT`.
- **Service `tts`:** `Dockerfile.tts` (CUDA 12.1, Python 3.10), preload FastPitch/HiFiGAN vao image, GPU 6-7.
- **compose`:** `docker-compose.yml` chay host network, healthcheck bang script Python, opentelemetry-instrument lam entrypoint.
- **RAG stack (tuy chon):** `compose.rag.yaml` de bat Text Embeddings Inference + Qdrant, script `scripts/ingest.py` chunk/embedding.

## 4. Tien do module NLP (production-ready roadmap)

### Giai doan 0: On dinh nen tang (da hoan thanh)
- Da build duoc `docker compose build --no-cache app tts` va khoi dong 2 service khong loi.
- Hoan tat cau hinh dependency chinh (torch, unsloth, llama-cpp, nemo_toolkit, opentelemetry...).

### Giai doan 1: Loi doi thoai & tuong tac (dang trien khai)
- **Tien do (09/2025):**
  - Hop nhat `CallHandler` streaming moi (`app/audio/stream.py`) noi ARI -> STT -> NLU -> TTS, ghi log danh gia JSONL.
  - Xoa hoan toan kien truc cu trong `src/` va cap nhat script (`scripts/setup/run.sh`) de chi dung module `app.*`.
  - Bo sung RTP header chuan (sequence/timestamp/SSRC) khi phat TTS va giam do tre chunk streaming trong `tts_server/api.py`.
- **Hang muc uu tien tiep theo:**
  1. **Input processing:** Hoan thien schema intent/slot, chuan hoa tieng Viet, post-ASR punctuation.
  2. **Dialogue state:** Mo rong `DialogState` va chinh sach hoi lai/chuyen may, ho tro barge-in thuc su.
  3. **Function calling an toan:** Duy tri GBNF/JSON-guided decoding, tang cuong sanitize truoc khi goi tool CRM.

### Giai doan 2: Toi uu do tre & tri thuc
- Muc tieu: E2E <500ms, chay streaming pipeline on dinh.
- Cong viec: VAD/endpointing nang cao, toi uu KV-cache, them speculative decoding (neu phu hop), kich hoat RAG bang Qdrant + TEI, pipeline ingest du lieu.

### Giai doan 3: An toan, giam sat & hoan thien
- Guardrails nang cao (Presidio, filter prompt), metrics latency theo tung buoc, bo testcase tieng Viet thuc te.
- Quan ly cau hinh tap trung (vd Hydra/pydantic-settings) va secret (Vault/Doppler) cho cac key ARI, Google, CRM.

## 5. Cong viec da giai quyet gan day
- Loai bo thu muc `src/` ke thua va tat ca tham chieu cu.
- Cap nhat `CallHandler` de su dung `DialogState`, `evaluation_tracker` (model_dump) va TTS streaming theo RTP chuan.
- Dieu chinh `scripts/setup/run.sh` de su dung virtualenv `.venv` va chay `python3 -m app.main`.
- Giam do tre chunk streaming TTS bang `await asyncio.sleep(0)`.

## 6. Viec con mo
1. Tai model Llama 4 Scout dinh dang GGUF (Q4_K_M) vao `models/nlp/Q4_K_M/` va xac nhan bien moi truong `LLAMA_MODEL_PATH`.
2. Chuan hoa pipeline barge-in (stop TTS khi phat hien nguoi dung noi) va bo sung test gia lap RTP.
3. Cap nhat tai lieu kien truc HLD/Sequence diagram (vd trong README) sau khi smoke test thuc te.
4. Mo rong logging/tracing sang metric latency va error budget, ket noi voi observability stack hien co.


