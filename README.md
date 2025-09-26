
# 📞 AI Agent for VoIP using Real-time Streaming

## 🌟 Project Introduction

This project provides a high-performance, low-latency AI agent capable of handling inbound phone calls on a VoIP platform. It uses a modern, real-time streaming architecture to deliver a natural and responsive conversational experience.

The system is built on **Asterisk 20** (within VitalPBX) on **Debian 12**, leveraging a powerful server with **8 NVIDIA V100 GPUs** to run state-of-the-art AI models for Vietnamese and English.

- **🤖 Core AI Technologies**: The agent uses Google's API for Speech-to-Text, a Llama 4-based model for natural language understanding, and an NVIDIA NeMo-based model for high-quality text-to-speech.
- **🎯 Key Goal**: Achieve an end-to-end latency of under 800ms through a fully streaming pipeline, from the moment the user speaks to the moment they hear the AI's response.
- **🔗 Integrations**: The system is designed with a modular architecture, allowing for flexible integration with CRM platforms like Zoho or Salesforce through function calling in the NLP model.

## 🏗️ System Architecture (Unified)

The project's architecture is designed for scalability and low latency, centered around the **Asterisk REST Interface (ARI)** for call control.

- **Hardware**: Server with 8x NVIDIA V100 GPUs (32GB VRAM).
- **VoIP**: VitalPBX / Asterisk 20.
- **Application Core**: An asynchronous Python application (`src/main.py`) that connects to Asterisk via ARI.

### Data Flow

The entire process is event-driven and built on streaming:

```
📞 Inbound Call (VitalPBX)
   ↓
🎙️ Asterisk sends 'StasisStart' event to the Python App via ARI (WebSocket)
   ↓
🐍 App creates a dedicated 'CallHandler' for the call (`src/core/call_handler.py`)
   1. Answers the call.
   2. Tells Asterisk to fork the incoming audio and stream it to a local UDP port.
   ↓
🎧 Real-time Audio Processing
   1. A UDP listener receives RTP packets and pushes the raw audio to the STT module (RTPAudioForwarder).
   2. **STT**: The audio is streamed to the **Google Cloud Speech-to-Text API** (STTModule).
   3. The transcribed text is received.
   ↓
🧠 AI Processing
   1. **NLP**: The text is sent to the locally-hosted **Llama 4 Scout** model for intent processing (NLPModule).
   2. The NLP model's response text is generated.
   ↓
🗣️ Speech Synthesis
   1. **TTS**: The response text is sent to a separate, dedicated **NVIDIA NeMo TTS Server** (`tts_server/server.py`) via a client (TTSModule).
   2. This server uses a two-step pipeline for high-quality audio:
      - **Step 1 (Spectrogram):** A **FastPitch** model converts text into a mel-spectrogram.
      - **Step 2 (Audio):** A **BigVGAN** model converts the spectrogram into an audible waveform.
   3. The generated audio is streamed back to the main application.
   ↓
🔊 Playback
   1. The `CallHandler` plays the received audio stream back to the caller via Asterisk.
   2. The system supports barge-in, allowing the user to interrupt the AI.
   ↓
👋 Hangup
```

### Core Components

- **🖥️ AI Backend**:
  - **STT**: **Google Cloud Speech-to-Text API**. Requires `GOOGLE_APPLICATION_CREDENTIALS` to be configured.
  - **NLP**: **Llama 4 Scout** model, loaded and managed by `src/core/nlp_module.py`.
  - **TTS**: A dedicated **FastAPI server** responsible for all Text-to-Speech processing.
    -   **Architecture**: Decouples heavy TTS processing from the core call-handling logic to ensure stability and performance.
    -   **Core Technology**: Uses a 2-step pipeline: **NeMo FastPitch** for spectrogram generation and **NVIDIA BigVGAN** as the vocoder for high-quality waveform synthesis.
    -   **Language Strategy**:
        -   **English**: Utilizes high-quality, pre-trained models from NVIDIA.
        -   **Vietnamese**: A custom **FastPitch** model fine-tuned on the `phoaudiobook` dataset. The fine-tuning process is managed by scripts in `scripts/training/`.
- **📡 VoIP Integration**:
  - Uses **ARI (Asterisk REST Interface)** for fine-grained, real-time call control. The `src/core/call_handler.py` contains all the logic for interacting with the Asterisk channel.

## 🚀 TTS Implementation Plan

This plan outlines the step-by-step process for deploying the complete bilingual TTS system.

### Stage 1: Foundation & Model Setup

1.  **Project Structure**: The codebase has been reorganized to support the new architecture. (Done)
2.  **Environment Setup**: All necessary libraries (`nemo_toolkit`, `fastapi`, etc.) are defined in `requirements.txt`. (Done)
3.  **Download Pre-trained Models**: Manually download the pre-trained models:
    *   **FastPitch (English):** `nvidia/tts_en_fastpitch` -> Placed in `models/tts/en/`
    *   **BigVGAN (Universal):** `nvidia/bigvgan_v2_22khz_80band_256x` -> Placed in `models/tts/vocoder/`
    *   **Status:** Done.
4.  **Docker Environment Configuration**: 
    *   Install Docker Engine and Docker Compose plugin (using `scripts/setup/install_docker.sh`).
    *   Create `.dockerignore` to optimize build context.
    *   Configure Docker to use `/data` partition for its data root.
    *   Optimize `Dockerfile` and `requirements.txt` for robust build (including system dependencies like `build-essential`, `sox`, `python3.11-dev`, and Python build-time dependencies like `numpy`, `typing_extensions`, `Cython`, `wheel`).
    *   **Status:** Done.
5.  **Update Environment Variables**: Update the `.env` file with the correct paths to the downloaded models. (Done)

### Stage 2: Server Launch (English TTS)

1.  **Configure Environment**: Update the `.env` file with the correct paths to the models downloaded in Stage 1. (Done)
2.  **Launch Server**: Start the FastAPI server in `tts_server/server.py`. At this point, the server will be fully capable of synthesizing high-quality English speech.

### Stage 3: Vietnamese Model Fine-tuning

1.  **Data Preparation**: Run the data preparation scripts (`prepare_audio_parquet.py`, `create_manifest.py`) to process the `phoaudiobook` dataset, creating train/validation/test manifests. (Done)
2.  **Run Fine-tuning**: Execute `run_finetune.py` with the `config_finetune_vi.yaml` config to create a custom Vietnamese FastPitch model. This will be saved in `models/tts/vi/`. (Configuration updated)

### Stage 4: Final Integration & Completion

1.  **Update Configuration**: Modify the `.env` file again to point to the newly trained Vietnamese model.
2.  **Restart Server**: Relaunch the TTS server.
3.  **Verification**: The system is now complete, supporting both English and Vietnamese TTS. Place calls to test both languages.
