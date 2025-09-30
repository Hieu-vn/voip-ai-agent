"""
TTS Server for VoIP AI Agent, using NeMo and FastAPI.

This server provides a high-performance, low-latency Text-to-Speech (TTS) service.
It is designed to be run as a separate microservice, as per the V1 architecture.

Key Features:
- **FastAPI**: For high-performance asynchronous request handling.
- **NeMo Toolkit**: Uses pre-trained FastPitch and HiFiGAN models for synthesis.
- **Model Pre-loading**: Models are loaded on startup (and pre-downloaded in the
  Docker build) to minimize cold-start latency.
- **Low-latency Audio**: Synthesizes audio and resamples it to 8kHz (slin16)
  for telephony.
- **Streaming Response**: Streams audio chunks to the client, allowing for
  faster playback.
- **Observability**: Integrated with structlog for structured logging and
  OpenTelemetry for distributed tracing.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import structlog
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from pydantic import BaseModel, Field

# --- Configuration ---
# Configure logging to be structured and JSON-formatted
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
)
log = structlog.get_logger()

# --- Global State & Models ---
class AppState:
    """Container for global application state, including ML models."""
    spec_generator: FastPitchModel | None = None
    vocoder: HifiGanModel | None = None
    device: torch.device | None = None

app_state = AppState()

# --- Pydantic Models for API ---
class SynthesisRequest(BaseModel):
    """Request model for the /synthesize endpoint."""
    text: str = Field(
        ...,
        description="The text to be synthesized into speech.",
        min_length=1,
        max_length=1000,
    )

class HealthResponse(BaseModel):
    """Response model for the /healthz endpoint."""
    status: str = "ok"
    models_loaded: bool

# --- Model Loading ---
def load_models():
    """
    Loads NeMo TTS models from local paths or HuggingFace cache.
    This function is called once on application startup.
    """
    log.info("Loading NeMo TTS models...")
    app_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {app_state.device}")

    # Define model paths from environment variables
    spec_path = os.getenv("SPEC_MODEL_PATH")
    vocoder_path = os.getenv("VOCODER_MODEL_PATH")

    # Load Spectrogram Generator (FastPitch)
    try:
        if spec_path and os.path.exists(spec_path):
            log.info(f"Loading FastPitch from local file: {spec_path}")
            app_state.spec_generator = FastPitchModel.restore_from(spec_path, map_location=app_state.device)
        else:
            model_name = os.getenv("FASTPITCH_MODEL_NAME", "tts_en_fastpitch") # Fallback to default pretrained name
            log.info(f"Loading FastPitch from pretrained model: {model_name}")
            app_state.spec_generator = FastPitchModel.from_pretrained(model_name, map_location=app_state.device)
        app_state.spec_generator.eval()
        log.info("FastPitch model loaded successfully.")
    except Exception as e:
        log.error("Failed to load FastPitch model", exc_info=e)
        raise

    # Load Vocoder (HiFiGAN)
    try:
        if vocoder_path and os.path.exists(vocoder_path):
            log.info(f"Loading HiFiGAN from local file: {vocoder_path}")
            app_state.vocoder = HifiGanModel.restore_from(vocoder_path, map_location=app_state.device)
        else:
            model_name = os.getenv("VOCODER_MODEL_NAME", "tts_hifigan") # Fallback to default pretrained name
            log.info(f"Loading HiFiGAN from pretrained model: {model_name}")
            app_state.vocoder = HifiGanModel.from_pretrained(model_name, map_location=app_state.device)
        app_state.vocoder.eval()
        log.info("HiFiGAN model loaded successfully.")
    except Exception as e:
        log.error("Failed to load HiFiGAN model", exc_info=e)
        raise

# --- FastAPI Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles application startup and shutdown events.
    - On startup: Loads the ML models.
    - On shutdown: Cleans up resources.
    """
    log.info("TTS Server starting up...")
    load_models()
    log.info("Models loaded, server is ready.")
    yield
    log.info("TTS Server shutting down...")
    # Clean up resources if needed
    app_state.spec_generator = None
    app_state.vocoder = None
    torch.cuda.empty_cache()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="VoIP AI Agent - TTS Service (NeMo)",
    description="Provides low-latency text-to-speech synthesis.",
    version="1.0.0",
    lifespan=lifespan,
)

# Instrument FastAPI for OpenTelemetry
FastAPIInstrumentor.instrument_app(app)
tracer = trace.get_tracer(__name__)

# --- Audio Synthesis and Streaming Logic ---
TARGET_SAMPLE_RATE = 8000

async def audio_synthesis_generator(text: str, request_id: str):
    """
    A generator that synthesizes audio from text and yields it in chunks.
    This enables streaming responses for low-latency playback.
    """
    log.info("Starting audio synthesis", text=text, request_id=request_id)
    
    if not app_state.spec_generator or not app_state.vocoder:
        log.error("TTS models are not loaded.", request_id=request_id)
        # This part of the code should ideally not be reached due to the health check
        # and lifespan management, but it's a safeguard.
        # In a real scenario, you might want to stream a pre-recorded error message.
        return

    try:
        synthesis_start_time = time.perf_counter()

        # 1. Parse text into tokens
        with tracer.start_as_current_span("tts.parse_text") as span:
            parsed = app_state.spec_generator.parse(text)
            span.set_attribute("text.length", len(text))

        # 2. Generate spectrogram
        with tracer.start_as_current_span("tts.generate_spectrogram"):
            spectrogram = app_state.spec_generator.generate_spectrogram(tokens=parsed)

        # 3. Convert spectrogram to audio (vocoder)
        with tracer.start_as_current_span("tts.vocode"):
            # The vocoder outputs audio at its native sample rate (e.g., 22050 Hz)
            audio_native = app_state.vocoder.convert_spectrogram_to_audio(spec=spectrogram)
            native_sr = app_state.spec_generator.cfg.sample_rate

        # 4. Resample to target sample rate for telephony (8kHz)
        with tracer.start_as_current_span("tts.resample") as span:
            # Use librosa for resampling if available, otherwise a simpler method.
            # For this project, we assume a high-quality resampler is needed.
            # Note: This is a CPU-bound operation and can add latency.
            # For extreme low-latency, consider a GPU-accelerated resampler.
            try:
                import librosa
                audio_resampled = librosa.resample(
                    y=audio_native.cpu().numpy().squeeze(),
                    orig_sr=native_sr,
                    target_sr=TARGET_SAMPLE_RATE
                )
                span.set_attribute("resampler", "librosa")
            except ImportError:
                # Fallback if librosa is not installed (lower quality)
                from scipy.signal import resample
                num_samples = int(len(audio_native.squeeze()) * TARGET_SAMPLE_RATE / native_sr)
                audio_resampled = resample(audio_native.cpu().numpy().squeeze(), num_samples)
                span.set_attribute("resampler", "scipy")

        synthesis_duration = (time.perf_counter() - synthesis_start_time) * 1000
        log.info("Synthesis complete", duration_ms=synthesis_duration, request_id=request_id)

        # 5. Convert to 16-bit PCM (s16le) and stream
        with tracer.start_as_current_span("tts.stream_chunks"):
            # Convert float audio to 16-bit integer
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
            
            # Yield audio in chunks
            chunk_size_bytes = 2048  # Send 2KB chunks
            for i in range(0, len(audio_int16.tobytes()), chunk_size_bytes):
                yield audio_int16.tobytes()[i:i+chunk_size_bytes]
                await asyncio.sleep(0)  # Yield control without adding latency

    except Exception as e:
        log.error("Error during audio synthesis", exc_info=e, request_id=request_id)
        # In case of an error, we stop the generator. The client will receive an incomplete stream.
        # A more robust implementation could stream a pre-recorded error message.
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


# --- API Endpoints ---
@app.post("/synthesize", response_class=StreamingResponse)
async def synthesize_speech(request: SynthesisRequest):
    """
    Synthesizes speech from the provided text and streams it back as raw
    16-bit PCM audio at 8kHz sample rate (audio/l16;rate=8000).
    """
    request_id = os.urandom(4).hex()
    log.info("Received synthesis request", text=request.text, request_id=request_id)
    
    if not app_state.spec_generator or not app_state.vocoder:
        raise HTTPException(status_code=503, detail="TTS models are not ready.")

    return StreamingResponse(
        audio_synthesis_generator(request.text, request_id),
        media_type="audio/l16;rate=8000", # SLIN16 format
    )

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Provides a health check endpoint for orchestration tools (e.g., Docker HEALTHCHECK).
    Returns 'ok' if the models are loaded and the server is responsive.
    """
    models_loaded = app_state.spec_generator is not None and app_state.vocoder is not None
    if not models_loaded:
        return HealthResponse(status="models_not_loaded", models_loaded=False)
    return HealthResponse(status="ok", models_loaded=True)

# --- Main entry point for Uvicorn ---
if __name__ == "__main__":
    import uvicorn
    # Note: For production, run with `opentelemetry-instrument python -m uvicorn ...`
    # as recommended in the user's analysis.
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
