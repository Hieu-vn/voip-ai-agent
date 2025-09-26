import asyncio
import logging
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

# Assuming these are the correct locations from your setup
from nemo.collections.tts.models import FastPitchModel, HifiGanModel

# --- Globals ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_LANGUAGES = {"en", "vi"}
# Placeholder maps - these would be populated at startup
fastpitch_models: Dict[str, FastPitchModel] = {}
vocoder_models: Dict[str, HifiGanModel] = {}
sample_rate_map: Dict[str, int] = {}
mel_cfg_map: Dict[str, Tuple[int, int]] = {}

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Pydantic Models ---
class SynthesisRequest(BaseModel):
    text: str
    language: str

# --- FastAPI App ---
app = FastAPI(
    title="NeMo TTS Server",
    description="High-quality TTS synthesis using NVIDIA NeMo (FastPitch + BigVGAN).",
    version="2.0.0",
)

# --- Model Loading ---
async def _load_model(lang: str, fp_path: str, vc_path: str):
    """Loads a language-specific model pair (FastPitch, Vocoder)."""
    global fastpitch_models, vocoder_models, sample_rate_map, mel_cfg_map
    logger.info(f"[{lang}] Loading FastPitch from: {fp_path}")
    fp_model = FastPitchModel.restore_from(fp_path, map_location=DEVICE)
    fp_model.eval()
    fastpitch_models[lang] = fp_model

    logger.info(f"[{lang}] Loading Vocoder from: {vc_path}")
    vc_model = HifiGanModel.restore_from(vc_path, map_location=DEVICE)
    vc_model.eval()
    vocoder_models[lang] = vc_model
    
    # Store metadata
    sample_rate_map[lang] = vc_model.cfg.sample_rate
    mel_cfg_map[lang] = (vc_model.cfg.n_mels, vc_model.cfg.fmax)
    logger.info(f"[{lang}] Models loaded. Sample Rate: {sample_rate_map[lang]}, Mel Bins: {mel_cfg_map[lang][0]}")

@app.on_event("startup")
async def startup_event():
    """Loads all models defined in environment variables."""
    async def _load_all() -> List[str]:
        errs = []
        # English
        fp_en = os.getenv("FASTPITCH_MODEL_EN_PATH")
        vc_en = os.getenv("VOCODER_MODEL_EN_PATH")
        if fp_en and vc_en:
            try:
                await _load_model("en", fp_en, vc_en)
            except Exception as e:
                errs.append(f"[en] {e}")
        # Vietnamese
        fp_vi = os.getenv("FASTPITCH_MODEL_VI_PATH")
        vc_vi = os.getenv("VOCODER_MODEL_VI_PATH")
        if fp_vi and vc_vi:
            try:
                await _load_model("vi", fp_vi, vc_vi)
            except Exception as e:
                errs.append(f"[vi] {e}")
        return errs

    errs = await asyncio.to_thread(_load_all)
    if errs:
        logger.warning("Some languages failed: " + "; ".join(errs))
    if not fastpitch_models or not vocoder_models:
        logger.error("No models loaded. Server will run but synthesis endpoints will return 503.")

# --- Endpoints ---
@app.get("/health")
async def health():
    states = {l: (l in fastpitch_models and l in vocoder_models) for l in SUPPORTED_LANGUAGES}
    return {
        "status": "ok",
        "device": DEVICE,
        "languages": states,
        "sample_rates": {l: sample_rate_map.get(l) for l in SUPPORTED_LANGUAGES},
        "mel_cfg": {l: {"n_mels": mel_cfg_map.get(l, (None, None))[0], "fmax": mel_cfg_map.get(l, (None, None))[1]} for l in SUPPORTED_LANGUAGES},
    }

def synth_wave(lang: str, text: str) -> Optional[torch.Tensor]:
    fp = fastpitch_models.get(lang)
    vc = vocoder_models.get(lang)
    if fp is None or vc is None:
        return None
    with torch.inference_mode():
        tokens = fp.parse(text)
        use_amp = DEVICE == "cuda"
        # FastPitch spec gen
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16 if use_amp else torch.float32):
            spec = fp.generate_spectrogram(tokens=tokens)
        # BigVGAN inference
        audio = vc.convert_spectrogram_to_audio(spec=spec)  # (B, T) float32 [-1,1]
        audio = audio.cpu()
    return audio

def save_wav_int16(path: Path, audio: torch.Tensor, sr: int):
    pcm16 = (audio.squeeze().numpy() * 32767.0).astype("int16")
    torchaudio.save(str(path), torch.from_numpy(pcm16).unsqueeze(0), sample_rate=sr,
                    format="wav", bits_per_sample=16)  # type: ignore[arg-type]

@app.post("/synthesize")
async def synthesize(req: SynthesisRequest, background_tasks: BackgroundTasks):
    """Trả file WAV 16-bit ở sample rate gốc của BigVGAN."""
    if not (fastpitch_models and vocoder_models):
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    lang = req.language.strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language '{lang}' not supported.")
    audio = await asyncio.to_thread(synth_wave, lang, req.text)
    if audio is None:
        raise HTTPException(status_code=503, detail=f"Models for '{lang}' not available.")

    sr = sample_rate_map.get(lang, 22050)
    out_path = Path(f"/tmp/tts_{os.urandom(6).hex()}.wav")

    await asyncio.to_thread(save_wav_int16, out_path, audio, sr)

    def _cleanup(p: Path):
        try:
            p.unlink(missing_ok=True)
        except Exception as e:
            logger.warning(f"Cleanup failed for {p}: {e}")

    background_tasks.add_task(_cleanup, out_path)
    return FileResponse(str(out_path), media_type="audio/wav", filename=out_path.name)

@app.post("/synthesize_pcm16")
async def synthesize_pcm16(req: SynthesisRequest):
    """Stream PCM s16le mono 16 kHz (không header) để phát trực tiếp qua ARI/RTP.
    Set PCM_TARGET_SR nếu muốn tần số khác (mặc định 16000).
    """
    if not (fastpitch_models and vocoder_models):
        raise HTTPException(status_code=503, detail="Models are not loaded.")
    lang = req.language.strip()
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Language '{lang}' not supported.")

    audio = await asyncio.to_thread(synth_wave, lang, req.text)
    if audio is None:
        raise HTTPException(status_code=503, detail=f"Models for '{lang}' not available.")

    src_sr = sample_rate_map.get(lang, 22050)
    tgt_sr = int(os.getenv("PCM_TARGET_SR", "16000"))
    if src_sr != tgt_sr:
        audio = torchaudio.functional.resample(audio, src_sr, tgt_sr)

    pcm16 = (audio.squeeze().numpy() * 32767.0).astype("int16").tobytes()
    return StreamingResponse(
        iter([pcm16]),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(tgt_sr), "X-Sample-Format": "s16le", "X-Channels": "1"},
    )

if __name__ == "__main__":
    # Gợi ý pin GPU: CUDA_VISIBLE_DEVICES=0 python tts_server_bigvgan.py
    uvicorn.run(app, host="0.0.0.0", port=8001)