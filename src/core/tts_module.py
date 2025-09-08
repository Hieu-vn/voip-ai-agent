import logging
import asyncio
from src.core.tts_google_cloud_client import TTSGoogleCloudClient

class TTSModule:
    def __init__(self, sample_rate_hz: int):
        self.client = TTSGoogleCloudClient(sample_rate_hz)

    def synthesize_speech(self, text: str, out_wav_path: str) -> str:
        logging.info(f"TTS: Đang tổng hợp: '{text}'")
        # In a real scenario, you might add logic here for caching, 
        # choosing different TTS models, or handling errors.
        generated_wav_path = self.client.synth_vi_wav(text, out_wav_path)
        logging.info(f"TTS: Đã tổng hợp thành công vào: '{generated_wav_path}'")
        return generated_wav_path
