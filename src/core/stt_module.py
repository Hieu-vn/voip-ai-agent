import logging
import asyncio
from src.core.stt_google_cloud_client import STTGoogleCloudClient

class STTModule:
    def __init__(self, language_code: str, sample_rate_hz: int):
        self.client = STTGoogleCloudClient(language_code, sample_rate_hz)

    def recognize_speech(self, audio_stream_fd=3) -> str:
        logging.info("STT: Bắt đầu nhận dạng giọng nói...")
        transcript = self.client.transcribe_single_utterance(audio_stream_fd)
        logging.info(f"STT: Kết quả: '{transcript}'")
        return transcript
