import logging
import asyncio
import aiohttp
import os
import tempfile
from typing import Optional

# --- TTS Module ---
# Module này hoạt động như một client cho NeMo TTS server.

logger = logging.getLogger(__name__)

# The TTS server URL is now expected to point to the NeMo server, e.g., http://localhost:8001
TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://localhost:8001")

async def tts_service_handler(text: str, language: str = "vi", **kwargs) -> Optional[str]:
    """
    Hàm xử lý TTS, gửi yêu cầu đến NeMo TTS server và trả về đường dẫn file âm thanh.
    Args:
        text (str): Văn bản cần chuyển thành giọng nói.
        language (str): Ngôn ngữ của văn bản ('vi' hoặc 'en').
    Returns:
        Optional[str]: Đường dẫn đến file âm thanh WAV đã được tạo ra, hoặc None nếu có lỗi.
    """
    # The new NeMo server doesn't need a speaker_wav_path
    payload = {
        "text": text,
        "language": language
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            # Using the /synthesize endpoint which returns a WAV file
            async with session.post(f"{TTS_SERVER_URL}/synthesize", json=payload, headers=headers) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)

                # Save the received audio to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                    audio_file_path = tmp_file.name

                logger.info(f"[TTS] Đã tạo giọng nói cho: '{text[:50]}...'. Lưu tại: {audio_file_path}")
                return audio_file_path

    except aiohttp.ClientError as e:
        logger.error(f"[TTS-Error] Lỗi kết nối đến TTS server {TTS_SERVER_URL}: {e}")
        return None
    except Exception as e:
        logger.error(f"[TTS-Error] Lỗi không xác định khi tạo giọng nói: {e}")
        return None

async def tts_stream_service_handler(pipe_path: str, text: str, language: str = "vi", **kwargs):
    """
    Hàm xử lý TTS streaming, gửi yêu cầu đến NeMo server và ghi trực tiếp vào pipe.
    Sử dụng endpoint /synthesize_pcm16 để nhận về audio PCM (s16le) đã được resample,
    phù hợp để phát trực tiếp qua Asterisk.
    Args:
        pipe_path (str): Đường dẫn đến named pipe để ghi dữ liệu audio.
        text (str): Văn bản cần chuyển thành giọng nói.
        language (str): Ngôn ngữ của văn bản ('vi' hoặc 'en').
    """
    payload = {
        "text": text,
        "language": language
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            # Use the more efficient streaming endpoint
            async with session.post(f"{TTS_SERVER_URL}/synthesize_pcm16", json=payload, headers=headers) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors

                # Write the received raw PCM audio directly to the named pipe
                with open(pipe_path, 'wb') as pipe_file:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        pipe_file.write(chunk)

                logger.info(f"[TTS-Stream] Đã gửi giọng nói PCM cho: '{text[:50]}...' đến pipe: {pipe_path}")

    except aiohttp.ClientError as e:
        logger.error(f"[TTS-Error] Lỗi kết nối đến TTS server {TTS_SERVER_URL} khi streaming: {e}")
    except Exception as e:
        logger.error(f"[TTS-Error] Lỗi không xác định khi streaming giọng nói: {e}")
    return