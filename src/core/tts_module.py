import logging
import asyncio
import aiohttp
import os
import tempfile

# --- TTS Module ---
# Module này tích hợp với Coqui XTTS-v2 TTS server.

logger = logging.getLogger(__name__)

TTS_SERVER_URL = os.getenv("TTS_SERVER_URL", "http://localhost:5002")

async def tts_service_handler(text: str, speaker_wav_path: str, language: str = "vi", **kwargs) -> str:
    """
    Hàm xử lý TTS, gửi yêu cầu đến Coqui XTTS-v2 server và trả về đường dẫn file âm thanh.
    Args:
        text (str): Văn bản cần chuyển thành giọng nói.
        speaker_wav_path (str): Đường dẫn đến file audio mẫu giọng nói (cho voice cloning).
        language (str): Ngôn ngữ của văn bản (mặc định là 'vi' cho tiếng Việt).
    Returns:
        str: Đường dẫn đến file âm thanh đã được tạo ra, hoặc None nếu có lỗi.
    """
    if not speaker_wav_path or not os.path.exists(speaker_wav_path):
        logger.error(f"[TTS-Error] Missing or invalid speaker_wav_path: {speaker_wav_path}")
        return None

    payload = {
        "text": text,
        "speaker_wav": speaker_wav_path,
        "language": language
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{TTS_SERVER_URL}/synthesize", json=payload, headers=headers) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Save the received audio to a temporary file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        tmp_file.write(chunk)
                    audio_file_path = tmp_file.name
                
                logger.info(f"[TTS] Đã tạo giọng nói cho: '{text}'. Lưu tại: {audio_file_path}")
                return audio_file_path

    except aiohttp.ClientError as e:
        logger.error(f"[TTS-Error] Lỗi kết nối đến TTS server {TTS_SERVER_URL}: {e}")
    except Exception as e:
        logger.error(f"[TTS-Error] Lỗi khi tạo giọng nói: {e}")
    return None

async def tts_stream_service_handler(pipe_path: str, text: str, speaker_wav_path: str, language: str = "vi", **kwargs):
    """
    Hàm xử lý TTS streaming, gửi yêu cầu đến Coqui XTTS-v2 server và ghi trực tiếp vào pipe.
    Args:
        pipe_path (str): Đường dẫn đến named pipe để ghi dữ liệu audio.
        text (str): Văn bản cần chuyển thành giọng nói.
        speaker_wav_path (str): Đường dẫn đến file audio mẫu giọng nói (cho voice cloning).
        language (str): Ngôn ngữ của văn bản (mặc định là 'vi' cho tiếng Việt).
    """
    if not speaker_wav_path or not os.path.exists(speaker_wav_path):
        logger.error(f"[TTS-Error] Missing or invalid speaker_wav_path for streaming: {speaker_wav_path}")
        return

    payload = {
        "text": text,
        "speaker_wav": speaker_wav_path,
        "language": language
    }
    headers = {"Content-Type": "application/json"}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{TTS_SERVER_URL}/synthesize", json=payload, headers=headers) as response:
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Write the received audio directly to the named pipe
                with open(pipe_path, 'wb') as pipe_file:
                    while True:
                        chunk = await response.content.read(8192)
                        if not chunk:
                            break
                        pipe_file.write(chunk)
                
                logger.info(f"[TTS-Stream] Đã gửi giọng nói cho: '{text}' đến pipe: {pipe_path}")

    except aiohttp.ClientError as e:
        logger.error(f"[TTS-Error] Lỗi kết nối đến TTS server {TTS_SERVER_URL} khi streaming: {e}")
    except Exception as e:
        logger.error(f"[TTS-Error] Lỗi khi streaming giọng nói: {e}")
    return
