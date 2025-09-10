import logging
import asyncio
import os
import hashlib
import time
from src.core.tts_google_cloud_client import TTSGoogleCloudClient
from src.config import TTS_CONFIG

# --- Khởi tạo các tài nguyên ở cấp module ---

# Khởi tạo client một lần và tái sử dụng trong toàn bộ ứng dụng
tts_client = TTSGoogleCloudClient()

# Lấy thông tin thư mục cache từ config và đảm bảo nó tồn tại
CACHE_DIR = TTS_CONFIG.get('cache_dir', '/tmp/tts_cache/')
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
    logging.info(f"TTS cache directory is set to: {CACHE_DIR}")
except OSError as e:
    logging.error(f"Không thể tạo thư mục cache TTS tại '{CACHE_DIR}': {e}. Caching sẽ bị vô hiệu hóa.")
    CACHE_DIR = None


def get_cache_key(text: str, voice_id: str, sample_rate: int, speaking_rate: float) -> str:
    """Tạo một cache key (tên file) duy nhất từ các tham số tổng hợp."""
    payload = f"{text}:{voice_id}:{sample_rate}:{speaking_rate}"
    return hashlib.sha256(payload.encode()).hexdigest()


from src.utils.metrics import TTS_LATENCY_SECONDS, TTS_ERRORS_TOTAL

# ... (imports and other functions remain the same) ...

@TTS_LATENCY_SECONDS.time()
def synthesize_speech_sync(text: str, voice_key: str = None, speaking_rate: float = None, sample_rate: int = 16000) -> str:
    """
    Hàm ĐỒNG BỘ (blocking) để tổng hợp giọng nói, có hỗ trợ cache và metrics.
    """
    # --- Xác định các tham số cuối cùng ---
    final_voice_key = voice_key or TTS_CONFIG.get('default_voice', 'female_a')
    final_voice_id = TTS_CONFIG.get('voices', {}).get(final_voice_key, 'vi-VN-Wavenet-A')
    final_speaking_rate = speaking_rate or TTS_CONFIG.get('speaking_rate', 1.0)

    # --- Logic Caching ---
    if CACHE_DIR:
        cache_key = get_cache_key(text, final_voice_id, sample_rate, final_speaking_rate)
        cached_file_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")

        if os.path.exists(cached_file_path):
            logging.info(f"TTS Cache HIT for key {cache_key}. Path: {cached_file_path}")
            return cached_file_path
        
        logging.info(f"TTS Cache MISS for key {cache_key}. Synthesizing audio...")
        output_path = cached_file_path
    else:
        output_path = f"/tmp/{hashlib.sha256(os.urandom(16)).hexdigest()}.wav"

    # --- Gọi API nếu không có trong cache ---
    try:
        tts_client.synth_to_wav(
            text=text,
            out_wav_path=output_path,
            voice_name=final_voice_id,
            sample_rate_hz=sample_rate,
            speaking_rate=final_speaking_rate
        )
        logging.info(f"TTS synthesized successfully: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"TTS synthesis failed: {e}", exc_info=True)
        TTS_ERRORS_TOTAL.inc() # Tăng bộ đếm lỗi
        return None



from src.utils.tracing import tracer

# ... (imports and other code) ...

async def tts_service_handler(text: str, **kwargs) -> str:
    with tracer.start_as_current_span("tts.service_handler") as span:
        span.set_attribute("tts.text.length", len(text))
        # ... (rest of the function logic) ...
    """
    Service handler BẤT ĐỒNG BỘ, là entry point cho các module khác gọi vào.
    Nó sẽ gọi hàm synthesize đồng bộ (blocking) trong một executor.
    """
    loop = asyncio.get_running_loop()
    
    # Chạy hàm blocking (ghi file, gọi API) trong một thread riêng để không ảnh hưởng
    # đến vòng lặp sự kiện chính của ứng dụng.
    wav_path = await loop.run_in_executor(
        None,
        synthesize_speech_sync,
        text,
        # Truyền các tham số tùy chọn vào hàm sync
        kwargs.get('voice'), # 'female_a', 'male_b', etc.
        kwargs.get('rate'),  # 1.0, 1.1, etc.
        kwargs.get('sample_rate', 16000)
    )
    return wav_path

# --- Streaming Functions ---

def _tts_stream_to_pipe_sync(pipe_path: str, text: str, voice_key: str = None, speaking_rate: float = None, sample_rate: int = 16000):
    """
    Hàm ĐỒNG BỘ: Lấy audio từ Google, sau đó ghi từng chunk vào một named pipe.
    """
    logging.info(f"TTS Streamer: Bắt đầu tiến trình ghi vào pipe '{pipe_path}'")
    # --- Xác định các tham số cuối cùng ---
    final_voice_key = voice_key or TTS_CONFIG.get('default_voice', 'female_a')
    final_voice_id = TTS_CONFIG.get('voices', {}).get(final_voice_key, 'vi-VN-Wavenet-A')
    final_speaking_rate = speaking_rate or TTS_CONFIG.get('speaking_rate', 1.0)

    try:
        # 1. Lấy toàn bộ nội dung audio từ Google
        audio_content = tts_client.get_audio_content(
            text=text,
            voice_name=final_voice_id,
            sample_rate_hz=sample_rate,
            speaking_rate=final_speaking_rate
        )

        if not audio_content:
            logging.error("TTS Streamer: Không nhận được nội dung audio từ client.")
            return

        # 2. Mở pipe để ghi và bắt đầu stream
        logging.debug(f"TTS Streamer: Đang mở pipe '{pipe_path}' để ghi...")
        with open(pipe_path, "wb") as pipe_out:
            logging.info(f"TTS Streamer: Pipe đã được mở. Bắt đầu ghi {len(audio_content)} bytes.")
            
            # 3. Tự chia nhỏ và ghi vào pipe để mô phỏng streaming
            chunk_size = 3200  # 100ms of 16kHz, 16-bit audio
            for i in range(0, len(audio_content), chunk_size):
                chunk = audio_content[i:i + chunk_size]
                pipe_out.write(chunk)
                time.sleep(0.05) 
            logging.info(f"TTS Streamer: Đã ghi xong vào pipe.")

    except Exception as e:
        logging.error(f"TTS Streamer: Lỗi trong tiến trình ghi vào pipe: {e}", exc_info=True)
    finally:
        logging.debug(f"TTS Streamer: Đã đóng pipe '{pipe_path}'.")


async def tts_stream_service_handler(pipe_path: str, text: str, **kwargs):
    """
    Service handler BẤT ĐỒNG BỘ cho streaming.
    Nó gọi hàm sync để thực hiện công việc trong một thread riêng.
    """
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        _tts_stream_to_pipe_sync,
        pipe_path,
        text,
        kwargs.get('voice'),
        kwargs.get('rate')
    )

