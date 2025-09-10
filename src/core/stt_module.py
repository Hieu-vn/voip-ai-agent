import logging
import asyncio
import time
from src.core.stt_google_cloud_client import STTGoogleCloudClient
from src.utils.metrics import STT_LATENCY_SECONDS, STT_ERRORS_TOTAL

class STTModule:
    def __init__(self, language_code: str):
        self.language_code = language_code

    @STT_LATENCY_SECONDS.time()
    from src.utils.tracing import tracer

# ... (other imports) ...

class STTModule:
    # ... (__init__ remains the same) ...
    @STT_LATENCY_SECONDS.time()
    async def stt_service_handler(self, audio_fd: int, sample_rate: int, call_id: str, adaptation_config: dict):
        with tracer.start_as_current_span("stt.service_handler") as span:
            span.set_attribute("call.id", call_id)
            span.set_attribute("stt.sample_rate", sample_rate)
            # ... (rest of the function logic) ...
        """
        Đây là một async generator, nó sẽ yield các kết quả STT.
        Đã được instrument với Prometheus metrics.
        """
        logging.info(f"STT Module [{call_id}]: Bắt đầu stream STT với sample rate {sample_rate}Hz.")
        
        stt_client = STTGoogleCloudClient(
            language_code=self.language_code,
            sample_rate_hz=sample_rate
        )
        
        blocking_generator = stt_client.streaming_recognize_generator(
            fd_audio=audio_fd,
            call_id=call_id,
            adaptation_config=adaptation_config
        )
        
        loop = asyncio.get_running_loop()

        try:
            while True:
                try:
                    result = await loop.run_in_executor(None, next, blocking_generator)
                    yield result
                    
                    if result.get('is_final'):
                        if result.get('error'): # Nếu client trả về lỗi
                            STT_ERRORS_TOTAL.labels(type='api_error').inc()
                        break
                except StopIteration:
                    logging.debug(f"STT Module [{call_id}]: Stream từ generator đã kết thúc.")
                    break
        except Exception as e:
            STT_ERRORS_TOTAL.labels(type='unknown_error').inc()
            logging.error(f"STT Module [{call_id}]: Lỗi không xác định trong stream: {e}", exc_info=True)
            raise
