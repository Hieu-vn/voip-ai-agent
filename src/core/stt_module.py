import logging
import asyncio
from collections import deque
from typing import Optional
from src.core.stt_google_cloud_client import STTGoogleCloudClient
from src.utils.metrics import STT_LATENCY_SECONDS, STT_ERRORS_TOTAL
from src.utils.tracing import tracer

logger = logging.getLogger(__name__)

class STTModule:
    def __init__(self, language_code: str):
        self.language_code = language_code
        self.sessions = {}

    async def start_session(self, call_id: str, sample_rate: int = 8000, adaptation_config: dict = None):
        """Khởi tạo một session STT mới cho một cuộc gọi."""
        if call_id in self.sessions:
            logger.warning(f"STT session for call_id {call_id} already exists.")
            return

        logger.info(f"Starting STT session for call_id: {call_id}")
        audio_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        session_state = {
            "audio_queue": audio_queue,
            "result_queue": result_queue,
            "has_partial": False,
            "task": None
        }
        self.sessions[call_id] = session_state

        stt_client = STTGoogleCloudClient(
            language_code=self.language_code,
            sample_rate_hz=sample_rate
        )

        session_state["task"] = asyncio.create_task(
            self._stt_service_handler(
                stt_client=stt_client,
                audio_queue=audio_queue,
                result_queue=result_queue,
                call_id=call_id,
                adaptation_config=adaptation_config or {}
            )
        )

    async def stop_session(self, call_id: str):
        """Dừng một session STT."""
        if call_id not in self.sessions:
            return
        
        logger.info(f"Stopping STT session for call_id: {call_id}")
        task = self.sessions[call_id]["task"]
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        del self.sessions[call_id]

    async def push_audio_chunk(self, call_id: str, chunk: bytes):
        """Đẩy một mẩu âm thanh vào session."""
        if call_id in self.sessions:
            await self.sessions[call_id]["audio_queue"].put(chunk)
        else:
            logger.warning(f"No active STT session for call_id {call_id} to push audio.")

    async def get_next_utterance(self, call_id: str) -> Optional[str]:
        """Lấy câu nói hoàn chỉnh tiếp theo từ kết quả STT."""
        if call_id not in self.sessions:
            logger.warning(f"No active STT session for call_id {call_id} to get utterance.")
            return None

        result_queue = self.sessions[call_id]["result_queue"]
        while True:
            try:
                result = await result_queue.get()
                if result.get("is_final") and result.get("transcript"): 
                    return result["transcript"]
                if result.get("type") == "stream_end":
                    return None
            except asyncio.CancelledError:
                return None

    async def has_any_partial(self, call_id: str) -> bool:
        """Kiểm tra xem đã có bất kỳ kết quả STT nào (kể cả tạm thời) hay chưa."""
        if call_id in self.sessions:
            return self.sessions[call_id]["has_partial"]
        return False

    @STT_LATENCY_SECONDS.time()
    async def _stt_service_handler(self, stt_client: STTGoogleCloudClient, audio_queue: asyncio.Queue, result_queue: asyncio.Queue, call_id: str, adaptation_config: dict):
        """
        Hàm xử lý nền cho một session STT.
        """
        with tracer.start_as_current_span("stt.service_handler") as span:
            span.set_attribute("call.id", call_id)
            
            try:
                async for result in stt_client.streaming_recognize_generator(
                    audio_queue=audio_queue,
                    call_id=call_id,
                    adaptation_config=adaptation_config
                ):
                    if call_id in self.sessions:
                        if result.get('transcript'):
                            self.sessions[call_id]["has_partial"] = True
                        await result_queue.put(result)
                    
                    if result.get('is_final') and result.get('error'):
                        STT_ERRORS_TOTAL.labels(type='api_error').inc()

            except Exception as e:
                STT_ERRORS_TOTAL.labels(type='unknown_error').inc()
                logger.error(f"STT Module [{call_id}]: Unknown error in stream: {e}", exc_info=True)
                span.record_exception(e)
                raise
            finally:
                if call_id in self.sessions:
                    await result_queue.put({"type": "stream_end"})