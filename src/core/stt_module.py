import asyncio
import logging
from typing import Optional, Callable, Awaitable

from src.core.stt_google_cloud_client import STTGoogleCloudClient
from src.utils.metrics import STT_LATENCY_SECONDS, STT_ERRORS_TOTAL
from src.utils.tracing import tracer

logger = logging.getLogger(__name__)

class STTModule:
    def __init__(self, language_code: str):
        self.language_code = language_code
        self.sessions = {}

    async def start_session(self, call_id: str, sample_rate: int = 8000, adaptation_config: dict = None):
        """Create a new STT session for the given call."""
        if call_id in self.sessions:
            logger.warning("STT session for call_id %s already exists.", call_id)
            return

        logger.info("Starting STT session for call_id: %s", call_id)
        audio_queue = asyncio.Queue()
        result_queue = asyncio.Queue()
        session_state = {
            "audio_queue": audio_queue,
            "result_queue": result_queue,
            "has_partial": False,
            "task": None,
            "partial_callback": None,
        }
        self.sessions[call_id] = session_state

        stt_client = STTGoogleCloudClient(
            language_code=self.language_code,
            sample_rate_hz=sample_rate,
        )

        session_state["task"] = asyncio.create_task(
            self._stt_service_handler(
                stt_client=stt_client,
                audio_queue=audio_queue,
                result_queue=result_queue,
                call_id=call_id,
                adaptation_config=adaptation_config or {},
            )
        )

    async def stop_session(self, call_id: str):
        """Stop the STT session and release resources."""
        if call_id not in self.sessions:
            return

        logger.info("Stopping STT session for call_id: %s", call_id)
        task = self.sessions[call_id]["task"]
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        del self.sessions[call_id]

    def register_partial_callback(
        self,
        call_id: str,
        callback: Callable[[str], Awaitable[None]],
    ) -> None:
        """Register a coroutine callback that fires when partial transcripts appear."""
        if call_id not in self.sessions:
            logger.warning("Cannot register partial callback; session %s not found.", call_id)
            return
        self.sessions[call_id]["partial_callback"] = callback

    async def push_audio_chunk(self, call_id: str, chunk: Optional[bytes]):
        """Feed a raw audio chunk into the session. None signals the end of stream."""
        if call_id in self.sessions:
            await self.sessions[call_id]["audio_queue"].put(chunk)
        else:
            logger.warning("No active STT session for call_id %s to push audio.", call_id)

    async def get_next_utterance(self, call_id: str) -> Optional[str]:
        """Return the next final transcript or None when the stream ends."""
        if call_id not in self.sessions:
            logger.warning("No active STT session for call_id %s to get utterance.", call_id)
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
        """Check whether any partial transcript has been observed."""
        if call_id in self.sessions:
            return self.sessions[call_id]["has_partial"]
        return False

    @STT_LATENCY_SECONDS.time()
    async def _stt_service_handler(
        self,
        stt_client: STTGoogleCloudClient,
        audio_queue: asyncio.Queue,
        result_queue: asyncio.Queue,
        call_id: str,
        adaptation_config: dict,
    ):
        """Background worker that streams audio to Google STT and buffers results."""
        with tracer.start_as_current_span("stt.service_handler") as span:
            span.set_attribute("call.id", call_id)

            try:
                async for result in stt_client.streaming_recognize_generator(
                    audio_queue=audio_queue,
                    call_id=call_id,
                    adaptation_config=adaptation_config,
                ):
                    if call_id in self.sessions:
                        if result.get("transcript"):
                            self.sessions[call_id]["has_partial"] = True
                        await result_queue.put(result)

                        if not result.get("is_final") and result.get("transcript"):
                            callback = self.sessions[call_id].get("partial_callback")
                            if callback:
                                asyncio.create_task(self._invoke_partial_callback(callback, result["transcript"]))

                        if result.get("is_final") and result.get("error"):
                            STT_ERRORS_TOTAL.labels(type="api_error").inc()

            except Exception as exc:
                STT_ERRORS_TOTAL.labels(type="unknown_error").inc()
                logger.error(
                    "STT Module [%s]: unknown error in stream: %s", call_id, exc, exc_info=True
                )
                span.record_exception(exc)
                raise
            finally:
                if call_id in self.sessions:
                    await result_queue.put({"type": "stream_end"})

    async def _invoke_partial_callback(
        self,
        callback: Callable[[str], Awaitable[None]],
        transcript: str,
    ) -> None:
        try:
            await callback(transcript)
        except Exception:
            logger.exception("Partial callback raised an exception.")

