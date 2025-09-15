import logging
import asyncio
import time
from src.core.stt_google_cloud_client import STTGoogleCloudClient
from src.utils.metrics import STT_LATENCY_SECONDS, STT_ERRORS_TOTAL
from src.utils.tracing import tracer

class STTModule:
    def __init__(self, language_code: str):
        self.language_code = language_code

    @STT_LATENCY_SECONDS.time()
    async def stt_service_handler(self, audio_queue: asyncio.Queue, sample_rate: int, call_id: str, adaptation_config: dict, result_queue: asyncio.Queue):
        """
        Recognizes speech from an asyncio.Queue and puts results into another asyncio.Queue.
        Instrumented with Prometheus metrics.
        """
        with tracer.start_as_current_span("stt.service_handler") as span:
            span.set_attribute("call.id", call_id)
            span.set_attribute("stt.sample_rate", sample_rate)
            
            logging.info(f"STT Module [{call_id}]: Starting STT stream with sample rate {sample_rate}Hz.")
            
            stt_client = STTGoogleCloudClient(
                language_code=self.language_code,
                sample_rate_hz=sample_rate
            )
            
            # Use async for to iterate over the async generator
            try:
                async for result in stt_client.streaming_recognize_generator(
                    audio_queue=audio_queue,
                    call_id=call_id,
                    adaptation_config=adaptation_config
                ):
                    await result_queue.put(result) # Put result into the queue
                    
                    if result.get('is_final'):
                        if result.get('error'): # If client returns error
                            STT_ERRORS_TOTAL.labels(type='api_error').inc()
                        # No break here, continue listening for more utterances
            except Exception as e:
                STT_ERRORS_TOTAL.labels(type='unknown_error').inc()
                logging.error(f"STT Module [{call_id}]: Unknown error in stream: {e}", exc_info=True)
                span.record_exception(e)
                # Raise the exception to be caught by the task runner
                raise
            finally:
                await result_queue.put({"type": "stream_end"}) # Signal end of stream
