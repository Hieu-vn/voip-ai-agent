import asyncio
import logging
from typing import AsyncIterator, Dict, Optional

from google.cloud import speech_v1p1beta1 as speech

logger = logging.getLogger(__name__)

class STTGoogleCloudClient:
    def __init__(self, language_code: str, sample_rate_hz: int):
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.client = speech.SpeechClient()

    async def streaming_recognize_generator(
        self,
        audio_queue: asyncio.Queue,
        call_id: str,
        adaptation_config: Optional[Dict],
        timeout: int = 120,
    ) -> AsyncIterator[Dict]:
        """
        Consume PCM audio chunks from an asyncio queue and stream them to Google STT.
        Yields dictionaries containing transcripts and flags from the API responses.
        """
        logger.debug("STT Client [%s]: sample rate %sHz", call_id, self.sample_rate_hz)

        streaming_config = self._build_streaming_config(adaptation_config or {})
        result_queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def request_iterator():
            while True:
                future = asyncio.run_coroutine_threadsafe(audio_queue.get(), loop)
                chunk = future.result()
                if chunk is None:
                    logger.debug("STT Client [%s]: received audio stream sentinel", call_id)
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        def run_recognition():
            try:
                responses = self.client.streaming_recognize(
                    streaming_config,
                    request_iterator(),
                    timeout=timeout,
                )
                for response in responses:
                    if not response.results:
                        continue
                    result = response.results[0]
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript
                    payload = {
                        "transcript": transcript,
                        "is_final": result.is_final,
                    }
                    asyncio.run_coroutine_threadsafe(result_queue.put(payload), loop)
            except Exception as exc:
                logger.error(
                    "STT Client [%s]: error during streaming recognition: %s",
                    call_id,
                    exc,
                    exc_info=True,
                )
                error_payload = {"transcript": "", "is_final": True, "error": str(exc)}
                asyncio.run_coroutine_threadsafe(result_queue.put(error_payload), loop)
            finally:
                asyncio.run_coroutine_threadsafe(result_queue.put(None), loop)

        recognition_future = loop.run_in_executor(None, run_recognition)

        try:
            while True:
                item = await result_queue.get()
                if item is None:
                    break
                yield item
        finally:
            await asyncio.wrap_future(recognition_future)

    def _build_streaming_config(self, adaptation_config: Dict) -> speech.StreamingRecognitionConfig:
        phrase_hints = adaptation_config.get("phrase_hints") or []
        boost = adaptation_config.get("boost", 1.0)
        enable_punctuation = adaptation_config.get("enable_automatic_punctuation", True)

        speech_adaptation = None
        if phrase_hints:
            phrase_set = speech.PhraseSet(
                phrases=[speech.PhraseSet.Phrase(value=p, boost=boost) for p in phrase_hints]
            )
            speech_adaptation = speech.SpeechAdaptation(phrase_sets=[phrase_set])
            logger.info(
                "STT Client: applying speech adaptation with %s hints", len(phrase_hints)
            )

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            enable_automatic_punctuation=enable_punctuation,
            adaptation=speech_adaptation,
        )
        return speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
        )
