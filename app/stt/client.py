"""
Asynchronous client for Google Cloud Speech-to-Text v2 API.

This module provides a streaming client for real-time speech recognition.
It's designed to be used within the CallHandler to process audio from the
VoIP stream.

Key Features:
- Implements the bi-directional streaming RPC for Google STT v2.
- Uses the `google-cloud-speech` library with async support.
- Configured for telephony use cases (8kHz, Vietnamese).
- Uses a callback mechanism to return transcription results (interim and final).
"""
import asyncio
import os
from typing import Awaitable, Callable, Optional

import structlog
from google.api_core import exceptions as google_exceptions
from google.cloud import speech_v2

log = structlog.get_logger()

# Type hint for the callback function
SttCallback = Callable[[str, bool], Awaitable[None]]

class SttClient:
    """A streaming client for the Google Cloud STT v2 API."""

    def __init__(
        self,
        on_result_callback: SttCallback,
        on_voice_event_callback: Optional[Callable[[str], Awaitable[None]]] = None,
    ):
        """
        Initializes the STT client.

        Args:
            on_result_callback: An async function to call with transcription results.
                                It receives two arguments: the transcript (str) and
                                a boolean indicating if the result is final (bool).
            on_voice_event_callback: Optional async callback fired when voice
                                     activity events (e.g., barge-in) are received.
        """
        self.client = speech_v2.SpeechAsyncClient()
        self.on_result = on_result_callback
        self.on_voice_event = on_voice_event_callback
        self.audio_queue = asyncio.Queue()
        self.is_active = False
        self._streaming_task: Optional[asyncio.Task] = None

        # --- STT Configuration ---
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")
        self.recognizer_id = os.getenv("GOOGLE_RECOGNIZER_ID", "_Default-Recognizer")
        self.language_code = "vi-VN"
        self.model = "telephony"

        self.recognition_config = speech_v2.RecognitionConfig(
            auto_decoding_config={},
            model=self.model,
            language_codes=[self.language_code],
            features=speech_v2.RecognitionFeatures(
                enable_automatic_punctuation=True,
            ),
        )

        self.streaming_config = speech_v2.StreamingRecognitionConfig(
            config=self.recognition_config,
            streaming_features=speech_v2.StreamingRecognitionFeatures(
                interim_results=True,
                enable_voice_activity_events=True, # Useful for barge-in
            ),
        )

    def set_result_callback(self, callback: SttCallback) -> None:
        self.on_result = callback

    def set_voice_event_callback(
        self, callback: Optional[Callable[[str], Awaitable[None]]]
    ) -> None:
        self.on_voice_event = callback

    async def _request_generator(self):
        """Generator that yields audio chunks from the queue to the API."""
        # The first request must contain the configuration
        yield speech_v2.StreamingRecognizeRequest(
            recognizer=self.recognizer_name,
            streaming_config=self.streaming_config,
        )

        # Subsequent requests contain the audio data
        while self.is_active:
            try:
                chunk = await self.audio_queue.get()
                if chunk is None: # Sentinel value to stop the stream
                    break
                yield speech_v2.StreamingRecognizeRequest(audio=chunk)
            except asyncio.CancelledError:
                break

    async def _process_responses(self, stream):
        """Processes the response stream from the STT API."""
        try:
            async for response in stream:
                if not self.is_active:
                    break

                if (
                    self.on_voice_event
                    and response.speech_event_type
                    != speech_v2.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED
                ):
                    event_name = speech_v2.StreamingRecognizeResponse.SpeechEventType(
                        response.speech_event_type
                    ).name
                    try:
                        await self.on_voice_event(event_name)
                    except Exception as callback_error:  # pragma: no cover - defensive
                        log.warning(
                            "STT voice event callback failed",
                            exc_info=callback_error,
                            event_name=event_name,
                        )
                
                # Process transcription results
                for result in response.results:
                    if not result.alternatives:
                        continue
                    transcript = result.alternatives[0].transcript
                    if transcript:
                        await self.on_result(transcript, result.is_final)

        except (google_exceptions.GoogleAPICallError, asyncio.CancelledError) as e:
            log.error("STT stream error", exc_info=e)
        finally:
            log.info("STT response processing finished.")

    async def start(self):
        """Starts the STT streaming process."""
        if self.is_active:
            log.warning("STT client is already active.")
            return

        log.info("Starting STT client...")
        self.is_active = True
        self.recognizer_name = f"projects/{self.project_id}/locations/global/recognizers/{self.recognizer_id}"

        # Create the bi-directional stream
        stream = await self.client.streaming_recognize(
            requests=self._request_generator()
        )

        # Start the task to process responses from the stream
        self._streaming_task = asyncio.create_task(self._process_responses(stream))
        log.info("STT client started and listening.")

    async def stop(self):
        """Stops the STT streaming process gracefully."""
        if not self.is_active:
            return
        
        log.info("Stopping STT client...")
        self.is_active = False
        
        # Send a sentinel value to unblock the request generator
        await self.audio_queue.put(None)

        if self._streaming_task and not self._streaming_task.done():
            self._streaming_task.cancel()
            try:
                await self._streaming_task
            except asyncio.CancelledError:
                pass # Expected cancellation
        
        self._streaming_task = None
        log.info("STT client stopped.")

    def write(self, audio_chunk: bytes):
        """Adds an audio chunk to the processing queue."""
        if self.is_active:
            try:
                self.audio_queue.put_nowait(audio_chunk)
            except asyncio.QueueFull:
                log.warning("STT audio queue is full, dropping packet.")
