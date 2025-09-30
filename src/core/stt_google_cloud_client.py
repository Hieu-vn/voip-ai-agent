import os
import logging
import asyncio # Import asyncio for Queue
from google.cloud import speech_v1p1beta1 as speech

class STTGoogleCloudClient:
    def __init__(self, language_code: str, sample_rate_hz: int):
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.client = speech.SpeechClient()

    async def streaming_recognize_generator(self, audio_queue: asyncio.Queue, call_id: str, adaptation_config: dict, timeout: int = 120):
        """
        Recognizes speech from an asyncio.Queue and yields results.
        
        :param audio_queue: An asyncio.Queue containing audio chunks (bytes).
        :param call_id: ID of the call for logging.
        :param adaptation_config: Dictionary containing speech adaptation configuration.
        :param timeout: Maximum timeout for the API call.
        :yield: A dictionary containing transcript and is_final flag.
        """
        chunk_interval_ms = 100
        # chunk_size = int(self.sample_rate_hz * 2 * (chunk_interval_ms / 1000)) # Not needed if reading from queue
        logging.debug(f"STT Client [{call_id}]: Sample rate {self.sample_rate_hz}Hz.")

        # --- Step 1: Build Speech Adaptation object from config ---
        speech_adaptation = None
        if adaptation_config and adaptation_config.get('phrase_hints'):
            phrase_set = speech.PhraseSet(
                phrases=[
                    speech.PhraseSet.Phrase(value=p, boost=adaptation_config.get('boost', 1.0))
                    for p in adaptation_config['phrase_hints']
                ]
            )
            speech_adaptation = speech.SpeechAdaptation(phrase_sets=[phrase_set])
            logging.info(f"STT Client [{call_id}]: Applying Speech Adaptation with {len(adaptation_config['phrase_hints'])} hints.")

        # --- Step 2: Build final Recognition Config ---
        enable_punctuation = adaptation_config.get('enable_automatic_punctuation', True)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            enable_automatic_punctuation=enable_punctuation,
            adaptation=speech_adaptation  # Attach adaptation to config
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        async def requests_gen():
            # This generator should yield audio content from the queue.
            while True:
                chunk = await audio_queue.get()
                if chunk is None: # Signal for end of stream
                    logging.debug(f"STT Client [{call_id}]: Received end of audio stream signal.")
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            logging.debug(f"STT Client [{call_id}]: Finished yielding audio stream.")

        responses = self.client.streaming_recognize(config=streaming_config, requests=requests_gen(), timeout=timeout)
        
        # --- Step 3: Process and yield results ---
        try:
            logging.info(f"STT Client [{call_id}]: Starting to receive stream results from Google API.")
            async for response in responses: # Use async for to iterate over async generator
                if not response.results or not response.results[0].alternatives: continue
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                logging.debug(f"STT Client [{call_id}]: Transcript (final={result.is_final}): '{transcript}'")
                yield { "transcript": transcript, "is_final": result.is_final }
                if result.is_final and streaming_config.single_utterance: break
        except Exception as e:
            logging.error(f"STT Client [{call_id}]: Lỗi khi xử lý stream từ Google API: {e}", exc_info=True)
            yield { "transcript": "", "is_final": True, "error": str(e) }