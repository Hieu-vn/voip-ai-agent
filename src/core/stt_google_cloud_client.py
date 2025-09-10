import os
import logging
from google.cloud import speech_v1p1beta1 as speech

class STTGoogleCloudClient:
    def __init__(self, language_code: str, sample_rate_hz: int):
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.client = speech.SpeechClient()

    def streaming_recognize_generator(self, fd_audio: int, call_id: str, adaptation_config: dict, timeout: int = 120):
        """
        Nhận dạng giọng nói từ một stream và trả về một generator các kết quả.
        
        :param fd_audio: File descriptor của luồng audio.
        :param call_id: ID của cuộc gọi để truy vết log.
        :param adaptation_config: Dictionary chứa cấu hình speech adaptation.
        :param timeout: Thời gian chờ tối đa cho API call.
        :yield: Một dictionary chứa transcript và cờ is_final.
        """
        chunk_interval_ms = 100
        chunk_size = int(self.sample_rate_hz * 2 * (chunk_interval_ms / 1000))
        logging.debug(f"STT Client [{call_id}]: Sample rate {self.sample_rate_hz}Hz, chunk size {chunk_size} bytes.")

        # --- Bước 1: Xây dựng đối tượng Speech Adaptation từ config ---
        speech_adaptation = None
        if adaptation_config and adaptation_config.get('phrase_hints'):
            phrase_set = speech.PhraseSet(
                phrases=[
                    speech.PhraseSet.Phrase(value=p, boost=adaptation_config.get('boost', 1.0))
                    for p in adaptation_config['phrase_hints']
                ]
            )
            speech_adaptation = speech.SpeechAdaptation(phrase_sets=[phrase_set])
            logging.info(f"STT Client [{call_id}]: Áp dụng Speech Adaptation với {len(adaptation_config['phrase_hints'])} gợi ý.")

        # --- Bước 2: Xây dựng Recognition Config cuối cùng ---
        enable_punctuation = adaptation_config.get('enable_automatic_punctuation', True)

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            enable_automatic_punctuation=enable_punctuation,
            model='phone_call',
            adaptation=speech_adaptation  # Gắn adaptation vào config
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True,
            single_utterance=True
        )

        def requests_gen():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            while True:
                try:
                    chunk = os.read(fd_audio, chunk_size)
                    if not chunk: break
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except (OSError, ValueError) as e:
                    logging.warning(f"STT Client [{call_id}]: Lỗi khi đọc audio stream: {e}")
                    break
            logging.debug(f"STT Client [{call_id}]: Hết audio stream.")

        responses = self.client.streaming_recognize(requests=requests_gen(), timeout=timeout)
        
        # --- Bước 3: Xử lý và yield kết quả ---
        try:
            logging.info(f"STT Client [{call_id}]: Bắt đầu nhận stream kết quả từ Google API.")
            for response in responses:
                if not response.results or not response.results[0].alternatives: continue
                result = response.results[0]
                transcript = result.alternatives[0].transcript
                logging.debug(f"STT Client [{call_id}]: Transcript (final={result.is_final}): '{transcript}'")
                yield { "transcript": transcript, "is_final": result.is_final }
                if result.is_final and streaming_config.single_utterance: break
        except Exception as e:
            logging.error(f"STT Client [{call_id}]: Lỗi khi xử lý stream từ Google API: {e}", exc_info=True)
            yield { "transcript": "", "is_final": True, "error": str(e) }
