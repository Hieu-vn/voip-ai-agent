import os
from google.cloud import speech_v1p1beta1 as speech

class STTGoogleCloudClient:
    def __init__(self, language_code: str, sample_rate_hz: int):
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.client = speech.SpeechClient()

    def transcribe_single_utterance(self, fd_audio=3) -> str:
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz,
            language_code=self.language_code,
            enable_automatic_punctuation=True
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=False,
            single_utterance=True  # kết thúc khi hết lượt nói
        )

        def requests_gen():
            yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
            while True:
                try:
                    chunk = os.read(fd_audio, 320)  # 20ms @ 8kHz, 16-bit
                except Exception:
                    break
                if not chunk:
                    break
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        responses = self.client.streaming_recognize(requests=requests_gen())
        final_text = ""
        for resp in responses:
            for result in resp.results:
                if result.is_final and result.alternatives:
                    final_text = result.alternatives[0].transcript.strip()
        return final_text