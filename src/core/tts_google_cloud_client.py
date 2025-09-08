import os
from google.cloud import texttospeech as tts

class TTSGoogleCloudClient:
    def __init__(self, sample_rate_hz: int):
        self.sample_rate_hz = sample_rate_hz
        self.client = tts.TextToSpeechClient()

    def synth_vi_wav(self, text: str, out_wav_path: str) -> str:
        synthesis_input = tts.SynthesisInput(text=text)
        voice = tts.VoiceSelectionParams(language_code="vi-VN", name="vi-VN-Wavenet-A")
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.sample_rate_hz
        )
        resp = self.client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
        with open(out_wav_path, "wb") as f:
            f.write(resp.audio_content)
        return out_wav_path