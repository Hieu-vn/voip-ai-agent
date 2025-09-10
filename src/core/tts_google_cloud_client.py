import os
from google.cloud import texttospeech as tts
import logging

class TTSGoogleCloudClient:
    def __init__(self):
        self.client = tts.TextToSpeechClient()

    def synth_to_wav(self, text: str, out_wav_path: str, voice_name: str, sample_rate_hz: int, speaking_rate: float = 1.0):
        """
        Tổng hợp văn bản thành file WAV với các tham số động.
        Hỗ trợ cả text thường và SSML (nếu text bắt đầu bằng <speak>).
        """
        audio_content = self.get_audio_content(text, voice_name, sample_rate_hz, speaking_rate)
        if audio_content:
            with open(out_wav_path, "wb") as f:
                f.write(audio_content)
            return out_wav_path
        return None

    def get_audio_content(self, text: str, voice_name: str, sample_rate_hz: int, speaking_rate: float = 1.0) -> bytes:
        """
        Gọi API và trả về nội dung audio dưới dạng bytes thô.
        """
        if text.strip().lower().startswith('<speak>'):
            synthesis_input = tts.SynthesisInput(ssml=text)
            logging.debug("TTS Client: Nhận diện input là SSML.")
        else:
            synthesis_input = tts.SynthesisInput(text=text)

        voice = tts.VoiceSelectionParams(
            language_code="vi-VN", 
            name=voice_name
        )
        
        audio_config = tts.AudioConfig(
            audio_encoding=tts.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate_hz,
            speaking_rate=speaking_rate
        )
        
        logging.debug(f"TTS Client: Gửi yêu cầu đến Google API: voice={voice_name}, rate={speaking_rate}, sample_rate={sample_rate_hz}")
        
        try:
            resp = self.client.synthesize_speech(
                input=synthesis_input, 
                voice=voice, 
                audio_config=audio_config
            )
            return resp.audio_content
        except Exception as e:
            logging.error(f"TTS Client: Lỗi khi gọi Google API: {e}", exc_info=True)
            return None
