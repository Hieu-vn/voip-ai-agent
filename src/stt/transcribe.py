import os
from google.cloud import speech
from loguru import logger
import sys
from pydub import AudioSegment

# Cấu hình logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/tmp/stt_transcribe.log", rotation="10 MB", level="DEBUG")

def transcribe_file(audio_path: str, sample_rate_hertz: int, language_code: str = "vi-VN") -> str:
    """
    Transcribes an audio file using Google Cloud Speech-to-Text API (synchronous).
    Assumes GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
    """
    client = speech.SpeechClient()

    try:
        with open(audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16, # Assuming LINEAR16 after normalization in agi_handler
            sample_rate_hertz=sample_rate_hertz,
            language_code=language_code,
            enable_automatic_punctuation=True,
        )

        logger.info(f"STT: Gửi file {audio_path} đến Google Cloud Speech-to-Text để xử lý...")
        response = client.recognize(config=config, audio=audio)

        transcript = "".join(result.alternatives[0].transcript for result in response.results)
        logger.info(f"STT: Google Cloud STT trả về: \"{transcript}\"")
        
        return transcript

    except Exception as e:
        logger.error(f"STT: Lỗi khi phiên âm audio: {e}")
        raise

if __name__ == "__main__":
    # Example usage (for testing transcribe.py directly)
    # Make sure you have a sample.wav file and GOOGLE_APPLICATION_CREDENTIALS set.
    audio_file_to_transcribe = "sample.wav"
    if not os.path.exists(audio_file_to_transcribe):
        logger.warning(f"'{audio_file_to_transcribe}' not found. Creating a dummy.")
        try:
            # Create a 5-second silent WAV file for testing
            sample_rate = 8000
            duration_ms = 5000
            samples = AudioSegment.silent(duration=duration_ms, frame_rate=sample_rate)
            samples.export(audio_file_to_transcribe, format="wav")
            logger.info(f"Created a dummy '{audio_file_to_transcribe}' for testing.")
        except Exception as e:
            logger.error(f"Could not create dummy WAV file: {e}")

    if os.path.exists(audio_file_to_transcribe):
        try:
            # Normalize the dummy WAV to 8kHz, mono, 16-bit
            audio_segment = AudioSegment.from_wav(audio_file_to_transcribe)
            audio_segment = audio_segment.set_frame_rate(8000).set_channels(1).set_sample_width(2)
            audio_segment.export(audio_file_to_transcribe, format="wav") # Overwrite with normalized

            transcribed_text = transcribe_file(audio_file_to_transcribe, 8000, "vi-VN")
            logger.info(f"Transcribed Text: {transcribed_text}")
        except Exception as e:
            logger.error(f"An error occurred during transcription: {e}")
    else:
        logger.warning("Skipping transcription test as no audio file is available.")