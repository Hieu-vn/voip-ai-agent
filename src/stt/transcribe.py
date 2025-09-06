import os
from pydub import AudioSegment
from google.cloud import speech
from loguru import logger
import sys

# Cấu hình logger để nhất quán với agi_handler
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/tmp/stt_transcribe.log", rotation="10 MB", level="DEBUG")

def convert_wav_to_flac(input_wav_path: str, output_flac_path: str):
    """
    Converts a WAV audio file to FLAC format.
    Requires ffmpeg to be installed and accessible in the system's PATH.
    """
    try:
        logger.debug(f"Bắt đầu chuyển đổi {input_wav_path} sang FLAC...")
        audio = AudioSegment.from_wav(input_wav_path)
        audio.export(output_flac_path, format="flac")
        logger.info(f"Đã chuyển đổi thành công sang {output_flac_path}")
    except Exception as e:
        logger.error(f"Lỗi khi chuyển đổi audio: {e}")
        raise

def transcribe_google_cloud(audio_path: str) -> str:
    """
    Transcribes an audio file using Google Cloud Speech-to-Text API.
    Assumes GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
    """
    client = speech.SpeechClient()

    # Chuyển đổi sang FLAC nếu cần
    flac_audio_path = audio_path
    if not audio_path.lower().endswith(".flac"):
        flac_audio_path = os.path.splitext(audio_path)[0] + ".flac"
        convert_wav_to_flac(audio_path, flac_audio_path)

    try:
        with open(flac_audio_path, "rb") as audio_file:
            content = audio_file.read()

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
            sample_rate_hertz=8000,  # Phù hợp với Asterisk, có thể cần điều chỉnh
            language_code="vi-VN",
        )

        logger.info("Bắt đầu gửi audio đến Google Cloud Speech-to-Text để xử lý...")
        response = client.recognize(config=config, audio=audio)

        transcript = "".join(result.alternatives[0].transcript for result in response.results)
        logger.info(f"Google Cloud STT trả về: \"{transcript}\"")
        
        return transcript

    finally:
        # Dọn dẹp file FLAC tạm nếu nó được tạo ra
        if flac_audio_path != audio_path and os.path.exists(flac_audio_path):
            logger.debug(f"Xóa file FLAC tạm: {flac_audio_path}")
            os.remove(flac_audio_path)

if __name__ == "__main__":
    

    # Example usage:
    # Make sure you have a sample.wav file in the src/stt directory
    # and GOOGLE_APPLICATION_CREDENTIALS environment variable set.
    audio_file_to_transcribe = "sample.wav" # Assuming sample.wav is in the same directory as transcribe.py
    
    # For testing, let's create a dummy sample.wav if it doesn't exist
    # In a real scenario, this would be provided by Asterisk
    if not os.path.exists(audio_file_to_transcribe):
        logger.warning(f"'{audio_file_to_transcribe}' not found. Please ensure it exists for testing.")
        logger.warning("You can create a dummy WAV file or use a real one.")
        # Example of creating a dummy WAV (requires pydub and numpy)
        try:
            from pydub import AudioSegment
            import numpy as np
            # Create a 1-second silent WAV file
            sample_rate = 8000
            duration_ms = 1000
            samples = np.zeros(int(sample_rate * duration_ms / 1000)).astype(np.int16)
            audio_segment = AudioSegment(
                samples.tobytes(), 
                frame_rate=sample_rate, 
                sample_width=samples.dtype.itemsize, 
                channels=1
            )
            audio_segment.export(audio_file_to_transcribe, format="wav")
            logger.info(f"Created a dummy '{audio_file_to_transcribe}' for testing.")
        except ImportError:
            logger.error("Install pydub and numpy to create a dummy WAV file: pip install pydub numpy")
        except Exception as e:
            logger.error(f"Could not create dummy WAV file: {e}")

    if os.path.exists(audio_file_to_transcribe):
        try:
            transcribed_text = transcribe_google_cloud(audio_file_to_transcribe)
            logger.info(f"Transcribed Text: {transcribed_text}")
        except Exception as e:
            logger.error(f"An error occurred during transcription: {e}")
            logger.error("Please ensure 'ffmpeg' is installed and 'GOOGLE_APPLICATION_CREDENTIALS' is set correctly.")
    else:
        logger.warning("Skipping transcription test as no audio file is available.")