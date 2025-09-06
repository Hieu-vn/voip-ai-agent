import os
from pydub import AudioSegment
from google.cloud import speech

def convert_wav_to_flac(input_wav_path: str, output_flac_path: str):
    """
    Converts a WAV audio file to FLAC format.
    Requires ffmpeg to be installed and accessible in the system's PATH.
    """
    try:
        audio = AudioSegment.from_wav(input_wav_path)
        audio.export(output_flac_path, format="flac")
        print(f"Converted {input_wav_path} to {output_flac_path}")
    except Exception as e:
        print(f"Error converting audio: {e}")
        raise

def transcribe_google_cloud(audio_path: str) -> str:
    """
    Transcribes an audio file using Google Cloud Speech-to-Text API.
    Assumes GOOGLE_APPLICATION_CREDENTIALS environment variable is set.
    """
    client = speech.SpeechClient()

    # Convert to FLAC if not already FLAC
    flac_audio_path = audio_path
    if not audio_path.lower().endswith(".flac"):
        flac_audio_path = audio_path.replace(".wav", ".flac")
        convert_wav_to_flac(audio_path, flac_audio_path)

    with open(flac_audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.FLAC,
        sample_rate_hertz=8000,  # Adjust based on your Asterisk setup
        language_code="vi-VN",  # Vietnamese language code
    )

    print(f"Sending audio for transcription to Google Cloud Speech-to-Text...")
    response = client.recognize(config=config, audio=audio)

    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript
    
    # Clean up the temporary FLAC file if it was created
    if flac_audio_path != audio_path and os.path.exists(flac_audio_path):
        os.remove(flac_audio_path)

    return transcript

if __name__ == "__main__":
    

    # Example usage:
    # Make sure you have a sample.wav file in the src/stt directory
    # and GOOGLE_APPLICATION_CREDENTIALS environment variable set.
    audio_file_to_transcribe = "sample.wav" # Assuming sample.wav is in the same directory as transcribe.py
    
    # For testing, let's create a dummy sample.wav if it doesn't exist
    # In a real scenario, this would be provided by Asterisk
    if not os.path.exists(audio_file_to_transcribe):
        print(f"'{audio_file_to_transcribe}' not found. Please ensure it exists for testing.")
        print("You can create a dummy WAV file or use a real one.")
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
            print(f"Created a dummy '{audio_file_to_transcribe}' for testing.")
        except ImportError:
            print("Install pydub and numpy to create a dummy WAV file: pip install pydub numpy")
        except Exception as e:
            print(f"Could not create dummy WAV file: {e}")

    if os.path.exists(audio_file_to_transcribe):
        try:
            transcribed_text = transcribe_google_cloud(audio_file_to_transcribe)
            print(f"Transcribed Text: {transcribed_text}")
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            print("Please ensure 'ffmpeg' is installed and 'GOOGLE_APPLICATION_CREDENTIALS' is set correctly.")
    else:
        print("Skipping transcription test as no audio file is available.")
