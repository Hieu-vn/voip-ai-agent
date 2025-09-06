import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from loguru import logger

# ========== Configuration ==========
TTS_MODEL_NAME = "facebook/mms-tts-vie"

# ========== Model Loading ==========
logger.info(f"Loading TTS model: {TTS_MODEL_NAME}...")
model = VitsModel.from_pretrained(TTS_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
logger.info("TTS model loaded successfully.")

def generate_audio(text: str, output_path: str):
    """
    Generates audio from text using the loaded TTS model.

    Args:
        text: The input text to synthesize.
        output_path: The path to save the generated WAV file.
    """
    try:
        inputs = tokenizer(text, return_tensors="pt")

        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Save as WAV file
        audio_data = output.cpu().float().numpy().squeeze() # Remove batch dimension
        sf.write(output_path, audio_data, model.config.sampling_rate)
        logger.info(f"Generated audio saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error generating audio: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    sample_text = "Xin chào, tôi là trợ lý ảo của bạn. Tôi có thể giúp gì cho bạn?"
    output_file = "sample_output.wav"
    generate_audio(sample_text, output_file)
    print(f"Generated audio for '{sample_text}' to {output_file}")
