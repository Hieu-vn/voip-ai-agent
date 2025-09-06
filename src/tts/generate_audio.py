import torch
from transformers import VitsModel, AutoTokenizer
import soundfile as sf
from loguru import logger
import sys

# Cấu hình logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/tmp/tts_generate_audio.log", rotation="10 MB", level="DEBUG")

# ========== Configuration ========== 
TTS_MODEL_NAME = "facebook/mms-tts-vie"

# ========== Model Loading and GPU Setup ==========
# Xác định thiết bị (GPU nếu có, không thì CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Sử dụng thiết bị: {device}")

logger.info(f"Bắt đầu tải model TTS: {TTS_MODEL_NAME}...")
# Chuyển model sang thiết bị đã chọn (GPU/CPU)
model = VitsModel.from_pretrained(TTS_MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
logger.info("Tải model TTS thành công.")

def generate_audio(text: str, output_path: str):
    """
    Generates audio from text using the loaded TTS model on the specified device.

    Args:
        text: The input text to synthesize.
        output_path: The path to save the generated WAV file.
    """
    try:
        logger.debug(f"Chuẩn bị tạo audio cho text: \"{text}\"")
        # Chuyển input tensors sang cùng thiết bị với model
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model(**inputs).waveform
        
        # Chuyển output về lại CPU để xử lý và lưu file
        audio_data = output.cpu().float().numpy().squeeze()
        
        sf.write(output_path, audio_data, model.config.sampling_rate)
        logger.info(f"Đã tạo và lưu file audio tại: {output_path}")
    except Exception as e:
        logger.error(f"Lỗi khi tạo audio: {e}")
        raise

if __name__ == "__main__":
    # Example usage
    sample_text = "Xin chào, tôi là trợ lý ảo của bạn. Tôi có thể giúp gì cho bạn?"
    output_file = "sample_output.wav"
    
    logger.info("Bắt đầu chạy thử nghiệm generate_audio...")
    generate_audio(sample_text, output_file)
    logger.info(f"Kiểm tra file output tại: {output_file}")