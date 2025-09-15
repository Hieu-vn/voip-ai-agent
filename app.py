import os
from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import torch
import io
import logging
import torchaudio

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize TTS model
# Use "cuda" if GPU is available, otherwise use "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# Load viXTTS model from Hugging Face
# This is an XTTS-v2 model fine-tuned for Vietnamese
tts = None
try:
    # Ensure the model is downloaded and loaded
    # The model will be downloaded to the default Hugging Face cache directory
    tts = TTS("thinhlpg/vixtts-demo").to(device)
    logging.info("viXTTS model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading viXTTS model: {e}")
    # Exit if the model cannot be loaded, as the server won't function
    exit(1)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    if tts is None:
        return jsonify({"error": "TTS model not loaded."}), 500

    data = request.json
    text = data.get('text')
    speaker_wav_path = data.get('speaker_wav') # Path to the reference speaker audio file
    language = data.get('language', 'vi') # Default to Vietnamese

    if not text:
        return jsonify({"error": "Missing 'text' parameter."}), 400
    
    # For XTTS-v2, a speaker_wav is crucial for voice cloning.
    # If not provided, we can use a default one or return an error.
    # For now, let's return an error if not provided or invalid.
    if not speaker_wav_path or not os.path.exists(speaker_wav_path):
        return jsonify({"error": "Missing or invalid 'speaker_wav' path. XTTS-v2 requires a reference speaker WAV for voice cloning."}), 400

    try:
        # Create a temporary audio file
        output_wav_path = "output.wav" # This will be overwritten on each request

        # Synthesize speech
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_wav_path,
            language=language,
            file_path=output_wav_path
        )
        logging.info(f"Successfully synthesized text: '{text}' with speaker: '{speaker_wav_path}'")

        # Return the audio file
        return send_file(output_wav_path, mimetype="audio/wav", as_attachment=True, download_name="synthesized_speech.wav")

    except Exception as e:
        logging.error(f"Error during synthesis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Coqui XTTS-v2 Vietnamese TTS Server is running. Use /synthesize endpoint for text-to-speech."

if __name__ == '__main__':
    # IMPORTANT: For XTTS-v2, you MUST provide a reference speaker WAV file (3-6 seconds)
    # For demonstration/testing, you might create a dummy one or instruct the user.
    # In a real application, this file should be properly managed.
    # For now, we'll just log a warning if it's not present.
    if not os.path.exists("vietnamese_speaker.wav"):
        logging.warning("vietnamese_speaker.wav not found. Please provide a reference WAV file for voice cloning.")
        logging.warning("You can record a short (3-6 seconds) audio clip of Vietnamese speech and save it as vietnamese_speaker.wav in the same directory as app.py.")

    app.run(host='0.0.0.0', port=5002, debug=True)
