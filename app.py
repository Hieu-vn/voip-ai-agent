# /data/voip-ai-agent/app.py
import anyio
import asyncari
import logging
import os
import threading
from flask import Flask, request, send_file, jsonify
from TTS.api import TTS
import torch
import wave

# --- General Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Asterisk ARI Configuration ---
AST_URL = "http://localhost:8088/ari"
AST_APP = "voip-ai-agent"
AST_USER = "vitalpbx"
AST_PASS = "zcWGYbNnPer2YUBTg433EMuVs"
MEDIA_IP = "127.0.0.1"  # IP address of this application server

# --- TTS Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = None

# --- Flask App for TTS ---
flask_app = Flask(__name__)

@flask_app.route('/synthesize', methods=['POST'])
def synthesize():
    if tts is None:
        return jsonify({"error": "TTS model not loaded."}), 500

    data = request.json
    text = data.get('text')
    speaker_wav_path = data.get('speaker_wav')
    language = data.get('language', 'vi')

    if not text or not speaker_wav_path or not os.path.exists(speaker_wav_path):
        return jsonify({"error": "Missing or invalid 'text' or 'speaker_wav' path."}), 400

    try:
        output_wav_path = f"/tmp/output_{os.urandom(4).hex()}.wav"
        tts.tts_to_file(text=text, speaker_wav=speaker_wav_path, language=language, file_path=output_wav_path)
        logging.info(f"Synthesized text to {output_wav_path}")
        return send_file(output_wav_path, mimetype="audio/wav")
    except Exception as e:
        logging.error(f"Error during synthesis: {e}")
        return jsonify({"error": str(e)}), 500

@flask_app.route('/')
def index():
    return "AI Agent Server is running. ARI client is active. TTS is on /synthesize."

def run_flask_app():
    # Ensure a default speaker wav exists for TTS testing if needed
    if not os.path.exists("vietnamese_speaker.wav"):
        logging.warning("vietnamese_speaker.wav not found. TTS endpoint /synthesize will require a valid 'speaker_wav' path.")
    flask_app.run(host='0.0.0.0', port=5002, debug=False) # Debug off in thread

# --- ARI Application Logic ---

async def audio_stream_handler(channel_id, media_port):
    """
    Listens for RTP packets on the given port and writes the audio to a file.
    """
    raw_audio_path = f"/tmp/{channel_id}.raw"
    wav_audio_path = f"/tmp/{channel_id}.wav"
    audio_data = bytearray()
    
    try:
        async with await anyio.create_udp_socket(local_port=media_port, local_host=MEDIA_IP) as udp:
            logging.info(f"[{channel_id}] UDP socket listening on {MEDIA_IP}:{media_port}")
            # Listen for 5 seconds of audio
            async with anyio.move_on_after(5):
                while True:
                    packet, (host, port) = await udp.receive()
                    # Strip 12-byte RTP header and append payload
                    audio_data.extend(packet[12:])
            logging.info(f"[{channel_id}] Finished receiving audio data ({len(audio_data)} bytes).")

    except Exception as e:
        logging.error(f"[{channel_id}] Error in UDP listener: {e}")
        return

    if not audio_data:
        logging.warning(f"[{channel_id}] No audio data received.")
        return

    # Save raw audio to a .wav file for easy playback/testing
    try:
        with wave.open(wav_audio_path, 'wb') as wf:
            wf.setnchannels(1)  # Mono
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(8000) # 8kHz sample rate (slin8)
            wf.writeframes(audio_data)
        logging.info(f"[{channel_id}] Saved captured audio to {wav_audio_path}")
    except Exception as e:
        logging.error(f"[{channel_id}] Error saving WAV file: {e}")


async def channel_handler(ari_client, channel):
    """
    Main handler for an incoming channel.
    """
    channel_id = channel.id
    logging.info(f"[{channel_id}] New call received. Answering.")
    
    try:
        await channel.answer()
        
        # Create an external media channel to receive audio from Asterisk
        logging.info(f"[{channel_id}] Creating external media stream...")
        media_info = await channel.externalMedia(
            app=AST_APP,
            external_host=f"{MEDIA_IP}:0", # Let the system pick a port
            format="slin" # 8kHz, 16-bit signed linear PCM
        )
        
        media_port = media_info['channel']['local_port']
        logging.info(f"[{channel_id}] Asterisk will send media to {MEDIA_IP}:{media_port}")

        # Start the audio stream handler in the background
        async with anyio.create_task_group() as tg:
            tg.start_soon(audio_stream_handler, channel_id, media_port)

        # Play a message indicating we are listening
        await channel.play(media="sound:im-sorry") # Placeholder sound
        
        # Wait for the audio handler to finish (or a timeout)
        await anyio.sleep(6) # Give it a bit more time than the listener

        logging.info(f"[{channel_id}] Finished processing. Hanging up.")
        await channel.play(media="sound:vm-goodbye")

    except Exception as e:
        logging.error(f"[{channel_id}] An error occurred in channel_handler: {e}")
    finally:
        try:
            if not channel.destroyed:
                await channel.hangup()
        except Exception as e:
            logging.error(f"[{channel_id}] Failed to hang up channel: {e}")


async def main_ari():
    """
    Connects to ARI and manages incoming calls.
    """
    global tts
    logging.info("Initializing TTS model...")
    try:
        # Load viXTTS model from Hugging Face
        tts = TTS("thinhlpg/vixtts-demo").to(device)
        logging.info("viXTTS model loaded successfully.")
    except Exception as e:
        logging.error(f"Fatal: Could not load viXTTS model: {e}")
        return # Exit if model fails

    async with asyncari.connect(AST_URL, AST_APP, AST_USER, AST_PASS) as ari:
        logging.info("ARI client connected. Waiting for calls...")
        async with ari.on_event('StasisStart') as stasis_start_events:
            async for ev in stasis_start_events:
                # When a new call enters Stasis, spawn a handler for it
                channel = ev['channel']
                anyio.create_task_group().start_soon(channel_handler, ari, channel)


if __name__ == '__main__':
    # Start Flask app in a background thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    logging.info("Flask TTS server starting in a background thread.")

    # Start the main ARI client
    try:
        anyio.run(main_ari)
    except KeyboardInterrupt:
        logging.info("Shutting down.")