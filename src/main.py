import asyncio
import logging
import os
from dotenv import load_dotenv
import threading
from prometheus_client import start_http_server
import socket
import struct
import audioop # For basic audio manipulation like uLaw to PCM
import websockets # For ARI connection handling
import anyio # Import anyio to catch its exceptions

# Compatibility shim for websockets>=12 (no websockets.exceptions)
try:
    ConnectionClosedOK = websockets.ConnectionClosedOK
    ConnectionClosedError = websockets.ConnectionClosedError
except AttributeError:
    # websockets<=11 fallback API
    from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from asyncari import connect

from src.core.stt_module import STTModule # Import STTModule class
from src.core.nlp_module import NLPModule
from src.core.tts_module import tts_service_handler, tts_stream_service_handler
from src.utils.tracing import initialize_tracer

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Asterisk ARI Configuration
AST_URL = os.getenv("ARI_URL")
AST_APP = os.getenv("ARI_APP_NAME", "voip-ai-agent") # Assuming ARI_APP_NAME is in .env or default
ARI_USERNAME = os.getenv("ARI_USERNAME")
ARI_PASSWORD = os.getenv("ARI_PASSWORD")

# External Media (UDP server) Configuration
EXTERNAL_MEDIA_HOST = os.getenv("EXTERNAL_MEDIA_HOST", "127.0.0.1") # IP of the Python server
EXTERNAL_MEDIA_PORT = int(os.getenv("EXTERNAL_MEDIA_PORT", 60000)) # Port to receive RTP from Asterisk
EXTERNAL_MEDIA_FORMAT = os.getenv("EXTERNAL_MEDIA_FORMAT", "ulaw") # Audio format Asterisk will send (e.g., ulaw, alaw, slin16)

# Global dictionary to hold active channels and their associated bridges/UDP sockets
active_calls = {}

# RTP Packetization Helper
# This is a simplified version. A full RTP library would handle more. 
# For ulaw @ 8kHz, 20ms of audio is 160 bytes. Payload type 0 for PCMU.
# Sequence number and timestamp need to be managed per stream.
# SSRC (Synchronization Source identifier) is a random 32-bit number.

def build_rtp_packet(sequence_number, timestamp, ssrc, payload, payload_type=0, marker=0):
    """
    Builds a basic RTP packet.
    """
    v_p_x_cc = 0x80  # Version 2, P=0, X=0, CC=0
    m_pt = (marker << 7) | (payload_type & 0x7F)
    
    # RTP header: V P X CC M PT SequenceNumber Timestamp SSRC
    header = struct.pack("!BBHII", v_p_x_cc, m_pt, sequence_number, timestamp, ssrc)
    return header + payload


async def handle_stasis_start(client, event, nlp_module_instance):
    """
    Handles StasisStart event when a new channel enters the ARI application.
    """
    channel = event['channel']
    channel_id = channel['id']
    logger.info(f"Channel {channel_id} entered Stasis application '{AST_APP}'")

    try:
        # 1. Answer the call
        await client.channels.answer(channelId=channel_id)
        logger.info(f"Answered channel {channel_id}")

        # 2. Create a mixing bridge
        bridge = await client.bridges.create(type='mixing')
        bridge_id = bridge['id']
        logger.info(f"Created mixing bridge {bridge_id}")

        # 3. Add the original channel to the bridge
        await client.bridges.addChannel(bridgeId=bridge_id, channel=channel_id)
        logger.info(f"Added channel {channel_id} to bridge {bridge_id}")

        # --- Dynamic UDP Port Binding & ExternalMedia Creation ---
        # Create UDP socket first to get an ephemeral port
        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp_sock.bind((EXTERNAL_MEDIA_HOST, 0)) # Bind to ephemeral port
        local_port = udp_sock.getsockname()[1]
        udp_sock.setblocking(False)
        logger.info(f"Bound UDP socket for {channel_id} to {EXTERNAL_MEDIA_HOST}:{local_port}")

        # 4. Create External Media channel using the dynamically assigned port
        external_media_channel = await client.channels.externalMedia(
            app=AST_APP,
            external_host=f"{EXTERNAL_MEDIA_HOST}:{local_port}", # Use dynamic port
            format=EXTERNAL_MEDIA_FORMAT,
            direction='both' # Allow two-way audio stream
        )
        external_media_channel_id = external_media_channel['id']
        logger.info(f"Created external media channel {external_media_channel_id} "
                     f"streaming to {EXTERNAL_MEDIA_HOST}:{local_port}")

        # 5. Add External Media channel to the bridge
        await client.bridges.addChannel(bridgeId=bridge_id, channel=external_media_channel_id)
        logger.info(f"Added external media channel {external_media_channel_id} to bridge {bridge_id}")

        # Store call information
        active_calls[channel_id] = {
            'bridge_id': bridge_id,
            'external_media_channel_id': external_media_channel_id,
            'udp_socket': udp_sock, # Store the bound socket
            'local_port': local_port, # Store the local port
            'remote_rtp_address': None, # Address to send RTP back to (discovered on first packet)
            'nlp_module_instance': nlp_module_instance, # Pass NLP instance
            'rtp_sequence': 0, # RTP sequence number
            'rtp_timestamp': 0, # RTP timestamp
            'rtp_ssrc': os.urandom(4) # Random SSRC for this stream
        }

        # Start a separate task to handle the UDP stream for this call
        asyncio.create_task(handle_udp_stream(channel_id))

    except Exception as e:
        logger.error(f"ARI Error for channel {channel_id}: {e}")
        # Try to hangup channel if error occurs
        try:
            await client.channels.hangup(channelId=channel_id)
        except Exception: # Catch all exceptions during hangup to avoid masking original error
            pass

async def handle_stasis_end(client, event):
    """
    Handles StasisEnd event when a channel leaves the ARI application.
    """
    channel_id = event['channel']['id']
    logger.info(f"Channel {channel_id} left Stasis application '{AST_APP}'")

    if channel_id in active_calls:
        call_info = active_calls.pop(channel_id)
        bridge_id = call_info['bridge_id']
        external_media_channel_id = call_info['external_media_channel_id']

        # Close UDP socket if it was opened
        if call_info['udp_socket']:
            call_info['udp_socket'].close()
            logger.info(f"Closed UDP socket for channel {channel_id}")

        try:
            # Hangup external media channel
            await client.channels.hangup(channelId=external_media_channel_id)
            logger.info(f"Hung up external media channel {external_media_channel_id}")
        except Exception as e:
            logger.warning(f"Could not hangup external media channel {external_media_channel_id}: {e}")

        try:
            # Destroy bridge
            await client.bridges.destroy(bridgeId=bridge_id)
            logger.info(f"Destroyed bridge {bridge_id}")
        except Exception as e:
            logger.warning(f"Could not destroy bridge {bridge_id}: {e}")

async def handle_udp_stream(channel_id):
    """
    Handles the UDP (RTP) stream for a specific call.
    This is where STT, NLP, TTS integration will happen.
    """
    logger.info(f"Starting UDP stream handler for channel {channel_id}")
    call_info = active_calls[channel_id]
    nlp_module_instance = call_info['nlp_module_instance'] # Retrieve NLP instance
    udp_sock = call_info['udp_socket'] # Retrieve the bound UDP socket

    # Prefer get_running_loop in modern asyncio
    loop = asyncio.get_running_loop()

    # STT setup
    stt_module_instance = STTModule(language_code="vi-VN") # Assuming Vietnamese
    stt_audio_queue = asyncio.Queue() # Queue to put audio chunks for STT
    stt_results_queue = asyncio.Queue() # Queue to receive STT results
    
    # Start STT recognition task
    stt_task = asyncio.create_task(stt_module_instance.stt_service_handler(
        audio_queue=stt_audio_queue, # Pass the audio queue
        sample_rate=8000, # Assuming 8kHz for phone calls
        call_id=channel_id,
        adaptation_config={},
        result_queue=stt_results_queue # Pass the result queue
    ))
    
    # TTS setup
    tts_pipe_path = f"/tmp/tts_audio_{channel_id}.fifo"
    # Create named pipe for TTS output
    try:
        os.mkfifo(tts_pipe_path)
        logger.info(f"Created TTS named pipe: {tts_pipe_path}")
    except FileExistsError:
        logger.warning(f"TTS named pipe already exists: {tts_pipe_path}")
    
    tts_fd = os.open(tts_pipe_path, os.O_RDONLY | os.O_NONBLOCK) # Open for reading TTS audio

    # Task to read from TTS pipe and send to Asterisk
    async def tts_playback_task(tts_fd, udp_sock, remote_rtp_address, rtp_sequence, rtp_timestamp, rtp_ssrc):
        logger.info(f"Starting TTS playback task for channel {channel_id}")
        seq = rtp_sequence
        ts = rtp_timestamp
        ssrc = int.from_bytes(rtp_ssrc, 'big') # Convert SSRC bytes to int
        
        try:
            while True:
                try:
                    tts_audio_chunk = os.read(tts_fd, 160) # Read 20ms ulaw chunks (8kHz * 1 byte/sample * 0.02s)
                    if not tts_audio_chunk:
                        await asyncio.sleep(0.001) # Wait a bit if no data
                        continue
                    
                    # Build RTP packet
                    # Payload type 0 for PCMU (ulaw)
                    rtp_packet = build_rtp_packet(seq, ts, ssrc, tts_audio_chunk, payload_type=0)
                    
                    await loop.sock_sendto(udp_sock, rtp_packet, remote_rtp_address)
                    # logger.debug(f"Sent {len(rtp_packet)} bytes of TTS audio to {remote_rtp_address}")
                    
                    seq = (seq + 1) & 0xFFFF # Increment sequence number
                    ts = (ts + 160) & 0xFFFFFFFF # Increment timestamp for 8kHz, 20ms (8000 * 0.02 = 160)

                except BlockingIOError:
                    await asyncio.sleep(0.001) # Wait a bit before trying again
                except Exception as e:
                    logger.error(f"Error in TTS playback task for channel {channel_id}: {e}")
                    break
        finally:
            logger.info(f"TTS playback task for channel {channel_id} finished.")
            os.close(tts_fd) # Close read end of pipe
            os.unlink(tts_pipe_path) # Unlink pipe

    tts_playback_task_handle = None
    full_transcript = ""
    
    try:
        while channel_id in active_calls:
            try:
                # Receive RTP data from Asterisk
                data, addr = await loop.sock_recvfrom(udp_sock, 2048) # RTP buffer size
                if not call_info['remote_rtp_address']:
                    call_info['remote_rtp_address'] = addr
                    # Initialize RTP sequence, timestamp, SSRC for outbound stream
                    call_info['rtp_sequence'] = 0
                    call_info['rtp_timestamp'] = 0
                    call_info['rtp_ssrc'] = os.urandom(4) # Random SSRC
                    logger.info(f"Received first RTP packet from {addr} for channel {channel_id}. Initializing outbound RTP.")
                    
                    # Start TTS playback task once remote address is known
                    tts_playback_task_handle = asyncio.create_task(
                        tts_playback_task(tts_fd, udp_sock, call_info['remote_rtp_address'], 
                                          call_info['rtp_sequence'], call_info['rtp_timestamp'], call_info['rtp_ssrc'])
                    )

                # Decode RTP packet
                rtp_payload = data[12:] # Simple payload extraction

                # Convert uLaw to PCM 16-bit
                if EXTERNAL_MEDIA_FORMAT == "ulaw":
                    pcm_data = audioop.ulaw2lin(rtp_payload, 2) # 2 bytes per sample for 16-bit PCM
                else:
                    pcm_data = rtp_payload # Assume already PCM or compatible format

                # Write PCM data to named pipe for STT
                try:
                    await stt_audio_queue.put(pcm_data) # Put raw audio data into queue for STT processing
                except asyncio.QueueFull:
                    logger.warning(f"STT input queue for {channel_id} is full. Dropping audio chunk.")
                
                # Process STT results from queue
                while not stt_results_queue.empty(): # Process all available STT results
                    stt_result = await stt_results_queue.get()
                    if stt_result.get("type") == "stream_end":
                        logger.info(f"STT stream ended for {channel_id}.")
                        break # Exit inner loop, STT task will be cancelled in finally
                    
                    if stt_result.get('transcript'):
                        logger.info(f"STT Transcript ({'Final' if stt_result.get('is_final') else 'Interim'}): {stt_result['transcript']}")
                        if stt_result.get('is_final'):
                            full_transcript = stt_result['transcript']
                            logger.info(f"Final Transcript for NLP: {full_transcript}")
                            
                            # 2. NLP: Process final transcript with Llama 4 Scout
                            nlp_response = await nlp_module_instance.process_user_input(full_transcript)
                            response_text = nlp_response.get("response_text", "Xin lỗi, tôi không hiểu yêu cầu của bạn.")
                            
                            # 3. TTS: Convert response_text to audio and write to TTS pipe
                            # Assuming a default speaker_wav for the response. This needs to be provided.
                            speaker_wav_path = os.getenv("DEFAULT_SPEAKER_WAV", "/data/voip-ai-agent/vietnamese_speaker.wav")
                            
                            # Call tts_stream_service_handler to write to the TTS named pipe
                            await tts_stream_service_handler(
                                pipe_path=tts_pipe_path,
                                text=response_text,
                                speaker_wav_path=speaker_wav_path
                            )
                            logger.info(f"TTS audio sent to pipe {tts_pipe_path} for playback.")
                            
                            # Reset full_transcript for next utterance
                            full_transcript = ""
                            
                    if stt_result.get('error'):
                        logger.error(f"STT Error: {stt_result['error']}. Sending 'I didn't hear you' message.")
                        response_text = "Xin lỗi, tôi không nghe rõ. Bạn có thể nhắc lại không?"
                        speaker_wav_path = os.getenv("DEFAULT_SPEAKER_WAV", "/data/voip-ai-agent/vietnamese_speaker.wav")
                        await tts_stream_service_handler(
                            pipe_path=tts_pipe_path,
                            text=response_text,
                            speaker_wav_path=speaker_wav_path
                        )
                        full_transcript = "" # Reset after error

            except BlockingIOError:
                await asyncio.sleep(0.001) # Wait a bit before trying again (very short sleep for low latency)
            except Exception as e:
                logger.error(f"Error in UDP stream for channel {channel_id}: {e}")
                break # Exit loop if critical error

    finally:
        # Cancel STT task
        stt_task.cancel()
        try:
            await stt_task # Await cancellation
        except asyncio.CancelledError:
            logger.info(f"STT task for {channel_id} cancelled.")

        # Cancel TTS playback task
        if tts_playback_task_handle:
            tts_playback_task_handle.cancel()
            try:
                await tts_playback_task_handle
            except asyncio.CancelledError:
                logger.info(f"TTS playback task for {channel_id} cancelled.")
        
        if channel_id in active_calls:
            del active_calls[channel_id]
        logger.info(f"Stopped UDP stream handler for channel {channel_id}")
        if udp_sock:
            udp_sock.close()

async def main():
    """
    Main function to connect to ARI and start the event loop.
    """
    # Validate essential environment variables early
    for k in ["ARI_URL", "ARI_USERNAME", "ARI_PASSWORD"]:
        if not os.getenv(k):
            logger.critical(f"Missing required environment variable: {k}")
            raise SystemExit(1)

    logger.info(f"Connecting to ARI at {AST_URL} for app '{AST_APP}'")
    try:
        async with connect(AST_URL, AST_APP, ARI_USERNAME, ARI_PASSWORD) as ari_client:
            logger.info("Connected to ARI.")

            # Instantiate NLPModule
            nlp_module_instance = NLPModule(
                llama_model=os.getenv("LLAMA_MODEL_PATH")
            )
            await nlp_module_instance.load_nlp_model() # Load the model at startup

            # Register event handlers
            ari_client.on_event('StasisStart', lambda client, event: handle_stasis_start(client, event, nlp_module_instance))
            ari_client.on_event('StasisEnd', handle_stasis_end) # Register StasisEnd handler
            
            logger.info(f"ARI application '{AST_APP}' is running. Waiting for calls.")
            # Keep the client running and processing events
            # This will block until the client is closed.
            async for event in ari_client:
                pass # We are handling events via on_event, so we just need to keep the loop running

    except ExceptionGroup as eg:
        logger.error(f"Failed to connect to ARI due to multiple errors:")
        for i, exc in enumerate(eg.exceptions):
            logger.error(f"  Sub-exception {i+1}: {exc}", exc_info=True)
    except ConnectionClosedOK:
        logger.info("ARI WebSocket connection closed gracefully.")
    except Exception as e:
        logger.error(f"Failed to connect to ARI or an error occurred: {e}", exc_info=True)

def start_metrics_server():
    """Khởi chạy server HTTP cho Prometheus trong một luồng nền."""
    try:
        port = int(os.getenv("METRICS_PORT", 9108)) # Use 9108 to avoid conflict with default Prometheus
        start_http_server(port)
        logger.info(f"Prometheus metrics server đang chạy trên port {port}")
    except Exception as e:
        logger.error(f"Không thể khởi chạy Prometheus metrics server: {e}")

if __name__ == "__main__":
    # Khởi tạo OpenTelemetry Tracer
    tracer_provider = initialize_tracer("voip-ai-agent")

    try:
        # Khởi chạy metrics server trong một daemon thread
        metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
        metrics_thread.start()

        logger.info("Khởi chạy ứng dụng AI Agent...")
        # Run the main ARI client
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Tắt ứng dụng.")
    except Exception as e:
        logger.critical(f"Lỗi nghiêm trọng ở tầng cao nhất: {e}", exc_info=True)
    finally:
        # Ensure tracer provider is shut down safely to send final spans
        if tracer_provider:
            logger.info("Shutting down OpenTelemetry Tracer Provider...")
            tracer_provider.shutdown()