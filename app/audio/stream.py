import socket
import structlog
import asyncio
import os
from typing import Optional

from app.tts.client import TTSClient
from app.stt.client import SttClient
from app.nlu.agent import Agent as NluAgent # Rename for clarity
from app.utils.text_normalizer import TextNormalizer
from app.evaluation.tracker import evaluation_tracker

log = structlog.get_logger()

# --- RTP Server for bi-directional audio ---

class RtpServerProtocol(asyncio.DatagramProtocol):
    """A Datagram Protocol to handle RTP traffic with Asterisk."""

    def __init__(self, call_handler: 'CallHandler'):
        self.call_handler = call_handler
        self.transport: Optional[asyncio.DatagramTransport] = None
        self.remote_addr = None
        super().__init__()

    def connection_made(self, transport: asyncio.DatagramTransport):
        """Called when the UDP socket is set up."""
        self.transport = transport
        self.call_handler.log.info("RTP server protocol connection made.")

    def datagram_received(self, data: bytes, addr: tuple):
        """Called when an RTP packet is received from Asterisk."""
        if self.remote_addr is None:
            self.remote_addr = addr  # Keep track of Asterisk's address
            self.call_handler.log.info("Received first RTP packet", remote_addr=addr)

        # A basic RTP header is 12 bytes. We just want the payload.
        if len(data) > 12:
            payload = data[12:]
            # Forward the raw SLIN16 audio payload to the STT stream
            self.call_handler.on_audio_received(payload)

    def error_received(self, exc: Exception):
        """Called when there is a UDP transport error."""
        self.call_handler.log.error("RTP server protocol error", exc_info=exc)

    def send_audio(self, audio_chunk: bytes):
        """Sends a raw audio chunk back to Asterisk, wrapped in a simple RTP header."""
        if self.transport and self.remote_addr:
            # This is a simplified RTP header. For production, a proper library
            # for RTP packetization (handling sequence numbers, timestamps) is recommended.
            header = os.urandom(12) # Placeholder for a real RTP header
            packet = header + audio_chunk
            try:
                self.transport.sendto(packet, self.remote_addr)
            except Exception as e:
                self.call_handler.log.error("Failed to send RTP packet", exc_info=e)

    def close(self):
        """Closes the UDP transport."""
        if self.transport:
            self.transport.close()
            self.call_handler.log.info("RTP server transport closed.")

# --- Main Call Handler Class ---

class CallHandler:
    """Orchestrates a single call from start to finish."""

    def __init__(self, ari_client, channel):
        self.ari_client = ari_client
        self.channel = channel
        self.log = structlog.get_logger(call_id=self.channel.id)
        self.rtp_server: Optional[RtpServerProtocol] = None
        self.external_media_channel = None
        self.is_active = True
        self.nlu_agent: Optional[NluAgent] = None
        self.turn_index = 0

        # Instantiate service clients and utilities
        self.tts_client = TTSClient()
        self.stt_client = SttClient(self.on_stt_result)
        self.normalizer = TextNormalizer()

    async def _setup_rtp_server(self) -> tuple[str, int]:
        """Creates a UDP server to receive RTP and returns its host and port."""
        loop = asyncio.get_running_loop()
        # Create a datagram endpoint, binding to an available port on all interfaces
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: RtpServerProtocol(self),
            local_addr=('0.0.0.0', 0) # 0 means OS picks a free port
        )
        self.rtp_server = protocol

        # Get the actual host and port we are bound to
        sock: socket.socket = transport.get_extra_info('socket')
        host, port = sock.getsockname()
        
        # Use the host's actual IP if possible, not 0.0.0.0, for the ARI command
        app_host = os.getenv("APP_HOST_IP", socket.gethostbyname(socket.gethostname()))
        self.log.info(f"RTP server listening on {app_host}:{port}")
        return app_host, port

    async def handle_call(self):
        """Main coroutine to manage the call lifecycle."""
        self.log.info("New call received, starting handler.")
        try:
            # 1. Create NLU agent instance for this call
            self.nlu_agent = await NluAgent.create()

            # 2. Answer call and set up media
            await self.channel.answer()
            self.log.info("Call answered.")

            rtp_host, rtp_port = await self._setup_rtp_server()
            
            self.external_media_channel = await self.ari_client.channels.externalMedia(
                channelId=self.channel.id,
                app=os.getenv("ARI_APP_NAME", "ai_app"),
                external_host=f"{rtp_host}:{rtp_port}",
                format="slin16",  # 16-bit signed linear PCM @ 8kHz
                transport="udp",
            )
            self.log.info("External media stream established.")

            # 3. Start the STT client
            await self.stt_client.start()

            await asyncio.sleep(0.5) # Wait for media to be active

            # 4. Welcome message
            await self.play_tts_audio("Xin chào, tôi là trợ lý ảo, tôi có thể giúp gì cho bạn?")

            # 5. Main loop - logic is now event-driven via callbacks
            while self.is_active:
                await asyncio.sleep(1)

        except Exception as e:
            self.log.error("An error occurred in the call handler", exc_info=e)
        finally:
            self.log.info("Call handler is shutting down.")
            await self.cleanup()

    def on_audio_received(self, audio_chunk: bytes):
        """Callback executed by RtpServerProtocol when audio arrives."""
        if self.stt_client:
            self.stt_client.write(audio_chunk)

    async def on_stt_result(self, transcript: str, is_final: bool):
        """Callback executed by the STT client with a transcription result."""
        self.log.debug(f"STT Result ({'final' if is_final else 'interim'}): {transcript}")
        
        if is_final and transcript and self.nlu_agent:
            # 1. Normalize the final transcript
            normalized_transcript = self.normalizer.normalize(transcript)
            self.log.info("STT Finalized", original=transcript, normalized=normalized_transcript)

            # 2. Pass normalized text to the NLU agent
            agent_state = await self.nlu_agent.respond(normalized_transcript)
            agent_reply_text = agent_state.get("reply", "")
            self.log.info("NLU Agent responded", response=agent_reply_text, state=agent_state)

            # 3. Log the turn for evaluation
            evaluation_tracker.log_turn(
                session_id=self.channel.id,
                turn_index=self.turn_index,
                user_text=normalized_transcript,
                bot_text=agent_reply_text,
                metadata=agent_state,
            )
            self.turn_index += 1

            # 4. Play the agent's response back to the user
            if agent_reply_text:
                await self.play_tts_audio(agent_reply_text)

    async def play_tts_audio(self, text: str):
        """Synthesizes text using the TTS client and streams it to Asterisk via RTP."""
        if not self.rtp_server:
            self.log.error("Cannot play TTS audio, RTP server is not initialized.")
            return

        self.log.info("Playing TTS for text", text=text)
        try:
            async for audio_chunk in self.tts_client.synthesize_stream(text):
                if self.is_active:
                    self.rtp_server.send_audio(audio_chunk)
                else:
                    self.log.info("Call ended, stopping TTS playback.")
                    break
        except Exception as e:
            self.log.error("Failed during TTS playback", exc_info=e)

    async def cleanup(self):
        """Gracefully cleans up all resources associated with the call."""
        if not self.is_active:
            return # Cleanup already in progress or done
        self.is_active = False
        self.log.info("Initiating cleanup...")

        # Stop all service clients
        cleanup_tasks = [self.tts_client.close(), self.stt_client.stop()]
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        if self.rtp_server:
            self.rtp_server.close()
        
        # Hangup channels
        hangup_tasks = []
        if self.external_media_channel:
            try:
                if self.external_media_channel.id in self.ari_client.channels:
                    hangup_tasks.append(self.external_media_channel.hangup())
            except Exception as e:
                self.log.warning("Could not schedule hangup for external media channel", exc_info=e)

        if self.channel.id in self.ari_client.channels:
            hangup_tasks.append(self.channel.hangup())
        
        if hangup_tasks:
            await asyncio.gather(*hangup_tasks, return_exceptions=True)
        
        self.log.info("Cleanup complete.")