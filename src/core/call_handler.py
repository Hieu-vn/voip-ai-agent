import asyncio
import logging
import math
import os
import struct
import time
from collections import defaultdict
from pathlib import Path
from typing import Awaitable, Callable, DefaultDict, Dict, List, Optional, Set, Tuple

from langchain_core.messages import AIMessage, BaseMessage

from src.config import SPEECH_ADAPTATION_CONFIG
from src.core.nlp_module import NLPModule
from src.core.stt_module import STTModule
from src.core.tts_module import tts_service_handler
from src.evaluation.tracker import evaluation_tracker
from src.utils import guardrails as guardrails_utils

GREETING_MEDIA = os.getenv("GREETING_MEDIA", "sound:hello-world")
REPROMPT_MEDIA = os.getenv("REPROMPT_MEDIA", "sound:vm-review")
BARGE_IN_DTMF_WINDOW = float(os.getenv("BARGE_IN_DTMF_WINDOW", "2.0"))
SILENCE_REPROMPT_TIMEOUT = float(os.getenv("SILENCE_REPROMPT_TIMEOUT", "6.0"))
GUARDRAIL_FALLBACK_MESSAGE = os.getenv(
    "GUARDRAIL_FALLBACK_MESSAGE",
    "Xin loi, toi khong the ho tro noi dung nay. Toi se chuyen ban toi nhan vien ho tro.",
)
DEFAULT_PLAYBACK_TIMEOUT = float(os.getenv("PLAYBACK_TIMEOUT_SECONDS", "15.0"))

logger = logging.getLogger(__name__)


class EnergyVAD:
    """Simple RMS-based voice activity detector."""

    def __init__(self, rms_threshold: float, activation_frames: int, release_frames: int) -> None:
        self.rms_threshold = rms_threshold
        self.activation_frames = activation_frames
        self.release_frames = release_frames
        self.active_frames = 0
        self.silence_frames = release_frames
        self.triggered = False

    def add_chunk(self, chunk: bytes) -> bool:
        if not chunk or len(chunk) < 2:
            return False
        if len(chunk) % 2:
            chunk = chunk[:-1]
        total_sq = 0.0
        count = 0
        for (sample,) in struct.iter_unpack('<h', chunk):
            total_sq += float(sample) * float(sample)
            count += 1
        if not count:
            return False
        rms = math.sqrt(total_sq / count)
        if rms >= self.rms_threshold:
            self.active_frames += 1
            self.silence_frames = 0
        else:
            self.silence_frames += 1
            if self.active_frames > 0:
                self.active_frames -= 1
        if not self.triggered and self.active_frames >= self.activation_frames:
            self.triggered = True
            self.silence_frames = 0
            return True
        if self.triggered and self.silence_frames >= self.release_frames:
            self.triggered = False
            self.active_frames = 0
        return False


class RTPAudioForwarder(asyncio.DatagramProtocol):
    def __init__(
        self,
        call_id: str,
        stt: STTModule,
        speech_hook: Optional[Callable[[str], Awaitable[None]]] = None,
        vad: Optional[EnergyVAD] = None,
    ) -> None:
        self.call_id = call_id
        self.stt = stt
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.speech_hook = speech_hook
        self.vad = vad

    def connection_made(self, transport: asyncio.transports.DatagramTransport) -> None:
        self.loop = asyncio.get_running_loop()

    def datagram_received(self, data: bytes, addr) -> None:
        payload = self._extract_payload(data)
        if not payload:
            return
        loop = self.loop or asyncio.get_running_loop()
        loop.create_task(self.stt.push_audio_chunk(self.call_id, payload))
        if self.vad:
            try:
                triggered = self.vad.add_chunk(payload)
            except Exception:
                logger.exception("VAD processing failed on call %s", self.call_id)
            else:
                if triggered and self.speech_hook:
                    loop.create_task(self.speech_hook(self.call_id))

    def error_received(self, exc: Exception) -> None:
        logger.error("External media UDP error on %s: %s", self.call_id, exc)

    @staticmethod
    def _extract_payload(packet: bytes) -> bytes:
        if len(packet) < 12:
            return b""
        first_byte = packet[0]
        version = first_byte >> 6
        if version != 2:
            return b""
        csrc_count = first_byte & 0x0F
        extension_flag = (first_byte >> 4) & 0x01
        padding_flag = (first_byte >> 5) & 0x01
        header_len = 12 + (csrc_count * 4)
        if len(packet) < header_len:
            return b""
        if extension_flag:
            if len(packet) < header_len + 4:
                return b""
            ext_length = struct.unpack_from('!H', packet, header_len + 2)[0]
            header_len += 4 + ext_length * 4
            if len(packet) < header_len:
                return b""
        payload = packet[header_len:]
        if padding_flag and payload:
            padding = payload[-1]
            if padding < len(payload):
                payload = payload[:-padding]
            else:
                payload = b""
        return payload


class CallHandler:
    def __init__(self, ari_client, stt: STTModule, nlp: NLPModule) -> None:
        self.ari = ari_client
        self.stt = stt
        self.nlp = nlp
        self.app_name = os.getenv("ARI_APP_NAME", "voip-ai-agent")
        self.external_media_host = os.getenv("EXTERNAL_MEDIA_HOST", "127.0.0.1")
        self.external_media_bind = os.getenv("EXTERNAL_MEDIA_BIND_IP", "0.0.0.0")
        self.external_media_format = os.getenv("EXTERNAL_MEDIA_FORMAT", "slin")
        self.language_code = os.getenv("LANGUAGE_CODE", "vi-VN")
        self.tts_speaker_wav = os.getenv("TTS_SPEAKER_WAV")
        self.tts_language = (
            self.language_code.split("-")[0] if "-" in self.language_code else self.language_code
        )
        self.speech_adaptation_cfg = SPEECH_ADAPTATION_CONFIG.get("default", {})
        self._active_playbacks: DefaultDict[str, Set[str]] = defaultdict(set)
        self._playback_monitors: Dict[str, asyncio.Task] = {}
        self.vad_enabled = os.getenv("VAD_BARGE_IN_ENABLED", "1") != "0"
        self.vad_rms_threshold = float(os.getenv("VAD_RMS_THRESHOLD", os.getenv("VAD_ENERGY_THRESHOLD", "700")))
        self.vad_activation_frames = int(os.getenv("VAD_ACTIVATION_FRAMES", "3"))
        self.vad_release_frames = int(os.getenv("VAD_RELEASE_FRAMES", "10"))
        self.stream_chunk_min_chars = int(os.getenv("STREAM_CHUNK_MIN_CHARS", "40"))
        self.stream_chunk_max_chars = int(os.getenv("STREAM_CHUNK_MAX_CHARS", "90"))
        self.stream_flush_punct = os.getenv("STREAM_FLUSH_PUNCT", ".,!?;:")
        self.playback_timeout = float(os.getenv("PLAYBACK_TIMEOUT_SECONDS", str(DEFAULT_PLAYBACK_TIMEOUT)))

    async def handle_call(self, channel) -> None:
        call_id = channel["id"]
        start_ts = time.monotonic()
        logger.info("Channel %s entered CallHandler", call_id)

        stt_started = False
        external_media: Optional[Tuple[str, asyncio.transports.DatagramTransport]] = None
        silence_task: Optional[asyncio.Task] = None

        try:
            await self.ari.channels.answer(channelId=call_id)
            logger.info("Answered channel=%s", call_id)

            await self.stt.start_session(
                call_id=call_id,
                sample_rate=8000,
                adaptation_config=self.speech_adaptation_cfg,
            )
            stt_started = True

            async def _partial_callback(_transcript: str) -> None:
                await self._on_user_speech_detected(call_id, _transcript)

            self.stt.register_partial_callback(call_id, _partial_callback)

            vad = self._create_vad()
            external_media = await self._attach_external_media(call_id, vad)

            await self._play_greeting_with_dtmf_bargein(channel)

            silence_task = asyncio.create_task(self._maybe_reprompt_on_silence(call_id, channel))

            await self._conversation_loop(call_id, channel)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Pipeline error on call %s: %s", call_id, exc, exc_info=True)
        finally:
            await self._safe_cancel(silence_task)
            await self._stop_active_playback(call_id, reason="call ending")
            await self._teardown_external_media(external_media)
            if stt_started:
                try:
                    await self.stt.push_audio_chunk(call_id, None)
                except Exception:
                    logger.debug("Failed to push STT sentinel for %s", call_id, exc_info=True)
                try:
                    await self.stt.stop_session(call_id)
                except Exception:
                    logger.debug("Failed to stop STT session for %s", call_id, exc_info=True)
            try:
                await self._hangup_channel(call_id)
            except Exception:
                pass
            logger.info("Call %s finished in %.2fs", call_id, time.monotonic() - start_ts)

    def _create_vad(self) -> Optional[EnergyVAD]:
        if not self.vad_enabled:
            return None
        return EnergyVAD(
            rms_threshold=self.vad_rms_threshold,
            activation_frames=self.vad_activation_frames,
            release_frames=self.vad_release_frames,
        )

    async def _attach_external_media(
        self, call_id: str, vad: Optional[EnergyVAD]
    ) -> Optional[Tuple[str, asyncio.transports.DatagramTransport]]:
        loop = asyncio.get_running_loop()
        speech_hook = self._handle_vad_trigger if vad else None
        transport, _ = await loop.create_datagram_endpoint(
            lambda: RTPAudioForwarder(call_id, self.stt, speech_hook=speech_hook, vad=vad),
            local_addr=(self.external_media_bind, 0),
        )
        bind_host, bind_port = transport.get_extra_info("sockname")
        target_host = self.external_media_host or bind_host

        logger.info(
            "Requesting external media for call %s to %s:%s", call_id, target_host, bind_port
        )
        try:
            response = await self.ari.channels.externalMedia(
                channelId=call_id,
                app=self.app_name,
                external_host=f"{target_host}:{bind_port}",
                format=self.external_media_format,
                direction="in",
                transport="udp",
            )
        except Exception:
            transport.close()
            raise

        media_channel_id = None
        if isinstance(response, dict):
            media_channel_id = response.get("id") or response.get("channel", {}).get("id")
        logger.info("External media channel %s established for call %s", media_channel_id, call_id)
        return (media_channel_id, transport)

    async def _teardown_external_media(
        self, media: Optional[Tuple[str, asyncio.transports.DatagramTransport]]
    ) -> None:
        if not media:
            return
        media_channel_id, transport = media
        if transport:
            transport.close()
        if media_channel_id:
            try:
                await self.ari.channels.hangup(channelId=media_channel_id)
            except Exception as exc:
                logger.debug(
                    "Failed to hang up external media channel %s: %s", media_channel_id, exc
                )

    def _should_flush_stream_chunk(self, text: str, is_final: bool = False) -> bool:
        if not text:
            return False
        stripped = text.rstrip()
        if is_final:
            return True
        if any(stripped.endswith(p) for p in self.stream_flush_punct):
            return True
        if len(stripped) >= self.stream_chunk_max_chars:
            return True
        return len(stripped) >= self.stream_chunk_min_chars

    def _register_playback(self, owner: str, playback_id: Optional[str]) -> None:
        if not playback_id:
            return
        self._active_playbacks[owner].add(playback_id)
        self._playback_monitors[playback_id] = asyncio.create_task(
            self._monitor_playback_end(owner, playback_id)
        )

    async def _stream_nlp_response(
        self,
        call_id: str,
        channel_id: str,
        user_text: str,
        history: List[BaseMessage],
        turn_start: float,
    ) -> Dict:
        chunks: List[str] = []
        pending_chunk: List[str] = []
        metadata: Dict[str, Optional[object]] = {}
        guardrail_violations: Dict[str, List[str]] = {}
        first_chunk_time: Optional[float] = None

        async for token in self.nlp.streaming_process_user_input(user_text, history=history):
            if token is None:
                continue
            chunks.append(token)
            pending_chunk.append(token)
            current_text = ''.join(chunks).strip()
            is_safe, violations = guardrails_utils.is_response_safe(current_text)
            if not is_safe:
                guardrail_violations = violations
                await self._stop_active_playback(call_id, reason="guardrail violation")
                await self._play_tts_response(channel_id, GUARDRAIL_FALLBACK_MESSAGE, owner_id=call_id)
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                metadata.update({
                    "response_text": GUARDRAIL_FALLBACK_MESSAGE,
                    "intent": "handoff_to_agent",
                    "emotion": "neutral",
                    "guardrail_violations": violations,
                })
                break

            partial_text = ''.join(pending_chunk).strip()
            if self._should_flush_stream_chunk(partial_text):
                await self._play_tts_response(
                    channel_id,
                    guardrails_utils.sanitize_response(partial_text),
                    owner_id=call_id,
                )
                if first_chunk_time is None:
                    first_chunk_time = time.monotonic()
                pending_chunk.clear()

        else:
            meta = self.nlp.pop_last_stream_result()
            if meta:
                metadata.update(meta)
            full_text = ''.join(chunks).strip()
            if pending_chunk:
                remaining = ''.join(pending_chunk).strip()
                if remaining:
                    await self._play_tts_response(
                        channel_id,
                        guardrails_utils.sanitize_response(remaining),
                        owner_id=call_id,
                    )
                    if first_chunk_time is None:
                        first_chunk_time = time.monotonic()
                pending_chunk.clear()
            metadata.setdefault("response_text", full_text)
            metadata.setdefault("intent", "continue_conversation")
            metadata.setdefault("emotion", self.nlp.analyze_emotion(user_text))

        metadata.setdefault("response_text", ''.join(chunks).strip())
        safe_final, violations = guardrails_utils.is_response_safe(metadata["response_text"] or "")
        if not safe_final:
            metadata["response_text"] = GUARDRAIL_FALLBACK_MESSAGE
            metadata["intent"] = "handoff_to_agent"
            metadata["emotion"] = "neutral"
            metadata["guardrail_violations"] = violations
            await self._stop_active_playback(call_id, reason="guardrail violation-final")
            await self._play_tts_response(
                channel_id,
                GUARDRAIL_FALLBACK_MESSAGE,
                owner_id=call_id,
            )
            if first_chunk_time is None:
                first_chunk_time = time.monotonic()

        if guardrail_violations and "guardrail_violations" not in metadata:
            metadata["guardrail_violations"] = guardrail_violations

        if first_chunk_time is not None:
            metadata["latency_ms"] = int((first_chunk_time - turn_start) * 1000)
        else:
            metadata.setdefault("latency_ms", None)
        return metadata

    async def _play_tts_response(self, channel_id: str, text: str, owner_id: Optional[str] = None) -> None:
        if not text.strip():
            return
        speaker_path = Path(self.tts_speaker_wav) if self.tts_speaker_wav else None
        try:
            if speaker_path and speaker_path.exists():
                audio_path = await tts_service_handler(
                    text=text,
                    speaker_wav_path=str(speaker_path),
                    language=self.tts_language,
                )
            else:
                if speaker_path and not speaker_path.exists():
                    logger.warning("TTS speaker WAV not found; using default voice")
                audio_path = await tts_service_handler(
                    text=text,
                    language=self.tts_language,
                )
        except Exception as exc:
            logger.error("TTS synthesis failed: %s", exc, exc_info=True)
            return
        if not audio_path:
            logger.warning("TTS synthesis returned no audio, skipping playback.")
            return
        media_uri = Path(audio_path).resolve().as_uri()
        try:
            playback = await self.ari.channels.play(channelId=channel_id, media=media_uri)
        except Exception as exc:
            logger.error("Failed to play TTS audio on channel %s: %s", channel_id, exc, exc_info=True)
            return

        playback_id = playback.get("id") if isinstance(playback, dict) else None
        owner = owner_id or channel_id
        self._register_playback(owner, playback_id)

    async def _monitor_playback_end(self, owner_id: str, playback_id: str) -> None:
        try:
            await self._wait_playback_finished(playback_id, timeout=self.playback_timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("Playback monitor error for %s: %s", playback_id, exc)
        finally:
            self._playback_monitors.pop(playback_id, None)
            owner_set = self._active_playbacks.get(owner_id)
            if owner_set:
                owner_set.discard(playback_id)
                if not owner_set:
                    self._active_playbacks.pop(owner_id, None)

    async def _stop_active_playback(self, owner_id: str, reason: str) -> None:
        playback_ids = list(self._active_playbacks.pop(owner_id, set()))
        if not playback_ids:
            return
        for playback_id in playback_ids:
            monitor = self._playback_monitors.pop(playback_id, None)
            if monitor:
                monitor.cancel()
            try:
                await self.ari.playbacks.stop(playbackId=playback_id)
                logger.info("Stopped playback %s for %s (%s)", playback_id, owner_id, reason)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Unable to stop playback %s: %s", playback_id, exc)

    async def _handle_vad_trigger(self, call_id: str) -> None:
        await self._stop_active_playback(call_id, reason="VAD speech detected")

    async def _on_user_speech_detected(self, call_id: str, transcript: str) -> None:
        await self._stop_active_playback(call_id, reason="caller speech detected")

    async def _hangup_channel(self, call_id: str) -> None:
        await self.ari.channels.hangup(channelId=call_id)

    async def _play_greeting_with_dtmf_bargein(self, channel) -> bool:
        unsub_dtmf = None
        playback_task: Optional[asyncio.Task] = None
        owner_id = f"{channel['id']}:greeting"
        try:
            dtmf_received = asyncio.get_running_loop().create_future()

            async def _dtmf_listener(client, evt):
                if evt.get("type") == "ChannelDtmfReceived" and evt.get("channel", {}).get("id") == channel["id"]:
                    if not dtmf_received.done():
                        dtmf_received.set_result(evt["digit"])

            unsub_dtmf = self.ari.on_event("ChannelDtmfReceived", _dtmf_listener)

            logger.info("Playing greeting media=%s", GREETING_MEDIA)
            playback = await self.ari.channels.play(channelId=channel["id"], media=GREETING_MEDIA)
            playback_id = playback.get("id") if isinstance(playback, dict) else None
            self._register_playback(owner_id, playback_id)

            if playback_id:
                playback_task = asyncio.create_task(self._wait_playback_finished(playback_id))
                wait_set = {dtmf_received, playback_task}
            else:
                wait_set = {dtmf_received}

            done, pending = await asyncio.wait(wait_set, return_when=asyncio.FIRST_COMPLETED)

            if dtmf_received in done:
                await self._stop_active_playback(owner_id, reason="greeting dtmf")
            else:
                try:
                    await self._wait_playback_finished(playback_id)
                except Exception:
                    pass

            for task in pending:
                task.cancel()
            return True
        except Exception:
            logger.exception("Greeting playback failed on channel=%s", channel["id"])
            return False
        finally:
            if playback_task:
                playback_task.cancel()
            if unsub_dtmf:
                try:
                    result = unsub_dtmf()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.debug("Failed to unsubscribe DTMF listener", exc_info=True)
            await self._stop_active_playback(owner_id, reason="greeting finished")

    async def _wait_playback_finished(self, playback_id: Optional[str], timeout: Optional[float] = None) -> None:
        if not playback_id:
            return
        fut = asyncio.get_running_loop().create_future()
        unsub = None

        async def _pb_listener(client, evt):
            if evt.get("type") == "PlaybackFinished" and evt.get("playback", {}).get("id") == playback_id:
                if not fut.done():
                    fut.set_result(True)

        unsub = self.ari.on_event("PlaybackFinished", _pb_listener)
        try:
            await asyncio.wait_for(fut, timeout=timeout or self.playback_timeout)
        finally:
            if unsub:
                try:
                    result = unsub()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception:
                    logger.debug("Failed to unsubscribe playback listener", exc_info=True)

    async def _maybe_reprompt_on_silence(self, call_id: str, channel) -> None:
        try:
            await asyncio.sleep(SILENCE_REPROMPT_TIMEOUT)
            if hasattr(self.stt, "has_any_partial") and await self.stt.has_any_partial(call_id):
                logger.debug("Skipping reprompt for %s because speech was detected", call_id)
                return
            logger.info("No speech after greeting; playing reprompt.")
            playback = await self.ari.channels.play(channelId=channel["id"], media=REPROMPT_MEDIA)
            playback_id = playback.get("id") if isinstance(playback, dict) else None
            self._register_playback(call_id, playback_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Reprompt playback failed.")

    async def _conversation_loop(self, call_id: str, channel) -> None:
        history: List[BaseMessage] = []
        turn_index = 0
        while True:
            try:
                user_utt = await self.stt.get_next_utterance(call_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to retrieve utterance for %s", call_id)
                break

            if user_utt is None:
                logger.info("No more utterances from STT, ending conversation for %s.", call_id)
                break

            logger.info("User[%s]: %s", call_id, user_utt)
            turn_start = time.monotonic()

            try:
                nlp_result = await self._stream_nlp_response(
                    call_id,
                    channel["id"],
                    user_utt,
                    history,
                    turn_start,
                )
            except Exception as exc:
                logger.error("NLP streaming error on call %s: %s", call_id, exc, exc_info=True)
                continue

            response_text = (nlp_result.get("response_text") or "").strip()
            intent = nlp_result.get("intent")

            if response_text:
                history.append(AIMessage(content=response_text))

            metadata_for_log = dict(nlp_result)
            metadata_for_log.setdefault("latency_ms", None)
            evaluation_tracker.log_turn(
                session_id=call_id,
                turn_index=turn_index,
                user_text=user_utt,
                bot_text=response_text,
                metadata=metadata_for_log,
            )
            turn_index += 1

            if intent == "end_conversation":
                logger.info("End intent detected for %s", call_id)
                break

    async def _safe_cancel(self, task: Optional[asyncio.Task]):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
