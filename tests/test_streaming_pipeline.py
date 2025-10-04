import asyncio
from types import SimpleNamespace

import pytest

from app.audio.stream import CallHandler


class FakeRtpServer:
    def __init__(self):
        self.sent = []

    def send_audio(self, chunk: bytes):
        self.sent.append(chunk)

    def close(self):
        pass


class FakeTTSClient:
    def __init__(self, started_event: asyncio.Event | None = None):
        self.requested_texts = []
        self._started_event = started_event

    async def synthesize_stream(self, text: str):
        self.requested_texts.append(text)
        if self._started_event and not self._started_event.is_set():
            self._started_event.set()
        # Yield a few chunks to simulate streaming audio
        for chunk in (b"chunk-1", b"chunk-2", b"chunk-3"):
            await asyncio.sleep(0)
            yield chunk

    async def close(self):
        pass


class FakeSttClient:
    def __init__(self):
        self._result_callback = None
        self._voice_callback = None

    def set_result_callback(self, callback):
        self._result_callback = callback

    def set_voice_event_callback(self, callback):
        self._voice_callback = callback

    async def start(self):
        pass

    async def stop(self):
        pass

    def write(self, _: bytes):
        pass


def _fake_channel():
    return SimpleNamespace(id="call-1", client=None)


def _fake_ari_client():
    # Minimal structure to satisfy cleanup logic
    return SimpleNamespace(channels={})


def test_call_handler_stops_tts_when_voice_activity_detected():
    asyncio.run(_run_call_handler_stops_tts())


async def _run_call_handler_stops_tts():
    rtp_server = FakeRtpServer()
    started_event = asyncio.Event()
    tts_client = FakeTTSClient(started_event)
    stt_client = FakeSttClient()

    handler = CallHandler(
        _fake_ari_client(),
        _fake_channel(),
        tts_client=tts_client,
        stt_client=stt_client,
    )
    handler.rtp_server = rtp_server

    playback_task = asyncio.create_task(handler.play_tts_audio("xin chao"))
    await started_event.wait()
    await asyncio.sleep(0)

    await handler.on_stt_voice_event("SPEECH_ACTIVITY_BEGIN")
    await asyncio.sleep(0)

    await handler.stop_tts_playback()
    await playback_task

    assert tts_client.requested_texts == ["xin chao"]
    assert handler._current_tts_task is None
    assert rtp_server.sent  # audio was streamed before cancellation


def test_stop_tts_playback_is_idempotent():
    asyncio.run(_run_stop_tts_idempotent())


async def _run_stop_tts_idempotent():
    handler = CallHandler(
        _fake_ari_client(),
        _fake_channel(),
        tts_client=FakeTTSClient(),
        stt_client=FakeSttClient(),
    )
    handler.rtp_server = FakeRtpServer()

    await handler.stop_tts_playback()  # No-op when nothing is playing
    assert handler._current_tts_task is None
