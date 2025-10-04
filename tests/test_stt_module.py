import asyncio
from types import SimpleNamespace

import pytest

from app.stt.client import SttClient
from google.cloud import speech_v2


class _FakeResponse:
    def __init__(self, *, results=None, event=None):
        self.results = results or []
        self.speech_event_type = event or speech_v2.StreamingRecognizeResponse.SpeechEventType.SPEECH_EVENT_TYPE_UNSPECIFIED


class _FakeStream:
    def __init__(self, responses):
        self._responses = responses

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._responses:
            raise StopAsyncIteration
        return self._responses.pop(0)


class _FakeSpeechClient:
    def __init__(self, responses):
        self._responses = responses

    async def streaming_recognize(self, requests):
        self._requests = requests
        return _FakeStream(self._responses)


def test_stt_client_triggers_callbacks(monkeypatch):
    asyncio.run(_run_stt_client_triggers_callbacks(monkeypatch))


async def _run_stt_client_triggers_callbacks(monkeypatch):
    transcripts = []
    voice_events = []
    result_event = asyncio.Event()

    async def on_result(text: str, is_final: bool):
        transcripts.append((text, is_final))
        if is_final:
            result_event.set()

    async def on_voice_event(event_name: str):
        voice_events.append(event_name)

    response_list = [
        _FakeResponse(event=speech_v2.StreamingRecognizeResponse.SpeechEventType.SPEECH_ACTIVITY_BEGIN),
        _FakeResponse(
            results=[
                SimpleNamespace(
                    alternatives=[SimpleNamespace(transcript="xin chao")],
                    is_final=True,
                )
            ]
        ),
    ]

    fake_client = _FakeSpeechClient(response_list)
    monkeypatch.setattr(speech_v2, "SpeechAsyncClient", lambda: fake_client)
    monkeypatch.setenv("GOOGLE_PROJECT_ID", "proj")
    monkeypatch.setenv("GOOGLE_RECOGNIZER_ID", "rec")

    stt_client = SttClient(on_result, on_voice_event)

    await stt_client.start()
    await result_event.wait()
    await stt_client.stop()

    assert ("xin chao", True) in transcripts
    assert "SPEECH_ACTIVITY_BEGIN" in voice_events
