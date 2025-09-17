import asyncio
from types import SimpleNamespace

import pytest

from src.core.call_handler import CallHandler


class DummyNLP:
    def __init__(self, chunks, metadata):
        self._chunks = chunks
        self._metadata = metadata

    async def streaming_process_user_input(self, user_text, history=None):
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk

    def pop_last_stream_result(self):
        return dict(self._metadata)

    def analyze_emotion(self, text: str) -> str:
        return self._metadata.get("emotion", "neutral")


@pytest.mark.asyncio
async def test_streaming_response_flushes_on_punctuation(monkeypatch):
    chunks = ["Xin ", "chao", " ban.", " Rat", " vui duoc", " ho tro."]
    metadata = {"response_text": "", "intent": "continue_conversation", "emotion": "neutral"}
    handler = CallHandler(SimpleNamespace(), SimpleNamespace(), DummyNLP(chunks, metadata))
    handler.stream_chunk_min_chars = 4
    handler.stream_chunk_max_chars = 80
    handler.stream_flush_punct = ".!?"

    played = []

    async def fake_play(channel_id, text, owner_id=None):
        played.append(text)

    handler._play_tts_response = fake_play  # type: ignore

    result = await handler._stream_nlp_response("call-1", "channel-1", "xin chao", [])

    assert played == ["Xin chao ban.", "Rat vui duoc ho tro."]
    assert result["response_text"] == "Xin chao ban. Rat vui duoc ho tro."
    assert result["intent"] == "continue_conversation"


def test_should_flush_stream_chunk_rules():
    handler = CallHandler(SimpleNamespace(), SimpleNamespace(), DummyNLP([], {"emotion": "neutral"}))
    handler.stream_chunk_min_chars = 5
    handler.stream_chunk_max_chars = 10
    handler.stream_flush_punct = ".!?"

    assert not handler._should_flush_stream_chunk("hi")
    assert handler._should_flush_stream_chunk("hello world")
    assert handler._should_flush_stream_chunk("bye.")
    assert handler._should_flush_stream_chunk("ok", is_final=True)
