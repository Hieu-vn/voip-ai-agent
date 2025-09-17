import asyncio

from src.core.stt_module import STTModule


def test_stt_partial_callback_invoked(monkeypatch):
    events = []

    class FakeClient:
        def __init__(self, language_code: str, sample_rate_hz: int):
            pass

        async def streaming_recognize_generator(self, audio_queue, call_id, adaptation_config, timeout=120):
            yield {"transcript": "xin chao", "is_final": False}
            yield {"transcript": "xin chao", "is_final": True}

    monkeypatch.setattr("src.core.stt_module.STTGoogleCloudClient", FakeClient)

    async def scenario():
        stt = STTModule(language_code="vi-VN")
        await stt.start_session("call-1")

        partial_event = asyncio.Event()

        async def partial_cb(transcript: str):
            events.append(transcript)
            partial_event.set()

        stt.register_partial_callback("call-1", partial_cb)

        await partial_event.wait()
        final_text = await stt.get_next_utterance("call-1")
        await stt.stop_session("call-1")
        return final_text

    final_transcript = asyncio.run(scenario())

    assert events and events[0] == "xin chao"
    assert final_transcript == "xin chao"
