import asyncio, os, json, structlog
from aiohttp import web, ClientSession
from google.cloud import speech_v1p1beta1 as speech
from app.nlu.agent import Agent
from app.audio.stream import AsteriskStream
from app.tts.client import TTSClient

log = structlog.get_logger()

async def handle_call(request: web.Request):
    chan_id = request.query.get("channel")
    stream = AsteriskStream(chan_id, ari_base=os.getenv("ARI_URL"))
    tts = TTSClient(base_url="http://127.0.0.1:8001")
    agent = await Agent.create()
    stt_client = speech.SpeechClient()

    # Bắt đầu pull RTP và đẩy STT streaming
    async with stream.audio_reader() as pcm_iter:
        stt_stream = stt_client.streaming_recognize(
            requests=speech.StreamingRecognizeRequest(
                streaming_config=speech.StreamingRecognitionConfig(
                    config=speech.RecognitionConfig(
                        language_code="vi-VN",
                        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                        sample_rate_hertz=8000,
                        enable_automatic_punctuation=True,
                    ),
                    interim_results=True,
                    single_utterance=False,
                )
            )
        )

        async def feed_stt():
            async for chunk in pcm_iter:
                await stt_stream.write(speech.StreamingRecognizeRequest(audio_content=chunk))
            await stt_stream.done_writing()

        async def consume_stt():
            async for resp in stt_stream:
                for r in resp.results:
                    if r.is_final or r.stability > 0.8:
                        text = r.alternatives[0].transcript.strip()
                        if text:
                            reply = await agent.respond(text)  # LangGraph + MCP lookup
                            wav = await tts.synthesize(reply, spk="female_vi", rate=1.05)
                            await stream.play_wav(wav)

        await asyncio.gather(feed_stt(), consume_stt())

    return web.Response(text="OK")

app = web.Application()
app.router.add_post("/call", handle_call)

if __name__ == "__main__":
    web.run_app(app, host="0.0.0.0", port=8000)
