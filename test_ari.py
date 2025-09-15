# /data/voip-ai-agent/test_ari.py
import anyio
from asyncari import connect

AST_URL = "http://localhost:8088/ari"   # lưu ý có /ari
APP     = "voip-ai-agent"               # khớp dialplan: Stasis("voip-ai-agent")
USER    = "vitalpbx"
PASS    = "zcWGYbNnPer2YUBTg433EMuVs"

async def main():
    async with connect(AST_URL, APP, USER, PASS) as ari:
        print("✅ Connected. Waiting for events...")
        async with ari.on_event('*') as events:
            async for ev in events:
                print("EV:", ev['type'])

anyio.run(main)
