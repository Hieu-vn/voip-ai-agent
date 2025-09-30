# /data/voip-ai-agent/test_ws.py
import asyncio, aiohttp, base64

AST_HOST = "http://localhost:8088"  # không có /ari ở cuối
APP      = "voip-ai-agent"
USER     = "vitalpbx"
PASS     = "wqpmDW40gfXc8FNnvRwM0zEbFNXZIF85"

async def main():
    api_key = f"{USER}:{PASS}"
    ws_url  = f"{AST_HOST}/ari/events?api_key={api_key}&app={APP}"
    # Nếu bạn muốn chỉ nhận event cho 1 kênh: &subscribeAll=false
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(ws_url) as ws:
            print("✅ WS connected. App is now registered. Waiting for events…")
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    print("EV:", msg.data[:500])
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print("WS error:", msg)
                    break

if __name__ == "__main__":
    asyncio.run(main())
