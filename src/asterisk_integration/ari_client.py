import ari
import asyncio
import aiohttp
import logging
from src.config import ARI_URL, ARI_USERNAME, ARI_PASSWORD, ARI_APP_NAME
from src.core.call_handler import AI_Call_Handler

async def on_stasis_start(channels, ev):
    channel = channels[0]
    logging.info(f"Cuộc gọi mới vào ứng dụng: {channel.id}")
    
    handler = AI_Call_Handler(channel)
    channel.on_event('StasisEnd', handler.on_hangup)
    
    asyncio.create_task(handler.handle_call())

async def start_ari_client():
    async with aiohttp.ClientSession() as session:
        client = await ari.connect(ARI_URL, ARI_USERNAME, ARI_PASSWORD, session=session)
        logging.info("Kết nối đến ARI thành công!")

        client.on_channel_event('StasisStart', on_stasis_start)

        logging.info(f"Ứng dụng '{ARI_APP_NAME}' đang chạy và chờ cuộc gọi...")
        await client.run(apps=ARI_APP_NAME)
