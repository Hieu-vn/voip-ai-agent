import logging
from src.services.ai import stt_service, nlp_service, tts_service

class AI_Call_Handler:
    def __init__(self, channel):
        self.channel = channel
        self.call_active = True

    async def handle_call(self):
        try:
            await self.channel.answer()
            logging.info(f"[{self.channel.id}] Đã trả lời cuộc gọi.")

            welcome_sound = await tts_service.tts_service_handler("Xin chào, tổng đài AI xin nghe.")
            await self.channel.play(media=welcome_sound)

            while self.call_active:
                transcript = await stt_service.stt_service_handler(None)

                if not transcript or not self.call_active:
                    break

                response_text = await nlp_service.nlp_agent_handler(transcript)
                response_sound = await tts_service.tts_service_handler(response_text)
                
                if not self.call_active: break
                await self.channel.play(media=response_sound)
                break

        except Exception as e:
            logging.error(f"[{self.channel.id}] Lỗi khi xử lý cuộc gọi: {e}", exc_info=True)
        finally:
            if self.call_active:
                logging.info(f"[{self.channel.id}] Kết thúc cuộc gọi.")
                await self.channel.hangup()

    def on_hangup(self, channel, ev):
        logging.info(f"[{channel.id}] Người dùng đã gác máy.")
        self.call_active = False
