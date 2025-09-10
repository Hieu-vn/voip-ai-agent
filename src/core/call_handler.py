import logging
import os
import uuid
import asyncio
from src.services.ai import stt_service, nlp_service, tts_service

class AI_Call_Handler:
    def __init__(self, channel):
        self.channel = channel
        self.call_active = True

    async def handle_call(self):
        pipe_path = None
        fd = -1
        live_recording = None
        
        try:
            await self.channel.answer()
            import logging
import os
import uuid
import asyncio
from src.services.ai import stt_service, nlp_service, tts_service
from src.config import SPEECH_ADAPTATION_CONFIG

class AI_Call_Handler:
    def __init__(self, channel):
        self.channel = channel
        self.call_active = True

    async def handle_call(self):
        pipe_path = None
        fd = -1
        live_recording = None
        
        try:
            await self.channel.answer()
            logging.info(f"[{self.channel.id}] Đã trả lời cuộc gọi.")

            # --- Thiết lập Audio & STT Context ---
            pipe_path = f"/tmp/rec_{self.channel.id}_{uuid.uuid4().hex}.pipe"
            os.mkfifo(pipe_path)
            live_recording = await self.channel.record(format='slin16@16000', name=pipe_path, beep=False, ifExists='overwrite')
            await asyncio.sleep(0.2)
            fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
            logging.info(f"[{self.channel.id}] Đã thiết lập audio stream qua pipe: {pipe_path} (fd: {fd})")

            # Lấy context adaptation mặc định từ config
            adaptation_context = SPEECH_ADAPTATION_CONFIG.get('default', {})
            if adaptation_context:
                logging.info(f"[{self.channel.id}] Áp dụng speech adaptation context: default")

            # --- Bắt đầu cuộc hội thoại ---
            welcome_sound = await tts_service.tts_service_handler("Xin chào, tổng đài AI xin nghe.")
            await self.channel.play(media=welcome_sound)

            while self.call_active:
                final_transcript_for_turn = ""
                
                logging.info(f"[{self.channel.id}] Bắt đầu lắng nghe lượt nói của người dùng...")
                async for stt_result in stt_service.stt_service_handler(
                    audio_fd=fd, 
                    sample_rate=16000, 
                    call_id=self.channel.id,
                    adaptation_config=adaptation_context # Truyền config vào service
                ):
                    if not stt_result.get('is_final'):
                        logging.debug(f"Call Handler [{self.channel.id}]: Interim STT: '{stt_result.get('transcript')}'")
                        continue
                    
                    final_transcript_for_turn = stt_result.get('transcript', '').strip()
                    logging.info(f"Call Handler [{self.channel.id}]: Final STT: '{final_transcript_for_turn}'")
                    break 

                if not final_transcript_for_turn or not self.call_active:
                    logging.warning(f"[{self.channel.id}] Không nhận được transcript cuối cùng hoặc người dùng đã gác máy.")
                    break

                response_obj = await nlp_service.nlp_agent_handler(final_transcript_for_turn)
                
                if not self.call_active: 
                    break

                response_sound = await tts_service.tts_service_handler(response_obj['response_text'])
                await self.channel.play(media=response_sound)

                if response_obj['intent'] == 'end_conversation':
                    logging.info(f"[{self.channel.id}] NLP xác nhận kết thúc cuộc gọi. Kết thúc hội thoại.")
                    break

        except Exception as e:
            logging.error(f"[{self.channel.id}] Lỗi nghiêm trọng khi xử lý cuộc gọi: {e}", exc_info=True)
        finally:
            # --- Dọn dẹp tài nguyên ---
            logging.info(f"[{self.channel.id}] Bắt đầu dọn dẹp tài nguyên cuộc gọi...")
            if live_recording: await live_recording.stop()
            if fd != -1: os.close(fd)
            if pipe_path and os.path.exists(pipe_path): os.remove(pipe_path)
            if self.call_active: await self.channel.hangup()

    def on_hangup(self, channel, ev):
        logging.info(f"[{channel.id}] Người dùng đã gác máy.")
        self.call_active = False


            # --- Bước 1: Thiết lập Named Pipe để nhận audio stream ---
            # Tạo một đường dẫn file pipe duy nhất
            pipe_path = f"/tmp/rec_{self.channel.id}_{uuid.uuid4().hex}.pipe"
            os.mkfifo(pipe_path)
            logging.info(f"[{self.channel.id}] Đã tạo named pipe: {pipe_path}")

            # Yêu cầu Asterisk ghi âm vào pipe với định dạng slin16@16000 (PCM 16-bit, 16kHz)
            # Asterisk sẽ tự động chuyển mã, chúng ta không cần dùng pydub ở đây nữa
            live_recording = await self.channel.record(format='slin16@16000', name=pipe_path, beep=False, ifExists='overwrite')
            logging.info(f"[{self.channel.id}] Đã yêu cầu Asterisk bắt đầu ghi âm vào pipe.")

            # Mở pipe để đọc (non-blocking). Cần một chút delay để Asterisk thực sự bắt đầu ghi.
            await asyncio.sleep(0.2) 
            fd = os.open(pipe_path, os.O_RDONLY | os.O_NONBLOCK)
            logging.info(f"[{self.channel.id}] Đã mở named pipe với file descriptor: {fd}")

            # --- Bước 2: Bắt đầu cuộc hội thoại ---
            welcome_sound = await tts_service.tts_service_handler("Xin chào, tổng đài AI xin nghe.")
            await self.channel.play(media=welcome_sound)

            while self.call_active:
                final_transcript_for_turn = ""
                
                # Vòng lặp con: xử lý một lượt nói (utterance) bằng cách tiêu thụ stream từ STT
                logging.info(f"[{self.channel.id}] Bắt đầu lắng nghe lượt nói của người dùng...")
                async for stt_result in stt_service.stt_service_handler(
                    audio_fd=fd, 
                    sample_rate=16000, 
                    call_id=self.channel.id
                ):
                    # Log kết quả tạm thời ở mức DEBUG
                    if not stt_result.get('is_final'):
                        logging.debug(f"Call Handler [{self.channel.id}]: Interim STT: '{stt_result.get('transcript')}'")
                        continue
                    
                    # Đã có kết quả cuối cùng cho lượt nói này
                    final_transcript_for_turn = stt_result.get('transcript', '').strip()
                    # Log transcript cuối cùng ở mức INFO để dễ theo dõi flow chính
                    logging.info(f"Call Handler [{self.channel.id}]: Final STT: '{final_transcript_for_turn}'")
                    # Thoát khỏi vòng lặp utterance để xử lý kết quả cuối cùng
                    break 

                # Kiểm tra xem có nhận được gì không sau khi kết thúc lượt nói
                if not final_transcript_for_turn or not self.call_active:
                    logging.warning(f"[{self.channel.id}] Không nhận được transcript cuối cùng hoặc người dùng đã gác máy.")
                    # Thay vì kết thúc ngay, có thể thêm logic hỏi lại "bạn có còn ở đó không?"
                    break

                # Đã có transcript cuối cùng, gửi đến NLP để xử lý
                response_obj = await nlp_service.nlp_agent_handler(final_transcript_for_turn)
                
                if not self.call_active: 
                    break

                # Phát phản hồi từ AI
                response_sound = await tts_service.tts_service_handler(response_obj['response_text'])
                await self.channel.play(media=response_sound)

                # Kiểm tra ý định để quyết định có tiếp tục vòng lặp hội thoại không
                if response_obj['intent'] == 'end_conversation':
                    logging.info(f"[{self.channel.id}] NLP xác nhận ý định kết thúc cuộc gọi. Kết thúc hội thoại.")
                    break

        except Exception as e:
            logging.error(f"[{self.channel.id}] Lỗi nghiêm trọng khi xử lý cuộc gọi: {e}", exc_info=True)
        finally:
            # --- Bước 3: Dọn dẹp tài nguyên ---
            logging.info(f"[{self.channel.id}] Bắt đầu dọn dẹp tài nguyên cuộc gọi...")
            
            if live_recording:
                try:
                    await live_recording.stop()
                    logging.info(f"[{self.channel.id}] Đã dừng ghi âm trên Asterisk.")
                except Exception as e:
                    logging.error(f"[{self.channel.id}] Lỗi khi dừng ghi âm: {e}")

            if fd != -1:
                os.close(fd)
                logging.info(f"[{self.channel.id}] Đã đóng file descriptor.")
            
            if pipe_path and os.path.exists(pipe_path):
                os.remove(pipe_path)
                logging.info(f"[{self.channel.id}] Đã xóa named pipe.")

            if self.call_active:
                logging.info(f"[{self.channel.id}] Gác máy.")
                await self.channel.hangup()

    def on_hangup(self, channel, ev):
        logging.info(f"[{channel.id}] Người dùng đã gác máy.")
        self.call_active = False
