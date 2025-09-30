import asyncio
import logging
import os
import time
from typing import Optional

# Giả định bạn đã có các module này trong codebase
# Nếu tên khác, đổi import cho đúng dự án của bạn
from src.core.stt_module import STTModule
from src.core.tts_module import tts_stream_service_handler  # vẫn giữ cho các lượt thoại sau
from src.core.nlp_module import NLPModule

GREETING_MEDIA = os.getenv("GREETING_MEDIA", "sound:hello-world")
REPROMPT_MEDIA = os.getenv("REPROMPT_MEDIA", "sound:vm-review")  # tạm dùng sound có sẵn
BARGE_IN_DTMF_WINDOW = float(os.getenv("BARGE_IN_DTMF_WINDOW", "2.0"))  # giây đầu cho DTMF
SILENCE_REPROMPT_TIMEOUT = float(os.getenv("SILENCE_REPROMPT_TIMEOUT", "6.0"))  # sau chào

logger = logging.getLogger(__name__)

class CallHandler:
    def __init__(self, ari_client, stt: STTModule, nlp: NLPModule):
        self.ari = ari_client
        self.stt = stt
        self.nlp = nlp

    async def handle_call(self, channel):
        """
        Pipeline mở đầu:
          1) Answer kênh.
          2) Khởi động STT ngay (song song để chuẩn bị barge-in).
          3) Phát lời chào từ file ghi sẵn.
          4) Cho phép DTMF barge-in trong BARGE_IN_DTMF_WINDOW giây đầu.
          5) Nếu im lặng quá SILENCE_REPROMPT_TIMEOUT sau chào -> reprompt 1 lần.
        """
        await self.ari.channels.answer(channelId=channel['id'])
        call_id = channel['id']
        t0 = time.monotonic()
        logger.info("Answered channel=%s", call_id)

        # Tạm thời bỏ qua STT và NLP để test playback cơ bản
        # stt_task = asyncio.create_task(self._start_stt_session(call_id))

        # Phát file WAV ngắn
        logger.info(f"Playing welcome.wav on channel {call_id}")
        try:
            await self.ari.channels.play(channelId=call_id, media='sound:/var/lib/asterisk/sounds/vi/welcome')
        except Exception as e:
            logger.error(f"Error playing welcome.wav on channel {call_id}: {e}")
        
        # Kết thúc cuộc gọi
        logger.info(f"Hanging up channel {call_id}")
        await self.ari.channels.hangup(channelId=call_id)

        # cleanup (nếu stt_task được tạo)
        # await self._safe_cancel(stt_task)

    async def _start_stt_session(self, call_id: str):
        try:
            # tuỳ interface STT của bạn; ví dụ:
            await self.stt.start_session(call_id=call_id)
            logger.info("STT session started for %s", call_id)
        except Exception:
            logger.exception("Failed to start STT session for %s", call_id)

    async def _play_greeting_with_dtmf_bargein(self, channel) -> bool:
        """
        Phát lời chào từ file; nếu nhận DTMF trong cửa sổ BARGE_IN_DTMF_WINDOW -> dừng phát (barge-in).
        Trả về True nếu đã phát xong (không bị lỗi), False nếu gặp lỗi.
        """
        unsub_dtmf = None
        try:
            # Đăng ký nghe DTMF trong cửa sổ đầu
            dtmf_received = asyncio.get_event_loop().create_future()

            async def _dtmf_listener(client, evt):
                if evt.get("type") == "ChannelDtmfReceived" and evt.get("channel", {}).get("id") == channel['id']:
                    if not dtmf_received.done():
                        dtmf_received.set_result(evt["digit"])
            
            unsub_dtmf = self.ari.on_event("ChannelDtmfReceived", _dtmf_listener)

            logger.info("Playing greeting media=%s", GREETING_MEDIA)
            pb = await self.ari.channels.play(channelId=channel['id'], media=GREETING_MEDIA)

            # Chờ hoặc DTMF hoặc PlaybackFinished hoặc timeout nhẹ để tránh kẹt
            done, _ = await asyncio.wait(
                [
                    dtmf_received, # Truyền future trực tiếp
                    asyncio.create_task(self._wait_playback_finished(pb['id'])),
                    asyncio.create_task(asyncio.sleep(BARGE_IN_DTMF_WINDOW)),
                ],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Nếu DTMF đến sớm -> dừng playback (barge-in)
            if dtmf_received.done():
                try:
                    await self.ari.playbacks.stop(playbackId=pb['id'])
                    logger.info("Greeting barged-in by DTMF on channel=%s", channel['id'])
                except Exception:
                    logger.debug("Stopping playback failed (maybe already finished).")
            else:
                # Nếu chưa xong, vẫn đợi playback kết thúc tự nhiên
                try:
                    await self._wait_playback_finished(pb['id'])
                except Exception:
                    pass
            
            return True
        except Exception:
            logger.exception("Greeting playback failed on channel=%s", channel['id'])
            return False
        finally:
            if unsub_dtmf:
                # Giả sử unsub_dtmf là một hàm có thể await
                # await unsub_dtmf()
                pass # Tạm thời pass nếu chưa rõ cách hủy

    async def _wait_playback_finished(self, playback_id: str, timeout: float = 5.0):
        """
        Chờ sự kiện PlaybackFinished cho playback_id (có timeout để không kẹt).
        """
        fut = asyncio.get_event_loop().create_future()
        unsub = None

        async def _pb_listener(client, evt):
            if evt.get("type") == "PlaybackFinished" and evt.get("playback", {}).get("id") == playback_id:
                if not fut.done():
                    fut.set_result(True)

        unsub = self.ari.on_event("PlaybackFinished", _pb_listener)
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        finally:
            if unsub:
                # await unsub()
                pass


    async def _maybe_reprompt_on_silence(self, call_id: str, channel) -> None:
        """
        Nếu sau SILENCE_REPROMPT_TIMEOUT không có partial STT nào -> phát reprompt 1 lần.
        Yêu cầu STT module có counter hoặc callback partial; ở đây dùng hàm giả .has_any_partial(call_id).
        Bạn điều chỉnh theo interface thật của STTModule.
        """
        try:
            await asyncio.sleep(SILENCE_REPROMPT_TIMEOUT)
            if hasattr(self.stt, "has_any_partial") and not await self.stt.has_any_partial(call_id):
                logger.info("No speech after greeting; playing reprompt.")
                try:
                                        await self.ari.channels.play(channelId=channel['id'], media=REPROMPT_MEDIA)
                except Exception:
                    logger.debug("Reprompt playback failed/ignored.")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Reprompt check failed.")

    async def _conversation_loop(self, call_id: str, channel) -> None:
        """
        Vòng thoại chính – giữ nguyên logic cũ của bạn:
        - nhận partial/final từ STT -> NLP -> sinh phản hồi -> TTS/stream
        - nhớ dừng TTS khi phát hiện VAD/double-talk (khi nâng cấp barge-in theo giọng nói)
        """
        # TODO: ghép với phần có sẵn của bạn.
        # Ví dụ khung:
        while True:
            try:
                user_utt = await self.stt.get_next_utterance(call_id)  # blocking theo event queue
                if user_utt is None:
                    logger.info("No more utterances from STT, ending conversation.")
                    break
                
                logger.info(f"User utterance: {user_utt}")
                # reply_text = await self.nlp.run(user_utt)
                # # Phát phản hồi bằng TTS (streaming của bạn)
                # await tts_stream_service_handler(channel['id'], reply_text)
                await asyncio.sleep(2) # Tạm thời sleep để tránh vòng lặp vô hạn

            except asyncio.CancelledError:
                logger.info("Conversation loop cancelled.")
                break
            except Exception:
                logger.exception("Error in conversation loop.")
                break


    async def _safe_cancel(self, task: Optional[asyncio.Task]):
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass