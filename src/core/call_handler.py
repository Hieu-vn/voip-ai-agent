import logging
import os
import uuid
import asyncio
from src.config import SPEECH_ADAPTATION_CONFIG
from src.services.ai import nlp_service, stt_service, tts_service
from src.utils.tracing import tracer

class AI_Call_Handler:
    def __init__(self, channel):
        self.channel = channel
        self.call_active = True
        self.active_tasks = set()

    async def _cancel_all_tasks(self):
        if not self.active_tasks: return
        logging.info(f"[{self.channel.id}] Cancelling {len(self.active_tasks)} tasks...")
        for task in list(self.active_tasks):
            if not task.done(): task.cancel()
        await asyncio.gather(*self.active_tasks, return_exceptions=True)

    def on_hangup(self, channel, ev):
        logging.info(f"[{self.channel.id}] User hung up.")
        self.call_active = False
        asyncio.create_task(self._cancel_all_tasks())

    async def _conversation_loop(self, stt_fd: int, adaptation_context: dict):
        history = []
        with tracer.start_as_current_span("welcome_playback") as span:
            welcome_path = await tts_service.tts_service_handler("Xin chào, tổng đài AI xin nghe.")
            if welcome_path and self.call_active:
                span.add_event("Playing welcome message")
                await self.channel.play(media=f"sound:{os.path.basename(welcome_path).replace('.wav', '')}")

        while self.call_active:
            with tracer.start_as_current_span("stt_turn") as stt_span:
                final_transcript = ""
                async for res in stt_service.stt_service_handler(stt_fd, 16000, self.channel.id, adaptation_context):
                    if res.get('is_final'):
                        final_transcript = res.get('transcript', '').strip()
                        break
                stt_span.set_attribute("transcript", final_transcript)
            
            if not final_transcript or not self.call_active: break
            logging.info(f"[{self.channel.id}] User: \"{final_transcript}\"\n")

            with tracer.start_as_current_span("nlp_tts_turn") as nlp_span:
                nlp_span.set_attribute("user.input", final_transcript)
                sentence_buffer = ""
                async for token in nlp_service.streaming_process_user_input(final_transcript, history):
                    if not self.call_active: break
                    sentence_buffer += token
                    if any(t in token for t in ".?!;"):
                        with tracer.start_as_current_span("tts_sentence_playback") as tts_span:
                            tts_span.set_attribute("sentence", sentence_buffer.strip())
                            audio_path = await tts_service.tts_service_handler(text=sentence_buffer.strip())
                            if audio_path and self.call_active:
                                await self.channel.play(media=f"sound:{os.path.basename(audio_path).replace('.wav', '')}")
                        sentence_buffer = ""
                
                if self.call_active and sentence_buffer.strip():
                    with tracer.start_as_current_span("tts_sentence_playback") as tts_span:
                        tts_span.set_attribute("sentence", sentence_buffer.strip())
                        audio_path = await tts_service.tts_service_handler(text=sentence_buffer.strip())
                        if audio_path and self.call_active:
                            await self.channel.play(media=f"sound:{os.path.basename(audio_path).replace('.wav', '')}")

            if any(kw in final_transcript.lower() for kw in ["tạm biệt", "kết thúc", "cảm ơn"]): break
    
    async def handle_call(self):
        with tracer.start_as_current_span("call") as root_span:
            root_span.set_attribute("call.id", self.channel.id)
            root_span.set_attribute("call.caller_id", self.channel.caller.number)

            stt_pipe_path, stt_fd, live_recording = None, -1, None
            try:
                await self.channel.answer()
                root_span.add_event("Call answered")

                stt_pipe_path = f"/tmp/rec_{self.channel.id}.pipe"
                os.mkfifo(stt_pipe_path)
                live_recording = await self.channel.record(format='slin16@16000', name=stt_pipe_path, beep=False, ifExists='overwrite')
                await asyncio.sleep(0.2)
                stt_fd = os.open(stt_pipe_path, os.O_RDONLY | os.O_NONBLOCK)
                
                adaptation_context = SPEECH_ADAPTATION_CONFIG.get('default', {})

                main_task = asyncio.create_task(self._conversation_loop(stt_fd, adaptation_context))
                self.active_tasks.add(main_task)
                await main_task

            except asyncio.CancelledError:
                logging.info(f"[{self.channel.id}] Main task cancelled.")
                root_span.set_status(trace.Status(trace.StatusCode.ERROR, "Call cancelled by hangup"))
            except Exception as e:
                logging.error(f"[{self.channel.id}] Unhandled exception in handle_call: {e}", exc_info=True)
                root_span.record_exception(e)
                root_span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            finally:
                logging.info(f"[{self.channel.id}] Final cleanup.")
                if live_recording: await live_recording.stop()
                await self._cancel_all_tasks()
                if stt_fd != -1: os.close(stt_fd)
                if stt_pipe_path and os.path.exists(stt_pipe_path): os.remove(stt_pipe_path)
                if self.call_active:
                    self.call_active = False
                    await self.channel.hangup()
                logging.info(f"[{self.channel.id}] Cleanup complete.")
