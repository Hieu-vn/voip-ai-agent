#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, requests
from loguru import logger
from asterisk.agi import AGI
from pydub import AudioSegment

# ===== Config =====
NLP_API_URL = os.environ.get("NLP_API_URL", "http://localhost:8000/v1/chat/completions")
NLP_MODEL = os.environ.get("NLP_MODEL", "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit")
SAMPLE_RATE = 8000  # 8kHz telephony
CHANNELS = 1        # mono
WELCOME_PROMPT = "welcome"         # sounds/welcome.wav
PLEASE_REPEAT = "please-repeat"    # sounds/please-repeat.wav (khuyến nghị tạo)
ERROR_STT = "error-stt"            # sounds/error-stt.wav (khuyến nghị tạo)
TTS_DIR = "/var/lib/asterisk/sounds/vi/custom"  # đã đúng cách anh làm

# ===== Logger =====
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/tmp/agi_handler.log", rotation="10 MB", level="DEBUG")

# ===== NLP =====
def process_nlp(text_input: str) -> str:
    if not text_input:
        return ""
    try:
        payload = {
            "model": NLP_MODEL,
            "messages": [{"role": "user", "content": text_input}],
            "temperature": 0.6,
            "max_tokens": 220,
        }
        logger.info(f"NLP → {NLP_API_URL} payload={payload}")
        r = requests.post(NLP_API_URL, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        reply = data["choices"][0]["message"]["content"]
        logger.info(f"NLP ← {reply}\n") # Added newline for clarity
        return reply
    except Exception as e:
        logger.exception(f"NLP error: {e}")
        return "Xin lỗi, hệ thống NLP đang bận. Vui lòng thử lại."

# ===== STT (Google sync) =====
def stt_google_sync(wav_path: str, language: str = "vi-VN") -> str:
    try:
        from google.cloud import speech
        client = speech.SpeechClient()
        with open(wav_path, "rb") as f:
            content = f.read()
        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            language_code=language,
            enable_automatic_punctuation=True,
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
        )
        resp = client.recognize(config=config, audio=audio)
        text = " ".join([r.alternatives[0].transcript for r in resp.results]).strip()
        logger.info(f"STT ← {text}")
        return text
    except Exception as e:
        logger.exception(f"STT error: {e}")
        return ""

# ===== TTS wrapper (dùng hàm của anh) =====
def tts_to_vi_custom(text: str, basename: str) -> str:
    """
    Gọi TTS -> wav, resample 8k/mono/16-bit, lưu về /var/lib/asterisk/sounds/vi/custom/<basename>.wav
    Trả về token tương đối 'vi/custom/<basename>' để AGI STREAM FILE.
    """
    from src.tts.generate_audio import generate_audio  # chính hàm của anh
    os.makedirs(TTS_DIR, exist_ok=True)
    tmp_out = f"/tmp/{basename}.wav"
    final_out = os.path.join(TTS_DIR, f"{basename}.wav")

    # 1) synth
    generate_audio(text, tmp_out)

    # 2) normalize format
    audio = AudioSegment.from_wav(tmp_out)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
    audio.export(final_out, format="wav")

    # 3) quyền đọc (optional)
    try:
        import pwd, grp
        uid = pwd.getpwnam("asterisk").pw_uid
        gid = grp.getgrnam("asterisk").gr_gid
        os.chown(final_out, uid, gid)
    except Exception:
        pass

    return f"vi/custom/{basename}"

# ===== Main AGI =====
def main():
    agi = AGI()
    unique_id = agi.get_variable('UNIQUEID') or "unknown"
    logger.info(f"AGI start call={unique_id}")

    try:
        agi.answer()
        try:
            agi.stream_file(WELCOME_PROMPT)
        except Exception as e:
            logger.warning(f"Welcome prompt missing or error: {e}")

        while True:
            logger.info("=== New turn ===")
            rec_base = f"/var/spool/asterisk/recordings/{unique_id}_user"
            os.makedirs(os.path.dirname(rec_base), exist_ok=True)

            # RECORD FILE <filename> <format> <escape_digits> [timeout(ms)] [offset_samples] [BEEP] [s=silence]
            try:
                agi.record_file(rec_base, 'wav', '#', 7000, beep=True, silence=2)
            except Exception as e:
                logger.exception(f"record_file error: {e}")
                break

            wav_in = rec_base + ".wav"
            if not os.path.exists(wav_in) or os.path.getsize(wav_in) < 1000:
                logger.info("No/too-small recorded audio. Ask to repeat.")
                try:
                    agi.stream_file(PLEASE_REPEAT)
                except Exception:
                    pass
                continue

            # Normalize input wav → 8k/mono/16-bit
            try:
                seg = AudioSegment.from_wav(wav_in)
                seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(CHANNELS).set_sample_width(2)
                seg.export(wav_in, format="wav")
            except Exception as e:
                logger.exception(f"Normalize wav error: {e}")
                try:
                    agi.stream_file(ERROR_STT)
                except Exception:
                    pass
                continue

            # STT
            user_text = stt_google_sync(wav_in, language="vi-VN")
            if not user_text:
                logger.info("Empty STT result. Ask to repeat.")
                try:
                    agi.stream_file(PLEASE_REPEAT)
                except Exception:
                    pass
                continue

            # NLP
            reply_text = process_nlp(user_text)

            # TTS → sounds/vi/custom/<basename>.wav
            basename = f"{unique_id}_assistant_{abs(hash(reply_text))%10_000_000}"
            token = tts_to_vi_custom(reply_text, basename)

            # Play back
            try:
                agi.stream_file(token)
            except Exception as e:
                logger.exception(f"STREAM FILE error: {e}")
                try:
                    agi.stream_file("vm-sorry")
                except Exception:
                    pass

            # Exit condition
            if "tạm biệt" in reply_text.lower():
                logger.info("Assistant requested to end call.")
                break

    except Exception as e:
        logger.exception(f"Main loop error: {e}")
    finally:
        try:
            agi.hangup()
        except Exception:
            pass
        logger.info("AGI end.")

if __name__ == "__main__":
    main()