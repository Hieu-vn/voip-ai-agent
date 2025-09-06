#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, uuid, threading
from loguru import logger
from pydub import AudioSegment
import requests

# Google Cloud
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import texttospeech as tts

# ========== Config ==========
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "sk-local")
LLAMA_MODEL     = os.getenv("LLAMA_MODEL", "llama-4")
LANGUAGE_CODE   = "vi-VN"
SAMPLE_RATE_HZ  = 8000
SOUNDS_DIR      = "/var/lib/asterisk/sounds/custom"

# ========== AGI helpers ==========
def read_agi_env(stdin=sys.stdin):
    env = {}
    while True:
        line = stdin.readline().strip()
        if line == "":
            break
        k, v = line.split(":", 1)
        env[k.strip()] = v.strip()
    return env

def agi_cmd(cmd, stdin=sys.stdin, stdout=sys.stdout):
    stdout.write(cmd.strip() + "
")
    stdout.flush()
    resp = stdin.readline().strip()
    logger.debug(f"AGI CMD: {cmd} -> {resp}")
    return resp

def agi_stream_wav(path_no_ext):
    # Asterisk STREAM FILE không kèm .wav
    return agi_cmd(f'STREAM FILE {path_no_ext} ""')

def agi_verbose(msg, level=1):
    return agi_cmd(f'VERBOSE "{msg}" {level}')

# ========== LLM ==========
def llm_chat(user_text, history=None, system_prompt=None):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type":"application/json"}
    messages = []
    if system_prompt:
        messages.append({"role":"system","content":system_prompt})
    if history:
        messages.extend(history)
    messages.append({"role":"user","content":user_text})
    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 256,
        "stream": False
    }
    r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

# ========== Google STT (single utterance) ==========
def transcribe_single_utterance(fd_audio=3):
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE_HZ,
        language_code=LANGUAGE_CODE,
        enable_automatic_punctuation=True
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=False,
        single_utterance=True  # kết thúc khi hết lượt nói
    )

    # Generator đọc từ FD=3 từng 20ms (160 samples = 320 bytes)
    def requests_gen():
        yield speech.StreamingRecognizeRequest(streaming_config=streaming_config)
        while True:
            try:
                chunk = os.read(fd_audio, 320)  # 20ms @ 8kHz, 16-bit
            except Exception:
                break
            if not chunk:
                break
            yield speech.StreamingRecognizeRequest(audio_content=chunk)

    responses = client.streaming_recognize(requests=requests_gen())
    final_text = ""
    for resp in responses:
        for result in resp.results:
            if result.is_final and result.alternatives:
                final_text = result.alternatives[0].transcript.strip()
    return final_text

# ========== Google TTS (vi-VN) ==========
def synth_vi_wav(text, out_wav_path):
    client = tts.TextToSpeechClient()
    synthesis_input = tts.SynthesisInput(text=text)
    voice = tts.VoiceSelectionParams(language_code="vi-VN", name="vi-VN-Wavenet-A")
    audio_config = tts.AudioConfig(
        audio_encoding=tts.AudioEncoding.LINEAR16,
        sample_rate_hertz=SAMPLE_RATE_HZ
    )
    resp = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with open(out_wav_path, "wb") as f:
        f.write(resp.audio_content)
    return out_wav_path

# ========== Main session ==========
def main():
    # Đọc biến môi trường AGI
    env = read_agi_env()
    callid = env.get("agi_uniqueid", str(uuid.uuid4()))
    logger.add(f"/tmp/agi_{callid}.log", level="DEBUG", rotation="2 MB")
    agi_verbose(f"AI-AGI start, callid={callid}")

    history = [{"role":"system","content":"Bạn là trợ lý CSKH nói giọng Việt, trả lời ngắn gọn, lịch sự. Nếu khách muốn kết thúc, xác nhận và chào."}]

    # Chào mừng:
    greet = "Xin chào, tôi là trợ lý ảo của công ty. Anh/chị cần hỗ trợ vấn đề gì ạ?"
    wav_name = f"{callid}_greet.wav"
    wav_path = os.path.join(SOUNDS_DIR, wav_name)
    synth_vi_wav(greet, wav_path)
    agi_stream_wav(f"custom/{os.path.splitext(os.path.basename(wav_path))[0]}")

    # Vòng lặp hội thoại
    while True:
        agi_verbose("Listening (single utterance)...")
        try:
            user_text = transcribe_single_utterance()
        except Exception as e:
            agi_verbose(f"STT error: {e}")
            break

        if not user_text:
            agi_verbose("No speech detected, bye.")
            break

        agi_verbose(f"User said: {user_text}")
        # Điều kiện kết thúc đơn giản
        if any(kw in user_text.lower() for kw in ["kết thúc", "bye", "tạm biệt", "ngưng", "đủ rồi"]):
            farewell = "Cảm ơn anh/chị đã gọi. Chúc một ngày tốt lành!"
            wav_name = f"{callid}_bye.wav"
            wav_path = os.path.join(SOUNDS_DIR, wav_name)
            synth_vi_wav(farewell, wav_path)
            agi_stream_wav(f"custom/{os.path.splitext(os.path.basename(wav_path))[0]}")
            break

        # Gọi Llama 4
        try:
            bot_text = llm_chat(user_text, history=history)
        except Exception as e:
            agi_verbose(f"LLM error: {e}")
            bot_text = "Xin lỗi, hệ thống đang bận. Vui lòng gọi lại sau."

        history.append({"role":"user","content":user_text})
        history.append({"role":"assistant","content":bot_text})
        agi_verbose(f"Bot: {bot_text}")

        # TTS & phát lại
        fbase = f"{callid}_{uuid.uuid4().hex}"
        wav_path = os.path.join(SOUNDS_DIR, f"{fbase}.wav")
        try:
            synth_vi_wav(bot_text, wav_path)
            agi_stream_wav(f"custom/{fbase}")
        except Exception as e:
            agi_verbose(f"TTS/Playback error: {e}")
            break

    agi_verbose("AI-AGI end.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        # Caller hangup
        pass
