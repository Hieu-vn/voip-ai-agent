import os, sys, json, time, uuid
from loguru import logger
import yaml

# Import core AI modules
from src.core.stt_module import STTModule
from src.core.nlp_module import NLPModule
from src.core.tts_module import TTSModule

# --- Configuration Loading ---
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "app_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables for ARI config if not already set in YAML
    # This is a simplified example, a dedicated config loader would be better
    ari_config = config.get('ari', {})
    ari_config['url'] = os.getenv('ARI_URL', ari_config.get('url', 'http://localhost:8088/'))
    ari_config['username'] = os.getenv('ARI_USERNAME', ari_config.get('username', 'asterisk'))
    ari_config['password'] = os.getenv('ARI_PASSWORD', ari_config.get('password', 'asterisk'))
    ari_config['app_name'] = os.getenv('ARI_APP_NAME', ari_config.get('app_name', 'voip-ai-agent'))
    config['ari'] = ari_config

    # Load other config values from environment variables if needed
    config['openai_base_url'] = os.getenv("OPENAI_BASE_URL", config.get('openai_base_url', "http://127.0.0.1:8000/v1"))
    config['openai_api_key'] = os.getenv("OPENAI_API_KEY", config.get('openai_api_key', "sk-local"))
    config['llama_model'] = os.getenv("LLAMA_MODEL", config.get('llama_model', "llama-4"))
    config['language_code'] = os.getenv("LANGUAGE_CODE", config.get('language_code', "vi-VN"))
    config['sample_rate_hz'] = int(os.getenv("SAMPLE_RATE_HZ", config.get('sample_rate_hz', 8000)))
    config['sounds_dir'] = os.getenv("SOUNDS_DIR", config.get('sounds_dir', "/var/lib/asterisk/sounds/custom"))

    return config

CONFIG = load_config()

# --- AGI helpers ---
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
    stdout.write(cmd.strip() + "")
    stdout.flush()
    resp = stdin.readline().strip()
    logger.debug(f"AGI CMD: {cmd} -> {resp}")
    return resp

def agi_stream_wav(path_no_ext):
    return agi_cmd(f'STREAM FILE {path_no_ext} ""')

def agi_verbose(msg, level=1):
    return agi_cmd(f'VERBOSE "{msg}" {level}')

# --- Main session ---
def main():
    # Initialize logging
    callid = str(uuid.uuid4())
    logger.add(f"/tmp/agi_{callid}.log", level="DEBUG", rotation="2 MB")
    agi_verbose(f"AI-AGI start, callid={callid}")

    # Initialize AI modules
    stt_module = STTModule(language_code=CONFIG['language_code'], sample_rate_hz=CONFIG['sample_rate_hz'])
    nlp_module = NLPModule(
        openai_base_url=CONFIG['openai_base_url'],
        openai_api_key=CONFIG['openai_api_key'],
        llama_model=CONFIG['llama_model']
    )
    tts_module = TTSModule(sample_rate_hz=CONFIG['sample_rate_hz'])

    history = [{"role":"system","content":"Bạn là trợ lý CSKH nói giọng Việt, trả lời ngắn gọn, lịch sự. Nếu khách muốn kết thúc, xác nhận và chào."}]

    # Greeting
    greet = "Xin chào, tôi là trợ lý ảo của công ty. Anh/chị cần hỗ trợ vấn đề gì ạ?"
    wav_name = f"{callid}_greet.wav"
    wav_path = os.path.join(CONFIG['sounds_dir'], wav_name)
    tts_module.synth_vi_wav(greet, wav_path)
    agi_stream_wav(f"custom/{os.path.splitext(os.path.basename(wav_path))[0]}")

    # Conversation loop
    while True:
        agi_verbose("Listening (single utterance)...")
        try:
            user_text = stt_module.transcribe_single_utterance()
        except Exception as e:
            agi_verbose(f"STT error: {e}")
            break

        if not user_text:
            agi_verbose("No speech detected, bye.")
            break

        agi_verbose(f"User said: {user_text}")
        # Simple termination condition
        if any(kw in user_text.lower() for kw in ["kết thúc", "bye", "tạm biệt", "ngưng", "đủ rồi"]):
            farewell = "Cảm ơn anh/chị đã gọi. Chúc một ngày tốt lành!"
            wav_name = f"{callid}_bye.wav"
            wav_path = os.path.join(CONFIG['sounds_dir'], wav_name)
            tts_module.synth_vi_wav(farewell, wav_path)
            agi_stream_wav(f"custom/{os.path.splitext(os.path.basename(wav_path))[0]}")
            break

        # Call Llama 4
        try:
            bot_text = nlp_module.llm_chat(user_text, history=history)
        except Exception as e:
            agi_verbose(f"LLM error: {e}")
            bot_text = "Xin lỗi, hệ thống đang bận. Vui lòng gọi lại sau."

        history.append({"role":"user","content":user_text})
        history.append({"role":"assistant","content":bot_text})
        agi_verbose(f"Bot: {bot_text}")

        # TTS & playback
        fbase = f"{callid}_{uuid.uuid4().hex}"
        wav_path = os.path.join(CONFIG['sounds_dir'], f"{fbase}.wav")
        try:
            tts_module.synth_vi_wav(bot_text, wav_path)
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
