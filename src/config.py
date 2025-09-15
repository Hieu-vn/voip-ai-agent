import os
import yaml
import logging

def load_config():
    """Tải cấu hình từ file YAML và biến môi trường."""
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'app_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            config = {}
    except FileNotFoundError:
        logging.warning(f"File cấu hình không tìm thấy tại: {config_path}. Sử dụng giá trị mặc định.")
        config = {}
    except Exception as e:
        logging.error(f"Lỗi khi đọc file cấu hình: {e}")
        config = {}

    # Load ARI config directly with correct defaults, ignoring YAML for this section
    # as it contains shell-style variables that are not expanded by PyYAML.
    ARI_URL = os.getenv('ARI_URL', 'http://localhost:8088/')
    ARI_USERNAME = os.getenv('ARI_USERNAME', 'vitalpbx')
    ARI_PASSWORD = os.getenv('ARI_PASSWORD', 'zcWGYbNnPer2YUBTg433EMuVs')
    ARI_APP_NAME = os.getenv('ARI_APP_NAME', 'voip-ai-agent')

    # Load other configs from YAML
    speech_adaptation_config = config.get('speech_adaptation', {})
    tts_config = config.get('tts', {})
    nlp_config = config.get('nlp', {})
    language_code = os.getenv("LANGUAGE_CODE", config.get('language_code', "vi-VN"))

    return {
        "ARI_URL": ARI_URL,
        "ARI_USERNAME": ARI_USERNAME,
        "ARI_PASSWORD": ARI_PASSWORD,
        "ARI_APP_NAME": ARI_APP_NAME,
        "SPEECH_ADAPTATION_CONFIG": speech_adaptation_config,
        "TTS_CONFIG": tts_config,
        "NLP_CONFIG": nlp_config,
        "LANGUAGE_CODE": language_code
    }

# Tải cấu hình ngay khi module được import
_config = load_config()

# Expose các biến để các module khác có thể import trực tiếp
ARI_URL = _config["ARI_URL"]
ARI_USERNAME = _config["ARI_USERNAME"]
ARI_PASSWORD = _config["ARI_PASSWORD"]
ARI_APP_NAME = _config["ARI_APP_NAME"]
SPEECH_ADAPTATION_CONFIG = _config["SPEECH_ADAPTATION_CONFIG"]
TTS_CONFIG = _config["TTS_CONFIG"]
NLP_CONFIG = _config["NLP_CONFIG"]
LANGUAGE_CODE = _config["LANGUAGE_CODE"]

logging.info("Đã tải cấu hình ứng dụng thành công.")
