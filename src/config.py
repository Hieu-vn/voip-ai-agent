import os
import yaml
import logging

def load_config():
    """Tải cấu hình từ file YAML và biến môi trường."""
    # Đường dẫn đến file config gốc
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'app_config.yaml')
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if not config:
            config = {}
    except FileNotFoundError:
        logging.error(f"File cấu hình không tìm thấy tại: {config_path}")
        config = {}
    except Exception as e:
        logging.error(f"Lỗi khi đọc file cấu hình: {e}")
        config = {}

    # Load ARI config, ưu tiên biến môi trường
    ari_config = config.get('ari', {})
    ARI_URL = os.getenv('ARI_URL', ari_config.get('url', 'http://localhost:8088/'))
    ARI_USERNAME = os.getenv('ARI_USERNAME', ari_config.get('username', 'asterisk'))
    ARI_PASSWORD = os.getenv('ARI_PASSWORD', ari_config.get('password', 'asterisk'))
    ARI_APP_NAME = os.getenv('ARI_APP_NAME', ari_config.get('app_name', 'voip-ai-agent'))

    # Load Speech Adaptation config
    speech_adaptation_config = config.get('speech_adaptation', {})

    # Load các cấu hình khác
    language_code = os.getenv("LANGUAGE_CODE", config.get('language_code', "vi-VN"))

    return {
        "ARI_URL": ARI_URL,
        "ARI_USERNAME": ARI_USERNAME,
        "ARI_PASSWORD": ARI_PASSWORD,
        "ARI_APP_NAME": ARI_APP_NAME,
        "SPEECH_ADAPTATION_CONFIG": speech_adaptation_config,
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
LANGUAGE_CODE = _config["LANGUAGE_CODE"]

logging.info("Đã tải cấu hình ứng dụng thành công.")
if SPEECH_ADAPTATION_CONFIG:
    logging.info(f"Đã tải {len(SPEECH_ADAPTATION_CONFIG)} context cho Speech Adaptation.")
