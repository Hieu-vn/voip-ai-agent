
import os
import yaml
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Load YAML config
config_path = os.path.join(os.path.dirname(__file__), "..", "config", "app_config.yaml")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# ARI Configuration
ARI_URL = os.getenv('ARI_URL', config['ari']['url'])
ARI_USERNAME = os.getenv('ARI_USERNAME', config['ari']['username'])
ARI_PASSWORD = os.getenv('ARI_PASSWORD', config['ari']['password'])
ARI_APP_NAME = os.getenv('ARI_APP_NAME', config['ari']['app_name'])

# Other configurations (from app_config.yaml or environment variables)
CONFIG = {
    "openai_base_url": os.getenv("OPENAI_BASE_URL", config.get('openai_base_url', "http://127.0.0.1:8000/v1")),
    "openai_api_key": os.getenv("OPENAI_API_KEY", config.get('openai_api_key', "sk-local")),
    "llama_model": os.getenv("LLAMA_MODEL", config.get('llama_model', "llama-4")),
    "language_code": os.getenv("LANGUAGE_CODE", config.get('language_code', "vi-VN")),
    "sample_rate_hz": int(os.getenv("SAMPLE_RATE_HZ", config.get('sample_rate_hz', 8000))),
    "sounds_dir": os.getenv("SOUNDS_DIR", config.get('sounds_dir', "/var/lib/asterisk/sounds/custom")),
}
