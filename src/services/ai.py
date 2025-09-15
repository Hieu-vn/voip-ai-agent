import logging
import asyncio
from src.core.stt_module import STTModule
from src.core.tts_module import tts_service_handler as tts_service_handler_func
from config import LANGUAGE_CODE, NLP_CONFIG

# Import the new local NLP handlers
from src.core.nlp_local_unsloth import LocalUnslothNLP
from src.core.nlp_local_llamacpp import LocalLlamaCppNLP

# --- Instantiate Services ---

# STT Service
stt_service = STTModule(language_code=LANGUAGE_CODE)
logging.info("STT service instantiated.")

# TTS Service
class TTSServiceWrapper:
    async def tts_service_handler(self, text, **kwargs):
        return await tts_service_handler_func(text, **kwargs)

tts_service = TTSServiceWrapper()
logging.info("TTS service instantiated.")

# NLP Service (New Factory Logic)

# This wrapper makes the non-streaming generate method compatible with the streaming interface expected by call_handler
class StreamingWrapper:
    def __init__(self, nlp_instance):
        self._nlp = nlp_instance

    async def streaming_process_user_input(self, user_text, history):
        # The new local NLP classes don't use history yet, this can be added later
        # This is a simplified stream, yielding the full response at once.
        # A true stream would require changes to the generate method in the local NLP class.
        response_text = self._nlp.generate(prompt=user_text)
        yield response_text

def get_nlp_service():
    """
    Factory function to instantiate the correct NLP service based on config.
    """
    if not NLP_CONFIG or not NLP_CONFIG.get("backend"):
        logging.warning("NLP configuration is missing 'backend'. Instantiating a dummy NLP service.")
        class DummyNLPService:
            async def streaming_process_user_input(self, user_text, history):
                logging.warning(f"Dummy NLP received input: {user_text}")
                yield "Xin lỗi, dịch vụ trí tuệ nhân tạo chưa được cấu hình."
        return DummyNLPService()

    backend = NLP_CONFIG["backend"]
    logging.info(f"Instantiating NLP service with backend: {backend}")

    try:
        if backend == "unsloth_transformers":
            nlp_instance = LocalUnslothNLP(
                model_path=NLP_CONFIG["model_path"],
                max_new_tokens=NLP_CONFIG.get("max_new_tokens", 128),
                temperature=NLP_CONFIG.get("temperature", 0.6),
                top_p=NLP_CONFIG.get("top_p", 0.9),
                stop=NLP_CONFIG.get("stop", []),
            )
            return StreamingWrapper(nlp_instance)
            
        elif backend == "llama_cpp":
            nlp_instance = LocalLlamaCppNLP(
                gguf_path=NLP_CONFIG["gguf_path"],
            )
            return StreamingWrapper(nlp_instance)

        else:
            raise RuntimeError(f"Unsupported NLP backend in config: {NLP_CONFIG['backend']}")
            
    except FileNotFoundError as e:
        logging.error(f"Model path not found: {e}. Check your app_config.yaml. Falling back to Dummy NLP.")
    except Exception as e:
        logging.error(f"Failed to instantiate NLP backend '{backend}': {e}. Falling back to Dummy NLP.")

    # Fallback to dummy if any error occurs during instantiation
    class DummyNLPService:
        async def streaming_process_user_input(self, user_text, history):
            logging.warning(f"Dummy NLP received input: {user_text}")
            yield "Xin lỗi, có lỗi xảy ra khi tải mô hình ngôn ngữ."
    return DummyNLPService()


# Instantiate the NLP service using the factory
nlp_service = get_nlp_service()