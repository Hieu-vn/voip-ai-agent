import asyncio
import logging
import os
from dotenv import load_dotenv
import threading
from prometheus_client import start_http_server

# Tương thích cho các phiên bản websockets khác nhau
try:
    import websockets.exceptions
    ConnectionClosedOK = websockets.exceptions.ConnectionClosedOK
except ImportError:
    from websockets.legacy.exceptions import ConnectionClosedOK

from asyncari import connect

# Import các module mới
from src.core.stt_module import STTModule
from src.core.nlp_module import NLPModule
from src.core.call_handler import CallHandler # Import CallHandler mới
from src.utils.tracing import initialize_tracer

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Asterisk ARI Configuration
AST_URL = os.getenv("ARI_URL", "http://localhost:8088")
AST_APP = os.getenv("ARI_APP_NAME", "voip-ai-agent")
ARI_USERNAME = os.getenv("ARI_USERNAME", "vitalpbx")
ARI_PASSWORD = os.getenv("ARI_PASSWORD", "zcWGYbNnPer2YUBTg433EMuVs")

async def handle_stasis_start(ari_client, event, stt_module, nlp_module):
    """
    Handles StasisStart event khi có kênh mới vào ứng dụng ARI.
    """
    channel = event['channel']
    logger.info(f"Channel {channel['id']} entered Stasis application '{AST_APP}'")

    # Tạo một CallHandler mới để xử lý cuộc gọi này
    handler = CallHandler(ari_client, stt_module, nlp_module)
    
    # Chạy handle_call trong một task nền để không block vòng lặp sự kiện chính
    asyncio.create_task(handler.handle_call(channel))

async def handle_stasis_end(client, event):
    """
    Handles StasisEnd event khi một kênh rời ứng dụng.
    """
    channel_id = event['channel']['id']
    logger.info(f"Channel {channel_id} left Stasis application.")
    # Logic dọn dẹp session (nếu cần) có thể được thêm ở đây, ví dụ: stt_module.stop_session(channel_id)

async def main():
    """
    Hàm chính để kết nối đến ARI và bắt đầu vòng lặp sự kiện.
    """
    for name, val in {"AST_URL": AST_URL, "ARI_USERNAME": ARI_USERNAME, "ARI_PASSWORD": ARI_PASSWORD}.items():
        if not val:
            logger.critical(f"Missing required configuration: {name}")
            raise SystemExit(1)

    # --- Khởi tạo các module AI một lần duy nhất ---
    logger.info("Initializing AI modules...")
    stt_module = STTModule(language_code=os.getenv("LANGUAGE_CODE", "vi-VN"))
    
    nlp_module = NLPModule(
        llama_model=os.getenv("LLAMA_MODEL_PATH"),
        llama_backend=os.getenv("LLAMA_BACKEND", "unsloth")
    )
    await nlp_module.load_nlp_model() # Tải mô hình NLP ngay khi khởi động
    logger.info("AI modules initialized successfully.")
    # ----------------------------------------------

    logger.info(f"Connecting to ARI at {AST_URL} for app '{AST_APP}'")
    try:
        async with connect(AST_URL, AST_APP, ARI_USERNAME, ARI_PASSWORD) as ari_client:
            logger.info(f"Connected to ARI and registered application '{AST_APP}'.")

            # Đăng ký các event handler với các module AI đã được khởi tạo
            ari_client.on_event('StasisStart', lambda client, event: handle_stasis_start(client, event, stt_module, nlp_module))
            ari_client.on_event('StasisEnd', handle_stasis_end)
            
            logger.info("Waiting for calls...")
            # Giữ cho client chạy và xử lý sự kiện
            while True:
                await asyncio.sleep(3600) # Ngủ một giấc dài, vì logic đã nằm trong event handlers

    except ConnectionClosedOK:
        logger.info("ARI WebSocket connection closed gracefully.")
    except Exception as e:
        logger.error(f"Failed to connect to ARI or an error occurred: {e}", exc_info=True)

def start_metrics_server():
    """Khởi chạy server HTTP cho Prometheus trong một luồng nền."""
    try:
        port = int(os.getenv("METRICS_PORT", 9108))
        start_http_server(port)
        logger.info(f"Prometheus metrics server is running on port {port}")
    except OSError as e:
        if e.errno == 98: # EADDRINUSE
            logger.warning(f"Metrics port {port} is already in use; skipping metrics server.")
        else:
            raise
    except Exception as e:
        logger.error(f"Could not start Prometheus metrics server: {e}")

if __name__ == "__main__":
    # Khởi tạo OpenTelemetry Tracer
    tracer_provider = initialize_tracer("voip-ai-agent")

    try:
        # Khởi chạy metrics server trong một daemon thread
        metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
        metrics_thread.start()

        logger.info("Starting AI Agent application...")
        asyncio.run(main())

    except KeyboardInterrupt:
        logger.info("Application shutting down.")
    except Exception as e:
        logger.critical(f"Critical error at top level: {e}", exc_info=True)
    finally:
        if tracer_provider:
            logger.info("Shutting down OpenTelemetry Tracer Provider...")
            # tracer_provider.shutdown() # Tùy thuộc vào phiên bản opentelemetry
            pass