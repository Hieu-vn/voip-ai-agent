import asyncio
import logging
import threading
from prometheus_client import start_http_server
from src.services import ari_client
from src.utils.tracing import initialize_tracer

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def start_metrics_server():
    """Khởi chạy server HTTP cho Prometheus trong một luồng nền."""
    try:
        port = 9090
        start_http_server(port)
        logging.info(f"Prometheus metrics server đang chạy trên port {port}")
    except Exception as e:
        logging.error(f"Không thể khởi chạy Prometheus metrics server: {e}")

if __name__ == "__main__":
    # Khởi tạo OpenTelemetry Tracer
    tracer_provider = initialize_tracer("voip-ai-agent")

    try:
        # Khởi chạy metrics server trong một daemon thread
        metrics_thread = threading.Thread(target=start_metrics_server, daemon=True)
        metrics_thread.start()

        logging.info("Khởi chạy ứng dụng AI Agent...")
        # Chạy client ARI
        asyncio.run(ari_client.start_ari_client())

    except KeyboardInterrupt:
        logging.info("Tắt ứng dụng.")
    except Exception as e:
        logging.critical(f"Lỗi nghiêm trọng ở tầng cao nhất: {e}", exc_info=True)
    finally:
        # Đảm bảo tracer provider được shutdown một cách an toàn để gửi đi các span cuối cùng
        if tracer_provider:
            logging.info("Shutting down OpenTelemetry Tracer Provider...")
            tracer_provider.shutdown()
