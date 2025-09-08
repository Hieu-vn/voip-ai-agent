import asyncio
import logging
from src.services import ari_client

# Cấu hình logging cơ bản
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    try:
        logging.info("Khởi chạy ứng dụng AI Agent...")
        # Chạy client ARI
        asyncio.run(ari_client.start_ari_client())
    except KeyboardInterrupt:
        logging.info("Tắt ứng dụng.")
    except Exception as e:
        logging.critical(f"Lỗi nghiêm trọng ở tầng cao nhất: {e}", exc_info=True)
