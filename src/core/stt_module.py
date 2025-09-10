import logging
import asyncio
from src.core.stt_google_cloud_client import STTGoogleCloudClient

class STTModule:
    def __init__(self, language_code: str):
        self.language_code = language_code

    async def stt_service_handler(self, audio_fd: int, sample_rate: int, call_id: str, adaptation_config: dict):
        """
        Đây là một async generator, nó sẽ yield các kết quả STT (tạm thời và cuối cùng).
        
        :param audio_fd: File descriptor của luồng audio.
        :param sample_rate: Tần số lấy mẫu.
        :param call_id: ID cuộc gọi để truy vết.
        :param adaptation_config: Dictionary chứa cấu hình speech adaptation.
        """
        logging.info(f"STT Module [{call_id}]: Bắt đầu stream STT với sample rate {sample_rate}Hz.")
        
        stt_client = STTGoogleCloudClient(
            language_code=self.language_code,
            sample_rate_hz=sample_rate
        )
        
        # Đây là một blocking generator, không thể dùng trực tiếp trong async code.
        blocking_generator = stt_client.streaming_recognize_generator(
            fd_audio=audio_fd,
            call_id=call_id,
            adaptation_config=adaptation_config # Truyền config xuống client
        )
        
        loop = asyncio.get_running_loop()

        while True:
            try:
                # Chạy hàm `next()` trên blocking_generator trong một thread khác
                # để lấy kết quả tiếp theo mà không block event loop.
                result = await loop.run_in_executor(None, next, blocking_generator)
                yield result
                
                # Nếu kết quả là cuối cùng (hoặc có lỗi), chúng ta có thể dừng.
                if result.get('is_final'):
                    break
            except StopIteration:
                # Generator đã kết thúc một cách bình thường.
                logging.debug(f"STT Module [{call_id}]: Stream từ generator đã kết thúc.")
                break
            except Exception as e:
                logging.error(f"STT Module [{call_id}]: Lỗi khi xử lý stream STT: {e}", exc_info=True)
                break
