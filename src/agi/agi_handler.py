#!/usr/bin/env python3
import sys
import os
import requests
from loguru import logger
from asterisk.agi import AGI

# Import các module STT và TTS đã được cấu hình
from ..stt.transcribe import transcribe_google_cloud
from ..tts.generate_audio import generate_audio


# Cấu hình logger
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("/tmp/agi_handler.log", rotation="10 MB", level="DEBUG")

# Cấu hình API cho NLP
NLP_API_URL = os.environ.get("NLP_API_URL", "http://localhost:8000/v1/chat/completions")

def process_nlp(text_input):
    """Gửi văn bản đến Llama 4 và nhận phản hồi."""
    if not text_input:
        return ""

    try:
        payload = {
            "model": "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit",
            "messages": [{"role": "user", "content": text_input}]
        }
        logger.info(f"NLP: Gửi payload đến Llama 4: {payload}")
        response = requests.post(NLP_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        assistant_response = result["choices"][0]["message"]["content"]
        logger.info(f"NLP: Nhận phản hồi từ Llama 4: {assistant_response}")
        return assistant_response
    except requests.exceptions.RequestException as e:
        logger.error(f"NLP: Lỗi gọi API: {e}")
        return "Đã có lỗi xảy ra khi kết nối đến bộ xử lý ngôn ngữ."
    except (KeyError, IndexError) as e:
        logger.error(f"NLP: Lỗi xử lý JSON response: {e}")
        return "Đã có lỗi xảy ra khi xử lý phản hồi từ AI."


def main():
    """Hàm chính điều khiển luồng hội thoại AGI."""
    agi = AGI()
    unique_id = agi.get_variable('UNIQUEID')
    logger.info(f"AGI handler started for call {unique_id}.")

    try:
        agi.answer()
        agi.stream_file('welcome') # Phát lời chào (cần có file welcome.wav/.gsm trong sounds dir)

        # Vòng lặp hội thoại
        while True:
            logger.info("Bắt đầu một lượt hội thoại mới.")
            
            # 1. Ghi âm giọng nói của người dùng
            # Ghi âm vào file wav, dừng khi có khoảng lặng 2 giây hoặc người dùng bấm phím #
            user_audio_path = f"/tmp/{unique_id}_user_input.wav"
            # AGI record file không cần phần mở rộng, nó sẽ tự thêm
            # Nó cũng cần file tồn tại trước. Hãy tạo file rỗng.
            open(user_audio_path, 'a').close()
            
            # Sử dụng lệnh `RECORD FILE` của AGI
            # Ghi âm file wav, thoát bằng phím #, timeout 5s, có phát beep, khoảng lặng 2s để dừng
            result = agi.record_file(user_audio_path[:-4], 'wav', '#', 5000, beep=True, silence=2)
            
            if result.get('result') == '0':
                logger.info("Người dùng không nói gì hoặc đã gác máy.")
                break

            # 2. Chuyển đổi giọng nói thành văn bản (STT)
            transcribed_text = transcribe_google_cloud(user_audio_path)
            logger.info(f"STT: {transcribed_text}")

            if not transcribed_text.strip():
                logger.info("STT không nhận diện được văn bản. Lặp lại.")
                agi.stream_file('please_repeat') # Yêu cầu nhắc lại
                continue

            # 3. Xử lý văn bản bằng NLP (Llama 4)
            assistant_response_text = process_nlp(transcribed_text)

            # Điều kiện kết thúc hội thoại
            if "tạm biệt" in assistant_response_text.lower():
                logger.info("NLP quyết định kết thúc hội thoại.")
                # Chúng ta vẫn sẽ generate audio cho lời chào tạm biệt
            
            # 4. Chuyển đổi văn bản phản hồi thành giọng nói (TTS)
            assistant_audio_path = f"/tmp/{unique_id}_assistant_response.wav"
            generate_audio(assistant_response_text, assistant_audio_path)

            # 5. Phát lại âm thanh cho người dùng
            # AGI stream file không cần phần mở rộng nếu file nằm trong sound dir
            # Nhưng với đường dẫn tuyệt đối, ta nên truyền cả đường dẫn
            agi.stream_file(assistant_audio_path[:-4])

            # Nếu NLP muốn kết thúc, chúng ta thoát vòng lặp sau khi đã trả lời
            if "tạm biệt" in assistant_response_text.lower():
                break

    except Exception as e:
        logger.error(f"Lỗi không mong muốn trong vòng lặp chính: {e}")
    finally:
        logger.info("Kết thúc AGI handler, gác máy.")
        agi.hangup()

if __name__ == "__main__":
    main()