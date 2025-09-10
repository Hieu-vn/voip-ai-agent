import json
import logging
import asyncio
import aiohttp

class LLMClient:
    def __init__(self, openai_base_url: str, openai_api_key: str, llama_model: str):
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.llama_model = llama_model
        # Tạo một session duy nhất để tái sử dụng kết nối (connection pooling)
        self._session = aiohttp.ClientSession()
        logging.info("LLMClient đã được khởi tạo với aiohttp.ClientSession.")

    async def close_session(self):
        """Nên được gọi khi ứng dụng tắt để đóng session một cách an toàn."""
        if self._session and not self._session.closed:
            await self._session.close()
            logging.info("LLMClient aiohttp session đã được đóng.")

    async def streaming_chat_generator(self, messages: list, temperature: float = 0.3, max_tokens: int = 256):
        """
        Đây là một async generator, nó yield từng token từ LLM.
        """
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.llama_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True  # Quan trọng: Bật streaming
        }
        
        url = f"{self.openai_base_url}/chat/completions"
        logging.debug(f"LLMClient: Gửi streaming request đến {url}")

        try:
            async with self._session.post(url, headers=headers, json=payload, timeout=120) as response:
                response.raise_for_status()
                
                # Xử lý stream theo định dạng Server-Sent Events (SSE)
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            logging.debug("LLMClient: Stream kết thúc.")
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            logging.warning(f"LLMClient: Không thể decode JSON từ stream: {data_str}")
                            continue
        except asyncio.TimeoutError:
            logging.error("LLMClient: Request timed out.")
            yield "<!-- LỖI TIMEOUT -->"
        except aiohttp.ClientError as e:
            logging.error(f"LLMClient: Lỗi aiohttp client: {e}", exc_info=True)
            yield "<!-- LỖI KẾT NỐI -->"

    async def chat_completion(self, messages: list, temperature: float = 0.3, max_tokens: int = 256) -> str:
        """
        Hàm non-streaming, bất đồng bộ. Sẽ tích lũy token từ stream để trả về kết quả cuối cùng.
        """
        full_response = ""
        async for token in self.streaming_chat_generator(messages, temperature, max_tokens):
            if "<!--" not in token: # Bỏ qua các token lỗi
                full_response += token
        return full_response
