import os, json
import requests

class LLMClient:
    def __init__(self, openai_base_url: str, openai_api_key: str, llama_model: str):
        self.openai_base_url = openai_base_url
        self.openai_api_key = openai_api_key
        self.llama_model = llama_model

    def chat_completion(self, messages: list, temperature: float = 0.3, max_tokens: int = 256) -> str:
        headers = {"Authorization": f"Bearer {self.openai_api_key}", "Content-Type":"application/json"}
        payload = {
            "model": self.llama_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        r = requests.post(f"{self.openai_base_url}/chat/completions", headers=headers, data=json.dumps(payload), timeout=60)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]