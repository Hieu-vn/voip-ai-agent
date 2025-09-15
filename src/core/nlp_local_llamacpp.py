# src/core/nlp_local_llamacpp.py
from llama_cpp import Llama

class LocalLlamaCppNLP:
    def __init__(self, gguf_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, threads: int = 8):
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, n_threads=threads, verbose=False)

    def generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.6, top_p: float = 0.9, **kw) -> str:
        out = self.llm(prompt=prompt, max_tokens=max_tokens, temperature=temperature, top_p=top_p, stop=["</s>", "###", "Human:", "Assistant:"])
        return out["choices"][0]["text"].strip()
