import os
from llama_cpp import Llama

_llm = None
def _get():
    global _llm
    if _llm: return _llm
    _llm = Llama(
        model_path=os.getenv("LLAMA_MODEL_PATH"),
        n_gpu_layers=999,                         # offload tối đa
        tensor_split=[0.2,0.2,0.2,0.15,0.15,0.1],# 6 GPU
        main_gpu=0,
        logits_all=False,
        n_ctx=4096,
        n_batch=512,
        n_threads=8,
        use_mlock=True,
        use_mmap=True,
        gpulayers_fast=True,
        flash_attn=True
    )
    return _llm

async def llama_infer(text: str):
    llm = _get()
    prompt = f"""Bạn là trợ lý CSKH tiếng Việt. Phân tích: intent, slot, emotion. Câu: {text}"""
    out = llm.create_completion(prompt, max_tokens=128, temperature=0.3)
    return parse_json(out["choices"][0]["text"])
