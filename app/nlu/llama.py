import os
import json
from typing import Optional, List, Any, Dict

import structlog
from llama_cpp import Llama, LlamaGrammar

log = structlog.get_logger()

# Load GBNF grammar for structured JSON output
GRAMMAR = LlamaGrammar.from_file("grammars/intent_json.gbnf")

_llm = None
def _get():
    global _llm
    if _llm: return _llm
    log.info("Initializing Llama model")
    _llm = Llama(
        model_path=os.getenv("LLAMA_MODEL_PATH"),
        n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", "999")), # offload tối đa
        tensor_split=json.loads(os.getenv("LLAMA_TENSOR_SPLIT", "[0.2,0.2,0.2,0.15,0.15,0.1]")), # 6 GPU
        main_gpu=int(os.getenv("LLAMA_MAIN_GPU", "0")),
        logits_all=False,
        n_ctx=int(os.getenv("LLAMA_N_CTX", "4096")),
        n_batch=int(os.getenv("LLAMA_N_BATCH", "512")),
        n_threads=int(os.getenv("LLAMA_N_THREADS", "8")),
        use_mlock=True,
        use_mmap=True,
        gpulayers_fast=True,
        flash_attn=True
    )
    log.info("Llama model initialized")
    return _llm

def parse_json(text: str) -> dict:
    """
    Parses JSON from LLM output. Assumes LLM output is valid JSON due to GBNF grammar.
    """
    log.debug("Attempting to parse JSON from LLM output", text=text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.error("Failed to parse JSON from LLM output", text=text, exc_info=e)
        return {"intent": "PARSE_ERROR", "slots": {}, "confidence": 0.0}

async def llama_infer(text: str, emotion: str = "neutral", rag_context: Optional[List[Dict[str, Any]]] = None) -> tuple[str, dict, str, str]:
    llm = _get()
    
    context_str = ""
    if rag_context:
        context_str = "\n\nThông tin bổ sung:\n"
        for item in rag_context:
            context_str += f"- {item.get('text', '')}\n"

    prompt = f"""Bối cảnh: Người dùng đang có cảm xúc {emotion}.{context_str}\nDựa vào đó, hãy phân tích câu sau đây để xác định ý định (intent), các thông tin quan trọng (slots), và đưa ra câu trả lời phù hợp.

Câu nói: "{text}""""
    log.info("Sending prompt to Llama", prompt=prompt)
    out = llm.create_completion(prompt, max_tokens=128, temperature=0.3, grammar=GRAMMAR)
    llm_response = out["choices"][0]["text"]
    log.info("Llama response received", response=llm_response)
    parsed_output = parse_json(llm_response)
    return parsed_output.get("intent", "UNKNOWN"), parsed_output.get("slots", {}), emotion, llm_response