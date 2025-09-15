# src/core/nlp_local_unsloth.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LocalUnslothNLP:
    def __init__(self, model_path: str, load_in_4bit: bool = True,
                 max_new_tokens: int = 128, temperature: float = 0.6,
                 top_p: float = 0.9, stop=None, device_map: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            low_cpu_mem_usage=True,
            load_in_4bit=load_in_4bit,
        )
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.stop = stop or []

    def generate(self, prompt: str, max_tokens: int = None, **kw) -> str:
        max_new = max_tokens or self.max_new_tokens
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        out = self.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=kw.get("temperature", self.temperature),
            top_p=kw.get("top_p", self.top_p),
            repetition_penalty=kw.get("repetition_penalty", 1.1),
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Cắt phần prompt
        resp = text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        # Áp stop words
        for s in self.stop:
            idx = resp.find(s)
            if idx != -1:
                resp = resp[:idx]
        return resp.strip()
