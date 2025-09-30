import logging
import json
import time
import os
import operator
import torch
import threading
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from unsloth import FastLanguageModel
from transformers import pipeline, AutoTokenizer, TextIteratorStreamer
from llama_cpp import Llama # Import Llama for GGUF backend

from src.core.nlp_llm_client import LLMClient
from src.utils.guardrails import redact_pii, unredact_pii
from src.utils.metrics import NLP_LATENCY_SECONDS, LLM_ERRORS_TOTAL
from src.utils.tracing import tracer

# --- Agent State Definition ---
class AgentState(TypedDict):
    # Use operator.add for a valid reducer
    messages: Annotated[Sequence[BaseMessage], operator.add]

# --- Simplified Agent Graph (No Tools) ---
class LangGraphAgent:
    def __init__(self, model, tokenizer, backend_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.backend_type = backend_type
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        return workflow.compile()

    async def _call_model(self, state: AgentState):
        """
        Node: Calls the language model based on the backend type.
        """
        logging.debug(f"AGENT GRAPH: Calling local Llama model ({self.backend_type} backend)")
        
        messages_for_llama = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                messages_for_llama.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_for_llama.append({"role": "assistant", "content": msg.content})

        if self.backend_type == "unsloth":
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(messages_for_llama, tokenize=False, add_generation_prompt=True)
            
            # Generate response using Unsloth model
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = self.model.generate(**inputs, max_new_tokens=256, use_cache=True) # Adjust max_new_tokens
            response_text = self.tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)[0]
        
        elif self.backend_type == "llama_cpp":
            # Generate response using llama_cpp.Llama
            # llama_cpp handles chat templating internally if model supports it
            response = self.model.create_chat_completion(
                messages=messages_for_llama,
                max_tokens=256,
                temperature=0.6,
                top_p=0.9,
                min_p=0.01,
                stream=False, # Not streaming in _call_model
            )
            response_text = response["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
        
        return {"messages": [AIMessage(content=response_text)]}

    async def stream(self, state: AgentState):
        """Returns chunks of text (async generator) based on the backend type."""
        messages_for_llama = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                messages_for_llama.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages_for_llama.append({"role": "assistant", "content": msg.content})

        if self.backend_type == "unsloth":
            prompt = self.tokenizer.apply_chat_template(
                messages_for_llama, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs = dict(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                use_cache=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
                streamer=streamer,
            )

            def _gen():
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                    self.model.generate(**gen_kwargs)

            th = threading.Thread(target=_gen, daemon=True)
            th.start()

            for token in streamer:
                yield token
        
        elif self.backend_type == "llama_cpp":
            response_stream = self.model.create_chat_completion(
                messages=messages_for_llama,
                max_tokens=256,
                temperature=0.6,
                top_p=0.9,
                min_p=0.01,
                stream=True, # Enable streaming
            )
            for chunk in response_stream:
                if "content" in chunk["choices"][0]["delta"]:
                    yield chunk["choices"][0]["delta"]["content"]
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")

# --- NLPModule Wrapper ---
class NLPModule:
    def __init__(self, llama_model: str, llama_backend: str = "unsloth", llama_ctx_size: int = 4096, llama_gpu_layers: int = 99, llama_temp: float = 0.6, llama_top_p: float = 0.9, llama_min_p: float = 0.01):
        self.llama_model_path = llama_model
        self.llama_backend = llama_backend
        self.llama_ctx_size = llama_ctx_size
        self.llama_gpu_layers = llama_gpu_layers
        self.llama_temp = llama_temp
        self.llama_top_p = llama_top_p
        self.llama_min_p = llama_min_p

        self.model = None
        self.tokenizer = None
        self.agent = None

        # Initialize sentiment analysis pipeline
        try:
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="wonrax/phobert-base-vietnamese-sentiment",
                device=0 if torch.cuda.is_available() else -1
            )
            logging.info("Sentiment analysis pipeline initialized.")
        except Exception as e:
            logging.warning(f"Could not load sentiment analysis model: {e}. Emotion detection will be disabled.")
            self.sentiment_pipeline = None

    async def load_nlp_model(self):
        """
        Loads the Llama 4 Scout model and tokenizer based on the backend type.
        This function should be called once at startup.
        """
        if not self.llama_model_path:
            logging.error("LLAMA_MODEL_PATH environment variable not set. Cannot load NLP model.")
            return

        if self.llama_backend == "unsloth":
            logging.info(f"Loading Llama 4 Scout model from {self.llama_model_path} using Unsloth backend...")
            try:
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name = self.llama_model_path,
                    max_seq_length = self.llama_ctx_size,
                    dtype = None,
                    load_in_4bit = True,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(self.model) # Optimize for inference
                logging.info("Llama 4 Scout model loaded successfully with Unsloth.")
                
                # Initialize LangGraph agent after model is loaded
                self.agent = LangGraphAgent(self.model, self.tokenizer, self.llama_backend)

                # Warmup the model
                logging.info("Warming up Llama 4 Scout model...")
                with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                    _ = self.model.generate(self.tokenizer("Hello", return_tensors="pt").to(self.model.device), max_new_tokens=1)
                logging.info("Llama 4 Scout model warmed up.")

            except Exception as e:
                logging.error(f"Failed to load Llama 4 Scout model with Unsloth: {e}", exc_info=True)
        
        elif self.llama_backend == "llama_cpp":
            logging.info(f"Loading Llama 4 Scout GGUF model from {self.llama_model_path} using llama_cpp backend...")
            try:
                self.model = Llama(
                    model_path=self.llama_model_path,
                    n_ctx=self.llama_ctx_size,
                    n_gpu_layers=self.llama_gpu_layers,
                    verbose=False,
                )
                # For llama_cpp, tokenizer is often part of the model or handled by its chat completion
                # We might need a dummy tokenizer or load a specific one if chat templating is external
                self.tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit") # Use Unsloth's tokenizer for chat templating
                logging.info("Llama 4 Scout GGUF model loaded successfully with llama_cpp.")

                # Initialize LangGraph agent after model is loaded
                self.agent = LangGraphAgent(self.model, self.tokenizer, self.llama_backend)

            except Exception as e:
                logging.error(f"Failed to load Llama 4 Scout GGUF model with llama_cpp: {e}", exc_info=True)
        else:
            logging.error(f"Unknown LLAMA_BACKEND specified: {self.llama_backend}. Model not loaded.")

    async def close_client_session(self):
        # No external client session to close if using local model
        pass

    def analyze_emotion(self, text: str) -> str:
        """
        Phân tích cảm xúc của văn bản bằng mô hình Hugging Face.
        Trả về 'positive', 'negative', hoặc 'neutral'.
        """
        if not self.sentiment_pipeline:
            return "neutral" # Fallback if pipeline not loaded

        try:
            result = self.sentiment_pipeline(text)[0]
            label = result['label']
            score = result['score']
            logging.debug(f"Emotion analysis result: {label} (score: {score:.2f})")

            # Map common sentiment labels to our categories for wonrax/phobert-base-vietnamese-sentiment
            if label == "POS":
                return "positive"
            elif label == "NEG":
                return "negative"
            elif label == "NEU":
                return "neutral"
            else:
                return "neutral" # Default fallback
        except Exception as e:
            logging.warning(f"Error during emotion analysis: {e}. Returning neutral.")
            return "neutral"

    async def process_user_input(self, user_text: str, history: list = None) -> dict:
        if not self.agent: # Check if agent is initialized
            logging.error("NLP Agent not initialized. Model not loaded.")
            return {"response_text": "Xin lỗi, hệ thống NLP chưa sẵn sàng.", "intent": "error", "emotion": "neutral"}

        logging.info(f"NLP Agent: Input gốc: '{user_text}'")
        redacted_text = user_text
        pii_map = {}
        if pii_map:
            logging.info(f"NLP Agent: Input đã lọc PII: '{redacted_text}'")

        messages = history or []
        messages.append(HumanMessage(content=redacted_text))

        t0 = time.time() # Start timer for Prometheus
        try:
            final_state = await self.agent.graph.ainvoke({"messages": messages})
            bot_response_raw = final_state["messages"][-1].content
        finally:
            duration = time.time() - t0
            try:
                NLP_LATENCY_SECONDS.observe(duration) # Observe latency
            except Exception: # Catch any error during metric observation
                pass

        final_bot_response = unredact_pii(bot_response_raw, pii_map)
        logging.info(f"NLP Agent: Phản hồi cuối cùng: '{final_bot_response}'")
        intent = "end_conversation" if any(kw in user_text.lower() for kw in ["tạm biệt", "kết thúc"]) else "continue_conversation"
        emotion = self.analyze_emotion(user_text) # Use self.analyze_emotion
        return {"response_text": final_bot_response, "intent": intent, "emotion": emotion}

    async def streaming_process_user_input(self, user_text: str, history: list = None):
        if not self.agent: # Check if agent is initialized
            logging.error("NLP Agent not initialized. Model not loaded.")
            yield "Xin lỗi, hệ thống NLP chưa sẵn sàng."
            return

        with tracer.start_as_current_span("nlp.streaming_process") as span:
            span.set_attribute("user.input.length", len(user_text))
            user_emotion = self.analyze_emotion(user_text) # Use self.analyze_emotion
            span.set_attribute("user.emotion", user_emotion)
            logging.info(f"NLP Agent (Streaming): Cảm xúc người dùng: {user_emotion}")
            
            messages = history or []
            messages.append(HumanMessage(content=user_text)) # Use original text for streaming agent

            start_time = time.time()
            try:
                async for chunk in self.agent.stream({"messages": messages}):
                    # PII redaction/unredaction for streaming needs careful handling
                    # For simplicity, PII is not redacted/unredacted per chunk here.
                    # It's assumed to be handled at the full utterance level or by the LLM itself.
                    yield chunk
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration.seconds", duration)
                try:
                    NLP_LATENCY_SECONDS.observe(duration)
                except Exception: # Catch any error during metric observation
                    pass
                logging.info(f"NLP Agent (Streaming): Hoàn tất stream trong {duration:.2f} giây.")