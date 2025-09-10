import logging
import json
import time
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.core.nlp_llm_client import LLMClient
from src.utils.guardrails import redact_pii, unredact_pii
from src.utils.metrics import NLP_LATENCY_SECONDS, LLM_ERRORS_TOTAL
from src.utils.tracing import tracer

# --- Agent State Definition ---
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]

# --- Simplified Agent Graph (No Tools) ---
class LangGraphAgent:
    def __init__(self, llm_client_config: dict):
        self.llm = ChatOpenAI(
            model=llm_client_config['llama_model'],
            api_key=llm_client_config['openai_api_key'],
            base_url=llm_client_config['openai_base_url'],
            temperature=0.3,
            streaming=True
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.set_entry_point("agent")
        workflow.add_edge("agent", END)
        return workflow.compile()

    async def _call_model(self, state: AgentState):
        """Node: Calls the language model."""
        logging.debug("AGENT GRAPH: Calling model (no tools)")
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": [response]}

# --- NLPModule Wrapper ---
class NLPModule:
    def __init__(self, openai_base_url: str, openai_api_key: str, llama_model: str):
        llm_config = {
            'openai_base_url': openai_base_url,
            'openai_api_key': openai_api_key,
            'llama_model': llama_model
        }
        self.agent = LangGraphAgent(llm_config)
        self.llm_client = LLMClient(openai_base_url, openai_api_key, llama_model)

    async def close_client_session(self):
        await self.llm_client.close_session()

    @NLP_LATENCY_SECONDS.time()
    async def process_user_input(self, user_text: str, history: list = None) -> dict:
        logging.info(f"NLP Agent: Input gốc: '{user_text}'")
        redacted_text, pii_map = redact_pii(user_text)
        if pii_map:
            logging.info(f"NLP Agent: Input đã lọc PII: '{redacted_text}'")

        messages = history or []
        messages.append(HumanMessage(content=redacted_text))

        try:
            final_state = await self.agent.graph.ainvoke({"messages": messages})
            bot_response_raw = final_state["messages"][-1].content
        except Exception as e:
            LLM_ERRORS_TOTAL.labels(type='agent_error').inc()
            logging.error(f"NLP Agent: Lỗi khi chạy graph: {e}", exc_info=True)
            bot_response_raw = "Xin lỗi, tôi đang gặp sự cố nội bộ."

        final_bot_response = unredact_pii(bot_response_raw, pii_map)
        logging.info(f"NLP Agent: Phản hồi cuối cùng: '{final_bot_response}'")
        intent = "end_conversation" if any(kw in user_text.lower() for kw in ["tạm biệt", "kết thúc"]) else "continue_conversation"
        return {"response_text": final_bot_response, "intent": intent, "emotion": "neutral"}

    async def streaming_process_user_input(self, user_text: str, history: list = None):
        with tracer.start_as_current_span("nlp.streaming_process") as span:
            span.set_attribute("user.input.length", len(user_text))
            start_time = time.time()
            try:
                logging.info(f"NLP Agent (Streaming): Đang xử lý: '{user_text}'")
                redacted_text, pii_map = redact_pii(user_text)
                if pii_map:
                    logging.info(f"NLP Agent (Streaming): Input đã lọc PII: '{redacted_text}'")

                messages = history or []
                messages.append(HumanMessage(content=redacted_text))

                async for event in self.agent.graph.astream_events({"messages": messages}, version="v1"):
                    kind = event["event"]
                    if kind == "on_chat_model_stream":
                        chunk = event["data"].get("chunk")
                        if chunk:
                            content = chunk.content
                            if content:
                                if "<!-- LỖI TIMEOUT -->" in content:
                                    LLM_ERRORS_TOTAL.labels(type='timeout').inc()
                                elif "<!-- LỖI KẾT NỐI -->" in content:
                                    LLM_ERRORS_TOTAL.labels(type='client_error').inc()
                                yield content
            finally:
                duration = time.time() - start_time
                span.set_attribute("duration.seconds", duration)
                NLP_LATENCY_SECONDS.observe(duration)
                logging.info(f"NLP Agent (Streaming): Hoàn tất stream trong {duration:.2f} giây.")