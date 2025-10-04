import os
from typing import Optional

import structlog
from langgraph.graph import START, StateGraph, END
from langchain_core.messages import HumanMessage
from app.tools.crm import crm_lookup, crm_update
from app.nlu.llama import llama_infer
from app.utils.guardrails import redact_pii, unredact_pii, is_content_safe
from app.utils.emotion_analyzer import EmotionAnalyzer
from app.dialog.state import DialogState
from app.nlu.policy import decide_next_action

log = structlog.get_logger()

# RAG retrieval is optional; skip if dependencies (e.g., Qdrant) are absent.
try:
    from app.rag.retriever import retrieve as rag_retrieve
except Exception:  # pragma: no cover - optional dependency
    rag_retrieve = None
    log.warning("RAG retriever unavailable; proceeding without knowledge base context.")

# Define a safe fallback message
GUARDRAIL_FALLBACK_MESSAGE = "Xin lỗi, tôi không thể hỗ trợ yêu cầu này. Bạn có cần giúp đỡ gì khác không?"

class Agent:
    def __init__(self, graph, emotion_analyzer):
        self.graph = graph
        self.emotion_analyzer = emotion_analyzer

    @classmethod
    async def create(cls):
        emotion_analyzer = EmotionAnalyzer()

        g = StateGraph(DialogState)
        
        async def rag_node(state: DialogState) -> DialogState:
            if rag_retrieve is None:
                state.slots["rag_context"] = []
                return state

            log.info("RAG node: Retrieving context", query=state.last_user_text, call_id=state.call_id)
            try:
                context = rag_retrieve(state.last_user_text)
            except Exception as exc:  # pragma: no cover - defensive logging
                log.warning("RAG retrieval failed; using empty context.", exc_info=exc, call_id=state.call_id)
                context = []
            state.slots["rag_context"] = context
            log.info("RAG node: Context retrieved", context=context, call_id=state.call_id)
            return state

        async def nlp_node(state: DialogState) -> DialogState:
            query = state.last_user_text
            log.info("NLU node received query", query=query, call_id=state.call_id)

            emotion = emotion_analyzer.analyze(query)
            log.info("Emotion analyzed", emotion=emotion, call_id=state.call_id)

            redacted_query, pii_map = redact_pii(query)
            if pii_map:
                log.info("PII redacted from query", redacted_query=redacted_query, call_id=state.call_id)

            # Pass RAG context to LLM
            rag_context = state.slots.get("rag_context", [])
            intent, slots, _, llm_response = await llama_infer(redacted_query, emotion=emotion, rag_context=rag_context)

            is_safe, violations = is_content_safe(llm_response)
            if not is_safe:
                log.warning("Unsafe LLM response detected, using fallback.", violations=violations, call_id=state.call_id)
                final_response = GUARDRAIL_FALLBACK_MESSAGE
            else:
                final_response = unredact_pii(llm_response, pii_map)
                if pii_map:
                    log.info("PII un-redacted in final response", final_response=final_response, call_id=state.call_id)

            state.intent = intent
            state.slots.update(slots)
            state.emotion = emotion
            state.last_bot_text = final_response
            state.turn += 1
            return state

        async def tool_node(state: DialogState) -> DialogState:
            if state.intent == "check_order" and "order_id" in state.slots:
                log.info("Calling CRM lookup tool", slots=state.slots, call_id=state.call_id)
                data = await crm_lookup(state.slots)
                state.slots["order_info"] = data
                log.info("CRM lookup tool result", result=data, call_id=state.call_id)
            return state

        async def policy_node(state: DialogState) -> DialogState:
            return decide_next_action(state)

        g.add_node("rag", rag_node)
        g.add_node("nlu", nlp_node)
        g.add_node("tool", tool_node)
        g.add_node("policy", policy_node)

        g.add_edge(START, "rag")
        g.add_edge("rag", "nlu")
        g.add_edge("nlu", "tool")
        g.add_edge("tool", "policy")
        g.add_edge("policy", END)
        
        compiled_graph = g.compile()
        return cls(graph=compiled_graph, emotion_analyzer=emotion_analyzer)

    async def respond(self, call_id: str, text: str, current_state: Optional[DialogState] = None) -> DialogState:
        if current_state is None:
            current_state = DialogState(call_id=call_id)
        current_state.last_user_text = text
        s = await self.graph.ainvoke(current_state)
        return s
