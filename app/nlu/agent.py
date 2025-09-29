from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from app.tools.crm import crm_lookup, crm_update
from app.nlu.llama import llama_infer

class Agent:
    @classmethod
    async def create(cls):
        g = StateGraph(dict)
        async def nlp_node(state):
            query = state["text"]
            intent, slots, emotions = await llama_infer(query)
            state.update(dict(intent=intent, slots=slots, emotions=emotions))
            return state
        async def tool_node(state):
            if state["intent"] in {"CHECK_ORDER", "CHECK_TICKET"}:
                data = await crm_lookup(state["slots"])
                state["tool_result"] = data
            return state
        async def reply_node(state):
            # render câu trả lời ngắn phù hợp cảm xúc
            state["reply"] = render_reply(state)
            return state

        g.add_node("nlp", nlp_node)
        g.add_node("tool", tool_node)
        g.add_node("reply", reply_node)
        g.add_edge(START, "nlp")
        g.add_edge("nlp", "tool")
        g.add_edge("tool", "reply")
        return cls(graph=g)

    def __init__(self, graph): self.graph = graph
    async def respond(self, text):
        s = await self.graph.ainvoke({"text": text})
        return s["reply"]
