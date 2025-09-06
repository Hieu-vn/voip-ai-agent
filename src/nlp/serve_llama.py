import sys
import os

# Add the project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, TypedDict
from unsloth import FastLanguageModel
from transformers import pipeline
import uvicorn
from loguru import logger
from langgraph.graph import StateGraph, END, START

# ========== Configuration ==========
MODEL_NAME = "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
DEVICE = "auto" # "auto" will use CUDA if available

# ========== Model Loading ==========
logger.info("Đang tải model Llama 4...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)
logger.info("Tải model Llama 4 thành công.")

FastLanguageModel.for_inference(model)
# Prompt template for tool-using agent
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

You have access to the following tools:
{tools_json}

Use the following format:

Instruction: the user's request, potentially influenced by their emotion.
Input: the user's direct input.
Thought: you should always think about what to do.
Tool Call: the tool to call, in valid JSON format, if any
Tool Output: the result of the tool call
Response: the final, natural language answer to the user

### Instruction:
{instruction_with_emotion}

### Input:
{user_message}

### Response:
{response}"""

# ========== Tool Definitions (Function Calling) ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_order_status",
            "description": "Lấy thông tin trạng thái đơn hàng của khách hàng dựa vào mã đơn hàng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Mã định danh duy nhất của đơn hàng.",
                    }
                },
                "required": ["order_id"],
            },
        },
    }
]

# Placeholder for tool execution
def execute_tool(tool_call: Dict[str, Any]) -> str:
    tool_name = tool_call["function"]["name"]
    try:
        tool_args = json.loads(tool_call["function"]["arguments"])
    except json.JSONDecodeError:
        return "Error: Invalid arguments format. Expected a valid JSON string."

    logger.info(f"Thực thi công cụ: {tool_name} với tham số: {tool_args}")

    if tool_name == "get_customer_order_status":
        order_id = tool_args.get("order_id")
        if order_id:
            # Logic giả lập, thay thế bằng kết nối CRM/DB thật
            if order_id == "12345":
                return "Đơn hàng 12345 đang được xử lý và dự kiến giao trong 2 ngày làm việc."
            elif order_id == "67890":
                return "Đơn hàng 67890 đã được giao thành công vào ngày 28/08/2025."
            else:
                return f"Không tìm thấy đơn hàng với mã {order_id}.."
        else:
            return "Cần có mã đơn hàng để tra cứu."
    else:
        return f"Công cụ không xác định: {tool_name}"

# ========== LangGraph Agent State ==========
class AgentState(TypedDict):
    messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]

# ========== Emotion Detection Model Loading ==========
logger.info("Đang tải model Nhận diện Cảm xúc...")
emotion_classifier = pipeline(
    "sentiment-analysis",
    model="clapAI/modernBERT-base-multilingual-sentiment",
    device=0 if torch.cuda.is_available() else -1
)
logger.info("Tải model Nhận diện Cảm xúc thành công.")

def detect_emotion(text: str) -> str:
    if not text.strip():
        return "NEUTRAL"
    result = emotion_classifier(text)
    return result[0]['label'].upper()

# ========== LangGraph Nodes ==========
def call_model(state: AgentState) -> dict:
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the last message is from a tool, format it correctly
    if last_message["role"] == "tool":
        observation = last_message["content"]
        prompt_response = f"Tool Output: {observation}\nResponse:"
    else:
        # It's a user message
        user_message = last_message["content"]
        detected_emotion = detect_emotion(user_message)
        logger.info(f"Cảm xúc nhận diện được: {detected_emotion}")
        instruction_with_emotion = f"Cảm xúc của người dùng là {detected_emotion}. Hãy phản hồi một cách phù hợp."
        prompt_response = alpaca_prompt.format(
            tools_json=json.dumps(TOOLS, indent=2),
            instruction_with_emotion=instruction_with_emotion,
            user_message=user_message,
            response="",
        )

    inputs = tokenizer([prompt_response], return_tensors="pt").to(DEVICE)
    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    # Extract content after the final "Response:"
    generated_text = decoded_output.split("### Response:")[-1].strip()
    logger.info(f"Model Output Thô: {generated_text}")

    tool_calls = []
    response_content = generated_text
    
    if "Tool Call:" in generated_text:
        tool_call_str = generated_text.split("Tool Call:")[-1].split("Tool Output:")[0].strip()
        try:
            tool_call_data = json.loads(tool_call_str)
            # Ensure it's wrapped in a list if it's a single call
            if isinstance(tool_call_data, dict):
                tool_calls.append({"type": "function", "function": tool_call_data})
            response_content = generated_text.split("Tool Call:")[0].strip()
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi parse JSON của tool call: {e} - Dữ liệu thô: {tool_call_str}")
            response_content = "Lỗi: Định dạng Tool Call không hợp lệ."

    if "Response:" in response_content:
        response_content = response_content.split("Response:")[-1].strip()

    new_message = {"role": "assistant", "content": response_content}
    return {"messages": [new_message], "tool_calls": tool_calls}

def call_tool_node(state: AgentState) -> dict:
    tool_calls = state["tool_calls"]
    if not tool_calls:
        return {}
    
    tool_call = tool_calls[0]
    tool_output = execute_tool(tool_call)
    
    return {"messages": [{"role": "tool", "content": tool_output}]}

def should_call_tool(state: AgentState) -> str:
    return "call_tool" if state.get("tool_calls") and len(state["tool_calls"]) > 0 else END

# ========== Build LangGraph ==========
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("call_tool", call_tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_call_tool, {"call_tool": "call_tool", END: END})
workflow.add_edge("call_tool", "agent")
agent_graph = workflow.compile()

# ========== FastAPI App ==========
app = FastAPI(
    title="Unsloth Llama Agent API",
    description="OpenAI-compatible API for a Llama agent using Unsloth and LangGraph.",
)

# --- Pydantic Models for OpenAI Compatibility ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: List[ChatCompletionChoice]

# --- API Endpoint ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    logger.info(f"Nhận request cho model: {request.model}")
    
    # Convert Pydantic models to dicts for LangGraph
    messages_dict = [msg.dict() for msg in request.messages]
    
    initial_state = {"messages": messages_dict, "tool_calls": []}
    
    # Execute the graph
    final_state = agent_graph.invoke(initial_state)
    
    # Extract the final assistant response
    response_message = final_state["messages"][-1]
    
    logger.info(f"Phản hồi cuối cùng của Agent: {response_message['content']}")

    return ChatCompletionResponse(
        model=request.model,
        choices=[ChatCompletionChoice(message=Message(**response_message))]
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Bắt đầu Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)