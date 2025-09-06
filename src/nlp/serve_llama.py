
import sys
import os

# Add the project root to sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import os
import torch
import uuid
from src.tts.generate_audio import generate_audio
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict
from unsloth import FastLanguageModel
from transformers import TextStreamer, pipeline
import uvicorn
from loguru import logger
from langgraph.graph import StateGraph, END, START

# ========== Configuration ==========
MODEL_NAME = "unsloth/Llama-4-Scout-17B-16E-Instruct-unsloth-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True
DEVICE = "auto" # "auto" will use CUDA if available

# ========== Model Loading ==========
logger.info("Loading Llama model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=LOAD_IN_4BIT,
)
logger.info("Model loaded successfully.")

FastLanguageModel.for_inference(model)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

You have access to the following tools:
{tools_json}

Use the following format:

Instruction: the input instruction
Input: the input to the instruction
Thought: you should always think about what to do
Tool Call: the tool to call, if any
Tool Output: the result of the tool call
Response: the final answer to the user

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# ========== Tool Definitions (for Function Calling) ==========
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_order_status",
            "description": "Get the status of a customer's order based on their order ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "The unique identifier of the order.",
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
    tool_args = tool_call["function"]["arguments"]
    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

    if tool_name == "get_customer_order_status":
        order_id = tool_args.get("order_id")
        if order_id:
            # In a real scenario, this would query a CRM or database
            if order_id == "12345":
                return "Order 12345 is currently being processed and is expected to ship within 2 business days."
            elif order_id == "67890":
                return "Order 67890 has been delivered on 2025-08-28."
            else:
                return f"Order {order_id} not found."
        else:
            return "Order ID is required to get order status."
    else:
        return f"Unknown tool: {tool_name}"

# ========== LangGraph Agent State ==========
class AgentState(TypedDict):
    """
    Represents the state of our agent.

    Attributes:
        messages: A list of messages in the current conversation.
        tool_calls: A list of tool calls made by the model.
        tool_output: The output from the last tool call.
    """
    messages: List[Dict[str, Any]]
    tool_calls: List[Dict[str, Any]]
    tool_output: Optional[str]

# ========== Emotion Detection Model Loading ==========
logger.info("Loading Emotion Detection model...")
emotion_classifier = pipeline(
    "sentiment-analysis",
    model="clapAI/modernBERT-base-multilingual-sentiment",
    tokenizer="clapAI/modernBERT-base-multilingual-sentiment",
    device=0 if torch.cuda.is_available() else -1 # Use GPU if available
)
logger.info("Emotion Detection model loaded successfully.")

def detect_emotion(text: str) -> str:
    """
    Detects the sentiment/emotion of the given text.
    Returns 'POSITIVE', 'NEGATIVE', or 'NEUTRAL'.
    """
    if not text.strip():
        return "NEUTRAL"
    result = emotion_classifier(text)
    return result[0]['label'].upper()

# ========== FastAPI App ==========
# ========== LangGraph Nodes ==========
def call_model(state: AgentState) -> AgentState:
    """
    Calls the Llama model with the current messages and returns its response.
    """
    messages = state["messages"]
    user_message = messages[-1]["content"] # Assuming last message is user's
    system_prompt = "You are a helpful assistant." # Default system prompt

    # Extract system prompt from messages if available
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]

    # Detect emotion
    detected_emotion = detect_emotion(user_message)
    logger.info(f"Detected Emotion: {detected_emotion}")

    # Incorporate emotion into the prompt
    instruction_with_emotion = f"The user's sentiment is {detected_emotion}. Respond accordingly: {system_prompt}"

    formatted_alpaca_prompt = alpaca_prompt.format(
        tools_json=json.dumps(TOOLS, indent=2),
        instruction_with_emotion=instruction_with_emotion,
        user_message=user_message,
        response="",
    )

    inputs = tokenizer(
        [formatted_alpaca_prompt], return_tensors="pt").to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_SEQ_LENGTH, # Use MAX_SEQ_LENGTH for model generation
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    decoded_output = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    generated_text = decoded_output.split("### Instruction:")[-1].strip()

    logger.info(f"Raw Model Output: {generated_text}")

    # Check for Tool Call
    tool_calls = []
    response_content = generated_text

    if "Tool Call:" in generated_text:
        tool_call_str = generated_text.split("Tool Call:")[-1].split("Tool Output:")[0].strip()
        try:
            tool_call = json.loads(tool_call_str)
            tool_calls.append(tool_call)
            response_content = generated_text.split("Tool Call:")[0].strip() # Content before tool call
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse tool call JSON: {e} - Raw: {tool_call_str}")
            response_content = f"Error: Invalid Tool Call format. {e}"
    
    # Extract only the final response part if available
    if "Response:" in response_content:
        response_content = response_content.split("Response:")[-1].strip()
    
    return {"messages": messages + [{"role": "assistant", "content": response_content}], "tool_calls": tool_calls}

def call_tool(state: AgentState) -> AgentState:
    """
    Executes the tool called by the model and returns the tool's output.
    """
    tool_calls = state["tool_calls"]
    tool_output = ""
    if tool_calls:
        # Assuming only one tool call per turn for simplicity
        tool_output = execute_tool(tool_calls[0])
    
    return {"tool_output": tool_output}

# ========== Build LangGraph ==========
graph_builder = StateGraph(AgentState)

# Add nodes
graph_builder.add_node("call_model", call_model)
graph_builder.add_node("call_tool", call_tool)

# Define edges
graph_builder.add_edge(START, "call_model")
graph_builder.add_edge("call_tool", "call_model") # After tool call, re-call model

# Define conditional edge from call_model
def should_continue(state: AgentState) -> str:
    if state["tool_calls"]:
        return "continue"
    else:
        return "end"

graph_builder.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "call_tool",
        "end": END,
    },
)

# Compile the graph
agent_graph = graph_builder.compile()

# ========== FastAPI App ==========
app = FastAPI(
    title="Unsloth Llama-4 Maverick API",
    description="An OpenAI-compatible API for the Unsloth Llama-4 Maverick model.",
)

# --- Pydantic Models for OpenAI Compatibility ---
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False

class ChatCompletionChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-local"
    object: str = "chat.completion"
    created: int = 0
    model: str
    choices: List[ChatCompletionChoice]

# --- API Endpoint ---
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint using LangGraph.
    """
    logger.info(f"Received request for model: {request.model}")

    user_message = ""
    system_prompt = "You are a helpful assistant." # Default system prompt

    # Extract system prompt and last user message
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = msg.content
        elif msg.role == "user":
            user_message = msg.content

    # Initial state for LangGraph
    initial_state = {
        "messages": [{"role": "user", "content": user_message}],
        "tool_calls": [],
        "tool_output": None,
    }

    # Execute the graph
    final_state = agent_graph.invoke(initial_state)

    # Extract the final response from the messages
    response_text = ""
    for msg in final_state["messages"]:
        if msg["role"] == "assistant":
            response_text = msg["content"]
            break # Take the first assistant response

    logger.info(f"Final Agent Response: {response_text}")

    # Generate audio from the final response text
    audio_output_filename = f"response_{uuid.uuid4()}.wav" # Generate unique filename
    audio_output_path = os.path.join("audio_responses", audio_output_filename) # Store in a dedicated folder

    # Ensure the audio_responses directory exists
    os.makedirs("audio_responses", exist_ok=True)

    try:
        generate_audio(response_text, audio_output_path)
        logger.info(f"Generated audio response at: {audio_output_path}")
    except Exception as e:
        logger.error(f"Failed to generate audio for response: {e}")
        audio_output_path = None # Indicate failure

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content=response_text)
            )
        ]
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    logger.info("Starting Uvicorn server...")
    # Note: Running this directly is for testing. 
    # For production, use: uvicorn src.nlp.serve_llama:app --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
