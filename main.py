import os
import base64
import tempfile
import requests
from typing import Annotated
from typing_extensions import TypedDict
from fastapi import FastAPI, Request
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient
import uvicorn

# ---------------------------------------------------------
# API Keys â€” all via os.getenv()
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ---------------------------------------------------------
# Tools (from agent notebook)
# ---------------------------------------------------------
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> str:
    """browser search tool to get latest news and confirms facts when not sure about that"""
    response = tavily_client.search(query=query, max_results=3)
    results = response.get("results", [])
    ans = "\n\n".join(f"{r['title']}: {r['content']}" for r in results)
    return ans

@tool
def search_docs(query: str) -> str:
    """Search internal documents for a given query."""
    return f"Found 3 results for '{query}'."

# ---------------------------------------------------------
# LLM + LangGraph Agent (from agent notebook)
# ---------------------------------------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=GROQ_API_KEY,
)

class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = [web_search, search_docs]
llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """You are a helpful assistant with access to:
1. web_search(query) - search the web for current information, facts, and real-world knowledge
2. search_docs(query) - search internal documents

CRITICAL: You MUST call web_search for ANY question about real-world facts, current events, people, places, dates, or anything requiring up-to-date information. NEVER answer factual questions from your own knowledge â€” always verify with web_search first. Only answer directly for simple greetings or conversational responses."""

def tool_calling_node(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(State)
builder.add_node("tool_calling_node", tool_calling_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "tool_calling_node")
builder.add_conditional_edges("tool_calling_node", tools_condition)
builder.add_edge("tools", "tool_calling_node")

graph = builder.compile()

# ---------------------------------------------------------
# Vision LLM (from telebot notebook â€” temperature 0.0 for vision)
# ---------------------------------------------------------
vision_llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.0,
    api_key=GROQ_API_KEY,
)

# ---------------------------------------------------------
# Helper: encode image to base64 (from telebot notebook)
# ---------------------------------------------------------
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# ---------------------------------------------------------
# Helper: download Telegram file (from telebot notebook)
# ---------------------------------------------------------
def download_telegram_file(file_id: str) -> str:
    response = requests.get(
        f"{TELEGRAM_API_URL}/getFile",
        params={"file_id": file_id},
    )
    file_info = response.json()
    file_path = file_info["result"]["file_path"]

    download_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
    file_response = requests.get(download_url)

    filename = os.path.join(tempfile.gettempdir(), os.path.basename(file_path))
    with open(filename, "wb") as f:
        f.write(file_response.content)
    return filename

# ---------------------------------------------------------
# Helper: analyze image with caption (from telebot notebook)
# ---------------------------------------------------------
def analyze_image(image_path: str, caption: str = None) -> str:
    base64_image = encode_image(image_path)

    if caption:
        text_prompt = f"What is in this image? Please describe it in detail. Caption: {caption}"
    else:
        text_prompt = "What is in this image? Please describe it in detail."

    message = HumanMessage(
        content=[
            {"type": "text", "text": text_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]
    )

    system_message = SystemMessage(
        content="You are a helpful assistant that analyzes images in detail and give the details in short and don't loose important information"
    )

    response = vision_llm.invoke([system_message, message])
    return response.content

# ---------------------------------------------------------
# Helper: send Telegram message (from GitHub repo)
# ---------------------------------------------------------
def send_message(chat_id: int, text: str):
    url = f"{TELEGRAM_API_URL}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": text})
    return response.json()

# ---------------------------------------------------------
# FastAPI App (from GitHub repo structure)
# ---------------------------------------------------------
app = FastAPI()

@app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    data = await request.json()

    message = data.get("message")
    if not message:
        return {"ok": True}

    chat_id = message["chat"]["id"]

    # --- Handle photo messages (from telebot notebook) ---
    if "photo" in message:
        photo = message["photo"][-2]
        file_id = photo["file_id"]
        caption = message.get("caption", None)

        try:
            filename = download_telegram_file(file_id)
            reply_text = analyze_image(filename, caption)
        except Exception as e:
            reply_text = f"Sorry, something went wrong: {e}"

        send_message(chat_id, reply_text)
        return {"ok": True}

    # --- Handle text messages (LangGraph agent) ---
    if "text" not in message:
        return {"ok": True}

    user_text = message["text"]

    try:
        messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)]
        result = graph.invoke({"messages": messages})
        reply_text = result["messages"][-1].content
    except Exception as e:
        reply_text = f"Sorry, something went wrong: {e}"

    send_message(chat_id, reply_text)
    return {"ok": True}

@app.get("/")
async def health_check():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
