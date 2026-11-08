import os
import base64
import tempfile
import json
import logging
import requests
import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any
from typing_extensions import TypedDict
from fastapi import FastAPI, Request
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from tavily import TavilyClient
import uvicorn
from dotenv import load_dotenv

# ---------------------------------------------------------
# Windows compatibility: Use SelectorEventLoop for psycopg
# ---------------------------------------------------------
if sys.platform == 'win32':
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

# ---------------------------------------------------------
# Load .env file — MUST happen before reading env vars
# ---------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------
# API Keys — all via os.getenv()
# ---------------------------------------------------------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# ---------------------------------------------------------
# Logging setup
# ---------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Memory configuration
# ---------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "384"))
SHORT_TERM_WINDOW = int(os.getenv("SHORT_TERM_WINDOW", "10"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_DIRECT = os.getenv("DATABASE_URL_DIRECT", DATABASE_URL)

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
# Memory tools (will be created with actual store at runtime)
# ---------------------------------------------------------
memory_tools = []
_store_ref = None  # Will hold the memory store reference

def register_memory_tools(store):
    """Register memory tools with the actual store instance."""
    global memory_tools, _store_ref
    _store_ref = store
    from memory.tools import get_memory_tools
    memory_tools = get_memory_tools(store, MEMORY_TOP_K)

# ---------------------------------------------------------
# LLM setup
# ---------------------------------------------------------
from langchain_groq import ChatGroq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    api_key=GROQ_API_KEY,
)
llm_with_tools = llm.bind_tools([web_search, search_docs])

SYSTEM_PROMPT = """You are a helpful assistant with access to:
1. web_search(query) - search the web for current information, facts, and real-world knowledge
2. search_docs(query) - search internal documents

CRITICAL: You MUST call web_search for ANY question about real-world facts, current events, people, places, dates, or anything requiring up-to-date information. NEVER answer factual questions from your own knowledge — always verify with web_search first. Only answer directly for simple greetings or conversational responses."""

# ---------------------------------------------------------
# State definition with memory support
# ---------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context_messages: Annotated[list, add_messages] = []  # Working context after memory loading
    thread_id: str = ""  # Telegram chat_id for thread identification
    summary: str = ""  # Running summary for long conversations
    user_profile: dict = {}  # Profile data from memory

# ---------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------
def simple_token_estimate(text: str) -> int:
    """Simple character-to-token estimation (approx 1 token per 4 chars)."""
    if isinstance(text, str):
        return len(text) // 4
    return 0

# ---------------------------------------------------------
# Memory node - loads user context and profile from store
# ---------------------------------------------------------
def load_memory_node(state: State) -> dict:
    """Load user memory from long-term store and assemble context."""
    thread_id = state.get("thread_id", "")

    messages = state.get("messages", [])
    recent = messages[-SHORT_TERM_WINDOW:] if len(messages) > SHORT_TERM_WINDOW else messages

    user_profile = {}
    retrieved_memories = []

    # If we have a store reference (set after async init), use it
    if _store_ref is not None and thread_id:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context - use run_in_executor for sync store access
                pass
            else:
                # Try direct sync access (PostgresStore has sync methods)
                profile_result = _store_ref.get(("users", thread_id, "profile"))
                if profile_result:
                    user_profile = profile_result.value if hasattr(profile_result, 'value') else profile_result
                    if not isinstance(user_profile, dict):
                        user_profile = {}

                search_results = _store_ref.search(
                    ("users", thread_id, "facts"),
                    query=" ".join(m.content for m in messages if isinstance(m, (HumanMessage, SystemMessage)))[:500],
                    limit=MEMORY_TOP_K
                )
                retrieved_memories = [r.value for r in search_results]
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

    # Build context messages
    context_messages = []

    # Add system prompt
    context_messages.append(SystemMessage(content=SYSTEM_PROMPT))

    # Add profile info if we have it
    if user_profile:
        context_messages.append(SystemMessage(content=f"User profile: {user_profile}"))

    # Add retrieved memories
    if retrieved_memories:
        context_messages.append(SystemMessage(content="Relevant memories: " + "; ".join(str(m) for m in retrieved_memories)))

    # Add recent messages
    context_messages.extend(recent)

    return {
        **state,
        "context_messages": context_messages,
        "user_profile": user_profile,
    }

# ---------------------------------------------------------
# LLM node - uses context_messages if available
# ---------------------------------------------------------
def call_llm_node(state: State) -> dict:
    """Main LLM node that uses context_messages if available."""
    messages = state.get("context_messages", state.get("messages", []))
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ---------------------------------------------------------
# Graph building function
# ---------------------------------------------------------
def build_graph():
    """Build and return the compiled graph."""
    builder = StateGraph(State)
    builder.add_node("load_memory", load_memory_node)
    builder.add_node("call_llm", call_llm_node)
    builder.add_node("tools", ToolNode([web_search, search_docs] + memory_tools))

    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "call_llm")
    builder.add_conditional_edges("call_llm", tools_condition)
    builder.add_edge("tools", "call_llm")

    return builder.compile()

# Compile initial graph (will be rebuilt after memory init)
graph = build_graph()

# ---------------------------------------------------------
# Vision LLM (from telebot notebook — temperature 0.0 for vision)
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
# Async memory initialization
# ---------------------------------------------------------
_checkpointer = None
_pool = None

async def initialize_memory():
    """Initialize the checkpointer and store at startup."""
    global graph, _checkpointer, _pool

    if not DATABASE_URL_DIRECT:
        logger.warning("DATABASE_URL_DIRECT not set, running without persistent memory")
        return

    try:
        from psycopg_pool import AsyncConnectionPool

        # Create pool with PgBouncer settings
        _pool = AsyncConnectionPool(
            conninfo=DATABASE_URL_DIRECT,
            max_size=10,
            min_size=1,
            kwargs={"prepare_threshold": 0, "autocommit": True},
        )

        # Create checkpointer for short-term memory
        async with _pool.connection() as conn:
            _checkpointer = PostgresSaver(conn)
            await _checkpointer.setup()

        # Create store for long-term memory
        async with _pool.connection() as conn:
            index_config = {
                "dims": EMBEDDING_DIMS,
                "embed": EMBEDDING_MODEL
            }
            store = PostgresStore(conn, index=index_config)
            await store.setup()

            # Register memory tools with store (sets _store_ref)
            register_memory_tools(store)

        # Rebuild graph with memory tools and checkpointer
        graph = build_graph()
        compiled = graph.compile(checkpointer=_checkpointer)

        logger.info("Memory layer initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize memory layer: {e}")
        import traceback
        traceback.print_exc()

# ---------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    """Initialize memory layer on app startup."""
    await initialize_memory()

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

    # --- Handle text messages (LangGraph agent with memory) ---
    if "text" not in message:
        return {"ok": True}

    user_text = message["text"]

    try:
        # Prepare initial state with thread_id for memory
        initial_state: State = {
            "thread_id": str(chat_id),  # Use chat_id as thread_id
            "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_text)],
        }

        # Invoke with thread_id for short-term memory persistence
        result = graph.invoke(initial_state, config={"configurable": {"thread_id": str(chat_id)}})
        reply_text = result["messages"][-1].content
    except Exception as e:
        logger.exception("Error processing message")
        reply_text = f"Sorry, something went wrong: {e}"

    send_message(chat_id, reply_text)
    return {"ok": True}

@app.get("/")
async def health_check():
    return {"status": "running"}

# ---------------------------------------------------------
# Webhook setup — call this once after starting the server
# to tell Telegram where to send messages
# Usage: GET /set-webhook?url=https://your-public-url/telegram-webhook
# ---------------------------------------------------------
@app.get("/set-webhook")
async def set_webhook(url: str = None):
    if not url:
        return {"error": "Provide ?url=https://your-public-url/telegram-webhook"}
    webhook_url = url.rstrip("/") + "/telegram-webhook"
    resp = requests.get(
        f"{TELEGRAM_API_URL}/setWebhook",
        params={"url": webhook_url},
    )
    return resp.json()

@app.get("/webhook-info")
async def webhook_info():
    resp = requests.get(f"{TELEGRAM_API_URL}/getWebhookInfo")
    return resp.json()

# ---------------------------------------------------------
# Memory management endpoints
# ---------------------------------------------------------
@app.post("/memory/search")
async def memory_search(query: str, namespace_key: str = '["users", "1", "facts"]'):
    """Search memories. For testing only."""
    return {"error": "Use direct database connection for memory operations"}

@app.post("/memory/delete-user")
async def memory_delete_user(chat_id: int):
    """Delete all memories for a user."""
    return {"error": "Use direct database connection for memory operations"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))