import os
import base64
import tempfile
import logging
import asyncio
import sys
import requests
from typing import Annotated
from typing_extensions import TypedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, BackgroundTasks
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg_pool import AsyncConnectionPool
from sentence_transformers import SentenceTransformer
from tavily import TavilyClient
from dotenv import load_dotenv
import uvicorn

load_dotenv()

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
BOT_TOKEN        = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY   = os.getenv("TAVILY_API_KEY", "")
DATABASE_URL     = os.getenv("DATABASE_URL")
DATABASE_URL_DIRECT = os.getenv("DATABASE_URL_DIRECT", DATABASE_URL)
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMS   = int(os.getenv("EMBEDDING_DIMS", "384"))
SHORT_TERM_WINDOW = int(os.getenv("SHORT_TERM_WINDOW", "10"))
MEMORY_TOP_K     = int(os.getenv("MEMORY_TOP_K", "5"))
ENABLE_MEMORY_EXTRACTION = os.getenv("ENABLE_MEMORY_EXTRACTION", "true").lower() == "true"
MEMORY_EXTRACTION_MODEL  = os.getenv("MEMORY_EXTRACTION_MODEL", "openai/gpt-oss-120b")
EPISODE_TTL_DAYS = int(os.getenv("EPISODE_TTL_DAYS", "90"))

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------
# Embedder (shared instance)
# ---------------------------------------------------------
_embedder = SentenceTransformer(EMBEDDING_MODEL)

def embed(texts: list[str]) -> list[list[float]]:
    return _embedder.encode(texts, normalize_embeddings=True).tolist()

# ---------------------------------------------------------
# LLMs
# ---------------------------------------------------------
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> str:
    """Search the web for current information, facts, and real-world knowledge."""
    results = tavily_client.search(query=query, max_results=3).get("results", [])
    return "\n\n".join(f"{r['title']}: {r['content']}" for r in results)

@tool
def search_docs(query: str) -> str:
    """Search internal documents for a given query."""
    return f"Found 3 results for '{query}'."

AGENT_TOOLS = [web_search, search_docs]

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5, api_key=GROQ_API_KEY)
vision_llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0, api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a helpful assistant with access to:
1. web_search(query) - search the web for current information
2. search_docs(query) - search internal documents

CRITICAL: Call web_search for ANY factual question. Only answer directly for simple greetings."""

# ---------------------------------------------------------
# State
# ---------------------------------------------------------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    context_messages: list
    thread_id: str
    summary: str
    user_profile: dict

# ---------------------------------------------------------
# Graph nodes (store injected via closure at startup)
# ---------------------------------------------------------
_store: AsyncPostgresStore | None = None

async def load_memory_node(state: State) -> dict:
    """Load profile + top-k facts/episodes, assemble context."""
    thread_id = state.get("thread_id", "")
    messages   = state.get("messages", [])
    recent     = messages[-SHORT_TERM_WINDOW:]

    user_profile, memories = {}, []

    if _store and thread_id:
        try:
            item = await _store.aget(("users", thread_id, "profile"), "profile")
            if item:
                user_profile = item.value if isinstance(item.value, dict) else {}
        except Exception as e:
            logger.warning(f"Profile load failed: {e}")

        try:
            query = " ".join(m.content for m in messages if hasattr(m, "content"))[-500:]
            facts = await _store.asearch(("users", thread_id, "facts"), query=query, limit=MEMORY_TOP_K)
            memories = [r.value for r in facts]
        except Exception as e:
            logger.warning(f"Facts search failed: {e}")

        try:
            eps = await _store.asearch(("users", thread_id, "episodes"), query="", limit=3)
            memories += [f"Past: {r.value.get('goal','')} → {r.value.get('outcome','')}" for r in eps if isinstance(r.value, dict)]
        except Exception as e:
            logger.warning(f"Episodes search failed: {e}")

    ctx: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
    if user_profile:
        ctx.append(SystemMessage(content=f"User profile: {user_profile}"))
    if memories:
        ctx.append(SystemMessage(content="Relevant memories: " + "; ".join(str(m) for m in memories)))
    summary = state.get("summary", "")
    if summary:
        ctx.append(SystemMessage(content=f"Conversation summary: {summary}"))
    ctx.extend(recent)

    return {**state, "context_messages": ctx, "user_profile": user_profile}


def call_llm_node(state: State) -> dict:
    messages = state.get("context_messages") or state.get("messages", [])
    response = llm.bind_tools(AGENT_TOOLS).invoke(messages)
    return {"messages": [response]}


# ---------------------------------------------------------
# Background memory extraction
# ---------------------------------------------------------
_last_extraction: dict[str, float] = {}
_DEBOUNCE = 30.0  # seconds

async def run_extraction(thread_id: str, messages: list, response_text: str):
    import time
    now = time.time()
    if now - _last_extraction.get(thread_id, 0) < _DEBOUNCE:
        return
    _last_extraction[thread_id] = now

    if not _store or not ENABLE_MEMORY_EXTRACTION:
        return

    try:
        from langmem import create_memory_store_manager
        extraction_llm = ChatGroq(model=MEMORY_EXTRACTION_MODEL, temperature=0.0, api_key=GROQ_API_KEY)
        manager = create_memory_store_manager(
            extraction_llm,
            namespace=("users", thread_id, "facts"),
            store=_store,
        )
        await manager.ainvoke({"messages": messages + [HumanMessage(content=response_text)]})
        logger.debug(f"Extraction done for {thread_id}")
    except Exception as e:
        logger.warning(f"Extraction failed for {thread_id}: {e}")


# ---------------------------------------------------------
# Graph + app lifecycle
# ---------------------------------------------------------
_graph = None
_pool: AsyncConnectionPool | None = None
_checkpointer: AsyncPostgresSaver | None = None

def build_graph(checkpointer=None):
    builder = StateGraph(State)
    builder.add_node("load_memory", load_memory_node)
    builder.add_node("call_llm", call_llm_node)
    builder.add_node("tools", ToolNode(AGENT_TOOLS))
    builder.add_edge(START, "load_memory")
    builder.add_edge("load_memory", "call_llm")
    builder.add_conditional_edges("call_llm", tools_condition)
    builder.add_edge("tools", "call_llm")
    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph, _pool, _checkpointer, _store

    if DATABASE_URL:
        try:
            _pool = AsyncConnectionPool(
                conninfo=DATABASE_URL,
                max_size=10,
                min_size=1,
                kwargs={"prepare_threshold": 0, "autocommit": True},
                open=False,
            )
            await _pool.open()

            _checkpointer = AsyncPostgresSaver(_pool)
            _store = AsyncPostgresStore(
                _pool,
                index={"dims": EMBEDDING_DIMS, "embed": embed},
            )
            _graph = build_graph(checkpointer=_checkpointer)
            logger.info("Memory layer initialized")
        except Exception as e:
            logger.error(f"Memory init failed: {e}; running without persistence")
            _graph = build_graph()
    else:
        logger.warning("DATABASE_URL not set; running without persistence")
        _graph = build_graph()

    yield

    if _pool:
        await _pool.close()


app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------
# Telegram helpers
# ---------------------------------------------------------
def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def download_telegram_file(file_id: str) -> str:
    info = requests.get(f"{TELEGRAM_API_URL}/getFile", params={"file_id": file_id}).json()
    file_path = info["result"]["file_path"]
    data = requests.get(f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}").content
    dest = os.path.join(tempfile.gettempdir(), os.path.basename(file_path))
    with open(dest, "wb") as f:
        f.write(data)
    return dest

def analyze_image(image_path: str, caption: str = None) -> str:
    b64 = encode_image(image_path)
    prompt = f"Describe this image in detail.{' Caption: ' + caption if caption else ''}"
    response = vision_llm.invoke([
        SystemMessage(content="Analyze images concisely without losing important details."),
        HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]),
    ])
    return response.content

def send_message(chat_id: int, text: str):
    requests.post(f"{TELEGRAM_API_URL}/sendMessage", json={"chat_id": chat_id, "text": text})

# ---------------------------------------------------------
# Webhook handler
# ---------------------------------------------------------
@app.post("/telegram-webhook")
async def telegram_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    message = data.get("message")
    if not message:
        return {"ok": True}

    chat_id = message["chat"]["id"]

    if "photo" in message:
        try:
            path = download_telegram_file(message["photo"][-2]["file_id"])
            reply = analyze_image(path, message.get("caption"))
        except Exception as e:
            reply = f"Sorry, image processing failed: {e}"
        send_message(chat_id, reply)
        return {"ok": True}

    if "text" not in message:
        return {"ok": True}

    user_text = message["text"]
    thread_id = str(chat_id)

    try:
        initial_state: State = {
            "thread_id": thread_id,
            "messages": [HumanMessage(content=user_text)],
            "context_messages": [],
            "summary": "",
            "user_profile": {},
        }
        result = await _graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        reply = result["messages"][-1].content

        # Fire background extraction (non-blocking)
        if ENABLE_MEMORY_EXTRACTION:
            background_tasks.add_task(run_extraction, thread_id, result["messages"], reply)

    except Exception as e:
        logger.exception("Error processing message")
        reply = f"Sorry, something went wrong: {e}"

    send_message(chat_id, reply)
    return {"ok": True}

# ---------------------------------------------------------
# Utility endpoints
# ---------------------------------------------------------
@app.get("/")
async def health():
    return {"status": "running", "memory": _store is not None}

@app.get("/set-webhook")
async def set_webhook(url: str = None):
    if not url:
        return {"error": "Provide ?url=https://your-public-url"}
    resp = requests.get(f"{TELEGRAM_API_URL}/setWebhook", params={"url": url.rstrip("/") + "/telegram-webhook"})
    return resp.json()

@app.get("/webhook-info")
async def webhook_info():
    return requests.get(f"{TELEGRAM_API_URL}/getWebhookInfo").json()

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop(asyncio.SelectorEventLoop())
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
