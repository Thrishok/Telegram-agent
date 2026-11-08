#!/usr/bin/env python
"""
Test script to simulate Telegram webhook and test the memory layer.
Run this and send messages to your Telegram bot.

Usage:
1. Start this script: python test_telegram.py
2. Send /webhook followed by your message to test manually
"""
import asyncio
import sys
import json
import requests
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Windows event loop fix
if sys.platform == 'win32':
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_DIRECT = os.getenv("DATABASE_URL_DIRECT", DATABASE_URL)

TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# State
messages_state = []
last_response = ""

# Tools
from tavily import TavilyClient
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    response = tavily_client.search(query=query, max_results=3)
    results = response.get("results", [])
    ans = "\n\n".join(f"{r['title']}: {r['content']}" for r in results)
    return ans

@tool
def search_docs(query: str) -> str:
    """Search internal documents."""
    return f"Found 3 results for '{query}'."

SYSTEM_PROMPT = """You are a helpful assistant with access to:
1. web_search(query) - search the web for current information
2. search_docs(query) - search internal documents

CRITICAL: You MUST call web_search for ANY question about real-world facts, current events, people, places, dates. NEVER answer from your own knowledge."""

class State(TypedDict):
    messages: Annotated[list, add_messages]
    thread_id: str

# Try to initialize memory
try:
    from langgraph.checkpoint.postgres import PostgresSaver
    from langgraph.store.postgres import PostgresStore
    from memory.config import EMBEDDING_DIMS, MEMORY_TOP_K
    from memory.tools import get_memory_tools
    from psycopg_pool import AsyncConnectionPool

    # Initialize database connection
    _pool = AsyncConnectionPool(
        conninfo=DATABASE_URL_DIRECT,
        max_size=5,
        min_size=1,
        kwargs={"prepare_threshold": 0, "autocommit": True},
    )

    # Create checkpointer and store
    init_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(init_loop)

    async def init_memory():
        async with _pool.connection() as conn:
            checkpointer = PostgresSaver(conn)
            await checkpointer.setup()
            store = PostgresStore(conn, index={"dims": EMBEDDING_DIMS, "embed": "all-MiniLM-L6-v2"})
            await store.setup()
            return checkpointer, store

    _checkpointer, _store = init_memory.run_until_complete() if hasattr(init_memory, 'run_until_complete') else init_memory()
    memory_tools = get_memory_tools(_store, MEMORY_TOP_K)
    memory_initialized = True
    print("✓ Memory layer initialized successfully")

except Exception as e:
    memory_tools = []
    memory_initialized = False
    print(f"! Memory layer not available (database connection failed): {e}")

# Build graph
from langgraph.graph import StateGraph

llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.5, api_key=GROQ_API_KEY)
llm_with_tools = llm.bind_tools([web_search, search_docs] + memory_tools)

def load_memory_node(state: State) -> dict:
    thread_id = state.get("thread_id", "")
    messages = state.get("messages", [])
    recent = messages[-10:] if len(messages) > 10 else messages

    user_profile = {}
    if _store and thread_id:
        try:
            profile_result = _store.get(("users", thread_id, "profile"))
            if profile_result:
                user_profile = profile_result.value if hasattr(profile_result, 'value') else profile_result
        except:
            pass

    context_messages = []
    context_messages.append(SystemMessage(content=SYSTEM_PROMPT))
    if user_profile:
        context_messages.append(SystemMessage(content=f"User profile: {user_profile}"))
    context_messages.extend(recent)

    return {
        **state,
        "context_messages": context_messages,
        "user_profile": user_profile,
    }

def call_llm_node(state: State) -> dict:
    messages = state.get("context_messages", state.get("messages", []))
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "user_profile": state.get("user_profile", {})}

builder = StateGraph(State)
builder.add_node("load_memory", load_memory_node)
builder.add_node("call_llm", call_llm_node)
builder.add_node("tools", ToolNode([web_search, search_docs] + memory_tools))

builder.add_edge("load_memory", "call_llm")
builder.add_conditional_edges("call_llm", tools_condition)
builder.add_edge("tools", "call_llm")

if memory_initialized and _checkpointer:
    graph = builder.compile(checkpointer=_checkpointer)
else:
    graph = builder.compile()

def invoke_agent(text: str, thread_id: str = "test-user") -> str:
    """Test the agent with a message."""
    state: State = {
        "thread_id": thread_id,
        "messages": [HumanMessage(content=text)],
    }
    result = graph.invoke(state, config={"configurable": {"thread_id": thread_id}})
    return result["messages"][-1].content

def send_message(chat_id: int, text: str):
    """Send a message to Telegram."""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    response = requests.post(url, json={"chat_id": chat_id, "text": text})
    return response.json()

def main():
    print("\n" + "="*60)
    print("Telegram Agent Test Script")
    print("="*60)

    while True:
        try:
            user_input = input("\nEnter message (or 'quit' to exit): ").strip()
            if user_input.lower() == 'quit':
                break

            # Simulate Telegram user_id
            thread_id = "test-telegram-user"

            print(f"\nProcessing: {user_input}")
            response = invoke_agent(user_input, thread_id)
            print(f"\nResponse:\n{response}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()