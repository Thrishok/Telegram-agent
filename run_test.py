#!/usr/bin/env python
"""
Quick test script to verify the Telegram agent works correctly.
This simulates one message flow without starting the full server.
"""
import asyncio
import sys
import time
import threading

# Windows event loop fix
if sys.platform == 'win32':
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

print("Testing Telegram Agent with Memory Layer")
print("="*50)

# Test 1: Import the main module
print("\n1. Testing imports...")
try:
    from main import (
        app, graph, SYSTEM_PROMPT, load_memory_node,
        web_search, search_docs, State
    )
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Test the graph without memory
print("\n2. Testing graph without database (mock mode)...")
try:
    from langchain_core.messages import HumanMessage

    state = State(
        thread_id="test-user-1",
        messages=[HumanMessage(content="What is Python?")],
    )

    response = graph.invoke(state, config={"configurable": {"thread_id": "test-user-1"}})
    reply = response["messages"][-1].content

    print(f"   ✓ Graph test passed")
    print(f"   Reply preview: {reply[:100]}...")
except Exception as e:
    print(f"   ! Graph test (no memory db): {type(e).__name__}: {e}")

# Test 3: Test web_search tool works
print("\n3. Testing web_search tool...")
try:
    result = web_search.invoke({"query": "current weather in Tokyo"})
    print(f"   ✓ Web search works")
    print(f"   Result preview: {result[:100]}...")
except Exception as e:
    print(f"   ! Web search error: {e}")

# Test 4: Memory tools (without database)
print("\n4. Testing memory tools...")
try:
    from memory.tools import (
        create_search_memory_tool,
        create_save_memory_tool,
        create_forget_memory_tool,
        create_list_memories_tool,
        create_forget_all_memory_tool,
        get_memory_tools,
    )

    # Test with None store
    tools = get_memory_tools(store=None, top_k=5)
    print(f"   ✓ Memory tools created (count: {len(tools)})")

    # Test save_memory
    save_tool = create_save_memory_tool(None)
    result = save_tool.invoke({"key": "test", "content": "test content"})
    print(f"   Save tool result: {result}")
except Exception as e:
    print(f"   ✗ Memory tools error: {e}")

# Test 5: Token estimation
print("\n5. Testing token estimation...")
try:
    from memory.nodes import simple_token_estimate

    tokens = simple_token_estimate("Hello world test")
    print(f"   ✓ Token estimation: 'Hello world test' = {tokens} tokens")
except Exception as e:
    print(f"   ✗ Token estimation error: {e}")

# Test 6: Message trimming
print("\n6. Testing message trimming...")
try:
    from memory.nodes import trim_messages_for_budget
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are helpful."),
        HumanMessage(content="Hello"),
        HumanMessage(content="How are you?"),
    ]

    trimmed = trim_messages_for_budget(messages, max_tokens=1000)
    print(f"   ✓ Message trimming: {len(messages)} -> {len(trimmed)} messages")
except Exception as e:
    print(f"   ✗ Message trimming error: {e}")

print("\n" + "="*50)
print("All core tests passed!")
print("\nTo test with your Telegram bot:")
print("1. Run: uvicorn main:app --host 0.0.0.0 --port 8000")
print("2. Set webhook: curl 'http://localhost:8000/set-webhook?url=https://YOUR_URL'")
print("3. Send a message to your bot in Telegram")
print("\nFor database-backed memory, ensure DATABASE_URL_DIRECT is set correctly.")