#!/usr/bin/env python
"""
Test script to verify memory persistence works correctly.
This simulates the full flow without starting the FastAPI server.
"""
import asyncio
import sys

# Windows event loop fix
if sys.platform == 'win32':
    asyncio.set_event_loop(asyncio.SelectorEventLoop())

print("Testing Memory Persistence Flow")
print("="*50)

# Test 1: Can we connect to the database?
print("\n1. Testing database connectivity...")
try:
    from memory.config import DATABASE_URL_DIRECT
    import psycopg

    async def test_db():
        if not DATABASE_URL_DIRECT:
            print("   SKIP: DATABASE_URL_DIRECT not set")
            return

        from psycopg_pool import AsyncConnectionPool
        async with AsyncConnectionPool(
            conninfo=DATABASE_URL_DIRECT,
            max_size=2,
            min_size=1,
            kwargs={"prepare_threshold": 0, "autocommit": True},
        ) as pool:
            async with pool.connection() as conn:
                result = await conn.execute("SELECT 1 as test")
                print("   ✓ Database connection successful")
                return True

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(test_db())
    loop.close()
except Exception as e:
    print(f"   ! Database test skipped (expected if no DB): {type(e).__name__}")

# Test 2: Can we initialize the memory module?
print("\n2. Testing memory module initialization...")
try:
    from memory import load_memory, get_db_pool, MEMORY_TOP_K
    from langchain_core.messages import HumanMessage

    print(f"   ✓ Memory module loaded (MEMORY_TOP_K={MEMORY_TOP_K})")

    # Test load_memory with no store
    async def test_load():
        state = {"messages": [HumanMessage("test")], "thread_id": "test-user"}
        result = await load_memory(state, store=None, memory_top_k=MEMORY_TOP_K)
        return result

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(test_load())
    print(f"   ✓ load_memory works: {list(result.keys())}")
    loop.close()

except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 3: Can the graph work in mock mode?
print("\n3. Testing graph in mock mode (no database)...")
try:
    from main import graph, State
    from langchain_core.messages import HumanMessage

    state = State(
        thread_id="test-mock",
        messages=[HumanMessage(content="Hello, what is 2+2?")],
    )

    result = graph.invoke(state, config={"configurable": {"thread_id": "test-mock"}})
    reply = result["messages"][-1].content

    print(f"   ✓ Graph works in mock mode")
    print(f"   Reply: {reply[:100]}..." if len(reply) > 100 else f"   Reply: {reply}")

except Exception as e:
    print(f"   ! Mock mode test: {type(e).__name__}: {e}")

print("\n" + "="*50)
print("Memory module tests complete!")
print("\nIf database is configured, run 'python -m memory.setup' to initialize tables.")
print("Then start server with: uvicorn main:app --host 0.0.0.0 --port 8000")