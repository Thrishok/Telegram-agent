#!/usr/bin/env python
"""
Startup script for the Telegram agent with memory layer.
This script initializes the database schema and starts the FastAPI server.

Usage:
    python run_agent.py

Or for production with uvicorn:
    uvicorn run_agent:app --host 0.0.0.0 --port 8000
"""
import asyncio
import sys
import os

# Windows event loop fix - MUST be set before any psycopg imports
if sys.platform == 'win32':
    loop = asyncio.SelectorEventLoop()
    asyncio.set_event_loop(loop)

from dotenv import load_dotenv
load_dotenv()

# Import main app (which will set up event loop correctly for uvicorn)
from main import app, initialize_memory, logger, _store_ref, memory_tools

async def run_with_memory():
    """Initialize memory and return configured app."""
    await initialize_memory()
    if memory_tools:
        logger.info(f"Memory tools initialized: {[t.name for t in memory_tools]}")
    else:
        logger.warning("Memory layer not available (no database connection)")
    return app

def setup_database():
    """Run database setup script."""
    print("\n" + "="*60)
    print("Setting up database...")
    print("="*60)

    try:
        from memory.setup import main as setup_main
        setup_main()
        print("\n✓ Database setup complete")
        return True
    except Exception as e:
        print(f"\n✗ Database setup failed: {e}")
        print("\nIf you don't have a database yet, the agent will run in mock mode.")
        return False

def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("Telegram Agent with Memory Layer")
    print("="*60)

    # Check if database URL is set
    from memory.config import DATABASE_URL_DIRECT
    if not DATABASE_URL_DIRECT:
        print("\n! DATABASE_URL_DIRECT not set - will run in mock mode")
        print("  Memory features will not be available.")
        print("  Set DATABASE_URL_DIRECT in .env to enable memory persistence.")
    else:
        print(f"\n✓ DATABASE_URL_DIRECT is configured")

    # Setup database tables
    setup_database()

    # Start server
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    print(f"\nStarting FastAPI server on http://0.0.0.0:{port}")
    print("\nTest endpoints:")
    print(f"  Health: http://localhost:{port}/")
    print(f"  Webhook info: http://localhost:{port}/webhook-info")
    print("\nPress Ctrl+C to stop")

    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()