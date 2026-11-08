import os
from typing import Optional
from psycopg_pool import AsyncConnectionPool
from dotenv import load_dotenv

load_dotenv()

# Database settings
DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL_DIRECT = os.getenv("DATABASE_URL_DIRECT", DATABASE_URL)

# Memory settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMS = int(os.getenv("EMBEDDING_DIMS", "384"))
SHORT_TERM_WINDOW = int(os.getenv("SHORT_TERM_WINDOW", "10"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))
EPISODE_TTL_DAYS = int(os.getenv("EPISODE_TTL_DAYS", "90"))
ENABLE_MEMORY_EXTRACTION = os.getenv("ENABLE_MEMORY_EXTRACTION", "true").lower() == "true"
MEMORY_EXTRACTION_MODEL = os.getenv("MEMORY_EXTRACTION_MODEL", "openai/gpt-oss-120b")

# Neon specific: Use DIRECT for setup, POOLED for runtime
# pooled connection string often has ?sslmode=require
def get_pool_kwargs():
    kwargs = {
        "max_size": 10,
        "min_size": 1,
        "kwargs": {
            "prepare_threshold": 0,  # Required for PgBouncer / Neon pooled
            "autocommit": True,
        }
    }
    return kwargs

_pool: Optional[AsyncConnectionPool] = None

async def get_db_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL environment variable is not set")

        # Neon cold start resilience: pool handles reconnection
        _pool = AsyncConnectionPool(
            conninfo=DATABASE_URL,
            **get_pool_kwargs()
        )
    return _pool

async def close_db_pool():
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
