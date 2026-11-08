from typing import AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import IndexConfig
from memory.config import get_db_pool, EMBEDDING_DIMS, EMBEDDING_MODEL

@asynccontextmanager
async def get_store() -> AsyncGenerator[PostgresStore, None]:
    pool = await get_db_pool()
    async with pool.connection() as conn:
        # Index config enables vector search with pgvector
        # The embed parameter is informational - actual embeddings come from our extraction
        index_config: IndexConfig = {
            "dims": EMBEDDING_DIMS,
            "embed": EMBEDDING_MODEL
        }
        store = PostgresStore(conn, index=index_config)
        yield store
