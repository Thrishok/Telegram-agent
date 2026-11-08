from typing import AsyncGenerator
from contextlib import asynccontextmanager
from langgraph.checkpoint.postgres import PostgresSaver
from memory.config import get_db_pool

@asynccontextmanager
async def get_checkpointer() -> AsyncGenerator[PostgresSaver, None]:
    pool = await get_db_pool()
    async with pool.connection() as conn:
        yield PostgresSaver(conn)
