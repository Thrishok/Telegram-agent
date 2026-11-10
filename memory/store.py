from typing import AsyncGenerator
from contextlib import asynccontextmanager
from sentence_transformers import SentenceTransformer
from langgraph.store.postgres.aio import AsyncPostgresStore
from memory.config import get_db_pool, EMBEDDING_DIMS, EMBEDDING_MODEL

_embedder = SentenceTransformer(EMBEDDING_MODEL)

def _embed(texts: list[str]) -> list[list[float]]:
    return _embedder.encode(texts, normalize_embeddings=True).tolist()

@asynccontextmanager
async def get_store() -> AsyncGenerator[AsyncPostgresStore, None]:
    pool = await get_db_pool()
    store = AsyncPostgresStore(pool, index={"dims": EMBEDDING_DIMS, "embed": _embed})
    yield store
