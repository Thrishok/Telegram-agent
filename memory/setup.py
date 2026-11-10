"""
Database setup script for memory module.
Run with: python -m memory.setup
"""
import asyncio
import os
import psycopg
from sentence_transformers import SentenceTransformer
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from dotenv import load_dotenv

# Load environment from .env in project root
load_dotenv()

DATABASE_URL_DIRECT = os.getenv('DATABASE_URL_DIRECT') or os.getenv('DATABASE_URL')
EMBEDDING_DIMS = int(os.getenv('EMBEDDING_DIMS', '384'))
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')


async def setup_database():
    print("Starting database setup...")
    if not DATABASE_URL_DIRECT:
        print("Error: DATABASE_URL_DIRECT not set")
        return

    try:
        async with await psycopg.AsyncConnection.connect(DATABASE_URL_DIRECT, autocommit=True) as conn:
            print("Enabling pgvector extension...")
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        print("Initializing PostgresSaver checkpointer...")
        async with AsyncPostgresSaver.from_conn_string(DATABASE_URL_DIRECT) as checkpointer:
            await checkpointer.setup()

        print("Initializing PostgresStore...")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        def embed(texts: list[str]) -> list[list[float]]:
            return _model.encode(texts, normalize_embeddings=True).tolist()

        async with AsyncPostgresStore.from_conn_string(DATABASE_URL_DIRECT, index={"dims": EMBEDDING_DIMS, "embed": embed}) as store:
            await store.setup()

        print("Database setup completed successfully.")
    except Exception as e:
        print(f"Error during database setup: {e}")
        import traceback
        traceback.print_exc()


def main():
    # Use SelectorEventLoop for psycopg on Windows
    loop = asyncio.SelectorEventLoop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(setup_database())
    finally:
        loop.close()


if __name__ == "__main__":
    main()