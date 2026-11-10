"""
Memory module for LangGraph agent with Postgres persistence.

Architecture:
- Short-term: PostgresSaver checkpointer (thread-level, full history in checkpoints)
- Long-term: PostgresStore with pgvector (cross-thread semantic search)

Namespaces in store:
- ("users", user_id, "profile"): User profile record
- ("users", user_id, "facts"): Semantic facts for retrieval
- ("users", user_id, "episodes"): Past actions (goal, action, outcome, timestamp)
- ("agent", "procedures"): Learned rules (feature-flagged)
"""

import os as _os

# Skip heavy imports when running setup script to avoid circular/premature init
if _os.environ.get("_MEMORY_SETUP") != "1":
    from memory.config import (
        get_db_pool,
        close_db_pool,
        DATABASE_URL,
        DATABASE_URL_DIRECT,
        EMBEDDING_MODEL,
        EMBEDDING_DIMS,
        SHORT_TERM_WINDOW,
        MEMORY_TOP_K,
        EPISODE_TTL_DAYS,
        ENABLE_MEMORY_EXTRACTION,
        MEMORY_EXTRACTION_MODEL,
    )
    from memory.checkpointer import get_checkpointer
    from memory.store import get_store
    from memory.nodes import load_memory, trim_messages_for_budget
    from memory.extraction import extract_memory, should_extract
    from memory.tools import get_memory_tools

    __all__ = [
        "get_db_pool",
        "close_db_pool",
        "DATABASE_URL",
        "DATABASE_URL_DIRECT",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMS",
        "SHORT_TERM_WINDOW",
        "MEMORY_TOP_K",
        "EPISODE_TTL_DAYS",
        "ENABLE_MEMORY_EXTRACTION",
        "MEMORY_EXTRACTION_MODEL",
        "get_checkpointer",
        "get_store",
        "load_memory",
        "trim_messages_for_budget",
        "extract_memory",
        "should_extract",
        "get_memory_tools",
    ]