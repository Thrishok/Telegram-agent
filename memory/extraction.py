import asyncio
import logging
from typing import Any, Optional, TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

if TYPE_CHECKING:
    from langgraph_store_core import KVStore

logger = logging.getLogger(__name__)

# Debounce state for extraction
_last_extraction_ts: dict[str, float] = {}
EXTRACTION_DEBOUNCE_SECONDS = 5.0  # Don't extract more than once per 5s per user

def should_extract(thread_id: str) -> bool:
    """Simple debounce check."""
    import time
    now = time.time()
    last = _last_extraction_ts.get(thread_id, 0)
    if now - last < EXTRACTION_DEBOUNCE_SECONDS:
        return False
    _last_extraction_ts[thread_id] = now
    return True


async def extract_memory(
    thread_id: str,
    messages: list,
    assistant_response: str,
    store: "KVStore | None" = None,
    extraction_model: str = "openai/gpt-oss-120b",
    groq_api_key: Optional[str] = None,
    enabled: bool = True,
) -> None:
    """
    Background extraction of memories after agent response.
    Updates facts and logs an episode.
    """
    if not enabled:
        return
    if not store:
        return
    if not should_extract(thread_id):
        return

    try:
        from langmem import create_memory_store_manager
        from langchain_groq import ChatGroq

        # We need a cheaper model for extraction if possible
        # For now, use the same model but this is a placeholder
        llm = ChatGroq(model=extraction_model, temperature=0.0, api_key=groq_api_key)

        # Use LangMem's store manager
        manager = create_memory_store_manager(
            store=store,
            namespace_keys=("users", thread_id),
            instructions="""
            You are an expert at extracting factual information from conversations.
            Extract ONLY:
            - User facts: explicit information about the user (name, preferences, timezone, etc.)
            - Important events: notable actions or outcomes that happened
            Do NOT extract: opinions, casual conversation, or things the user asked you to forget.
            """
        )

        # Run extraction
        async for update_info in manager.almstream(messages + [HumanMessage(content=assistant_response)]):
            # update_info contains extracted facts/episodes
            pass  # The manager updates the store directly

        logger.debug(f"Memory extraction completed for thread {thread_id}")

    except Exception as e:
        logger.warning(f"Memory extraction failed for thread {thread_id}: {e}")


# Simple fact extraction (fallback if LangMem doesn't work perfectly)
async def simple_fact_extract(
    thread_id: str,
    messages: list,
    store=None,
) -> None:
    """Simple template-based fact extraction as fallback."""
    if not store:
        return

    try:
        # Look for patterns like "my name is", "I am in", "I prefer", etc.
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                if "my name is" in content:
                    name = content.split("my name is")[-1].strip().split(".")[0].strip()
                    await store.put(
                        ("users", thread_id, "profile"),
                        {"name": name},
                    )
                elif "i am" in content or "i'm" in content:
                    location = content.split("i am")[-1].strip().split(".")[0].strip()
                    if "timezone" not in str(store.get(("users", thread_id, "profile"))):
                        profile = await store.get(("users", thread_id, "profile"))
                        existing = profile.value if profile else {}
                        existing["timezone"] = location
                        await store.put(("users", thread_id, "profile"), existing)
    except Exception as e:
        logger.warning(f"Simple fact extract failed: {e}")