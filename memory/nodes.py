import asyncio
import logging
from typing import Any, Union, TYPE_CHECKING

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

if TYPE_CHECKING:
    from langgraph_store_core import KVStore

logger = logging.getLogger(__name__)

# Token budget constants
MAX_TOKENS_BUDGET = 8000  # Conservative for GPT-oss models
PROXIMITY_MARGIN = 1000  # Keep some buffer

async def load_memory(
    state: dict,
    store: "KVStore | None" = None,
    memory_top_k: int = 5,
    short_term_window: int = 10,
) -> dict:
    """
    Load user memory from the long-term store and assemble the working context.

    Returns a dict with 'context_messages' and 'user_profile' keys.
    """
    thread_id = state.get("thread_id")  # This is telegram chat_id as string
    if not thread_id:
        # Return default context structure without memory lookups
        return {
            **state,
            "context_messages": state.get("messages", []),
            "user_profile": {},
        }

    user_profile = {}
    retrieved_memories = []

    if store is not None:
        # 1. Get user profile (strict schema)
        try:
            profile_result = await store.get(("users", thread_id, "profile"))
            if profile_result:
                user_profile = profile_data = profile_result.value if hasattr(profile_result, 'value') else profile_result
                if not isinstance(user_profile, dict):
                    user_profile = {}
        except Exception as e:
            logger.warning(f"Failed to load user profile: {e}")
            user_profile = {}

        # 2. Get top-k relevant facts via semantic search
        try:
            query = " ".join(
                msg.content for msg in state.get("messages", [])
                if isinstance(msg, (HumanMessage, SystemMessage))
            )[-500:]  # Truncate for query efficiency
            if query:
                search_results = await store.search(
                    ("users", thread_id, "facts"),
                    query=query,
                    limit=memory_top_k
                )
                retrieved_memories = [r.value for r in search_results]
        except Exception as e:
            logger.warning(f"Failed to search facts: {e}")

        # 3. Get recent episodes (actions/outcomes)
        try:
            episode_results = await store.search(
                ("users", thread_id, "episodes"),
                query="",
                limit=memory_top_k
            )
            retrieved_memories.extend(
                f"Earlier action: {r.value.get('goal', '')} -> {r.value.get('outcome', '')}"
                for r in episode_results
            )
        except Exception as e:
            logger.warning(f"Failed to search episodes: {e}")

    # Build context messages
    system_prompt = state.get("system_prompt", "")
    instructions = []

    # Order: System prompt, Instructions (procedures), Profile, Retrieved Memories, Summary, Recent Messages
    if user_profile:
        preferences = user_profile.get("preferences", {})
        tone = user_profile.get("tone", "helpful")
        instructions.append(f"User profile: {user_profile}")

    if retrieved_memories:
        instructions.append("Relevant memories: " + "; ".join(str(m) for m in retrieved_memories))

    # Get running summary if available
    summary = state.get("summary", "")

    messages = state.get("messages", [])
    recent_messages = messages[-short_term_window:] if len(messages) > short_term_window else messages

    context_messages = []
    if system_prompt:
        context_messages.append(SystemMessage(content=system_prompt))
    if instructions:
        context_messages.append(SystemMessage(content="\n\n".join(instructions)))
    if summary:
        context_messages.append(SystemMessage(content=f"Summary of earlier conversation: {summary}"))
    context_messages.extend(recent_messages)

    return {
        **state,
        "context_messages": context_messages,
        "user_profile": user_profile,
    }


# Helper to estimate tokens (simple char count based)
def simple_token_estimate(text: str) -> int:
    return len(text) // 4


def trim_messages_for_budget(
    messages: list[BaseMessage],
    max_tokens: int = MAX_TOKENS_BUDGET,
    system_prompt: str = ""
) -> list[BaseMessage]:
    """Trim messages to fit within token budget."""
    system_tk = simple_token_estimate(system_prompt)
    available = max_tokens - system_tk

    tokens = 0
    result = []

    # Keep system message first
    for msg in reversed(messages):
        tk = simple_token_estimate(msg.content if isinstance(msg.content, str) else "")
        if tokens + tk > available:
            break
        result.append(msg)
        tokens += tk

    return list(reversed(result))