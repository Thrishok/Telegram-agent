import json
import logging
from typing import Any, TYPE_CHECKING, Union

from langchain_core.tools import tool

if TYPE_CHECKING:
    from langgraph_store_core import KVStore

logger = logging.getLogger(__name__)


def _serialize_key(key: tuple) -> str:
    """Serialize a tuple key to JSON string for LangChain tools."""
    return json.dumps(list(key))


def _deserialize_key(key_str: str) -> tuple:
    """Deserialize a JSON string key back to tuple."""
    return tuple(json.loads(key_str))


# These tools need store passed in - we'll make them closures
# NOTE: The store passed here MUST have sync methods (search, put, delete, etc.)
# For async stores (PostgresStore async), wrap calls in asyncio.to_thread or use async-compatible tool
def create_search_memory_tool(store: "KVStore | None" = None, top_k: int = 5):
    @tool
    def search_memory(query: str, namespace_key: str = '["users"]') -> str:
        """Search your saved memories for relevant information.

        Args:
            query: Search query string
            namespace_key: JSON-escaped namespace key (e.g., '["users", "user123", "facts"]')
        """
        if store is None:
            return "Memory store not available."
        try:
            ns = _deserialize_key(namespace_key)
            results = store.search(ns, query=query, limit=top_k)
            if not results:
                return "No memories found."
            return "\n".join(str(r.value) for r in results)
        except Exception as e:
            return f"Error searching memories: {e}"
    return search_memory


def create_save_memory_tool(store: "KVStore | None" = None):
    @tool
    def save_memory(key: str, content: str) -> str:
        """Save a piece of information to your long-term memory.

        Args:
            key: JSON-escaped namespace key (e.g., '["users", "user123", "profile"]')
            content: Content to save (as JSON string or plain text)
        """
        if store is None:
            return "Memory store not available."
        try:
            parsed_key = _deserialize_key(key)
            # Try to parse content as JSON, otherwise store as-is
            try:
                parsed_content = json.loads(content)
            except json.JSONDecodeError:
                parsed_content = content
            store.put(parsed_key, parsed_content)
            return f"Saved: {key}"
        except Exception as e:
            return f"Error saving memory: {e}"
    return save_memory


def create_forget_memory_tool(store: "KVStore | None" = None):
    @tool
    def forget_memory(key: str) -> str:
        """Forget a saved piece of information by key.

        Args:
            key: JSON-escaped namespace key to delete
        """
        if store is None:
            return "Memory store not available."
        try:
            parsed_key = _deserialize_key(key)
            store.delete(parsed_key)
            return f"Forgotten: {key}"
        except Exception as e:
            return f"Error forgetting memory: {e}"
    return forget_memory


def create_list_memories_tool(store: "KVStore | None" = None):
    @tool
    def list_memories(namespace_key: str = '["users"]') -> str:
        """List all your saved memories in a namespace.

        Args:
            namespace_key: JSON-escaped namespace prefix (e.g., '["users", "user123"]')
        """
        if store is None:
            return "Memory store not available."
        try:
            ns = _deserialize_key(namespace_key)
            # Use search with empty query to list all items
            items = store.search(ns, query="", limit=100)
            if not items:
                return "No memories saved."
            lines = []
            for item in items:
                lines.append(f"- {item.key}: {item.value}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error listing memories: {e}"
    return list_memories


def create_forget_all_memory_tool(store: "KVStore | None" = None):
    @tool
    def forget_all_memory() -> str:
        """Forget all your memories (profile, facts, episodes)."""
        if store is None:
            return "Memory store not available."
        try:
            store.delete_namespace(("users",))  # Deletes all user memories
            return "All memories have been forgotten."
        except Exception as e:
            return f"Error forgetting all memories: {e}"
    return forget_all_memory


# Convenience function to get all memory tools
def get_memory_tools(store: "KVStore | None" = None, top_k: int = 5) -> list:
    """Return a list of all memory-related tools."""
    return [
        create_search_memory_tool(store, top_k),
        create_save_memory_tool(store),
        create_forget_memory_tool(store),
        create_list_memories_tool(store),
        create_forget_all_memory_tool(store),
    ]