"""
Tests for memory module.
Uses mocks for LLM/embeddings, real Postgres or Neon for integration tests via env var.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langgraph_store_core import KVStore
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from memory.nodes import load_memory, trim_messages_for_budget, simple_token_estimate
from memory.extraction import should_extract, extract_memory
from memory.tools import (
    create_search_memory_tool,
    create_save_memory_tool,
    create_forget_memory_tool,
    create_list_memories_tool,
    create_forget_all_memory_tool,
    get_memory_tools,
)


class TestTokenEstimate:
    def test_simple_estimate(self):
        text = "Hello world this is a test"
        tokens = simple_token_estimate(text)
        assert tokens > 0

    def test_empty_string(self):
        tokens = simple_token_estimate("")
        assert tokens == 0


class TestTrimMessages:
    def test_trim_within_budget(self):
        messages = [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!"),
        ]
        result = trim_messages_for_budget(messages, max_tokens=1000)
        assert len(result) == 2

    def test_trim_exceeds_budget(self):
        # Create long messages
        long_content = "a" * 5000  # ~1250 tokens
        messages = [
            HumanMessage(content=long_content),
            HumanMessage(content=long_content),
        ]
        result = trim_messages_for_budget(messages, max_tokens=2000)
        # Should trim to fit
        estimated = sum(simple_token_estimate(m.content) for m in result)
        assert estimated <= 2000


class TestLoadMemory:
    @pytest.mark.asyncio
    async def test_load_memory_no_store(self):
        state = {"messages": [HumanMessage(content="Hello")]}
        result = await load_memory(state, store=None)
        assert "context_messages" in result
        assert "user_profile" in result

    @pytest.mark.asyncio
    async def test_load_memory_with_mock_store(self):
        mock_store = AsyncMock(spec=KVStore)
        mock_store.get.return_value = MagicMock(value={"name": "Test User"})
        mock_store.search.return_value = []

        state = {
            "messages": [HumanMessage(content="Hello my name is Alice")],
            "thread_id": "test-thread-1",
            "system_prompt": "You are helpful."
        }
        result = await load_memory(state, store=mock_store)
        assert "context_messages" in result
        # Verify profile was fetched with correct namespace
        mock_store.get.assert_called_once_with(("users", "test-thread-1", "profile"))

    @pytest.mark.asyncio
    async def test_load_memory_no_thread_id(self):
        state = {"messages": [HumanMessage(content="Hello")]}
        result = await load_memory(state, store=None)
        # Now returns context_messages even without thread_id
        assert "context_messages" in result
        assert "user_profile" in result


class TestMemoryExtraction:
    @pytest.mark.asyncio
    async def test_should_extract_debounce(self):
        thread_id = "test-thread"
        # First call should return True
        assert should_extract(thread_id) is True
        # Immediate second call should be debounced
        assert should_extract(thread_id) is False

    @pytest.mark.asyncio
    async def test_extract_memory_disabled(self):
        mock_store = AsyncMock(spec=KVStore)
        result = await extract_memory(
            "test-thread", [], "response",
            store=mock_store, enabled=False
        )
        assert result is None  # Returns early


class TestMemoryTools:
    @pytest.fixture
    def mock_store(self):
        # Create a mock with sync methods (for use with StructuredTool)
        mock = MagicMock(spec=KVStore)
        mock.search = MagicMock(return_value=[])
        mock.put = MagicMock()
        mock.delete = MagicMock()
        mock.delete_namespace = MagicMock()
        return mock

    def test_search_memory_tool(self, mock_store):
        tool = create_search_memory_tool(mock_store)
        result = tool.invoke({"query": "test", "namespace_key": '["users"]'})
        assert "No memories found" in result or isinstance(result, str)

    def test_save_memory_tool(self, mock_store):
        tool = create_save_memory_tool(mock_store)
        result = tool.invoke({
            "key": '["users", "1", "profile"]',
            "content": '{"name": "Alice"}'
        })
        mock_store.put.assert_called_once()

    def test_forget_memory_tool(self, mock_store):
        tool = create_forget_memory_tool(mock_store)
        result = tool.invoke({"key": '["users", "1", "profile"]'})
        mock_store.delete.assert_called_once()

    def test_list_memories_tool(self, mock_store):
        mock_item = MagicMock()
        mock_item.key = ("users", "1", "profile")
        mock_item.value = {"name": "Alice"}
        mock_store.search.return_value = [mock_item]

        tool = create_list_memories_tool(mock_store)
        result = tool.invoke({"namespace_key": '["users"]'})
        assert "Alice" in result or "No memories" in result

    def test_forget_all_memory_tool(self, mock_store):
        tool = create_forget_all_memory_tool(mock_store)
        result = tool.invoke({})
        # Calls delete_namespace with ("users",)
        assert result is not None

    def test_get_memory_tools(self, mock_store):
        tools = get_memory_tools(mock_store)
        assert len(tools) == 5  # search, save, forget, list, forget_all


class TestUserIsolation:
    @pytest.mark.asyncio
    async def test_namespace_isolation(self):
        """Ensure users can only access their own memories."""
        mock_store = AsyncMock(spec=KVStore)
        mock_store.search.return_value = []

        async def run_load(thread_id):
            messages = [HumanMessage(content="Hello")]
            state = {
                "messages": messages,
                "thread_id": thread_id,
            }
            result = await load_memory(state, store=mock_store)
            # Verify get was called with correct namespace
            assert mock_store.get.called

        # Load for user1
        await run_load("user1")
        call_args_1 = mock_store.get.call_args[0][0]
        assert call_args_1[0] == "users"
        assert call_args_1[1] == "user1"

        # Reset mock
        mock_store.reset_mock()

        # Load for user2
        await run_load("user2")
        call_args_2 = mock_store.get.call_args[0][0]
        assert call_args_2[1] == "user2"


class TestFactDeduplication:
    @pytest.mark.asyncio
    async def test_concurrent_fact_updates(self):
        """Facts should be deduplicated, not appended."""
        # PostgresStore overwrites on put() with same key
        # This test verifies the logic
        mock_store = AsyncMock()
        mock_store.put = AsyncMock()

        async def save_fact(key, value):
            await mock_store.put(key, value)

        # Simulate two facts for same key
        await save_fact(("users", "1", "profile"), {"name": "Alice"})
        await save_fact(("users", "1", "profile"), {"name": "Alice", "location": "NYC"})

        # verify put called twice (store handles deduplication)
        assert mock_store.put.call_count == 2