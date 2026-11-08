# Telegram Agent with LangGraph

A Telegram bot powered by LangGraph with production-grade context and memory management backed by Neon PostgreSQL.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Telegram Webhook                             │
│                        ( FastAPI endpoint )                         │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ┌─────────────┐                                  │
│                    │  load_      │  ← Fetches user profile, facts,  │
│                    │  memory     │    recent episodes from Neon      │
│                    │    node     │    (short-circuit for new users)  │
└────────┬───────────┴───────┬─────┴──────────────────────────────────┘
         │                   │
         │ context           │
         ▼  messages         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   ┌───────────────────┐                            │
│                   │     LLM Agent     │  ← Full conv. in          │
│                   │   (langgraph)     │    PostgresSaver           │
│                   └────────┬──────────┘    (checkpointer)           │
│                            │                                        │
│                            ▼                                        │
│                 ┌────────────────────┐                             │
│                 │    Assistant       │                             │
│                 │     Response       │                             │
│                 └────────┬───────────┘                             │
└──────────────────────────┼─────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  ┌──────────────────┐                             │
│                  │ extract_           │  ← Background memory       │
│                  │ memory (langmem)   │    extraction               │
│                  └────────┬───────────┘                             │
└────────────────────────────┼─────────────────────────────────────────┘
                              │
                              ▼
                  ┌──────────────────────┐
                  │    Postgres Store    │  ← Long-term semantic memory
                  │   (pgvector search)  │    with namespaces:
                  │                      │    - users/{id}/profile
                  │                      │    - users/{id}/facts
                  │                      │    - users/{id}/episodes
                  └──────────────────────┘
```

### Memory Layers

1. **Short-term memory (thread-level)**: `PostgresSaver` stores full conversation history in checkpoints. Before each LLM call, we trim to `SHORT_TERM_WINDOW` messages + running summary.

2. **Long-term memory (cross-thread)**: `PostgresStore` with `pgvector` enables semantic search. Namespaces:
   - `("users", user_id, "profile")`: User preferences (name, tone, timezone)
   - `("users", user_id, "facts")`: Semantic facts for retrieval
   - `("users", user_id, "episodes")`: Past actions with TTL (default 90 days)
   - `("agent", "procedures")`: Learned rules (feature-flagged)

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | (required) | Neon pooled connection string (PgBouncer) |
| `DATABASE_URL_DIRECT` | `DATABASE_URL` | Direct connection for setup/migrations |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer for pgvector |
| `EMBEDDING_DIMS` | `384` | Embedding dimension (must match model) |
| `SHORT_TERM_WINDOW` | `10` | Recent messages to include |
| `MEMORY_TOP_K` | `5` | Memories to retrieve per turn |
| `EPISODE_TTL_DAYS` | `90` | Days before episodes expire |
| `ENABLE_MEMORY_EXTRACTION` | `true` | Toggle background extraction |
| `MEMORY_EXTRACTION_MODEL` | `openai/gpt-oss-120b` | Model for memory extraction |

```bash
# Copy .env.example to .env and fill in your values
cp .env.example .env
```

## Setup

1. **Create a Neon branch with pgvector:**
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

2. **Run the setup script:**
   ```bash
   # Install dependencies
   pip install -r requirements.txt

   # Run setup (creates tables and enables pgvector)
   python -m memory.setup
   ```

   Or via FastAPI:
   ```bash
   uvicorn memory.setup:app --port 8000
   ```

3. **Start the agent:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

4. **Set the webhook:**
   ```bash
   curl "http://localhost:8000/set-webhook?url=https://your-public-url"
   ```

## Memory Administration

### View memory via Python
```python
from memory.store import get_store
from langgraph.store.postgres import PostgresStore

async def inspect_memory(thread_id: str):
    async with get_store() as store:
        # List all user memories
        profile = await store.get(("users", thread_id, "profile"))
        facts = list(await store.search(("users", thread_id, "facts"), ""))
        episodes = list(await store.search(("users", thread_id, "episodes"), ""))
        
        print("Profile:", profile.value if profile else None)
        print("Facts:", [f.value for f in facts])
        print("Episodes:", [e.value for e in episodes])
```

### Forget a user's memories
```python
store.delete_namespace(("users", chat_id))  # Deletes all memories for a user
```

### Forget specific memory by key
```python
store.delete(("users", chat_id, "profile"))  # Just the profile
```

## Testing

```bash
# Unit tests only (mock database)
pytest tests/

# Integration tests (requires real database)
PYTEST_INTEGRATION=1 pytest tests/
```

## Files

- `main.py` - FastAPI server and LangGraph agent
- `memory/` - Memory module package
  - `config.py` - Environment variables and connection pool
  - `setup.py` - Database migration script
  - `checkpointer.py` - PostgresSaver for short-term memory
  - `store.py` - PostgresStore for long-term semantic memory
  - `nodes.py` - Context loading node
  - `extraction.py` - Background memory extraction
  - `tools.py` - Memory management tools for the agent
- `tests/` - Test suite