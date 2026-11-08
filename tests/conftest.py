import pytest
import asyncio
import os

# Try to load .env if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check if we should run integration tests (requires NEON_TEST_BRANCH_URL or local DB)
PYTEST_INTEGRATION = os.getenv("PYTEST_INTEGRATION", "").lower() in ("1", "true", "yes")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks integration tests (requires external DB)")


def pytest_collection_modifyitems(config, items):
    if PYTEST_INTEGRATION:
        # Run all tests including integration ones
        return
    # Skip integration tests if not explicitly enabled
    skip_integration = pytest.mark.skip(
        reason="Integration test set PYTEST_INTEGRATION=1 to run"
    )
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)