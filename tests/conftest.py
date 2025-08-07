"""Test fixtures and configuration for LeanVibe Agent Hive tests."""

import asyncio
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Test configuration
TEST_DATABASE_URL = (
    "postgresql+asyncpg://test_user:test_pass@localhost:5433/test_leanvibe_hive"
)
TEST_REDIS_URL = "redis://localhost:6381"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create a Redis client for testing."""
    client = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.flushdb()  # Clean up after test
        await client.close()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session


@pytest.fixture
def mock_cli_tools() -> dict:
    """Mock CLI tools for testing."""
    return {
        "opencode": {
            "name": "OpenCode",
            "command": "opencode",
            "execute_pattern": 'opencode "{prompt}"',
            "available": True,
        },
        "claude": {
            "name": "Claude Code CLI",
            "command": "claude",
            "execute_pattern": 'claude --no-interactive "{prompt}"',
            "available": True,
        },
    }


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.agent_type = "test"
    agent.status = "active"
    agent.execute_with_cli_tool = AsyncMock(return_value="Mock response")
    return agent


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    from src.core.task_queue import Task

    return Task(
        id="test-task-123",
        title="Test Task",
        description="A test task for unit testing",
        task_type="test",
        payload={"test_data": "value"},
        priority=5,
    )


class AsyncMockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self.data = {}
        self.lists = {}
        self.pubsub_messages = []

    async def hset(self, key: str, mapping: dict) -> None:
        if key not in self.data:
            self.data[key] = {}
        self.data[key].update(mapping)

    async def hgetall(self, key: str) -> dict:
        return self.data.get(key, {})

    async def lpush(self, key: str, value: str) -> None:
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].insert(0, value)

    async def brpop(self, key: str, timeout: int = 1) -> tuple:
        if key in self.lists and self.lists[key]:
            return (key, self.lists[key].pop())
        return None

    async def publish(self, channel: str, message: str) -> None:
        self.pubsub_messages.append((channel, message))

    async def ping(self) -> bool:
        return True


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    return AsyncMockRedis()
