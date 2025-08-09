import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import uuid
from datetime import datetime, timezone
from src.core.async_db import (
    AsyncDatabaseManager,
    DatabaseConnectionError,
    DatabaseOperationError,
)


class MockAgentModel:
    def __init__(self, name, agent_type, role, capabilities=None, tmux_session=None):
        self.id = uuid.uuid4()
        self.name = name
        self.type = agent_type
        self.role = role
        self.capabilities = capabilities or {}
        self.status = "active"
        self.tmux_session = tmux_session
        self.last_heartbeat = datetime.now(timezone.utc)
        self.short_id = None
        self.created_at = datetime.now(timezone.utc)


class MockContext:
    def __init__(self, agent_id, content, **kwargs):
        self.id = uuid.uuid4()
        self.agent_id = uuid.UUID(agent_id) if isinstance(agent_id, str) else agent_id
        self.content = content
        self.importance_score = kwargs.get("importance_score", 0.5)
        self.category = kwargs.get("category", "general")
        self.topic = kwargs.get("topic")
        self.meta_data = kwargs.get("metadata", {})
        self.created_at = datetime.now(timezone.utc)
        self.last_accessed = datetime.now(timezone.utc)


class MockSystemMetric:
    def __init__(self, metric_name, metric_type, value, **kwargs):
        self.id = uuid.uuid4()
        self.metric_name = metric_name
        self.metric_type = metric_type
        self.value = value
        self.unit = kwargs.get("unit")
        self.agent_id = kwargs.get("agent_id")
        self.task_id = kwargs.get("task_id")
        self.session_id = kwargs.get("session_id")
        self.labels = kwargs.get("labels", {})
        self.timestamp = datetime.now(timezone.utc)


class AsyncContextManagerMock:
    def __init__(self, session_mock):
        self.session_mock = session_mock

    async def __aenter__(self):
        return self.session_mock

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest_asyncio.fixture
async def db_manager():
    """Create a database manager with mocked engine and session."""
    with (
        patch("src.core.async_db.create_async_engine") as mock_engine,
        patch("src.core.async_db.async_sessionmaker") as mock_session_maker,
    ):
        # Create database manager
        db_manager = AsyncDatabaseManager("postgresql://test:test@localhost/test")

        # Mock the engine
        mock_engine.return_value = AsyncMock()
        db_manager.engine = mock_engine.return_value

        # Create a callable that returns the context manager
        def mock_session_factory():
            return AsyncContextManagerMock(AsyncMock())

        db_manager.async_session_maker = mock_session_factory

        yield db_manager


@pytest.mark.asyncio
async def test_health_check_success(db_manager):
    """Test successful database health check."""
    # Create a mock session that returns 1 for SELECT 1
    mock_session = AsyncMock()
    mock_result = AsyncMock()
    mock_result.scalar = MagicMock(
        return_value=1
    )  # Use MagicMock, not AsyncMock for scalar
    mock_session.execute.return_value = mock_result

    # Override the session factory to return our specific mock
    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    result = await db_manager.health_check()
    assert result is True
    mock_session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_health_check_failure(db_manager):
    """Test database health check failure."""
    # Mock session that raises an exception
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Database connection failed")

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    result = await db_manager.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_register_agent_new(db_manager):
    """Test registering a new agent."""
    mock_session = AsyncMock()

    # Mock the SELECT query to return None (agent doesn't exist)
    mock_result = AsyncMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)

    # Mock the short ID query to return empty set
    mock_short_id_result = AsyncMock()
    mock_short_id_result.fetchall = MagicMock(return_value=[])

    # Set up multiple execute calls in sequence
    mock_session.execute.side_effect = [mock_result, mock_short_id_result]

    # Mock session operations
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.refresh = AsyncMock()

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    with (
        patch("src.core.async_db.AgentModel") as MockAgentClass,
        patch("sqlalchemy.select") as mock_select,  # Patch the original import
        patch(
            "src.core.short_id.ShortIDGenerator"
        ) as mock_short_id_gen,  # Correct import path
    ):
        # Mock the AgentModel class to return a proper mock instance
        mock_agent_instance = MockAgentModel("test", "meta", "role")
        MockAgentClass.return_value = mock_agent_instance

        mock_select.return_value = MagicMock()
        mock_short_id_gen.generate_unique_agent_short_id.return_value = "test-abc123"

        agent_id = await db_manager.register_agent("test", "meta", "role")
        assert isinstance(agent_id, str)
        uuid.UUID(agent_id)  # Validate it's a valid UUID


@pytest.mark.asyncio
async def test_get_active_agents_success(db_manager):
    """Test successful retrieval of active agents."""
    mock_session = AsyncMock()

    # Create mock agents
    agents = [
        MockAgentModel("agent1", "meta", "role1"),
        MockAgentModel("agent2", "qa", "role2"),
    ]

    # Mock the query result chain: execute -> scalars -> all
    mock_scalars = MagicMock()
    mock_scalars.all = MagicMock(
        return_value=agents
    )  # Use MagicMock for synchronous call
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(return_value=mock_scalars)
    mock_session.execute.return_value = mock_result

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    with patch("sqlalchemy.select") as mock_select:
        mock_select.return_value = MagicMock()

        result = await db_manager.get_active_agents()
        assert len(result) == 2
        assert all(isinstance(agent, MockAgentModel) for agent in result)


@pytest.mark.asyncio
async def test_get_active_agents_failure(db_manager):
    """Test get active agents failure."""
    mock_session = AsyncMock()
    mock_session.execute.side_effect = Exception("Database error")

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    with patch("sqlalchemy.select") as mock_select:
        mock_select.return_value = MagicMock()

        result = await db_manager.get_active_agents()
        assert result == []


@pytest.mark.asyncio
async def test_close(db_manager):
    """Test database connection closing."""
    mock_engine = AsyncMock()
    db_manager.engine = mock_engine

    await db_manager.close()
    mock_engine.dispose.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_agent_by_name_success(db_manager):
    """Test successful agent retrieval by name."""
    mock_session = AsyncMock()

    # Create mock agent
    agent = MockAgentModel("test_agent", "meta", "role")

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=agent)
    mock_session.execute.return_value = mock_result

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    with patch("sqlalchemy.select") as mock_select:
        mock_select.return_value = MagicMock()

        result = await db_manager.get_agent_by_name("test_agent")
        assert result == agent


@pytest.mark.asyncio
async def test_get_agent_by_name_not_found(db_manager):
    """Test agent retrieval by name when agent not found."""
    mock_session = AsyncMock()

    mock_result = AsyncMock()
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_session.execute.return_value = mock_result

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    with patch("sqlalchemy.select") as mock_select:
        mock_select.return_value = MagicMock()

        result = await db_manager.get_agent_by_name("nonexistent")
        assert result is None


@pytest.mark.asyncio
async def test_update_agent_heartbeat_success(db_manager):
    """Test successful agent heartbeat update."""
    mock_session = AsyncMock()

    mock_result = AsyncMock()
    mock_result.rowcount = 1
    mock_session.execute.return_value = mock_result
    mock_session.commit = AsyncMock()

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    with patch("sqlalchemy.update") as mock_update:
        mock_update.return_value.where.return_value.values.return_value = MagicMock()

        result = await db_manager.update_agent_heartbeat("test_agent")
        assert result is True


@pytest.mark.asyncio
async def test_update_agent_heartbeat_not_found(db_manager):
    """Test agent heartbeat update when agent not found."""
    mock_session = AsyncMock()

    mock_result = AsyncMock()
    mock_result.rowcount = 0
    mock_session.execute.return_value = mock_result
    mock_session.commit = AsyncMock()

    db_manager.async_session_maker = lambda: AsyncContextManagerMock(mock_session)

    with patch("sqlalchemy.update") as mock_update:
        mock_update.return_value.where.return_value.values.return_value = MagicMock()

        result = await db_manager.update_agent_heartbeat("nonexistent")
        assert result is False
