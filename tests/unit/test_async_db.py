import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import uuid
from src.core.async_db import AsyncDatabaseManager


class DummyAgent:
    def __init__(self, name, id):
        self.name = name
        self.id = id
        self.status = None


class AsyncSessionMock(AsyncMock):
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        pass


@pytest_asyncio.fixture
def db_manager_and_session():
    # Use a dummy URL and patch engine/session creation
    with (
        patch("src.core.async_db.create_async_engine"),
        patch("src.core.async_db.async_sessionmaker") as session_maker,
    ):
        session = AsyncSessionMock()
        session_maker_mock = MagicMock()
        session_maker_mock.__call__ = MagicMock(return_value=session)
        session_maker.return_value = session_maker_mock
        yield AsyncDatabaseManager("postgresql://dummy"), session


@pytest.mark.asyncio
async def test_health_check_success(db_manager_and_session):
    db_manager, session = db_manager_and_session
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def session_cm():
        yield session

    db_manager.async_session_maker = MagicMock(return_value=session_cm())

    class ExecuteResult:
        async def scalar(self):
            return 1

    session.execute = AsyncMock(return_value=ExecuteResult())
    result = await db_manager.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_health_check_failure(db_manager_and_session):
    db_manager, session = db_manager_and_session
    session.execute.side_effect = Exception("fail")
    assert await db_manager.health_check() is False


@pytest.mark.asyncio
async def test_register_agent_new(db_manager_and_session):
    db_manager, session = db_manager_and_session

    # Patch the async session maker to return our test session
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session():
        yield session

    db_manager.async_session_maker = lambda: mock_session()

    # Set up the mock chain for scalar_one_or_none to return None (new agent)
    execute_result = MagicMock()
    execute_result.scalar_one_or_none = AsyncMock(return_value=None)
    session.execute = AsyncMock(return_value=execute_result)
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()

    with (
        patch("src.core.async_db.AgentModel") as MockAgent,
        patch("sqlalchemy.select") as mock_select,
    ):
        agent_instance = DummyAgent("test", str(uuid.uuid4()))
        MockAgent.return_value = agent_instance
        mock_select.return_value = MagicMock()  # Mock the select statement

        agent_id = await db_manager.register_agent("test", "meta", "role")
        assert isinstance(agent_id, str)


@pytest.mark.asyncio
async def test_store_context_success(db_manager_and_session):
    db_manager, session = db_manager_and_session
    with patch("src.core.async_db.Context") as Context:
        context_instance = MagicMock(id=str(uuid.uuid4()))
        Context.return_value = context_instance
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        context_id = await db_manager.store_context(str(uuid.uuid4()), "content")
        # Assert that context_id is a valid UUID string
        uuid.UUID(context_id)


@pytest.mark.asyncio
async def test_store_context_failure(db_manager_and_session):
    db_manager, session = db_manager_and_session
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    with patch("src.core.async_db.Context") as Context:
        Context.side_effect = Exception("fail")
        with pytest.raises(Exception):
            await db_manager.store_context(str(uuid.uuid4()), "content")


@pytest.mark.asyncio
async def test_record_system_metric_success(db_manager_and_session):
    db_manager, session = db_manager_and_session
    with patch("src.core.async_db.SystemMetric") as SystemMetric:
        metric_instance = MagicMock(id=str(uuid.uuid4()))
        SystemMetric.return_value = metric_instance
        session.add = MagicMock()
        session.commit = AsyncMock()
        session.refresh = AsyncMock()
        metric_id = await db_manager.record_system_metric("cpu", "gauge", 0.5)
        # Assert that metric_id is a valid UUID string
        uuid.UUID(metric_id)


@pytest.mark.asyncio
async def test_record_system_metric_failure(db_manager_and_session):
    db_manager, session = db_manager_and_session
    session.commit = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()
    with patch("src.core.async_db.SystemMetric") as SystemMetric:
        SystemMetric.side_effect = Exception("fail")
        with pytest.raises(Exception):
            await db_manager.record_system_metric("cpu", "gauge", 0.5)


@pytest.mark.asyncio
async def test_get_active_agents_success(db_manager_and_session):
    db_manager, session = db_manager_and_session

    # Patch the async session maker to return our test session
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session():
        yield session

    db_manager.async_session_maker = lambda: mock_session()

    agents = [MagicMock(), MagicMock()]

    # Set up the mock chain for scalars().all() to return agents
    scalars_result = MagicMock()
    scalars_result.all = AsyncMock(return_value=agents)
    execute_result = MagicMock()
    execute_result.scalars = AsyncMock(return_value=scalars_result)
    session.execute = AsyncMock(return_value=execute_result)

    result = await db_manager.get_active_agents()
    assert result == agents


@pytest.mark.asyncio
async def test_get_active_agents_failure(db_manager_and_session):
    db_manager, session = db_manager_and_session

    # Patch the async session maker to return our test session
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def mock_session():
        yield session

    db_manager.async_session_maker = lambda: mock_session()

    session.execute = AsyncMock(side_effect=Exception("fail"))
    result = await db_manager.get_active_agents()
    assert result == []


@pytest.mark.asyncio
async def test_close(db_manager_and_session):
    db_manager, _ = db_manager_and_session
    db_manager.engine = AsyncMock()
    await db_manager.close()
    db_manager.engine.dispose.assert_awaited()
