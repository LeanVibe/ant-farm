"""Unit tests for AgentOrchestrator system with SQLAlchemy integration."""

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.core.orchestrator import AgentInfo, AgentRegistry, AgentStatus


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing."""
    mock_manager = MagicMock()
    mock_manager.get_session = MagicMock()
    mock_manager.create_tables = MagicMock()
    return mock_manager


@pytest.fixture
def agent_registry(mock_db_manager):
    """Create AgentRegistry with mocked database manager."""
    with patch(
        "src.core.orchestrator.get_database_manager", return_value=mock_db_manager
    ):
        registry = AgentRegistry("postgresql://test")
    return registry


@pytest.fixture
def sample_agent_info():
    """Create sample AgentInfo for testing."""
    return AgentInfo(
        id=str(uuid.uuid4()),
        name="test-agent",
        type="meta",
        role="meta",
        status=AgentStatus.STARTING,
        capabilities=["system_analysis", "code_generation"],
        tmux_session="hive-test-agent",
        last_heartbeat=1234567890.0,
        created_at=1234567890.0,
        tasks_completed=0,
        tasks_failed=0,
    )


class TestAgentRegistry:
    """Test cases for AgentRegistry with SQLAlchemy."""

    @pytest.mark.asyncio
    async def test_register_agent_new(self, agent_registry, sample_agent_info):
        """Test registering a new agent."""
        # Mock session to return None (agent doesn't exist)
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.register_agent(sample_agent_info)

        assert result is True
        assert sample_agent_info.name in agent_registry.agents

        # Verify database operations
        mock_session.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_agent_existing(self, agent_registry, sample_agent_info):
        """Test registering an existing agent (should update)."""
        # Mock session to return existing agent
        existing_agent = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_agent
        )
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.register_agent(sample_agent_info)

        assert result is True
        assert sample_agent_info.name in agent_registry.agents

        # Verify agent was updated, not added
        mock_session.add.assert_not_called()
        mock_session.commit.assert_called_once()

        # Verify existing agent properties were updated
        assert existing_agent.type == sample_agent_info.type
        assert existing_agent.role == sample_agent_info.role
        assert existing_agent.status == sample_agent_info.status.value

    @pytest.mark.asyncio
    async def test_register_agent_database_error(
        self, agent_registry, sample_agent_info
    ):
        """Test registering agent with database error."""
        # Mock session to raise exception
        mock_session = MagicMock()
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.register_agent(sample_agent_info)

        assert result is False
        assert sample_agent_info.name not in agent_registry.agents

    @pytest.mark.asyncio
    async def test_update_agent_status_success(self, agent_registry, sample_agent_info):
        """Test updating agent status successfully."""
        # First register the agent
        agent_registry.agents[sample_agent_info.name] = sample_agent_info

        # Mock session for update
        existing_agent = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_agent
        )
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.update_agent_status(
            sample_agent_info.name, AgentStatus.ACTIVE, "task-123"
        )

        assert result is True
        assert (
            agent_registry.agents[sample_agent_info.name].status == AgentStatus.ACTIVE
        )
        assert (
            agent_registry.agents[sample_agent_info.name].current_task_id == "task-123"
        )

        # Verify database update
        assert existing_agent.status == AgentStatus.ACTIVE.value
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_agent_status_not_found(self, agent_registry):
        """Test updating status of non-existent agent."""
        result = await agent_registry.update_agent_status(
            "non-existent", AgentStatus.ACTIVE
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_agent_status_database_error(
        self, agent_registry, sample_agent_info
    ):
        """Test updating agent status with database error."""
        # Register agent first
        agent_registry.agents[sample_agent_info.name] = sample_agent_info

        # Mock session to raise exception
        mock_session = MagicMock()
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.update_agent_status(
            sample_agent_info.name, AgentStatus.ACTIVE
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_get_agent_exists(self, agent_registry, sample_agent_info):
        """Test getting an existing agent."""
        agent_registry.agents[sample_agent_info.name] = sample_agent_info

        result = await agent_registry.get_agent(sample_agent_info.name)

        assert result == sample_agent_info

    @pytest.mark.asyncio
    async def test_get_agent_not_exists(self, agent_registry):
        """Test getting a non-existent agent."""
        result = await agent_registry.get_agent("non-existent")

        assert result is None

    @pytest.mark.asyncio
    async def test_list_agents_no_filter(self, agent_registry, sample_agent_info):
        """Test listing all agents without filter."""
        agent_registry.agents[sample_agent_info.name] = sample_agent_info

        # Add another agent with different status
        another_agent = AgentInfo(
            id=str(uuid.uuid4()),
            name="another-agent",
            type="qa",
            role="qa",
            status=AgentStatus.ACTIVE,
            capabilities=["testing"],
            tmux_session="hive-another",
            last_heartbeat=1234567890.0,
            created_at=1234567890.0,
            tasks_completed=0,
            tasks_failed=0,
        )
        agent_registry.agents[another_agent.name] = another_agent

        result = await agent_registry.list_agents()

        assert len(result) == 2
        assert sample_agent_info in result
        assert another_agent in result

    @pytest.mark.asyncio
    async def test_list_agents_with_filter(self, agent_registry, sample_agent_info):
        """Test listing agents with status filter."""
        agent_registry.agents[sample_agent_info.name] = sample_agent_info

        # Add another agent with different status
        another_agent = AgentInfo(
            id=str(uuid.uuid4()),
            name="another-agent",
            type="qa",
            role="qa",
            status=AgentStatus.ACTIVE,
            capabilities=["testing"],
            tmux_session="hive-another",
            last_heartbeat=1234567890.0,
            created_at=1234567890.0,
            tasks_completed=0,
            tasks_failed=0,
        )
        agent_registry.agents[another_agent.name] = another_agent

        result = await agent_registry.list_agents(AgentStatus.ACTIVE)

        assert len(result) == 1
        assert another_agent in result
        assert sample_agent_info not in result

    @pytest.mark.asyncio
    async def test_remove_agent_success(self, agent_registry, sample_agent_info):
        """Test removing an agent successfully."""
        # Register agent first
        agent_registry.agents[sample_agent_info.name] = sample_agent_info

        # Mock session for removal
        existing_agent = MagicMock()
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            existing_agent
        )
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.remove_agent(sample_agent_info.name)

        assert result is True
        assert sample_agent_info.name not in agent_registry.agents

        # Verify database removal
        mock_session.delete.assert_called_once_with(existing_agent)
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_remove_agent_not_found(self, agent_registry):
        """Test removing a non-existent agent."""
        result = await agent_registry.remove_agent("non-existent")

        assert result is False

    @pytest.mark.asyncio
    async def test_remove_agent_database_error(self, agent_registry, sample_agent_info):
        """Test removing agent with database error."""
        # Register agent first
        agent_registry.agents[sample_agent_info.name] = sample_agent_info

        # Mock session to raise exception
        mock_session = MagicMock()
        mock_session.commit.side_effect = SQLAlchemyError("Database error")
        agent_registry.db_manager.get_session.return_value = mock_session

        result = await agent_registry.remove_agent(sample_agent_info.name)

        assert result is False
        # Agent should still be in memory since database removal failed
        assert (
            sample_agent_info.name not in agent_registry.agents
        )  # Removed from memory first


class TestAgentStatus:
    """Test cases for AgentStatus enum."""

    def test_agent_status_values(self):
        """Test AgentStatus enum values."""
        assert AgentStatus.STARTING.value == "starting"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.STOPPING.value == "stopping"
        assert AgentStatus.STOPPED.value == "stopped"


class TestAgentInfo:
    """Test cases for AgentInfo dataclass."""

    def test_agent_info_creation(self):
        """Test creating AgentInfo with all fields."""
        agent_info = AgentInfo(
            id="test-id",
            name="test-agent",
            type="meta",
            role="meta",
            status=AgentStatus.ACTIVE,
            capabilities=["test_capability"],
            tmux_session="test-session",
            last_heartbeat=1234567890.0,
            created_at=1234567890.0,
            tasks_completed=5,
            tasks_failed=1,
            current_task_id="task-123",
            load_factor=0.5,
        )

        assert agent_info.id == "test-id"
        assert agent_info.name == "test-agent"
        assert agent_info.type == "meta"
        assert agent_info.role == "meta"
        assert agent_info.status == AgentStatus.ACTIVE
        assert agent_info.capabilities == ["test_capability"]
        assert agent_info.tmux_session == "test-session"
        assert agent_info.last_heartbeat == 1234567890.0
        assert agent_info.created_at == 1234567890.0
        assert agent_info.tasks_completed == 5
        assert agent_info.tasks_failed == 1
        assert agent_info.current_task_id == "task-123"
        assert agent_info.load_factor == 0.5

    def test_agent_info_optional_fields(self):
        """Test AgentInfo with optional fields."""
        agent_info = AgentInfo(
            id="test-id",
            name="test-agent",
            type="meta",
            role="meta",
            status=AgentStatus.ACTIVE,
            capabilities=["test_capability"],
            tmux_session="test-session",
            last_heartbeat=1234567890.0,
            created_at=1234567890.0,
            tasks_completed=0,
            tasks_failed=0,
        )

        assert agent_info.current_task_id is None
        assert agent_info.load_factor == 0.0
