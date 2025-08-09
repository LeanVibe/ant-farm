"""Unit tests for AgentOrchestrator system with SQLAlchemy integration."""

import asyncio
import subprocess
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import SQLAlchemyError

from src.core.orchestrator import (
    AgentInfo,
    AgentOrchestrator,
    AgentRegistry,
    AgentSpawner,
    AgentStatus,
    HealthMonitor,
    SystemHealth,
)


@pytest.fixture
def mock_db_manager():
    """Mock database manager for testing."""
    mock_manager = AsyncMock()
    mock_manager.get_session = MagicMock()
    mock_manager.create_tables = AsyncMock()
    mock_manager.get_active_agents = AsyncMock(return_value=[])
    return mock_manager


@pytest.fixture
async def agent_registry(mock_db_manager):
    """Create AgentRegistry with mocked database manager."""
    registry = AgentRegistry("postgresql://test")
    registry.db_manager = mock_db_manager

    # Mock the async method that gets called during initialization
    registry.load_agents_from_database = AsyncMock()

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


@pytest.fixture
def agent_spawner():
    """Create AgentSpawner for testing."""
    return AgentSpawner(Path("/test/project"))


@pytest.fixture
def health_monitor(agent_registry):
    """Create HealthMonitor for testing."""
    return HealthMonitor(agent_registry, heartbeat_interval=5)


@pytest.fixture
def mock_task_queue():
    """Mock task queue for testing."""
    with patch("src.core.orchestrator.task_queue") as mock_queue:
        mock_queue.initialize = AsyncMock()
        mock_queue.get_task = AsyncMock(return_value=None)
        mock_queue.get_queue_stats = AsyncMock()
        mock_queue.cleanup_expired_tasks = AsyncMock()
        mock_queue.fail_task = AsyncMock()
        yield mock_queue


@pytest.fixture
def mock_message_broker():
    """Mock message broker for testing."""
    with patch("src.core.message_broker.message_broker") as mock_broker:
        mock_broker.send_message = AsyncMock()
        yield mock_broker


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
        """Test registering an existing agent (should not update)."""
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

        # Verify agent was not added again (early return when existing)
        mock_session.add.assert_not_called()
        # Commit is not called when agent already exists (early return)
        mock_session.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_register_agent_database_error(
        self, agent_registry, sample_agent_info
    ):
        """Test registering agent with database error."""
        # Mock the sync operation to raise an exception
        original_sync_store = agent_registry._store_agent_in_db

        async def failing_store(agent_info):
            raise SQLAlchemyError("Database error")

        agent_registry._store_agent_in_db = failing_store

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


class TestAgentSpawner:
    """Test cases for AgentSpawner."""

    @pytest.mark.asyncio
    async def test_spawn_agent_success(self, agent_spawner):
        """Test successful agent spawning."""
        with patch("subprocess.run") as mock_run:
            # Mock successful tmux commands
            mock_run.side_effect = [
                MagicMock(returncode=0),  # tmux new-session
                MagicMock(returncode=0),  # tmux has-session
            ]

            result = await agent_spawner.spawn_agent("meta", "test-agent")

            assert result == "hive-test-agent"
            assert mock_run.call_count == 2

    @pytest.mark.asyncio
    async def test_spawn_agent_tmux_failure(self, agent_spawner):
        """Test agent spawning with tmux failure."""
        with patch("subprocess.run") as mock_run:
            # Mock failed tmux command
            mock_run.side_effect = subprocess.CalledProcessError(1, "tmux")

            result = await agent_spawner.spawn_agent("meta", "test-agent")

            assert result is None

    @pytest.mark.asyncio
    async def test_spawn_agent_session_not_found(self, agent_spawner):
        """Test agent spawning where session creation appears to succeed but session not found."""
        with patch("subprocess.run") as mock_run:
            # Mock successful creation but failed verification
            mock_run.side_effect = [
                MagicMock(returncode=0),  # tmux new-session
                MagicMock(returncode=1),  # tmux has-session (not found)
            ]

            result = await agent_spawner.spawn_agent("meta", "test-agent")

            assert result is None

    @pytest.mark.asyncio
    async def test_terminate_agent_success(self, agent_spawner):
        """Test successful agent termination."""
        with patch("subprocess.run") as mock_run:
            # Mock successful tmux commands
            mock_run.return_value = MagicMock(returncode=0)

            result = await agent_spawner.terminate_agent(
                "test-agent", "hive-test-agent"
            )

            assert result is True
            assert mock_run.call_count == 2  # send-keys and kill-session

    @pytest.mark.asyncio
    async def test_terminate_agent_failure(self, agent_spawner):
        """Test agent termination failure."""
        with patch("subprocess.run") as mock_run:
            # Mock failed tmux command
            mock_run.side_effect = subprocess.CalledProcessError(1, "tmux")

            result = await agent_spawner.terminate_agent(
                "test-agent", "hive-test-agent"
            )

            assert result is False


class TestHealthMonitor:
    """Test cases for HealthMonitor."""

    @pytest.mark.asyncio
    async def test_start_monitoring(self, health_monitor):
        """Test starting health monitoring."""
        # Mock the monitoring loop to stop after one iteration
        original_check = health_monitor._check_agent_health
        health_monitor._check_agent_health = AsyncMock()

        # Start monitoring and let it run briefly
        monitoring_task = asyncio.create_task(health_monitor.start_monitoring())

        # Stop after brief period
        await asyncio.sleep(0.1)
        await health_monitor.stop_monitoring()

        # Cancel the task
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

        assert health_monitor._check_agent_health.called

    @pytest.mark.asyncio
    async def test_check_agent_health_responsive(
        self, health_monitor, sample_agent_info
    ):
        """Test health check for responsive agent."""
        # Set agent as active with recent heartbeat
        sample_agent_info.status = AgentStatus.ACTIVE
        sample_agent_info.last_heartbeat = time.time()
        health_monitor.registry.agents[sample_agent_info.name] = sample_agent_info

        # Mock ping to return True (responsive)
        health_monitor._ping_agent = AsyncMock(return_value=True)

        await health_monitor._check_agent_health()

        # Agent should remain in registry and no special handling needed
        assert sample_agent_info.name in health_monitor.registry.agents

    @pytest.mark.asyncio
    async def test_check_agent_health_unresponsive(
        self, health_monitor, sample_agent_info
    ):
        """Test health check for unresponsive agent."""
        # Set agent as active with old heartbeat
        sample_agent_info.status = AgentStatus.ACTIVE
        sample_agent_info.last_heartbeat = time.time() - 200  # Very old
        sample_agent_info.current_task_id = "task-123"
        health_monitor.registry.agents[sample_agent_info.name] = sample_agent_info

        # Mock ping to return False (unresponsive)
        health_monitor._ping_agent = AsyncMock(return_value=False)
        health_monitor._handle_dead_agent = AsyncMock()

        await health_monitor._check_agent_health()

        # Should handle dead agent
        health_monitor._handle_dead_agent.assert_called_once_with(
            sample_agent_info.name
        )

    @pytest.mark.asyncio
    async def test_check_starting_agent_becomes_active(
        self, health_monitor, sample_agent_info
    ):
        """Test that starting agent becomes active after timeout."""
        # Set agent as starting with old creation time
        sample_agent_info.status = AgentStatus.STARTING
        sample_agent_info.created_at = time.time() - 15  # 15 seconds ago
        health_monitor.registry.agents[sample_agent_info.name] = sample_agent_info
        health_monitor.registry.update_agent_status = AsyncMock()

        with patch("subprocess.run") as mock_run:
            # Mock successful session check
            mock_run.return_value = MagicMock(returncode=0)

            await health_monitor._check_starting_agent(
                sample_agent_info.name, sample_agent_info
            )

            # Should mark as active
            health_monitor.registry.update_agent_status.assert_called_once_with(
                sample_agent_info.name, AgentStatus.ACTIVE
            )

    @pytest.mark.asyncio
    async def test_check_starting_agent_session_missing(
        self, health_monitor, sample_agent_info
    ):
        """Test starting agent with missing tmux session."""
        sample_agent_info.status = AgentStatus.STARTING
        health_monitor.registry.agents[sample_agent_info.name] = sample_agent_info
        health_monitor.registry.update_agent_status = AsyncMock()

        with patch("subprocess.run") as mock_run:
            # Mock session not found
            mock_run.return_value = MagicMock(returncode=1)

            await health_monitor._check_starting_agent(
                sample_agent_info.name, sample_agent_info
            )

            # Should mark as stopped
            health_monitor.registry.update_agent_status.assert_called_once_with(
                sample_agent_info.name, AgentStatus.STOPPED
            )

    @pytest.mark.asyncio
    async def test_handle_dead_agent(
        self, health_monitor, sample_agent_info, mock_task_queue
    ):
        """Test handling dead agent."""
        sample_agent_info.current_task_id = "task-123"
        sample_agent_info.tmux_session = "hive-test"
        health_monitor.registry.update_agent_status = AsyncMock()
        health_monitor.registry.get_agent = AsyncMock(return_value=sample_agent_info)

        with patch("subprocess.run") as mock_run:
            await health_monitor._handle_dead_agent(sample_agent_info.name)

            # Should mark as error and fail task
            health_monitor.registry.update_agent_status.assert_called_once_with(
                sample_agent_info.name, AgentStatus.ERROR
            )
            mock_task_queue.fail_task.assert_called_once()
            mock_run.assert_called_once()  # tmux kill-session


class TestAgentOrchestrator:
    """Test cases for AgentOrchestrator."""

    @pytest.fixture
    def orchestrator(self, mock_task_queue, mock_message_broker):
        """Create AgentOrchestrator for testing."""
        with patch("src.core.orchestrator.get_async_database_manager"):
            orch = AgentOrchestrator(
                db_url="postgresql://test",
                project_root=Path("/test"),
                max_agents=10,
                heartbeat_interval=5,
            )
            return orch

    @pytest.mark.asyncio
    async def test_orchestrator_start(self, orchestrator, mock_task_queue):
        """Test orchestrator start."""
        await orchestrator.start()

        assert orchestrator.running is True
        mock_task_queue.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_orchestrator_stop(self, orchestrator):
        """Test orchestrator stop."""
        orchestrator.running = True
        orchestrator.task_assignment_running = True
        orchestrator.health_monitor.stop_monitoring = AsyncMock()

        await orchestrator.stop()

        assert orchestrator.running is False
        assert orchestrator.task_assignment_running is False
        orchestrator.health_monitor.stop_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_agent_success(self, orchestrator):
        """Test successful agent spawning."""
        # Mock spawner and registry
        orchestrator.spawner.spawn_agent = AsyncMock(return_value="hive-test-agent")
        orchestrator.registry.register_agent = AsyncMock(return_value=True)

        with patch("src.core.orchestrator.ShortIDGenerator") as mock_short_id:
            mock_short_id.generate_agent_short_id.return_value = "short123"

            result = await orchestrator.spawn_agent("meta", "test-agent")

            assert result == "test-agent"
            orchestrator.spawner.spawn_agent.assert_called_once()
            orchestrator.registry.register_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_spawn_agent_max_limit_reached(self, orchestrator):
        """Test spawning agent when max limit reached."""
        # Fill up to max agents
        for i in range(orchestrator.max_agents):
            orchestrator.registry.agents[f"agent-{i}"] = MagicMock()

        result = await orchestrator.spawn_agent("meta", "test-agent")

        assert result is None

    @pytest.mark.asyncio
    async def test_spawn_agent_name_exists(self, orchestrator):
        """Test spawning agent with existing name."""
        orchestrator.registry.agents["test-agent"] = MagicMock()

        result = await orchestrator.spawn_agent("meta", "test-agent")

        assert result is None

    @pytest.mark.asyncio
    async def test_spawn_agent_spawner_failure(self, orchestrator):
        """Test spawning agent with spawner failure."""
        orchestrator.spawner.spawn_agent = AsyncMock(return_value=None)

        result = await orchestrator.spawn_agent("meta", "test-agent")

        assert result is None

    @pytest.mark.asyncio
    async def test_spawn_agent_registration_failure(self, orchestrator):
        """Test spawning agent with registration failure."""
        orchestrator.spawner.spawn_agent = AsyncMock(return_value="hive-test-agent")
        orchestrator.registry.register_agent = AsyncMock(return_value=False)
        orchestrator.spawner.terminate_agent = AsyncMock(return_value=True)

        with patch("src.core.orchestrator.ShortIDGenerator") as mock_short_id:
            mock_short_id.generate_agent_short_id.return_value = "short123"

            result = await orchestrator.spawn_agent("meta", "test-agent")

            assert result is None
            orchestrator.spawner.terminate_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_agent_success(self, orchestrator, sample_agent_info):
        """Test successful agent termination."""
        orchestrator.registry.get_agent = AsyncMock(return_value=sample_agent_info)
        orchestrator.registry.update_agent_status = AsyncMock()
        orchestrator.spawner.terminate_agent = AsyncMock(return_value=True)
        orchestrator.registry.remove_agent = AsyncMock()

        result = await orchestrator.terminate_agent("test-agent")

        assert result is True
        orchestrator.registry.update_agent_status.assert_called_once_with(
            "test-agent", AgentStatus.STOPPING
        )
        orchestrator.registry.remove_agent.assert_called_once()

    @pytest.mark.asyncio
    async def test_terminate_agent_not_found(self, orchestrator):
        """Test terminating non-existent agent."""
        orchestrator.registry.get_agent = AsyncMock(return_value=None)

        result = await orchestrator.terminate_agent("non-existent")

        assert result is False

    @pytest.mark.asyncio
    async def test_get_system_health(self, orchestrator, mock_task_queue):
        """Test getting system health metrics."""
        # Mock agents with different statuses
        agents = [
            MagicMock(status=AgentStatus.ACTIVE, load_factor=0.5),
            MagicMock(status=AgentStatus.IDLE, load_factor=0.1),
            MagicMock(status=AgentStatus.BUSY, load_factor=0.8),
            MagicMock(status=AgentStatus.ERROR, load_factor=0.0),
        ]
        orchestrator.registry.list_agents = AsyncMock(return_value=agents)

        # Mock queue stats
        mock_queue_stats = MagicMock()
        mock_queue_stats.completed_tasks = 100
        mock_queue_stats.queue_size_by_priority = {1: 5, 2: 10}
        mock_task_queue.get_queue_stats.return_value = mock_queue_stats

        health = await orchestrator.get_system_health()

        assert isinstance(health, SystemHealth)
        assert health.total_agents == 4
        assert health.active_agents == 1
        assert health.idle_agents == 1
        assert health.busy_agents == 1
        assert health.error_agents == 1
        assert health.avg_load_factor == 0.35  # (0.5 + 0.1 + 0.8 + 0.0) / 4
        assert health.queue_size == 15  # 5 + 10

    @pytest.mark.asyncio
    async def test_get_default_capabilities(self, orchestrator):
        """Test getting default capabilities for different agent types."""
        meta_caps = orchestrator._get_default_capabilities("meta")
        assert "system_analysis" in meta_caps
        assert "code_generation" in meta_caps

        dev_caps = orchestrator._get_default_capabilities("developer")
        assert "code_generation" in dev_caps
        assert "testing" in dev_caps

        unknown_caps = orchestrator._get_default_capabilities("unknown")
        assert unknown_caps == ["general"]


class TestSystemHealth:
    """Test cases for SystemHealth dataclass."""

    def test_system_health_creation(self):
        """Test creating SystemHealth with all fields."""
        health = SystemHealth(
            total_agents=10,
            active_agents=5,
            idle_agents=3,
            busy_agents=2,
            error_agents=0,
            avg_load_factor=0.6,
            queue_size=15,
            tasks_per_minute=12.5,
        )

        assert health.total_agents == 10
        assert health.active_agents == 5
        assert health.idle_agents == 3
        assert health.busy_agents == 2
        assert health.error_agents == 0
        assert health.avg_load_factor == 0.6
        assert health.queue_size == 15
        assert health.tasks_per_minute == 12.5


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
