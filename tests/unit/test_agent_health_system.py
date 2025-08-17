"""Comprehensive tests for the Agent Health Monitoring System.

Tests cover:
- Health monitoring functionality
- Circuit breaker behavior
- Recovery mechanisms
- Resource management
- Lifecycle state machine
"""

import asyncio
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch


# Test the health monitoring system
class TestAgentHealthMonitor:
    """Test the AgentHealthMonitor class."""

    @pytest.fixture
    def health_monitor(self):
        """Create a health monitor for testing."""
        from src.core.agent_health import AgentHealthMonitor

        return AgentHealthMonitor("test_agent", check_interval=0.1)

    @pytest.mark.asyncio
    async def test_health_monitor_initialization(self, health_monitor):
        """Test health monitor initialization."""
        assert health_monitor.agent_name == "test_agent"
        assert health_monitor.check_interval == 0.1
        assert health_monitor.health_status.value == "unknown"
        assert len(health_monitor.health_checks) == 0
        assert len(health_monitor.circuit_breakers) == 0

    @pytest.mark.asyncio
    async def test_add_health_check(self, health_monitor):
        """Test adding health checks."""

        def dummy_check():
            return True

        health_monitor.add_health_check("test_check", dummy_check)
        assert "test_check" in health_monitor.health_checks
        assert health_monitor.health_checks["test_check"] == dummy_check

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, health_monitor):
        """Test circuit breaker functionality."""
        from src.core.agent_health import CircuitBreakerConfig

        # Add circuit breaker
        config = CircuitBreakerConfig(failure_threshold=2, timeout=0.1)
        health_monitor.add_circuit_breaker("test_breaker", config)

        # Test successful execution
        async def success_func():
            return "success"

        result = await health_monitor.execute_with_circuit_breaker(
            "test_breaker", success_func
        )
        assert result == "success"

        # Test failure handling
        async def failing_func():
            raise Exception("Test failure")

        # First failure - should work
        with pytest.raises(Exception):
            await health_monitor.execute_with_circuit_breaker(
                "test_breaker", failing_func
            )

        # Second failure - should work
        with pytest.raises(Exception):
            await health_monitor.execute_with_circuit_breaker(
                "test_breaker", failing_func
            )

        # Third failure - circuit should open
        from src.core.agent_health import CircuitBreakerOpenError

        with pytest.raises(CircuitBreakerOpenError):
            await health_monitor.execute_with_circuit_breaker(
                "test_breaker", failing_func
            )

    @pytest.mark.asyncio
    async def test_health_check_execution(self, health_monitor):
        """Test health check execution."""
        check_called = False

        async def async_check():
            nonlocal check_called
            check_called = True
            return True

        health_monitor.add_health_check("async_test", async_check)

        status = await health_monitor.perform_health_check()
        assert check_called
        assert status.value == "healthy"

    @pytest.mark.asyncio
    async def test_monitoring_loop(self, health_monitor):
        """Test the monitoring loop."""
        check_count = 0

        def counting_check():
            nonlocal check_count
            check_count += 1
            return True

        health_monitor.add_health_check("counter", counting_check)

        # Start monitoring for a short time
        await health_monitor.start_monitoring()
        await asyncio.sleep(0.25)  # Should trigger at least 2 checks
        await health_monitor.stop_monitoring()

        assert check_count >= 2

    def test_health_report_generation(self, health_monitor):
        """Test health report generation."""
        report = health_monitor.get_health_report()

        assert "agent_name" in report
        assert "current_status" in report
        assert "metrics" in report
        assert "health_checks" in report
        assert "circuit_breakers" in report
        assert report["agent_name"] == "test_agent"


class TestAgentRecovery:
    """Test the AgentRecoveryManager class."""

    @pytest.fixture
    def recovery_manager(self):
        """Create a recovery manager for testing."""
        from src.core.agent_recovery import AgentRecoveryManager, RecoveryPolicy

        policy = RecoveryPolicy(max_restart_attempts=2, restart_backoff_base=0.1)
        return AgentRecoveryManager("test_agent", policy)

    @pytest.mark.asyncio
    async def test_recovery_strategy_selection(self, recovery_manager):
        """Test recovery strategy selection."""
        from src.core.agent_recovery import AgentFailure, FailureCategory

        # Test transient failure -> restart
        failure = AgentFailure(
            agent_name="test_agent",
            timestamp=time.time(),
            failure_category=FailureCategory.TRANSIENT,
            error_message="Temporary error",
        )

        strategy = recovery_manager._select_recovery_strategy(failure)
        assert strategy.value == "restart"

        # Test critical failure -> terminate
        critical_failure = AgentFailure(
            agent_name="test_agent",
            timestamp=time.time(),
            failure_category=FailureCategory.CRITICAL,
            error_message="Critical error",
        )

        strategy = recovery_manager._select_recovery_strategy(critical_failure)
        assert strategy.value == "terminate"

    @pytest.mark.asyncio
    async def test_failure_handling(self, recovery_manager):
        """Test failure handling and recovery initiation."""
        from src.core.agent_recovery import AgentFailure, FailureCategory, RecoveryState

        failure = AgentFailure(
            agent_name="test_agent",
            timestamp=time.time(),
            failure_category=FailureCategory.TRANSIENT,
            error_message="Test failure",
        )

        # Mock the agent factory
        recovery_manager.agent_factory = AsyncMock(return_value="new_agent")

        await recovery_manager.handle_agent_failure(failure)

        # Should have recorded the failure
        assert len(recovery_manager.failure_history) == 1
        assert recovery_manager.failure_history[0] == failure

    def test_backup_agent_management(self, recovery_manager):
        """Test backup agent management."""
        recovery_manager.add_backup_agent("backup_agent_1")
        recovery_manager.add_backup_agent("backup_agent_2")

        assert "backup_agent_1" in recovery_manager.backup_agents
        assert "backup_agent_2" in recovery_manager.backup_agents
        assert len(recovery_manager.backup_agents) == 2

    def test_recovery_status_reporting(self, recovery_manager):
        """Test recovery status reporting."""
        status = recovery_manager.get_recovery_status()

        assert "agent_name" in status
        assert "recovery_state" in status
        assert "failure_count" in status
        assert "backup_agents" in status
        assert status["agent_name"] == "test_agent"


class TestResourceManagement:
    """Test the ResourceManager class."""

    @pytest.fixture
    def resource_manager(self):
        """Create a resource manager for testing."""
        from src.core.resource_management import ResourceManager, ResourceQuota

        quota = ResourceQuota(max_memory_mb=100, max_file_handles=10)
        return ResourceManager("test_agent", quota)

    @pytest.mark.asyncio
    async def test_resource_registration(self, resource_manager):
        """Test resource registration."""
        from src.core.resource_management import ResourceType

        test_resource = "test_data"

        managed_resource = resource_manager.register_resource(
            "test_resource",
            ResourceType.CUSTOM,
            test_resource,
            metadata={"type": "test"},
        )

        assert managed_resource.resource_id == "test_resource"
        assert managed_resource.resource_obj == test_resource
        assert "test_resource" in resource_manager.resources

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, resource_manager):
        """Test resource cleanup."""
        from src.core.resource_management import ResourceType

        cleanup_called = False

        def cleanup_func(resource):
            nonlocal cleanup_called
            cleanup_called = True

        test_resource = "test_data"
        managed_resource = resource_manager.register_resource(
            "cleanup_test",
            ResourceType.CUSTOM,
            test_resource,
            cleanup_func=cleanup_func,
        )

        await managed_resource.cleanup()
        assert cleanup_called
        assert managed_resource.state.value == "released"

    @pytest.mark.asyncio
    async def test_quota_checking(self, resource_manager):
        """Test quota violation checking."""
        # Mock usage that exceeds quota
        resource_manager.current_usage.memory_mb = 150  # Exceeds 100MB limit
        resource_manager.current_usage.file_handles = 15  # Exceeds 10 limit

        violations = resource_manager.current_usage.exceeds_quota(
            resource_manager.quota
        )

        assert len(violations) == 2
        assert any("memory" in v for v in violations)
        assert any("file_handles" in v for v in violations)

    def test_resource_status_reporting(self, resource_manager):
        """Test resource status reporting."""
        status = resource_manager.get_resource_status()

        assert "agent_name" in status
        assert "current_usage" in status
        assert "quota" in status
        assert "resource_count" in status
        assert status["agent_name"] == "test_agent"


class TestAgentLifecycle:
    """Test the AgentLifecycle class."""

    @pytest.fixture
    def lifecycle(self):
        """Create a lifecycle manager for testing."""
        from src.core.agent_lifecycle import AgentLifecycle, LifecycleConfig

        config = LifecycleConfig(startup_timeout=1.0, health_check_interval=0.1)
        return AgentLifecycle("test_agent", config)

    @pytest.mark.asyncio
    async def test_lifecycle_initialization(self, lifecycle):
        """Test lifecycle initialization."""
        from src.core.agent_lifecycle import AgentState

        assert lifecycle.agent_name == "test_agent"
        assert lifecycle.current_state == AgentState.CREATED
        assert lifecycle.previous_state is None

    @pytest.mark.asyncio
    async def test_state_transitions(self, lifecycle):
        """Test valid state transitions."""
        from src.core.agent_lifecycle import LifecycleEvent, AgentState

        # Test valid transition sequence
        assert await lifecycle.trigger_event(LifecycleEvent.START)
        assert lifecycle.current_state == AgentState.INITIALIZING

        assert await lifecycle.trigger_event(LifecycleEvent.INITIALIZE)
        assert lifecycle.current_state == AgentState.READY

        assert await lifecycle.trigger_event(LifecycleEvent.ACTIVATE)
        assert lifecycle.current_state == AgentState.ACTIVE

    @pytest.mark.asyncio
    async def test_invalid_transitions(self, lifecycle):
        """Test invalid state transitions."""
        from src.core.agent_lifecycle import LifecycleEvent

        # Try invalid transition (ACTIVATE from CREATED)
        result = await lifecycle.trigger_event(LifecycleEvent.ACTIVATE)
        assert not result  # Should fail

    @pytest.mark.asyncio
    async def test_event_handlers(self, lifecycle):
        """Test event handlers."""
        from src.core.agent_lifecycle import LifecycleEvent

        handler_called = False
        event_data = None

        def test_handler(lc, event, metadata):
            nonlocal handler_called, event_data
            handler_called = True
            event_data = (event, metadata)

        lifecycle.add_event_handler(LifecycleEvent.START, test_handler)

        await lifecycle.trigger_event(LifecycleEvent.START, {"test": "data"})

        assert handler_called
        assert event_data[0] == LifecycleEvent.START
        assert event_data[1]["test"] == "data"

    @pytest.mark.asyncio
    async def test_task_tracking(self, lifecycle):
        """Test task tracking functionality."""
        from src.core.agent_lifecycle import AgentState

        # Set up lifecycle in active state
        await lifecycle.trigger_event(LifecycleEvent.START)
        await lifecycle.trigger_event(LifecycleEvent.INITIALIZE)
        await lifecycle.trigger_event(LifecycleEvent.ACTIVATE)

        # Add tasks
        lifecycle.add_task("task1")
        lifecycle.add_task("task2")

        assert len(lifecycle.active_tasks) == 2
        assert "task1" in lifecycle.active_tasks
        assert "task2" in lifecycle.active_tasks

        # Remove task
        lifecycle.remove_task("task1")
        assert len(lifecycle.active_tasks) == 1
        assert "task1" not in lifecycle.active_tasks

    def test_lifecycle_status_reporting(self, lifecycle):
        """Test lifecycle status reporting."""
        status = lifecycle.get_lifecycle_status()

        assert "agent_name" in status
        assert "current_state" in status
        assert "time_in_state" in status
        assert "active_tasks" in status
        assert "performance_metrics" in status
        assert status["agent_name"] == "test_agent"

    def test_operational_state_checking(self, lifecycle):
        """Test operational state checking."""
        from src.core.agent_lifecycle import AgentState

        # Initially not operational (CREATED state)
        assert not lifecycle.is_operational()

        # Set to operational state
        lifecycle.current_state = AgentState.ACTIVE
        assert lifecycle.is_operational()

        # Set to non-operational state
        lifecycle.current_state = AgentState.ERROR
        assert not lifecycle.is_operational()


class TestEnhancedBaseAgent:
    """Test the EnhancedBaseAgent integration."""

    @pytest.fixture
    def enhanced_agent(self):
        """Create an enhanced agent for testing."""
        # Mock the imports since they may not be available in test environment
        with patch("src.agents.enhanced_base_agent.BaseAgent"):
            from src.agents.enhanced_base_agent import EnhancedBaseAgent

            return EnhancedBaseAgent("test_agent", "test_type", "test_role")

    @pytest.mark.asyncio
    async def test_enhanced_agent_initialization(self, enhanced_agent):
        """Test enhanced agent initialization."""
        assert enhanced_agent.name == "test_agent"
        assert enhanced_agent.agent_type == "test_type"
        assert hasattr(enhanced_agent, "health_monitor")
        assert hasattr(enhanced_agent, "recovery_manager")
        assert hasattr(enhanced_agent, "resource_manager")
        assert hasattr(enhanced_agent, "lifecycle")

    @pytest.mark.asyncio
    async def test_health_check_integration(self, enhanced_agent):
        """Test health check integration."""
        # Mock the health monitor
        enhanced_agent.health_monitor.perform_health_check = AsyncMock(
            return_value=MagicMock(value="healthy")
        )

        status = await enhanced_agent.health_check()
        assert status.value == "healthy"

    def test_comprehensive_status_reporting(self, enhanced_agent):
        """Test comprehensive status reporting."""
        # Mock all the subsystem status methods
        enhanced_agent.health_monitor.get_health_report = MagicMock(
            return_value={"status": "healthy"}
        )
        enhanced_agent.lifecycle.get_lifecycle_status = MagicMock(
            return_value={"state": "active"}
        )
        enhanced_agent.recovery_manager.get_recovery_status = MagicMock(
            return_value={"recovery_state": "none"}
        )
        enhanced_agent.resource_manager.get_resource_status = MagicMock(
            return_value={"resource_count": 0}
        )

        status = enhanced_agent.get_comprehensive_status()

        assert "agent_name" in status
        assert "health" in status
        assert "lifecycle" in status
        assert "recovery" in status
        assert "resources" in status
        assert "base_status" in status


class TestSystemIntegration:
    """Test system-wide integration."""

    @pytest.mark.asyncio
    async def test_health_monitor_registry(self):
        """Test health monitor registry."""
        from src.core.agent_health import (
            get_health_monitor_registry,
            AgentHealthMonitor,
        )

        registry = get_health_monitor_registry()
        monitor = AgentHealthMonitor("integration_test")

        await registry.register_monitor("integration_test", monitor)

        retrieved_monitor = registry.get_monitor("integration_test")
        assert retrieved_monitor == monitor

        # Test system health report
        report = await registry.get_system_health_report()
        assert "system_health" in report
        assert "agent_reports" in report

        await registry.unregister_monitor("integration_test")

    @pytest.mark.asyncio
    async def test_recovery_system_integration(self):
        """Test recovery system integration."""
        from src.core.agent_recovery import get_recovery_system

        recovery_system = get_recovery_system()
        recovery_system.register_agent("integration_test")

        status = recovery_system.get_system_status()
        assert "total_agents" in status
        assert "recovering_agents" in status
        assert "agent_status" in status

        recovery_system.unregister_agent("integration_test")

    @pytest.mark.asyncio
    async def test_resource_system_integration(self):
        """Test resource system integration."""
        from src.core.resource_management import get_resource_system

        resource_system = get_resource_system()
        resource_system.register_agent("integration_test")

        status = resource_system.get_system_status()
        assert "total_agents" in status
        assert "system_usage" in status
        assert "agents" in status

        resource_system.unregister_agent("integration_test")

    @pytest.mark.asyncio
    async def test_lifecycle_system_integration(self):
        """Test lifecycle system integration."""
        from src.core.agent_lifecycle import get_lifecycle_system

        lifecycle_system = get_lifecycle_system()
        lifecycle_system.register_agent("integration_test")

        status = lifecycle_system.get_system_status()
        assert "total_agents" in status
        assert "operational_agents" in status
        assert "agents" in status

        lifecycle_system.unregister_agent("integration_test")


# Utility functions for testing
def create_mock_agent():
    """Create a mock agent for testing."""
    mock_agent = MagicMock()
    mock_agent.name = "mock_agent"
    mock_agent.agent_type = "test"
    mock_agent.status = "active"
    return mock_agent


async def wait_for_condition(condition_func, timeout=1.0, interval=0.1):
    """Wait for a condition to become true."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if condition_func():
            return True
        await asyncio.sleep(interval)
    return False


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""

    @pytest.mark.asyncio
    async def test_health_check_performance(self):
        """Test health check performance under load."""
        from src.core.agent_health import AgentHealthMonitor

        monitor = AgentHealthMonitor("perf_test", check_interval=0.01)

        # Add multiple health checks
        for i in range(10):
            monitor.add_health_check(f"check_{i}", lambda: True)

        start_time = time.time()

        # Perform multiple health checks
        for _ in range(100):
            await monitor.perform_health_check()

        duration = time.time() - start_time

        # Should complete 100 health checks in reasonable time
        assert duration < 5.0  # Less than 5 seconds

    @pytest.mark.asyncio
    async def test_circuit_breaker_under_load(self):
        """Test circuit breaker behavior under load."""
        from src.core.agent_health import AgentHealthMonitor, CircuitBreakerConfig

        monitor = AgentHealthMonitor("circuit_test")
        config = CircuitBreakerConfig(failure_threshold=5)
        monitor.add_circuit_breaker("load_test", config)

        success_count = 0
        failure_count = 0

        async def test_func():
            # Simulate 70% success rate
            import random

            if random.random() < 0.7:
                return "success"
            else:
                raise Exception("Random failure")

        # Execute many operations
        for _ in range(1000):
            try:
                result = await monitor.execute_with_circuit_breaker(
                    "load_test", test_func
                )
                success_count += 1
            except Exception:
                failure_count += 1

        # Should have completed all operations
        assert success_count + failure_count == 1000


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
