"""Tests for Emergency Intervention System."""

import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.core.safety.emergency_intervention import (
    CriticalFailureType,
    EmergencyEvent,
    EmergencyInterventionSystem,
    EmergencyTrigger,
    InterventionLevel,
)


@pytest.fixture
def project_path():
    """Test project path."""
    return Path("/tmp/test_project")


@pytest.fixture
def session_id():
    """Test session ID."""
    return "test-session-123"


@pytest.fixture
def emergency_system(project_path, session_id):
    """Emergency intervention system fixture."""
    return EmergencyInterventionSystem(project_path, session_id)


@pytest.mark.asyncio
class TestEmergencyInterventionSystem:
    """Test emergency intervention system functionality."""

    async def test_initialization(self, emergency_system, project_path, session_id):
        """Test emergency system initialization."""
        assert emergency_system.project_path == project_path
        assert emergency_system.session_id == session_id
        assert not emergency_system.active
        assert not emergency_system.monitoring_active
        assert len(emergency_system.triggers) > 0
        assert len(emergency_system.emergency_events) == 0

    async def test_start_stop_monitoring(self, emergency_system):
        """Test starting and stopping monitoring."""
        # Start monitoring
        await emergency_system.start_monitoring()
        assert emergency_system.active
        assert emergency_system.monitoring_active

        # Stop monitoring
        await emergency_system.stop_monitoring()
        assert not emergency_system.active
        assert not emergency_system.monitoring_active

    async def test_default_triggers_configuration(self, emergency_system):
        """Test default triggers are properly configured."""
        trigger_types = [trigger.failure_type for trigger in emergency_system.triggers]

        expected_types = [
            CriticalFailureType.RESOURCE_EXHAUSTION,
            CriticalFailureType.INFINITE_LOOP,
            CriticalFailureType.REPEATED_FAILURES,
            CriticalFailureType.SECURITY_VIOLATION,
            CriticalFailureType.DATA_CORRUPTION,
            CriticalFailureType.SYSTEM_INSTABILITY,
        ]

        for expected_type in expected_types:
            assert expected_type in trigger_types

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.disk_usage")
    async def test_resource_exhaustion_detection(
        self, mock_disk, mock_cpu, mock_memory, emergency_system
    ):
        """Test resource exhaustion detection."""
        # Mock high resource usage
        mock_memory.return_value.percent = 98.0
        mock_cpu.return_value = 95.0
        mock_disk.return_value.percent = 90.0

        # Test resource exhaustion check
        result = await emergency_system._check_resource_exhaustion(0.95)
        assert result  # Should detect exhaustion

        # Mock normal resource usage
        mock_memory.return_value.percent = 50.0
        mock_cpu.return_value = 30.0
        mock_disk.return_value.percent = 40.0

        result = await emergency_system._check_resource_exhaustion(0.95)
        assert not result  # Should not detect exhaustion

    async def test_repeated_failures_detection(self, emergency_system):
        """Test repeated failures detection."""
        current_time = time.time()

        # Add multiple unresolved events
        for i in range(3):
            event = EmergencyEvent(
                timestamp=current_time - (i * 60),  # Events in last 3 minutes
                failure_type=CriticalFailureType.SYSTEM_INSTABILITY,
                intervention_level=InterventionLevel.WARNING,
                session_id=emergency_system.session_id,
                description=f"Test failure {i}",
            )
            emergency_system.emergency_events.append(event)

        # Should detect repeated failures
        result = await emergency_system._check_repeated_failures(3, 5)
        assert result

        # Should not detect if threshold is higher
        result = await emergency_system._check_repeated_failures(5, 5)
        assert not result

    async def test_human_intervention_request(self, emergency_system):
        """Test human intervention request."""
        assert not emergency_system.human_intervention_requested

        emergency_system.request_human_intervention("Test reason")
        assert emergency_system.human_intervention_requested

    async def test_emergency_event_resolution(self, emergency_system):
        """Test resolving emergency events."""
        # Add an event
        event = EmergencyEvent(
            timestamp=time.time(),
            failure_type=CriticalFailureType.REPEATED_FAILURES,
            intervention_level=InterventionLevel.WARNING,
            session_id=emergency_system.session_id,
            description="Test event",
        )
        emergency_system.emergency_events.append(event)

        # Resolve the event
        emergency_system.resolve_emergency(0, "Manual resolution")

        assert emergency_system.emergency_events[0].resolved
        assert emergency_system.emergency_events[0].resolution_timestamp is not None

    async def test_emergency_status_reporting(self, emergency_system):
        """Test emergency status reporting."""
        # Add test events
        for i in range(2):
            event = EmergencyEvent(
                timestamp=time.time() - i,
                failure_type=CriticalFailureType.SYSTEM_INSTABILITY,
                intervention_level=InterventionLevel.WARNING,
                session_id=emergency_system.session_id,
                description=f"Test event {i}",
            )
            emergency_system.emergency_events.append(event)

        status = emergency_system.get_emergency_status()

        assert status["total_events"] == 2
        assert status["unresolved_events"] == 2
        assert len(status["recent_events"]) == 2

    async def test_callback_registration(self, emergency_system):
        """Test registering emergency callbacks."""
        termination_callback = AsyncMock()
        rollback_callback = AsyncMock()
        pause_callback = AsyncMock()
        alert_callback = AsyncMock()

        emergency_system.register_callbacks(
            session_termination_callback=termination_callback,
            rollback_callback=rollback_callback,
            pause_callback=pause_callback,
            alert_callback=alert_callback,
        )

        assert emergency_system.session_termination_callback == termination_callback
        assert emergency_system.rollback_callback == rollback_callback
        assert emergency_system.pause_callback == pause_callback
        assert emergency_system.alert_callback == alert_callback

    async def test_warning_intervention(self, emergency_system):
        """Test warning level intervention."""
        alert_callback = AsyncMock()
        emergency_system.register_callbacks(alert_callback=alert_callback)

        event = EmergencyEvent(
            timestamp=time.time(),
            failure_type=CriticalFailureType.SYSTEM_INSTABILITY,
            intervention_level=InterventionLevel.WARNING,
            session_id=emergency_system.session_id,
            description="Test warning",
        )

        await emergency_system._handle_warning(event)
        alert_callback.assert_called_once_with(event)

    async def test_termination_intervention(self, emergency_system):
        """Test termination level intervention."""
        termination_callback = AsyncMock()
        emergency_system.register_callbacks(
            session_termination_callback=termination_callback
        )

        event = EmergencyEvent(
            timestamp=time.time(),
            failure_type=CriticalFailureType.SECURITY_VIOLATION,
            intervention_level=InterventionLevel.TERMINATE,
            session_id=emergency_system.session_id,
            description="Security violation",
        )

        # Start monitoring first
        await emergency_system.start_monitoring()
        assert emergency_system.active

        await emergency_system._handle_termination(event)

        termination_callback.assert_called_once_with(event)
        assert not emergency_system.active  # Should stop monitoring

    async def test_escalation_intervention(self, emergency_system):
        """Test escalation level intervention."""
        emergency_system.emergency_contacts = ["admin@example.com"]

        event = EmergencyEvent(
            timestamp=time.time(),
            failure_type=CriticalFailureType.SYSTEM_INSTABILITY,
            intervention_level=InterventionLevel.ESCALATE,
            session_id=emergency_system.session_id,
            description="System instability",
        )

        await emergency_system._handle_escalation(event)
        assert emergency_system.human_intervention_requested

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.disk_usage")
    async def test_emergency_metrics_collection(
        self, mock_disk, mock_cpu, mock_memory, emergency_system
    ):
        """Test emergency metrics collection."""
        # Mock system metrics
        mock_memory.return_value.percent = 85.0
        mock_cpu.return_value = 70.0
        mock_disk.return_value.percent = 60.0

        metrics = await emergency_system._collect_emergency_metrics()

        assert metrics["memory_percent"] == 85.0
        assert metrics["cpu_percent"] == 70.0
        assert metrics["disk_percent"] == 60.0
        assert "timestamp" in metrics
        assert metrics["session_id"] == emergency_system.session_id

    async def test_trigger_evaluation_resource_exhaustion(self, emergency_system):
        """Test trigger evaluation for resource exhaustion."""
        trigger = EmergencyTrigger(
            CriticalFailureType.RESOURCE_EXHAUSTION,
            InterventionLevel.PAUSE,
            0.95,
            window_minutes=2,
            description="Test trigger",
        )

        with patch.object(
            emergency_system, "_check_resource_exhaustion", return_value=True
        ):
            result = await emergency_system._evaluate_trigger(trigger, time.time())
            assert result

        with patch.object(
            emergency_system, "_check_resource_exhaustion", return_value=False
        ):
            result = await emergency_system._evaluate_trigger(trigger, time.time())
            assert not result

    async def test_trigger_evaluation_human_intervention(self, emergency_system):
        """Test trigger evaluation for human intervention request."""
        trigger = EmergencyTrigger(
            CriticalFailureType.HUMAN_INTERVENTION_REQUESTED,
            InterventionLevel.ESCALATE,
            1,
            window_minutes=1,
            description="Human intervention",
        )

        # Not requested initially
        result = await emergency_system._evaluate_trigger(trigger, time.time())
        assert not result

        # Request intervention
        emergency_system.human_intervention_requested = True
        result = await emergency_system._evaluate_trigger(trigger, time.time())
        assert result


@pytest.mark.asyncio
class TestEmergencyTriggerConfiguration:
    """Test emergency trigger configuration."""

    def test_trigger_creation(self):
        """Test creating emergency triggers."""
        trigger = EmergencyTrigger(
            CriticalFailureType.RESOURCE_EXHAUSTION,
            InterventionLevel.PAUSE,
            0.90,
            window_minutes=5,
            description="Test trigger",
        )

        assert trigger.failure_type == CriticalFailureType.RESOURCE_EXHAUSTION
        assert trigger.intervention_level == InterventionLevel.PAUSE
        assert trigger.threshold == 0.90
        assert trigger.window_minutes == 5
        assert trigger.description == "Test trigger"


@pytest.mark.asyncio
class TestEmergencyEvent:
    """Test emergency event handling."""

    def test_event_creation(self):
        """Test creating emergency events."""
        timestamp = time.time()
        event = EmergencyEvent(
            timestamp=timestamp,
            failure_type=CriticalFailureType.REPEATED_FAILURES,
            intervention_level=InterventionLevel.ROLLBACK,
            session_id="test-session",
            description="Test event",
            metrics={"test": "data"},
        )

        assert event.timestamp == timestamp
        assert event.failure_type == CriticalFailureType.REPEATED_FAILURES
        assert event.intervention_level == InterventionLevel.ROLLBACK
        assert event.session_id == "test-session"
        assert event.description == "Test event"
        assert event.metrics == {"test": "data"}
        assert not event.resolved
        assert event.resolution_timestamp is None
