"""
Unit tests for ADW Session Manager.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.adw.session_manager import ADWSession, ADWSessionConfig, SessionPhase


@pytest.mark.asyncio
async def test_adw_session_initialization():
    """Test ADW session initialization."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        config = ADWSessionConfig()

        session = ADWSession(project_path, config)

        assert session.project_path == project_path
        assert session.config == config
        assert session.metrics.current_phase == SessionPhase.RECONNAISSANCE
        assert not session.active
        assert session.consecutive_failures == 0


@pytest.mark.asyncio
async def test_adw_session_phases():
    """Test ADW session phase execution."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session = ADWSession(project_path)

        # Mock the safety systems
        session.rollback_system.create_safety_checkpoint = AsyncMock(
            return_value="checkpoint_hash"
        )

        # Mock resource status with proper attributes
        mock_resource_status = MagicMock()
        mock_resource_status.memory_percent = 50.0
        mock_resource_status.cpu_percent = 30.0
        mock_resource_status.disk_percent = 70.0
        mock_resource_status.warnings = []
        mock_resource_status.critical_alerts = []
        session.resource_guardian.get_current_status = AsyncMock(
            return_value=mock_resource_status
        )

        session.resource_guardian.start_monitoring = AsyncMock()
        session.resource_guardian.stop_monitoring = AsyncMock()

        # Mock quality gate result with proper attributes
        mock_quality_result = MagicMock()
        mock_quality_result.passed = True
        mock_quality_result.score = 0.8
        session.quality_gates.run_all_gates = AsyncMock(
            return_value=[mock_quality_result]
        )

        # Mock test optimizer
        session.resource_guardian.test_optimizer.measure_test_runtime = AsyncMock(
            return_value=30.0
        )
        session.resource_guardian.test_optimizer.baseline_runtime = None
        session.resource_guardian.test_optimizer.set_baseline_runtime = MagicMock()

        # Mock git commands
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = ""

            # Test reconnaissance phase
            recon_result = await session._run_reconnaissance_phase()
            assert isinstance(recon_result, dict)
            assert "resource_status" in recon_result


@pytest.mark.asyncio
async def test_micro_iteration():
    """Test micro-development iteration."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session = ADWSession(project_path)

        # Mock dependencies
        session.rollback_system.create_safety_checkpoint = AsyncMock(
            return_value="checkpoint_hash"
        )
        session.quality_gates.run_all_gates = AsyncMock(return_value=[])
        session.resource_guardian.get_current_status = AsyncMock()

        # Mock the current status to return safe values
        mock_status = MagicMock()
        mock_status.critical_alerts = []
        session.resource_guardian.get_current_status.return_value = mock_status

        result = await session._run_micro_iteration(1)

        assert isinstance(result, dict)
        assert "status" in result
        assert "iteration" in result
        assert "duration" in result


@pytest.mark.asyncio
async def test_integration_validation():
    """Test integration validation phase."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session = ADWSession(project_path)

        # Mock dependencies
        session.quality_gates.run_all_gates = AsyncMock(return_value=[])
        session.resource_guardian.test_optimizer.measure_test_runtime = AsyncMock(
            return_value=30.0
        )
        session.resource_guardian.test_optimizer.baseline_runtime = 25.0
        session.resource_guardian.test_optimizer.is_runtime_regression = MagicMock(
            return_value=False
        )

        # Mock subprocess for pytest
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            result = await session._run_integration_validation_phase()

            assert isinstance(result, dict)
            assert "test_suite" in result
            assert "overall_success" in result


@pytest.mark.asyncio
async def test_meta_learning():
    """Test meta-learning phase."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session = ADWSession(project_path)

        # Mock dependencies
        session.rollback_system.get_rollback_statistics = MagicMock(
            return_value={
                "total_attempts": 0,
                "success_rate": 1.0,
                "failure_type_distribution": {},
            }
        )

        result = await session._run_meta_learning_phase()

        assert isinstance(result, dict)
        assert "patterns_discovered" in result
        assert "performance_improvements" in result
        assert "next_priorities" in result


@pytest.mark.asyncio
async def test_phase_failure_handling():
    """Test phase failure handling and rollback."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session = ADWSession(project_path)

        # Mock rollback system
        session.rollback_system.handle_failure = AsyncMock(return_value=True)

        await session._handle_phase_failure(
            SessionPhase.MICRO_DEVELOPMENT, "test error"
        )

        # Verify rollback was triggered
        session.rollback_system.handle_failure.assert_called_once()
        assert session.metrics.rollbacks_triggered == 1


def test_session_config():
    """Test session configuration."""
    config = ADWSessionConfig()

    assert config.total_duration_hours == 4.0
    assert config.reconnaissance_minutes == 15
    assert config.integration_validation_minutes == 30
    assert config.meta_learning_minutes == 15
    assert config.micro_iteration_minutes == 30
    assert config.quality_gates_enabled is True
    assert config.test_first_enforced is True


def test_session_metrics():
    """Test session metrics tracking."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session = ADWSession(project_path)

        # Test initial metrics
        assert session.metrics.commits_made == 0
        assert session.metrics.tests_written == 0
        assert session.metrics.current_phase == SessionPhase.RECONNAISSANCE

        # Test duration calculation
        duration = session.metrics.duration()
        assert duration >= 0
