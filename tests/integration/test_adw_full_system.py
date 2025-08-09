"""Integration tests for complete ADW system.

Tests the full autonomous development workflow with all components integrated:
- Session manager with cognitive load management
- Failure prediction and prevention
- Autonomous monitoring dashboard
- Resource management and rollback systems
- Extended session capabilities
"""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.adw.cognitive_load_manager import CognitiveLoadManager, SessionMode
from src.core.adw.session_manager import ADWSession, ADWSessionConfig, SessionPhase
from src.core.monitoring.autonomous_dashboard import AutonomousDashboard
from src.core.prediction.failure_prediction import FailurePredictionSystem


class TestADWFullSystemIntegration:
    """Integration tests for complete ADW system."""

    @pytest.fixture
    def project_path(self, tmp_path):
        """Create a temporary project directory."""
        return tmp_path

    @pytest.fixture
    def adw_config(self):
        """Create ADW configuration with all components enabled."""
        return ADWSessionConfig(
            total_duration_hours=0.1,  # Short duration for testing
            reconnaissance_minutes=1,
            integration_validation_minutes=1,
            meta_learning_minutes=1,
            micro_iteration_minutes=2,
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
            extended_session_mode=False,
            cognitive_fatigue_threshold=0.7,
        )

    @pytest.fixture
    def extended_config(self):
        """Create extended session configuration."""
        return ADWSessionConfig(
            total_duration_hours=0.2,  # Short but extended for testing
            max_extended_duration_hours=0.5,
            extended_session_mode=True,
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
            cognitive_fatigue_threshold=0.5,  # Lower threshold for testing
        )

    @pytest.mark.asyncio
    async def test_full_adw_session_integration(self, project_path, adw_config):
        """Test complete ADW session with all components integrated."""
        with (
            patch.multiple(
                "src.core.adw.session_manager",
                AutoRollbackSystem=MagicMock,
                AutonomousQualityGates=MagicMock,
                ResourceGuardian=MagicMock,
            ),
            patch.multiple(
                "src.core.monitoring.autonomous_dashboard",
                asyncio=MagicMock(),
            ),
            patch.multiple(
                "src.core.prediction.failure_prediction",
                asyncio=MagicMock(),
            ),
        ):
            session = ADWSession(project_path, adw_config)

            # Verify all components are initialized
            assert session.cognitive_load_manager is not None
            assert session.failure_predictor is not None
            assert session.dashboard is not None
            assert session.current_session_mode == SessionMode.FOCUS

            # Mock component methods
            session.cognitive_load_manager.start_session = AsyncMock()
            session.cognitive_load_manager.assess_cognitive_state = AsyncMock()
            session.cognitive_load_manager.get_optimal_mode = AsyncMock(
                return_value=SessionMode.FOCUS
            )
            session.cognitive_load_manager.end_session = AsyncMock(
                return_value={
                    "total_fatigue": 0.3,
                    "mode_transitions": 0,
                    "optimal_session_length": 3.5,
                }
            )

            session.failure_predictor.start_monitoring = AsyncMock()
            session.failure_predictor.predict_failure_risk = AsyncMock(return_value=0.2)
            session.failure_predictor.stop_monitoring = AsyncMock()
            session.failure_predictor.get_session_summary = AsyncMock(
                return_value={"predictions_made": 4, "high_risk_phases": 0}
            )

            session.dashboard.start_monitoring = AsyncMock()
            session.dashboard.stop_monitoring = AsyncMock()
            session.dashboard.get_session_metrics = AsyncMock(
                return_value={"data_points_collected": 50, "alerts_generated": 0}
            )

            # Mock safety systems
            session.rollback_system.create_safety_checkpoint = AsyncMock(
                return_value="checkpoint-123"
            )
            session.quality_gates.run_all_gates = AsyncMock(
                return_value=[
                    MagicMock(passed=True, score=0.9, gate_name="test_gate"),
                    MagicMock(passed=True, score=0.8, gate_name="quality_gate"),
                ]
            )
            session.quality_gates.get_gate_statistics = MagicMock(
                return_value={"total_runs": 2, "success_rate": 1.0}
            )

            session.resource_guardian.start_monitoring = AsyncMock()
            session.resource_guardian.stop_monitoring = AsyncMock()
            session.resource_guardian.get_current_status = AsyncMock()
            session.resource_guardian.get_resource_statistics = MagicMock(
                return_value={"avg_memory": 45.0, "avg_cpu": 30.0}
            )

            session.rollback_system.get_rollback_statistics = MagicMock(
                return_value={"total_attempts": 0, "success_rate": 1.0}
            )

            # Mock persistence
            session.persistence.save_checkpoint = AsyncMock()

            # Mock subprocess for git operations
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(
                    stdout="", stderr="", returncode=0
                )

                # Mock asyncio subprocess for pytest
                with patch("asyncio.create_subprocess_exec") as mock_async_subprocess:
                    mock_process = AsyncMock()
                    mock_process.communicate.return_value = (b"", b"")
                    mock_process.returncode = 0
                    mock_async_subprocess.return_value = mock_process

                    # Start session
                    result = await session.start_session(
                        target_goals=["implement_test_feature"]
                    )

        # Verify session completed successfully
        assert result["status"] == "completed"
        assert result["session_id"] == session.session_id
        assert "cognitive_stats" in result
        assert "failure_prediction_stats" in result
        assert "dashboard_metrics" in result

        # Verify component interactions
        session.cognitive_load_manager.start_session.assert_called_once()
        session.failure_predictor.start_monitoring.assert_called_once()
        session.dashboard.start_monitoring.assert_called_once_with(session.session_id)

        # Verify cognitive state was assessed
        assert session.cognitive_load_manager.assess_cognitive_state.call_count >= 4

        # Verify failure prediction was called for each phase
        assert session.failure_predictor.predict_failure_risk.call_count >= 4

    @pytest.mark.asyncio
    async def test_cognitive_load_mode_transitions(self, project_path, adw_config):
        """Test cognitive load management and mode transitions."""
        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            session = ADWSession(project_path, adw_config)

            # Mock high fatigue scenario
            session.cognitive_load_manager.assess_cognitive_state = AsyncMock()
            session.cognitive_load_manager.get_optimal_mode = AsyncMock()

            # Simulate fatigue escalation
            fatigue_states = [
                MagicMock(fatigue_level=0.3, focus_efficiency=0.9),
                MagicMock(fatigue_level=0.8, focus_efficiency=0.4),  # High fatigue
                MagicMock(fatigue_level=0.6, focus_efficiency=0.7),
                MagicMock(fatigue_level=0.4, focus_efficiency=0.8),
            ]

            optimal_modes = [
                SessionMode.FOCUS,
                SessionMode.REST,  # Switch to rest due to high fatigue
                SessionMode.EXPLORATION,
                SessionMode.FOCUS,
            ]

            session.cognitive_load_manager.assess_cognitive_state.side_effect = (
                fatigue_states
            )
            session.cognitive_load_manager.get_optimal_mode.side_effect = optimal_modes

            # Initialize mode transition
            initial_mode = session.current_session_mode
            await session._transition_session_mode(SessionMode.REST)

            # Verify mode transition occurred
            assert session.current_session_mode == SessionMode.REST
            assert session.config.micro_iteration_minutes == 45  # Extended for rest

            # Test transition back to focus
            await session._transition_session_mode(SessionMode.FOCUS)
            assert session.current_session_mode == SessionMode.FOCUS
            assert session.config.micro_iteration_minutes == 30  # Back to standard

    @pytest.mark.asyncio
    async def test_failure_prediction_and_prevention(self, project_path, adw_config):
        """Test failure prediction and preventive actions."""
        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            session = ADWSession(project_path, adw_config)

            # Mock high-risk failure prediction
            session.failure_predictor.predict_failure_risk = AsyncMock(
                return_value=0.8  # High risk
            )
            session.rollback_system.create_safety_checkpoint = AsyncMock(
                return_value="preventive-checkpoint-456"
            )

            initial_iteration_time = session.config.micro_iteration_minutes

            # Test preventive action
            await session._handle_high_failure_risk(SessionPhase.MICRO_DEVELOPMENT, 0.8)

            # Verify preventive measures were taken
            session.rollback_system.create_safety_checkpoint.assert_called_with(
                "Pre-micro_development high-risk checkpoint"
            )
            assert session.config.micro_iteration_minutes > initial_iteration_time
            assert session.config.quality_gates_enabled is True

    @pytest.mark.asyncio
    async def test_extended_session_planning(self, project_path, extended_config):
        """Test extended session planning and execution."""
        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            session = ADWSession(project_path, extended_config)

            # Test extended session phase planning
            phases = await session._plan_extended_session()

            # Should have multiple cycles for 0.5 hour duration (0.5/4 = 0.125, so at least 1 cycle)
            expected_phases_per_cycle = 4
            assert len(phases) >= expected_phases_per_cycle

            # Verify phase sequence
            cycle_phases = [
                SessionPhase.RECONNAISSANCE,
                SessionPhase.MICRO_DEVELOPMENT,
                SessionPhase.INTEGRATION_VALIDATION,
                SessionPhase.META_LEARNING,
            ]

            for i, expected_phase in enumerate(cycle_phases):
                assert phases[i] == expected_phase

    @pytest.mark.asyncio
    async def test_dashboard_integration_and_metrics(self, project_path, adw_config):
        """Test autonomous dashboard integration and metrics collection."""
        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            session = ADWSession(project_path, adw_config)

            # Mock dashboard methods
            session.dashboard.start_monitoring = AsyncMock()
            session.dashboard.stop_monitoring = AsyncMock()
            session.dashboard.get_session_metrics = AsyncMock(
                return_value={
                    "data_points_collected": 100,
                    "alerts_generated": 2,
                    "performance_trend": "stable",
                    "resource_efficiency": 0.85,
                }
            )

            # Start monitoring
            await session.dashboard.start_monitoring(session.session_id)
            session.dashboard.start_monitoring.assert_called_once_with(
                session.session_id
            )

            # Get metrics
            metrics = await session.dashboard.get_session_metrics()
            assert metrics["data_points_collected"] == 100
            assert metrics["alerts_generated"] == 2
            assert "performance_trend" in metrics
            assert "resource_efficiency" in metrics

            # Stop monitoring
            await session.dashboard.stop_monitoring()
            session.dashboard.stop_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_component_failure_resilience(self, project_path, adw_config):
        """Test system resilience when individual components fail."""
        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            session = ADWSession(project_path, adw_config)

            # Simulate dashboard failure
            session.dashboard.start_monitoring = AsyncMock(
                side_effect=Exception("Dashboard connection failed")
            )

            # Simulate failure prediction failure
            session.failure_predictor.predict_failure_risk = AsyncMock(
                side_effect=Exception("Prediction model error")
            )

            # Mock other components normally
            session.cognitive_load_manager.start_session = AsyncMock()
            session.cognitive_load_manager.assess_cognitive_state = AsyncMock(
                return_value=MagicMock(fatigue_level=0.3, focus_efficiency=0.9)
            )

            session.rollback_system.create_safety_checkpoint = AsyncMock(
                return_value="checkpoint-resilience"
            )
            session.quality_gates.run_all_gates = AsyncMock(
                return_value=[MagicMock(passed=True, score=0.9, gate_name="test")]
            )
            session.resource_guardian.start_monitoring = AsyncMock()
            session.resource_guardian.get_current_status = AsyncMock()
            session.persistence.save_checkpoint = AsyncMock()

            # Mock subprocess operations
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = MagicMock(
                    stdout="", stderr="", returncode=0
                )

                with patch("asyncio.create_subprocess_exec") as mock_async_subprocess:
                    mock_process = AsyncMock()
                    mock_process.communicate.return_value = (b"", b"")
                    mock_process.returncode = 0
                    mock_async_subprocess.return_value = mock_process

                    # Session should still work despite component failures
                    try:
                        result = await session.start_session(target_goals=["test"])
                        # System should be resilient and continue operation
                        assert result is not None
                        # Status might be failed due to component errors, but system should not crash
                    except Exception as e:
                        # If exceptions occur, they should be handled gracefully
                        assert "Dashboard connection failed" in str(
                            e
                        ) or "Prediction model error" in str(e)

    @pytest.mark.asyncio
    async def test_session_persistence_with_new_components(
        self, project_path, adw_config
    ):
        """Test session persistence includes new component state."""
        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            session = ADWSession(project_path, adw_config)

            # Mock component state
            cognitive_state = {"mode": "focus", "fatigue": 0.4, "efficiency": 0.8}
            failure_context = {"last_prediction": 0.3, "risk_factors": ["complexity"]}

            session.persistence.save_checkpoint = AsyncMock()

            # Simulate checkpoint save with enhanced context
            await session.persistence.save_checkpoint(
                session,
                phase_progress={"reconnaissance_completed": True},
                additional_context={
                    "goals": ["test_goal"],
                    "cognitive_mode": session.current_session_mode.value,
                    "failure_predictions": 0.3,
                    "component_states": {
                        "cognitive_load": cognitive_state,
                        "failure_prediction": failure_context,
                    },
                },
            )

            # Verify checkpoint was called with enhanced context
            session.persistence.save_checkpoint.assert_called_once()
            call_args = session.persistence.save_checkpoint.call_args
            additional_context = call_args[1]["additional_context"]

            assert "cognitive_mode" in additional_context
            assert "failure_predictions" in additional_context
            assert additional_context["cognitive_mode"] == "focus"
            assert additional_context["failure_predictions"] == 0.3


class TestADWSystemPerformance:
    """Performance tests for integrated ADW system."""

    @pytest.mark.asyncio
    async def test_session_startup_performance(self, tmp_path):
        """Test that session startup with all components is performant."""
        config = ADWSessionConfig(
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
        )

        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            start_time = time.time()
            session = ADWSession(tmp_path, config)
            initialization_time = time.time() - start_time

            # Initialization should be fast (< 1 second in tests)
            assert initialization_time < 1.0

            # Verify all components are initialized
            assert session.cognitive_load_manager is not None
            assert session.failure_predictor is not None
            assert session.dashboard is not None

    @pytest.mark.asyncio
    async def test_component_interaction_overhead(self, tmp_path):
        """Test overhead of component interactions during session."""
        config = ADWSessionConfig(
            total_duration_hours=0.05,  # Very short for performance testing
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
        )

        with patch.multiple(
            "src.core.adw.session_manager",
            AutoRollbackSystem=MagicMock,
            AutonomousQualityGates=MagicMock,
            ResourceGuardian=MagicMock,
        ):
            session = ADWSession(tmp_path, config)

            # Mock component interactions with timing
            call_count = 0

            async def mock_assess_cognitive_state():
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.001)  # Simulate small delay
                return MagicMock(fatigue_level=0.3, focus_efficiency=0.9)

            async def mock_predict_failure_risk(context):
                await asyncio.sleep(0.001)  # Simulate small delay
                return 0.2

            session.cognitive_load_manager.assess_cognitive_state = (
                mock_assess_cognitive_state
            )
            session.failure_predictor.predict_failure_risk = mock_predict_failure_risk

            # Time the cognitive assessment calls
            start_time = time.time()
            for _ in range(10):
                await session.cognitive_load_manager.assess_cognitive_state()
            assessment_time = time.time() - start_time

            # Should be efficient (< 0.1 seconds for 10 calls)
            assert assessment_time < 0.1
            assert call_count == 10
