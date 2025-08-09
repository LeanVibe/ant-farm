"""Tests for Extended Session Testing Framework."""

import asyncio
import json
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.core.testing.extended_session_tester import (
    ExtendedSessionTester,
    analyze_test_results,
    compare_test_results,
)


class TestExtendedSessionTester:
    """Tests for the extended session testing framework."""

    @pytest.fixture
    def project_path(self, tmp_path):
        """Create a temporary project directory."""
        return tmp_path

    @pytest.fixture
    def output_dir(self, tmp_path):
        """Create output directory for test results."""
        output_dir = tmp_path / "test_outputs"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def tester(self, project_path, output_dir):
        """Create extended session tester instance."""
        return ExtendedSessionTester(project_path, output_dir)

    @pytest.mark.asyncio
    async def test_endurance_test_basic(self, tester):
        """Test basic endurance test functionality."""
        with patch(
            "src.core.testing.extended_session_tester.ADWSession"
        ) as MockSession:
            # Mock session results
            mock_session_instance = AsyncMock()
            mock_session_instance.start_session.return_value = {
                "status": "completed",
                "session_id": "test-session-1",
                "duration_hours": 4.0,
                "metrics": {
                    "commits_made": 8,
                    "tests_written": 12,
                    "tests_passed": 10,
                    "quality_gate_passes": 3,
                    "quality_gate_failures": 1,
                    "rollbacks_triggered": 0,
                },
            }
            MockSession.return_value = mock_session_instance

            # Mock dashboard
            with (
                patch.object(tester, "_start_monitoring", new=AsyncMock()),
                patch.object(tester, "_stop_monitoring", new=AsyncMock()),
            ):
                # Run short endurance test
                result = await tester.run_endurance_test(
                    duration_hours=0.1,  # Very short for testing
                    session_goals=["test_endurance_feature"],
                )

                # Verify test results
                assert result["test_type"] == "endurance"
                assert result["session_summary"]["planned"] >= 1
                assert result["session_summary"]["success_rate"] > 0
                assert "performance_summary" in result
                assert "resource_summary" in result

                # Verify session was called
                MockSession.assert_called()
                mock_session_instance.start_session.assert_called()

    @pytest.mark.asyncio
    async def test_stress_test_configuration(self, tester):
        """Test stress test with specific configuration."""
        with patch(
            "src.core.testing.extended_session_tester.ADWSession"
        ) as MockSession:
            mock_session_instance = AsyncMock()
            mock_session_instance.start_session.return_value = {
                "status": "completed",
                "session_id": "stress-session-1",
                "duration_hours": 2.0,
                "metrics": {
                    "commits_made": 3,  # Lower due to stress
                    "tests_written": 6,
                    "tests_passed": 4,
                    "quality_gate_passes": 1,
                    "quality_gate_failures": 2,
                    "rollbacks_triggered": 1,
                },
            }
            MockSession.return_value = mock_session_instance

            with (
                patch.object(tester, "_start_monitoring", new=AsyncMock()),
                patch.object(tester, "_stop_monitoring", new=AsyncMock()),
            ):
                stress_factors = {
                    "goals": ["stress_feature_1", "stress_feature_2"],
                    "complexity_factor": 1.5,
                }

                result = await tester.run_stress_test(
                    duration_hours=0.05,  # Very short for testing
                    stress_factors=stress_factors,
                )

                # Verify stress test specifics
                assert result["test_type"] == "stress"
                assert result["test_context"]["goals"] == stress_factors["goals"]

                # Verify session configuration (would be stress-optimized)
                MockSession.assert_called()
                call_args = MockSession.call_args
                config = call_args[0][1]  # Second argument is config
                assert config.cognitive_fatigue_threshold == 0.8  # Higher for stress
                assert config.micro_iteration_minutes == 20  # Shorter for stress

    @pytest.mark.asyncio
    async def test_recovery_test_failure_handling(self, tester):
        """Test recovery test with simulated failures."""
        with patch(
            "src.core.testing.extended_session_tester.ADWSession"
        ) as MockSession:
            # Mock session with failures
            mock_session_instance = AsyncMock()

            # Simulate pattern: success, failure, success (recovery)
            mock_session_instance.start_session.side_effect = [
                {
                    "status": "completed",
                    "session_id": "recovery-session-1",
                    "duration_hours": 3.0,
                    "metrics": {
                        "commits_made": 6,
                        "tests_written": 8,
                        "tests_passed": 7,
                    },
                },
                {
                    "status": "failed",
                    "session_id": "recovery-session-2",
                    "duration_hours": 1.5,
                    "error": "simulated_failure",
                    "metrics": {
                        "commits_made": 2,
                        "tests_written": 3,
                        "tests_passed": 1,
                    },
                },
                {
                    "status": "completed",
                    "session_id": "recovery-session-3",
                    "duration_hours": 3.5,
                    "metrics": {
                        "commits_made": 7,
                        "tests_written": 9,
                        "tests_passed": 8,
                    },
                },
            ]
            MockSession.return_value = mock_session_instance

            with (
                patch.object(tester, "_start_monitoring", new=AsyncMock()),
                patch.object(tester, "_stop_monitoring", new=AsyncMock()),
            ):
                result = await tester.run_recovery_test(
                    duration_hours=0.2,  # Short test with multiple sessions
                    failure_injection_rate=0.3,
                )

                # Verify recovery behavior
                assert result["test_type"] == "recovery"
                assert result["session_summary"]["failed"] >= 1
                assert result["session_summary"]["completed"] >= 1

                # Should have recorded failure types
                assert "failure_analysis" in result
                assert (
                    result["failure_analysis"]["failure_types"]["simulated_failure"]
                    >= 1
                )

    @pytest.mark.asyncio
    async def test_efficiency_test_performance_tracking(self, tester):
        """Test efficiency test with performance targets."""
        with patch(
            "src.core.testing.extended_session_tester.ADWSession"
        ) as MockSession:
            mock_session_instance = AsyncMock()

            # Mock progressively improving performance
            mock_session_instance.start_session.side_effect = [
                {
                    "status": "completed",
                    "duration_hours": 4.0,
                    "metrics": {
                        "commits_made": 6,
                        "tests_written": 8,
                        "tests_passed": 6,
                    },
                },
                {
                    "status": "completed",
                    "duration_hours": 4.0,
                    "metrics": {
                        "commits_made": 8,
                        "tests_written": 10,
                        "tests_passed": 9,
                    },
                },
                {
                    "status": "completed",
                    "duration_hours": 4.0,
                    "metrics": {
                        "commits_made": 10,
                        "tests_written": 12,
                        "tests_passed": 11,
                    },
                },
            ]
            MockSession.return_value = mock_session_instance

            with (
                patch.object(tester, "_start_monitoring", new=AsyncMock()),
                patch.object(tester, "_stop_monitoring", new=AsyncMock()),
            ):
                efficiency_targets = {
                    "min_commits_per_hour": 2.0,
                    "min_test_success_rate": 0.8,
                    "max_rollback_rate": 0.1,
                }

                result = await tester.run_efficiency_test(
                    duration_hours=0.15,  # Short test
                    efficiency_targets=efficiency_targets,
                )

                # Verify efficiency tracking
                assert result["test_type"] == "efficiency"
                assert "performance_summary" in result

                # Check performance improvement trend
                commits_per_hour = result["raw_metrics"]["commits_per_hour"]
                assert len(commits_per_hour) >= 2
                # Should show improvement: 6/4=1.5, 8/4=2.0, 10/4=2.5
                assert commits_per_hour[-1] > commits_per_hour[0]

    @pytest.mark.asyncio
    async def test_cognitive_progression_test(self, tester):
        """Test cognitive progression tracking."""
        with patch(
            "src.core.testing.extended_session_tester.ADWSession"
        ) as MockSession:
            mock_session_instance = AsyncMock()
            mock_session_instance.start_session.return_value = {
                "status": "completed",
                "session_id": "cognitive-session-1",
                "duration_hours": 4.0,
                "metrics": {"commits_made": 5, "tests_written": 7, "tests_passed": 6},
            }
            MockSession.return_value = mock_session_instance

            with (
                patch.object(tester, "_start_monitoring", new=AsyncMock()),
                patch.object(tester, "_stop_monitoring", new=AsyncMock()),
            ):
                result = await tester.run_cognitive_progression_test(
                    duration_hours=0.1  # Short test
                )

                # Verify cognitive test specifics
                assert result["test_type"] == "cognitive"
                assert "cognitive_summary" in result

                # Should use lower fatigue threshold for more transitions
                MockSession.assert_called()
                call_args = MockSession.call_args
                config = call_args[0][1]
                assert config.cognitive_fatigue_threshold == 0.5

    @pytest.mark.asyncio
    async def test_continuous_monitoring(self, tester):
        """Test continuous monitoring during extended sessions."""
        # Mock the dashboard instead of creating a real one
        with patch(
            "src.core.testing.extended_session_tester.AutonomousDashboard"
        ) as MockDashboard:
            mock_dashboard_instance = AsyncMock()
            mock_dashboard_instance.start_monitoring = AsyncMock()
            mock_dashboard_instance.stop_monitoring = AsyncMock()
            MockDashboard.return_value = mock_dashboard_instance

            # Start monitoring
            await tester._start_monitoring("test-monitoring")

            # Verify monitoring components
            assert tester.dashboard is not None
            assert tester.monitoring_task is not None
            assert not tester.monitoring_task.done()

            # Verify start_monitoring was called correctly
            mock_dashboard_instance.start_monitoring.assert_called_once_with(
                session_id="test-monitoring", interval_minutes=0.5
            )

            # Let monitoring run briefly
            await asyncio.sleep(0.1)

            # Stop monitoring
            await tester._stop_monitoring()

            # Verify cleanup
            mock_dashboard_instance.stop_monitoring.assert_called_once()
            assert tester.monitoring_task.done() or tester.monitoring_task.cancelled()

    @pytest.mark.asyncio
    async def test_test_abort_functionality(self, tester):
        """Test aborting a running test."""
        with patch(
            "src.core.testing.extended_session_tester.ADWSession"
        ) as MockSession:
            mock_session_instance = AsyncMock()
            mock_session_instance.abort_session = AsyncMock()
            MockSession.return_value = mock_session_instance

            # Start a test that would run forever
            async def long_running_session(*args, **kwargs):
                await asyncio.sleep(0.1)  # Short delay for testing
                return {"status": "completed"}

            mock_session_instance.start_session = long_running_session

            with (
                patch.object(tester, "_start_monitoring", new=AsyncMock()),
                patch.object(tester, "_stop_monitoring", new=AsyncMock()),
            ):
                # Start test in background
                test_task = asyncio.create_task(
                    tester.run_endurance_test(duration_hours=1.0)
                )

                # Wait a bit for test to start
                await asyncio.sleep(0.1)
                assert tester.active_test is True

                # Abort the test
                await tester.abort_test("test_abort")

                # Verify abort
                assert tester.active_test is False
                mock_session_instance.abort_session.assert_called_with("test_abort")

                # Cancel the background task
                test_task.cancel()
                try:
                    await test_task
                except asyncio.CancelledError:
                    pass

    def test_analyze_test_results(self, tmp_path):
        """Test test result analysis functionality."""
        # Create sample test report
        report_data = {
            "test_id": "test_analysis",
            "test_type": "endurance",
            "session_summary": {
                "success_rate": 0.75,  # Below threshold
            },
            "raw_metrics": {
                "commits_per_hour": [3.0, 2.8, 2.5, 2.0],  # Declining trend
            },
            "resource_summary": {
                "max_memory_usage": 85.0,  # High memory usage
            },
        }

        report_file = tmp_path / "test_report.json"
        with open(report_file, "w") as f:
            json.dump(report_data, f)

        # Analyze results
        analysis = analyze_test_results(report_file)

        # Verify analysis identified issues
        assert analysis["test_passed"] is False
        assert (
            len(analysis["issues"]) >= 2
        )  # Low success rate and performance degradation
        assert len(analysis["recommendations"]) >= 2

        # Check specific issues
        issues_text = " ".join(analysis["issues"])
        assert "success rate" in issues_text.lower()
        assert (
            "performance degradation" in issues_text.lower()
            or "memory usage" in issues_text.lower()
        )

    def test_compare_test_results(self, tmp_path):
        """Test comparing multiple test results."""
        # Create sample reports showing improvement
        report1_data = {
            "performance_summary": {
                "avg_commits_per_hour": 2.0,
                "avg_test_success_rate": 0.8,
                "avg_quality_gate_rate": 0.7,
            },
            "session_summary": {
                "success_rate": 0.75,
            },
        }

        report2_data = {
            "performance_summary": {
                "avg_commits_per_hour": 2.5,
                "avg_test_success_rate": 0.85,
                "avg_quality_gate_rate": 0.8,
            },
            "session_summary": {
                "success_rate": 0.85,
            },
        }

        report1_file = tmp_path / "report1.json"
        report2_file = tmp_path / "report2.json"

        with open(report1_file, "w") as f:
            json.dump(report1_data, f)
        with open(report2_file, "w") as f:
            json.dump(report2_data, f)

        # Compare results
        comparison = compare_test_results([report1_file, report2_file])

        # Verify comparison detected improvements
        assert comparison["reports_compared"] == 2
        assert comparison["stability_comparison"]["stability_trend"] == "improving"
        assert comparison["stability_comparison"]["avg_success_rate"] > 0.75

        # Check performance trends
        commits_trend = comparison["performance_trends"]["avg_commits_per_hour"]
        assert commits_trend["trend"] == "improving"
        assert commits_trend["change_percent"] > 0


class TestExtendedSessionMetrics:
    """Tests for extended session metrics collection."""

    def test_metrics_duration_calculation(self):
        """Test duration calculation in metrics."""
        from src.core.testing.extended_session_tester import ExtendedSessionMetrics

        start_time = time.time()
        metrics = ExtendedSessionMetrics(start_time=start_time)

        # Test ongoing duration
        time.sleep(0.1)
        duration1 = metrics.duration()
        assert duration1 > 0
        assert duration1 < 1  # Should be much less than 1 hour

        # Test completed duration
        end_time = start_time + 3600  # 1 hour later
        metrics.end_time = end_time
        duration2 = metrics.duration()
        assert abs(duration2 - 1.0) < 0.01  # Should be very close to 1 hour

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        from src.core.testing.extended_session_tester import ExtendedSessionMetrics

        metrics = ExtendedSessionMetrics(start_time=time.time())

        # No sessions
        assert metrics.success_rate() == 0.0

        # Some sessions
        metrics.total_sessions_planned = 10
        metrics.total_sessions_completed = 8
        metrics.total_sessions_failed = 2

        assert metrics.success_rate() == 0.8


class TestExtendedSessionIntegration:
    """Integration tests for extended session testing."""

    @pytest.mark.asyncio
    async def test_full_test_lifecycle(self, tmp_path):
        """Test complete extended session test lifecycle."""
        tester = ExtendedSessionTester(tmp_path)

        with patch(
            "src.core.testing.extended_session_tester.ADWSession"
        ) as MockSession:
            mock_session_instance = AsyncMock()
            mock_session_instance.start_session.return_value = {
                "status": "completed",
                "session_id": "lifecycle-test",
                "duration_hours": 4.0,
                "metrics": {
                    "commits_made": 10,
                    "tests_written": 15,
                    "tests_passed": 13,
                    "quality_gate_passes": 4,
                    "quality_gate_failures": 1,
                    "rollbacks_triggered": 0,
                },
            }
            MockSession.return_value = mock_session_instance

            with (
                patch.object(tester, "_start_monitoring", new=AsyncMock()),
                patch.object(tester, "_stop_monitoring", new=AsyncMock()),
            ):
                # Run test
                result = await tester.run_endurance_test(duration_hours=0.1)

                # Verify complete result structure
                required_keys = [
                    "test_id",
                    "test_type",
                    "start_time",
                    "end_time",
                    "duration_hours",
                    "session_summary",
                    "performance_summary",
                    "resource_summary",
                    "cognitive_summary",
                    "failure_analysis",
                    "raw_metrics",
                ]

                for key in required_keys:
                    assert key in result, f"Missing key in result: {key}"

                # Verify report file was created
                report_files = list(tester.output_dir.glob("*_report.json"))
                assert len(report_files) >= 1

                # Verify report file contents
                with open(report_files[0]) as f:
                    saved_report = json.load(f)

                assert saved_report["test_type"] == result["test_type"]
                assert saved_report["session_summary"] == result["session_summary"]
