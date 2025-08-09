"""Tests for Autonomous Monitoring Dashboard."""

import asyncio
import json
import time
from pathlib import Path

import pytest

from src.core.monitoring.autonomous_dashboard import (
    AutonomousDashboard,
    AutonomousMetrics,
    AutonomyScore,
    AutonomyScoreCalculator,
    MetricsCollector,
    VelocityMetrics,
)


@pytest.fixture
def temp_project_path(tmp_path):
    """Create a temporary project path."""
    project_path = Path(tmp_path)

    # Create basic project structure
    (project_path / "src").mkdir()
    (project_path / "tests").mkdir()
    (project_path / "tests" / "unit").mkdir()

    # Create some test files
    (project_path / "tests" / "test_example.py").write_text("""
def test_example():
    assert True

def test_another():
    assert 1 + 1 == 2
""")

    (project_path / "src" / "example.py").write_text("""
def hello_world():
    return "Hello, World!"

class ExampleClass:
    def method(self):
        return 42
""")

    return project_path


@pytest.fixture
def metrics_collector(temp_project_path):
    """Create a MetricsCollector instance."""
    return MetricsCollector(temp_project_path)


@pytest.fixture
def autonomy_calculator():
    """Create an AutonomyScoreCalculator instance."""
    return AutonomyScoreCalculator()


@pytest.fixture
def autonomous_dashboard(temp_project_path):
    """Create an AutonomousDashboard instance."""
    return AutonomousDashboard(temp_project_path)


@pytest.fixture
def sample_velocity_metrics():
    """Sample velocity metrics for testing."""
    return VelocityMetrics(
        commits_per_hour=1.5,
        tests_per_hour=3.0,
        lines_of_code_per_hour=150.0,
        bugs_introduced_per_hour=0.1,
        quality_score=0.8,
        technical_debt_trend=0.05,
    )


@pytest.fixture
def sample_autonomous_metrics(sample_velocity_metrics):
    """Sample autonomous metrics for testing."""
    return AutonomousMetrics(
        session_duration_hours=4.5,
        velocity=sample_velocity_metrics,
        autonomy_score=AutonomyScore(overall_score=75.0),
        task_completion_rate=0.8,
        error_recovery_rate=0.9,
        test_coverage_percentage=85.0,
        performance_regression_count=1,
        memory_efficiency_score=0.7,
        cpu_efficiency_score=0.8,
        disk_usage_trend=0.9,
        rollback_events_count=2,
        quality_gate_failures=1,
        time_since_last_rollback_hours=6.0,
    )


class TestVelocityMetrics:
    """Test VelocityMetrics class."""

    def test_velocity_metrics_creation(self):
        """Test creating velocity metrics."""
        metrics = VelocityMetrics(
            commits_per_hour=2.0,
            tests_per_hour=5.0,
            lines_of_code_per_hour=200.0,
            bugs_introduced_per_hour=0.2,
            quality_score=0.9,
            technical_debt_trend=-0.1,
        )

        assert metrics.commits_per_hour == 2.0
        assert metrics.tests_per_hour == 5.0
        assert metrics.lines_of_code_per_hour == 200.0
        assert metrics.bugs_introduced_per_hour == 0.2
        assert metrics.quality_score == 0.9
        assert metrics.technical_debt_trend == -0.1


class TestAutonomyScore:
    """Test AutonomyScore class."""

    def test_autonomy_score_creation(self):
        """Test creating autonomy score."""
        score = AutonomyScore(
            overall_score=80.0,
            reliability_component=0.9,
            quality_component=0.8,
            velocity_component=0.7,
            efficiency_component=0.85,
            learning_component=0.6,
        )

        assert score.overall_score == 80.0
        assert score.reliability_component == 0.9
        assert score.quality_component == 0.8
        assert score.velocity_component == 0.7
        assert score.efficiency_component == 0.85
        assert score.learning_component == 0.6


class TestMetricsCollector:
    """Test MetricsCollector class."""

    @pytest.mark.asyncio
    async def test_git_velocity_metrics_collection(self, metrics_collector):
        """Test git velocity metrics collection."""
        # This test requires git repo - mock the subprocess calls
        import unittest.mock

        with unittest.mock.patch("asyncio.create_subprocess_exec") as mock_subprocess:
            # Mock git rev-list output
            mock_process = unittest.mock.AsyncMock()
            mock_process.communicate.return_value = (b"3\n", b"")
            mock_subprocess.return_value = mock_process

            metrics = await metrics_collector._collect_git_velocity_metrics(1.0)

            assert "commits_per_hour" in metrics
            assert "lines_per_hour" in metrics
            assert metrics["commits_per_hour"] >= 0

    @pytest.mark.asyncio
    async def test_test_velocity_metrics_collection(self, metrics_collector):
        """Test test velocity metrics collection."""
        metrics = await metrics_collector._collect_test_velocity_metrics(1.0)

        assert "tests_per_hour" in metrics
        assert metrics["tests_per_hour"] >= 0

    @pytest.mark.asyncio
    async def test_quality_metrics_collection(self, metrics_collector):
        """Test quality metrics collection."""
        # Mock pytest subprocess
        import unittest.mock

        with unittest.mock.patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = unittest.mock.AsyncMock()
            mock_process.communicate.return_value = (
                b"TOTAL    100    50    50%\n",
                b"",
            )
            mock_subprocess.return_value = mock_process

            metrics = await metrics_collector._collect_quality_metrics()

            assert "composite_score" in metrics
            assert "debt_trend" in metrics
            assert 0.0 <= metrics["composite_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, metrics_collector):
        """Test performance metrics collection."""
        metrics = await metrics_collector.collect_performance_metrics()

        assert isinstance(metrics, dict)
        # Should contain various efficiency metrics
        assert (
            "memory_efficiency" in metrics or len(metrics) == 0
        )  # Might be empty if psutil unavailable

    @pytest.mark.asyncio
    async def test_velocity_metrics_collection(self, metrics_collector):
        """Test full velocity metrics collection."""
        velocity = await metrics_collector.collect_velocity_metrics(1.0)

        assert isinstance(velocity, VelocityMetrics)
        assert velocity.commits_per_hour >= 0
        assert velocity.tests_per_hour >= 0
        assert velocity.lines_of_code_per_hour >= 0
        assert velocity.bugs_introduced_per_hour >= 0
        assert 0.0 <= velocity.quality_score <= 1.0


class TestAutonomyScoreCalculator:
    """Test AutonomyScoreCalculator class."""

    def test_reliability_score_calculation(self, autonomy_calculator):
        """Test reliability score calculation."""
        # Test various time periods
        test_cases = [
            (0.5, 0.1),  # 30 minutes - low score
            (4.0, 0.5),  # 4 hours - medium score
            (12.0, 0.8),  # 12 hours - good score
            (24.0, 1.0),  # 24+ hours - perfect score
            (48.0, 1.0),  # 48 hours - still perfect
        ]

        for hours, expected_min in test_cases:
            score = autonomy_calculator._calculate_reliability_score(hours)
            assert score >= expected_min
            assert 0.0 <= score <= 1.0

    def test_quality_score_calculation(self, autonomy_calculator):
        """Test quality score calculation."""
        # High quality scenario
        score = autonomy_calculator._calculate_quality_score(
            test_coverage=0.9, quality_metric=0.8, quality_gate_failures=0
        )
        assert score >= 0.8

        # Low quality scenario with failures
        score = autonomy_calculator._calculate_quality_score(
            test_coverage=0.6, quality_metric=0.5, quality_gate_failures=3
        )
        assert score <= 0.6

    def test_velocity_score_calculation(
        self, autonomy_calculator, sample_velocity_metrics
    ):
        """Test velocity score calculation."""
        score = autonomy_calculator._calculate_velocity_score(
            sample_velocity_metrics, 0.8
        )

        assert 0.0 <= score <= 1.0

    def test_efficiency_score_calculation(self, autonomy_calculator):
        """Test efficiency score calculation."""
        score = autonomy_calculator._calculate_efficiency_score(0.8, 0.7)
        assert score == 0.75  # Average of 0.8 and 0.7

    def test_learning_score_calculation(
        self, autonomy_calculator, sample_autonomous_metrics
    ):
        """Test learning score calculation."""
        # Create historical data showing improvement
        historical_data = []
        for i in range(10):
            metrics = AutonomousMetrics(
                task_completion_rate=0.6 + (i * 0.02)  # Gradual improvement
            )
            historical_data.append(metrics)

        current_metrics = AutonomousMetrics(task_completion_rate=0.8)

        score = autonomy_calculator._calculate_learning_score(
            current_metrics, historical_data
        )

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should show improvement

    def test_overall_autonomy_score_calculation(
        self, autonomy_calculator, sample_autonomous_metrics
    ):
        """Test overall autonomy score calculation."""
        historical_data = [sample_autonomous_metrics]

        score = autonomy_calculator.calculate_autonomy_score(
            sample_autonomous_metrics, historical_data
        )

        assert isinstance(score, AutonomyScore)
        assert 0.0 <= score.overall_score <= 100.0
        assert 0.0 <= score.reliability_component <= 1.0
        assert 0.0 <= score.quality_component <= 1.0
        assert 0.0 <= score.velocity_component <= 1.0
        assert 0.0 <= score.efficiency_component <= 1.0
        assert 0.0 <= score.learning_component <= 1.0


class TestAutonomousDashboard:
    """Test AutonomousDashboard class."""

    @pytest.mark.asyncio
    async def test_metrics_collection_and_recording(self, autonomous_dashboard):
        """Test metrics collection and recording."""
        metrics = await autonomous_dashboard.collect_and_record_metrics()

        assert isinstance(metrics, AutonomousMetrics)
        assert metrics.session_duration_hours >= 0
        assert len(autonomous_dashboard.metrics_history) == 1

    def test_dashboard_data_generation(
        self, autonomous_dashboard, sample_autonomous_metrics
    ):
        """Test dashboard data generation."""
        # Add sample metrics to history
        autonomous_dashboard.metrics_history.append(sample_autonomous_metrics)

        dashboard_data = autonomous_dashboard.get_current_dashboard_data()

        assert dashboard_data["status"] == "active"
        assert "session_duration_hours" in dashboard_data
        assert "current_autonomy_score" in dashboard_data
        assert "current_velocity" in dashboard_data
        assert "resource_efficiency" in dashboard_data
        assert "safety_metrics" in dashboard_data
        assert "autonomy_components" in dashboard_data

    def test_dashboard_data_no_metrics(self, autonomous_dashboard):
        """Test dashboard data when no metrics available."""
        dashboard_data = autonomous_dashboard.get_current_dashboard_data()

        assert dashboard_data["status"] == "no_data"

    def test_metrics_persistence(
        self, autonomous_dashboard, sample_autonomous_metrics, tmp_path
    ):
        """Test metrics persistence to file."""
        # Set custom metrics file
        metrics_file = tmp_path / "test_metrics.json"
        autonomous_dashboard.metrics_file = metrics_file

        # Add metrics and save
        autonomous_dashboard.metrics_history.append(sample_autonomous_metrics)
        autonomous_dashboard._save_metrics_history()

        # Verify file was created
        assert metrics_file.exists()

        # Load and verify data
        with open(metrics_file) as f:
            data = json.load(f)

        assert "metrics_history" in data
        assert len(data["metrics_history"]) == 1

    def test_metrics_loading(
        self, autonomous_dashboard, sample_autonomous_metrics, tmp_path
    ):
        """Test loading metrics from file."""
        # Create metrics file
        metrics_file = tmp_path / "test_metrics.json"
        autonomous_dashboard.metrics_file = metrics_file

        # Save some metrics
        autonomous_dashboard.metrics_history.append(sample_autonomous_metrics)
        autonomous_dashboard._save_metrics_history()

        # Create new dashboard and load
        new_dashboard = AutonomousDashboard(tmp_path, metrics_file)

        assert len(new_dashboard.metrics_history) == 1
        loaded_metrics = new_dashboard.metrics_history[0]
        assert (
            loaded_metrics.session_duration_hours
            == sample_autonomous_metrics.session_duration_hours
        )

    @pytest.mark.asyncio
    async def test_monitoring_start_stop(self, autonomous_dashboard):
        """Test monitoring start and stop."""
        assert not autonomous_dashboard.monitoring_active

        # Start monitoring in background
        monitoring_task = asyncio.create_task(
            autonomous_dashboard.start_monitoring(
                interval_minutes=0.01
            )  # Very short interval
        )

        # Let it run briefly
        await asyncio.sleep(0.1)
        assert autonomous_dashboard.monitoring_active

        # Stop monitoring
        await autonomous_dashboard.stop_monitoring()
        assert not autonomous_dashboard.monitoring_active

        # Cancel the task
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_integration_dashboard_workflow(autonomous_dashboard):
    """Test complete dashboard workflow."""
    # 1. Collect initial metrics
    metrics = await autonomous_dashboard.collect_and_record_metrics()
    assert isinstance(metrics, AutonomousMetrics)

    # 2. Get dashboard data
    dashboard_data = autonomous_dashboard.get_current_dashboard_data()
    assert dashboard_data["status"] == "active"

    # 3. Collect more metrics to show trends
    await asyncio.sleep(0.1)  # Small delay to ensure different timestamp
    await autonomous_dashboard.collect_and_record_metrics()

    # 4. Get updated dashboard data with trends
    updated_data = autonomous_dashboard.get_current_dashboard_data()
    assert "trends" in updated_data

    # 5. Test statistics
    assert len(autonomous_dashboard.metrics_history) == 2


@pytest.mark.asyncio
async def test_error_handling_robustness(autonomous_dashboard):
    """Test error handling in dashboard components."""
    # Test with minimal environment (might not have git, pytest, etc.)
    try:
        metrics = await autonomous_dashboard.collect_and_record_metrics()
        # Should not crash, even if some collection methods fail
        assert isinstance(metrics, AutonomousMetrics)
    except Exception as e:
        pytest.fail(f"Dashboard should handle missing tools gracefully: {e}")


def test_autonomy_score_components_balance(autonomy_calculator):
    """Test that autonomy score components are properly balanced."""
    # Test with all components at different levels
    test_metrics = AutonomousMetrics(
        session_duration_hours=8.0,
        velocity=VelocityMetrics(
            commits_per_hour=1.0, quality_score=0.8, bugs_introduced_per_hour=0.1
        ),
        task_completion_rate=0.7,
        test_coverage_percentage=85.0,
        memory_efficiency_score=0.8,
        cpu_efficiency_score=0.7,
        time_since_last_rollback_hours=12.0,
        quality_gate_failures=1,
    )

    score = autonomy_calculator.calculate_autonomy_score(test_metrics, [test_metrics])

    # All components should contribute to overall score
    assert score.reliability_component > 0
    assert score.quality_component > 0
    assert score.velocity_component > 0
    assert score.efficiency_component > 0

    # Overall score should be reasonable combination
    assert 30.0 <= score.overall_score <= 90.0


def test_metrics_history_management(autonomous_dashboard):
    """Test metrics history size management."""
    # Add many metrics to test size limiting
    for i in range(1005):
        metrics = AutonomousMetrics(timestamp=time.time() + i)
        autonomous_dashboard.metrics_history.append(metrics)

    # Trigger collection to test size limiting
    asyncio.run(autonomous_dashboard.collect_and_record_metrics())

    # Should be limited to 1000
    assert len(autonomous_dashboard.metrics_history) <= 1000
