"""Tests for Failure Prediction System."""

import json
import time
from pathlib import Path

import pytest

from src.core.prediction.failure_prediction import (
    FailureCategory,
    FailurePattern,
    FailurePredictionSystem,
    FailureRiskLevel,
    HistoricalFailure,
    PatternAnalyzer,
    RiskAssessment,
    RiskMonitor,
)


@pytest.fixture
def temp_project_path(tmp_path):
    """Create a temporary project path."""
    return Path(tmp_path)


@pytest.fixture
def pattern_analyzer():
    """Create a PatternAnalyzer instance."""
    return PatternAnalyzer()


@pytest.fixture
def risk_monitor():
    """Create a RiskMonitor instance."""
    return RiskMonitor()


@pytest.fixture
def prediction_system(temp_project_path):
    """Create a FailurePredictionSystem instance."""
    return FailurePredictionSystem(temp_project_path)


@pytest.fixture
def sample_historical_failures():
    """Sample historical failure data for testing."""
    return [
        HistoricalFailure(
            timestamp=time.time() - 86400,  # 1 day ago
            category=FailureCategory.RESOURCE_EXHAUSTION,
            severity=3,
            precursor_metrics={
                "memory_percent": 85.0,
                "cpu_percent": 90.0,
                "error_frequency": 0.2,
            },
            resolution_time_minutes=15.0,
            mitigation_used="memory_optimization",
            context={"session_duration": 8.0},
        ),
        HistoricalFailure(
            timestamp=time.time() - 172800,  # 2 days ago
            category=FailureCategory.RESOURCE_EXHAUSTION,
            severity=4,
            precursor_metrics={
                "memory_percent": 88.0,
                "cpu_percent": 95.0,
                "error_frequency": 0.25,
            },
            resolution_time_minutes=20.0,
            mitigation_used="cpu_throttling",
            context={"session_duration": 12.0},
        ),
        HistoricalFailure(
            timestamp=time.time() - 259200,  # 3 days ago
            category=FailureCategory.QUALITY_REGRESSION,
            severity=2,
            precursor_metrics={
                "test_coverage": 0.6,
                "error_frequency": 0.3,
                "task_completion_rate": 0.4,
            },
            resolution_time_minutes=30.0,
            mitigation_used="rollback_to_checkpoint",
            context={"session_duration": 6.0},
        ),
    ]


@pytest.fixture
def sample_historical_metrics():
    """Sample historical metrics data for testing."""
    base_time = time.time() - 86400
    return [
        {
            "timestamp": base_time - 3600,  # 1 hour before failure
            "memory_percent": 75.0,
            "cpu_percent": 80.0,
            "error_frequency": 0.15,
        },
        {
            "timestamp": base_time - 1800,  # 30 min before failure
            "memory_percent": 82.0,
            "cpu_percent": 88.0,
            "error_frequency": 0.18,
        },
        {
            "timestamp": base_time - 900,  # 15 min before failure
            "memory_percent": 85.0,
            "cpu_percent": 90.0,
            "error_frequency": 0.2,
        },
    ]


@pytest.fixture
def sample_current_metrics():
    """Sample current metrics for risk assessment."""
    return {
        "memory_percent": 70.0,
        "cpu_percent": 60.0,
        "error_frequency": 0.1,
        "task_completion_rate": 0.8,
        "test_coverage": 0.85,
        "context_retention_score": 0.9,
    }


@pytest.fixture
def sample_session_context():
    """Sample session context for testing."""
    return {
        "session_duration_hours": 6.0,
        "cognitive_load_score": 0.4,
        "current_mode": "normal",
    }


class TestFailurePattern:
    """Test FailurePattern class."""

    def test_failure_pattern_creation(self):
        """Test creating a failure pattern."""
        pattern = FailurePattern(
            category=FailureCategory.RESOURCE_EXHAUSTION,
            precursor_indicators=["high_memory_usage", "high_cpu_usage"],
            failure_probability=0.7,
            time_to_failure_hours=2.0,
            confidence_score=0.8,
            historical_occurrences=5,
            mitigation_strategies=[
                "trigger_memory_optimization",
                "activate_cpu_throttling",
            ],
        )

        assert pattern.category == FailureCategory.RESOURCE_EXHAUSTION
        assert len(pattern.precursor_indicators) == 2
        assert pattern.failure_probability == 0.7
        assert pattern.time_to_failure_hours == 2.0
        assert pattern.confidence_score == 0.8
        assert pattern.historical_occurrences == 5
        assert len(pattern.mitigation_strategies) == 2


class TestHistoricalFailure:
    """Test HistoricalFailure class."""

    def test_historical_failure_creation(self):
        """Test creating a historical failure record."""
        failure = HistoricalFailure(
            timestamp=time.time(),
            category=FailureCategory.PERFORMANCE_DEGRADATION,
            severity=3,
            precursor_metrics={"cpu_percent": 90.0},
            resolution_time_minutes=25.0,
            mitigation_used="rollback_to_stable",
            context={"session_duration": 8.0},
        )

        assert failure.category == FailureCategory.PERFORMANCE_DEGRADATION
        assert failure.severity == 3
        assert failure.precursor_metrics["cpu_percent"] == 90.0
        assert failure.resolution_time_minutes == 25.0
        assert failure.mitigation_used == "rollback_to_stable"


class TestPatternAnalyzer:
    """Test PatternAnalyzer class."""

    @pytest.mark.asyncio
    async def test_failure_pattern_analysis(
        self, pattern_analyzer, sample_historical_failures, sample_historical_metrics
    ):
        """Test failure pattern analysis."""
        patterns = await pattern_analyzer.analyze_failure_patterns(
            sample_historical_failures, sample_historical_metrics
        )

        assert isinstance(patterns, list)
        # Should find patterns for categories with multiple failures
        resource_patterns = [
            p for p in patterns if p.category == FailureCategory.RESOURCE_EXHAUSTION
        ]
        assert len(resource_patterns) >= 1

        if patterns:
            pattern = patterns[0]
            assert isinstance(pattern, FailurePattern)
            assert pattern.failure_probability > 0
            assert pattern.confidence_score > 0
            assert pattern.historical_occurrences > 0

    @pytest.mark.asyncio
    async def test_precursor_indicator_identification(
        self, pattern_analyzer, sample_historical_failures, sample_historical_metrics
    ):
        """Test identification of precursor indicators."""
        # Focus on resource exhaustion failures
        resource_failures = [
            f
            for f in sample_historical_failures
            if f.category == FailureCategory.RESOURCE_EXHAUSTION
        ]

        indicators = await pattern_analyzer._find_precursor_indicators(
            resource_failures, sample_historical_metrics
        )

        assert isinstance(indicators, list)
        # Should identify high memory/CPU usage as indicators
        expected_indicators = {"high_memory_usage", "high_cpu_usage", "high_error_rate"}
        found_indicators = set(indicators)
        assert len(found_indicators.intersection(expected_indicators)) > 0

    def test_metric_anomaly_detection(self, pattern_analyzer):
        """Test anomaly detection in metrics."""
        high_usage_metrics = [
            {"memory_percent": 85, "cpu_percent": 92, "error_frequency": 0.25}
        ]

        anomalies = pattern_analyzer._detect_metric_anomalies(high_usage_metrics)

        assert "high_memory_usage" in anomalies
        assert "high_cpu_usage" in anomalies
        assert "high_error_rate" in anomalies

    def test_failure_probability_calculation(
        self, pattern_analyzer, sample_historical_failures, sample_historical_metrics
    ):
        """Test failure probability calculation."""
        probability = pattern_analyzer._calculate_failure_probability(
            sample_historical_failures, sample_historical_metrics
        )

        assert 0.1 <= probability <= 0.9  # Should be within bounds

    def test_pattern_confidence_calculation(
        self, pattern_analyzer, sample_historical_failures
    ):
        """Test pattern confidence calculation."""
        confidence = pattern_analyzer._calculate_pattern_confidence(
            sample_historical_failures
        )

        assert 0.1 <= confidence <= 0.95
        # Should be higher with more failures
        single_failure = [sample_historical_failures[0]]
        single_confidence = pattern_analyzer._calculate_pattern_confidence(
            single_failure
        )
        assert confidence >= single_confidence

    def test_mitigation_strategy_generation(self, pattern_analyzer):
        """Test mitigation strategy generation."""
        strategies = pattern_analyzer._generate_mitigation_strategies(
            FailureCategory.RESOURCE_EXHAUSTION
        )

        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert "trigger_memory_optimization" in strategies
        assert "activate_cpu_throttling" in strategies


class TestRiskMonitor:
    """Test RiskMonitor class."""

    @pytest.mark.asyncio
    async def test_risk_assessment(
        self, risk_monitor, sample_current_metrics, sample_session_context
    ):
        """Test current risk assessment."""
        # Create a simple learned pattern
        learned_patterns = [
            FailurePattern(
                category=FailureCategory.RESOURCE_EXHAUSTION,
                precursor_indicators=["high_memory_usage"],
                failure_probability=0.6,
                time_to_failure_hours=2.0,
                confidence_score=0.8,
                historical_occurrences=3,
                mitigation_strategies=["trigger_memory_optimization"],
            )
        ]

        assessment = await risk_monitor.assess_current_risk(
            sample_current_metrics, learned_patterns, sample_session_context
        )

        assert isinstance(assessment, RiskAssessment)
        assert assessment.overall_risk_level in [
            FailureRiskLevel.LOW,
            FailureRiskLevel.MEDIUM,
            FailureRiskLevel.HIGH,
            FailureRiskLevel.CRITICAL,
        ]
        assert 0.0 <= assessment.confidence_score <= 1.0

    @pytest.mark.asyncio
    async def test_pattern_risk_evaluation(
        self, risk_monitor, sample_current_metrics, sample_session_context
    ):
        """Test risk evaluation for specific patterns."""
        pattern = FailurePattern(
            category=FailureCategory.RESOURCE_EXHAUSTION,
            precursor_indicators=["high_memory_usage", "high_cpu_usage"],
            failure_probability=0.8,
            time_to_failure_hours=4.0,
            confidence_score=0.9,
            historical_occurrences=5,
        )

        # Test with low-risk metrics
        low_risk_metrics = {
            "memory_percent": 50.0,
            "cpu_percent": 40.0,
            "error_frequency": 0.05,
        }

        risk_score = await risk_monitor._evaluate_pattern_risk(
            pattern, low_risk_metrics, sample_session_context
        )

        assert 0.0 <= risk_score <= 1.0
        assert risk_score < 0.5  # Should be low risk

        # Test with high-risk metrics
        high_risk_metrics = {
            "memory_percent": 85.0,
            "cpu_percent": 95.0,
            "error_frequency": 0.25,
        }

        high_risk_score = await risk_monitor._evaluate_pattern_risk(
            pattern, high_risk_metrics, sample_session_context
        )

        assert high_risk_score > risk_score  # Should be higher risk

    def test_indicator_presence_checking(
        self, risk_monitor, sample_current_metrics, sample_session_context
    ):
        """Test checking for indicator presence."""
        # Test various indicators
        test_cases = [
            ("high_memory_usage", {"memory_percent": 85}, True),
            ("high_memory_usage", {"memory_percent": 50}, False),
            ("high_cpu_usage", {"cpu_percent": 95}, True),
            ("high_error_rate", {"error_frequency": 0.25}, True),
            ("low_completion_rate", {"task_completion_rate": 0.3}, True),
            ("extended_session", {}, True),  # sample_session_context has 6 hours
        ]

        for indicator, metrics, expected in test_cases:
            result = risk_monitor._check_indicator_present(
                indicator, metrics, sample_session_context
            )
            assert result == expected, f"Failed for indicator {indicator}"

    def test_time_factor_calculation(self, risk_monitor):
        """Test time-based risk factor calculation."""
        # Test different session durations vs pattern time to failure
        pattern_time = 8.0  # 8 hours to failure

        test_cases = [
            (2.0, 0.3),  # Early in session - low factor
            (6.0, 0.5),  # Approaching window - medium factor
            (8.0, 1.0),  # At failure time - high factor
            (10.0, 1.0),  # Past failure time - high factor
        ]

        for duration, expected_min in test_cases:
            factor = risk_monitor._calculate_time_factor(duration, pattern_time)
            assert factor >= expected_min
            assert 0.0 <= factor <= 1.0


class TestFailurePredictionSystem:
    """Test FailurePredictionSystem class."""

    @pytest.mark.asyncio
    async def test_system_initialization(
        self, prediction_system, sample_historical_metrics
    ):
        """Test prediction system initialization."""
        await prediction_system.initialize_prediction_system(sample_historical_metrics)

        # Should have learned patterns after initialization
        assert len(prediction_system.pattern_analyzer.learned_patterns) >= 0

    @pytest.mark.asyncio
    async def test_failure_prediction_and_prevention(
        self, prediction_system, sample_current_metrics, sample_session_context
    ):
        """Test failure prediction and prevention action generation."""
        # Initialize with some patterns
        await prediction_system.initialize_prediction_system([])

        risk_assessment, actions = await prediction_system.predict_and_prevent_failures(
            sample_current_metrics, sample_session_context
        )

        assert isinstance(risk_assessment, RiskAssessment)
        assert isinstance(actions, list)

        # Actions should be strings
        for action in actions:
            assert isinstance(action, str)

    @pytest.mark.asyncio
    async def test_failure_event_recording(
        self, prediction_system, sample_current_metrics
    ):
        """Test recording of failure events."""
        initial_count = len(prediction_system.failure_history)

        await prediction_system.record_failure_event(
            category=FailureCategory.RESOURCE_EXHAUSTION,
            severity=3,
            context={
                "precursor_metrics": sample_current_metrics,
                "historical_metrics": [],
            },
            resolution_time_minutes=15.0,
            mitigation_used="memory_optimization",
        )

        assert len(prediction_system.failure_history) == initial_count + 1

        recorded_failure = prediction_system.failure_history[-1]
        assert recorded_failure.category == FailureCategory.RESOURCE_EXHAUSTION
        assert recorded_failure.severity == 3
        assert recorded_failure.resolution_time_minutes == 15.0

    @pytest.mark.asyncio
    async def test_immediate_action_determination(
        self, prediction_system, sample_current_metrics, sample_session_context
    ):
        """Test immediate action determination based on risk level."""
        # Create high-risk assessment
        high_risk_assessment = RiskAssessment(
            overall_risk_level=FailureRiskLevel.CRITICAL,
            recommended_actions=["create_safety_checkpoint"],
        )

        actions = await prediction_system._determine_immediate_actions(
            high_risk_assessment, sample_current_metrics, sample_session_context
        )

        assert "create_emergency_checkpoint" in actions
        assert "activate_safe_mode" in actions

    def test_action_prioritization(self, prediction_system):
        """Test action prioritization."""
        test_actions = [
            "monitor_resource_usage_closely",
            "create_emergency_checkpoint",
            "switch_to_conservative_mode",
            "activate_safe_mode",
        ]

        risk_assessment = RiskAssessment(overall_risk_level=FailureRiskLevel.HIGH)

        prioritized = prediction_system._prioritize_actions(
            test_actions, risk_assessment
        )

        # Emergency checkpoint should be first
        assert prioritized[0] == "create_emergency_checkpoint"
        assert prioritized[1] == "activate_safe_mode"

    def test_failure_history_persistence(self, prediction_system, tmp_path):
        """Test failure history persistence."""
        # Set custom history file
        history_file = tmp_path / "test_history.json"
        prediction_system.history_file = history_file

        # Add a failure and save
        failure = HistoricalFailure(
            timestamp=time.time(),
            category=FailureCategory.RESOURCE_EXHAUSTION,
            severity=2,
            precursor_metrics={"memory_percent": 80},
            resolution_time_minutes=10.0,
            mitigation_used="optimization",
        )
        prediction_system.failure_history.append(failure)
        prediction_system._save_failure_history()

        # Verify file was created
        assert history_file.exists()

        # Load and verify data
        with open(history_file) as f:
            data = json.load(f)

        assert "failure_history" in data
        assert len(data["failure_history"]) == 1

    def test_prediction_statistics(self, prediction_system, sample_historical_failures):
        """Test prediction statistics generation."""
        # Add some failure history
        prediction_system.failure_history.extend(sample_historical_failures)

        # Add some learned patterns
        pattern = FailurePattern(
            category=FailureCategory.RESOURCE_EXHAUSTION,
            precursor_indicators=["high_memory_usage"],
            failure_probability=0.6,
            time_to_failure_hours=2.0,
            confidence_score=0.8,
            historical_occurrences=3,
        )
        prediction_system.pattern_analyzer.learned_patterns.append(pattern)

        stats = prediction_system.get_prediction_statistics()

        assert stats["total_failures_in_history"] == len(sample_historical_failures)
        assert stats["learned_patterns"] == 1
        assert "pattern_categories" in stats
        assert "failure_categories_seen" in stats
        assert stats["average_resolution_time_minutes"] > 0


@pytest.mark.asyncio
async def test_integration_prediction_workflow(
    prediction_system,
    sample_historical_failures,
    sample_historical_metrics,
    sample_current_metrics,
    sample_session_context,
):
    """Test complete prediction workflow integration."""
    # 1. Add historical data
    prediction_system.failure_history.extend(sample_historical_failures)

    # 2. Initialize system
    await prediction_system.initialize_prediction_system(sample_historical_metrics)

    # 3. Make predictions
    risk_assessment, actions = await prediction_system.predict_and_prevent_failures(
        sample_current_metrics, sample_session_context
    )

    assert isinstance(risk_assessment, RiskAssessment)
    assert isinstance(actions, list)

    # 4. Record new failure (simulating a failure that occurred)
    await prediction_system.record_failure_event(
        category=FailureCategory.COGNITIVE_OVERLOAD,
        severity=2,
        context={"historical_metrics": sample_historical_metrics},
        resolution_time_minutes=20.0,
        mitigation_used="session_break",
    )

    # 5. Get updated statistics
    stats = prediction_system.get_prediction_statistics()
    assert stats["total_failures_in_history"] > len(sample_historical_failures)


@pytest.mark.asyncio
async def test_error_handling_robustness(prediction_system):
    """Test error handling in prediction system."""
    # Test with invalid data
    invalid_metrics = {"invalid": "data"}
    invalid_context = {}

    # Should not crash
    risk_assessment, actions = await prediction_system.predict_and_prevent_failures(
        invalid_metrics, invalid_context
    )

    assert isinstance(risk_assessment, RiskAssessment)
    assert isinstance(actions, list)


def test_risk_level_transitions():
    """Test risk level enumeration and transitions."""
    levels = [
        FailureRiskLevel.LOW,
        FailureRiskLevel.MEDIUM,
        FailureRiskLevel.HIGH,
        FailureRiskLevel.CRITICAL,
    ]

    # Should have all expected levels
    assert len(levels) == 4

    # Should be able to compare levels (enum ordering)
    assert FailureRiskLevel.LOW.value == "low"
    assert FailureRiskLevel.CRITICAL.value == "critical"


def test_failure_categories_completeness():
    """Test that all expected failure categories are defined."""
    expected_categories = [
        "resource_exhaustion",
        "performance_degradation",
        "quality_regression",
        "cognitive_overload",
        "system_instability",
    ]

    actual_categories = [cat.value for cat in FailureCategory]

    for expected in expected_categories:
        assert expected in actual_categories
