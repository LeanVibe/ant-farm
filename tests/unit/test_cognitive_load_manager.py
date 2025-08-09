"""Tests for Cognitive Load Manager."""

import asyncio
import time
from pathlib import Path

import pytest

from src.core.adw.cognitive_load_manager import (
    CognitiveLoadManager,
    CognitiveLoadMetrics,
    ContextCompressionEngine,
    SessionMode,
    TaskComplexityAdapter,
)


@pytest.fixture
def temp_project_path(tmp_path):
    """Create a temporary project path."""
    return Path(tmp_path)


@pytest.fixture
def cognitive_load_manager(temp_project_path):
    """Create a CognitiveLoadManager instance."""
    return CognitiveLoadManager(temp_project_path)


@pytest.fixture
def context_compressor():
    """Create a ContextCompressionEngine instance."""
    return ContextCompressionEngine()


@pytest.fixture
def task_adapter():
    """Create a TaskComplexityAdapter instance."""
    return TaskComplexityAdapter()


@pytest.fixture
def sample_performance_data():
    """Sample performance data for testing."""
    return {
        "task_completion_rate": 0.8,
        "error_frequency": 0.1,
        "avg_decision_time_ms": 1500,
        "context_retention_score": 0.9,
        "recent_tasks": [
            {"success": True, "complexity": 0.5},
            {"success": True, "complexity": 0.7},
            {"success": False, "complexity": 0.8},
        ],
        "recent_errors": [{"type": "syntax_error"}, {"type": "test_failure"}],
    }


@pytest.fixture
def sample_context():
    """Sample session context for testing."""
    return {
        "current_task_objectives": ["implement feature X", "fix bug Y"],
        "recent_decisions_rationale": {"decision1": "rationale1"},
        "active_code_patterns": {"pattern1": "details"},
        "test_strategies": ["unit tests", "integration tests"],
        "known_issues": ["issue1", "issue2"],
        "other_data": {"key": "value"},
        "timestamp_data": {"timestamp": time.time()},
    }


@pytest.fixture
def sample_tasks():
    """Sample tasks for complexity adaptation testing."""
    return [
        {
            "type": "refactoring",
            "description": "refactor module X",
            "estimated_hours": 2.0,
            "dependencies": [],
        },
        {
            "type": "new_feature",
            "description": "implement new API endpoint",
            "estimated_hours": 4.0,
            "dependencies": ["feature1", "feature2"],
        },
        {
            "type": "bug_fixing",
            "description": "fix critical bug",
            "estimated_hours": 1.0,
            "dependencies": [],
        },
        {
            "type": "documentation",
            "description": "update API docs",
            "estimated_hours": 0.5,
            "dependencies": [],
        },
    ]


class TestCognitiveLoadMetrics:
    """Test CognitiveLoadMetrics class."""

    def test_metrics_creation(self):
        """Test creating cognitive load metrics."""
        metrics = CognitiveLoadMetrics(
            session_duration_hours=4.5,
            task_completion_rate=0.8,
            error_frequency=0.1,
            decision_latency_ms=1200,
            context_retention_score=0.9,
            complexity_handling_score=0.7,
        )

        assert metrics.session_duration_hours == 4.5
        assert metrics.task_completion_rate == 0.8
        assert metrics.error_frequency == 0.1
        assert metrics.decision_latency_ms == 1200
        assert metrics.context_retention_score == 0.9
        assert metrics.complexity_handling_score == 0.7
        assert isinstance(metrics.timestamp, float)


class TestContextCompressionEngine:
    """Test ContextCompressionEngine class."""

    @pytest.mark.asyncio
    async def test_context_compression(self, context_compressor, sample_context):
        """Test context compression functionality."""
        compressed = await context_compressor.compress_session_context(
            sample_context, target_compression_ratio=0.3
        )

        # Critical keys should be preserved
        for key in context_compressor.critical_context_keys:
            if key in sample_context:
                assert key in compressed

        # Result should be smaller than original
        assert len(str(compressed)) <= len(str(sample_context))

    @pytest.mark.asyncio
    async def test_importance_scoring(self, context_compressor, sample_context):
        """Test importance scoring algorithm."""
        scores = await context_compressor._calculate_importance_scores(sample_context)

        assert isinstance(scores, dict)
        for key, score in scores.items():
            assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_compression_history(self, context_compressor, sample_context):
        """Test compression history tracking."""
        initial_history_len = len(context_compressor.compression_history)

        await context_compressor.compress_session_context(sample_context)

        assert len(context_compressor.compression_history) == initial_history_len + 1

        latest_compression = context_compressor.compression_history[-1]
        assert "original_size" in latest_compression
        assert "compressed_size" in latest_compression
        assert "compression_ratio" in latest_compression


class TestTaskComplexityAdapter:
    """Test TaskComplexityAdapter class."""

    @pytest.mark.asyncio
    async def test_task_complexity_calculation(self, task_adapter, sample_tasks):
        """Test task complexity calculation."""
        for task in sample_tasks:
            complexity = task_adapter._calculate_task_complexity(task)
            assert 0.0 <= complexity <= 1.0

    @pytest.mark.asyncio
    async def test_task_adaptation_normal_mode(self, task_adapter, sample_tasks):
        """Test task adaptation in normal mode."""
        metrics = CognitiveLoadMetrics(
            session_duration_hours=2.0,
            task_completion_rate=0.8,
            error_frequency=0.1,
            decision_latency_ms=1000,
            context_retention_score=0.9,
            complexity_handling_score=0.8,
        )

        adapted_tasks = await task_adapter.adapt_task_complexity(
            sample_tasks, SessionMode.NORMAL, metrics
        )

        # Should return tasks with complexity scores
        for task in adapted_tasks:
            assert "calculated_complexity" in task
            assert "suitability_score" in task

    @pytest.mark.asyncio
    async def test_task_adaptation_conservative_mode(self, task_adapter, sample_tasks):
        """Test task adaptation in conservative mode."""
        metrics = CognitiveLoadMetrics(
            session_duration_hours=10.0,
            task_completion_rate=0.6,
            error_frequency=0.3,
            decision_latency_ms=2000,
            context_retention_score=0.6,
            complexity_handling_score=0.5,
        )

        adapted_tasks = await task_adapter.adapt_task_complexity(
            sample_tasks, SessionMode.CONSERVATIVE, metrics
        )

        # Should filter out high complexity tasks
        for task in adapted_tasks:
            assert task["calculated_complexity"] <= 0.5

    def test_performance_multiplier_calculation(self, task_adapter):
        """Test performance multiplier calculation."""
        high_performance_metrics = CognitiveLoadMetrics(
            session_duration_hours=2.0,
            task_completion_rate=0.9,
            error_frequency=0.05,
            decision_latency_ms=800,
            context_retention_score=0.95,
            complexity_handling_score=0.9,
        )

        multiplier = task_adapter._calculate_performance_multiplier(
            high_performance_metrics
        )
        assert 0.7 <= multiplier <= 1.0

        low_performance_metrics = CognitiveLoadMetrics(
            session_duration_hours=2.0,
            task_completion_rate=0.3,
            error_frequency=0.4,
            decision_latency_ms=3000,
            context_retention_score=0.4,
            complexity_handling_score=0.3,
        )

        multiplier = task_adapter._calculate_performance_multiplier(
            low_performance_metrics
        )
        assert 0.1 <= multiplier <= 0.5


class TestCognitiveLoadManager:
    """Test CognitiveLoadManager class."""

    @pytest.mark.asyncio
    async def test_cognitive_load_assessment(
        self, cognitive_load_manager, sample_performance_data
    ):
        """Test cognitive load assessment."""
        metrics = await cognitive_load_manager.assess_cognitive_load(
            sample_performance_data
        )

        assert isinstance(metrics, CognitiveLoadMetrics)
        assert metrics.session_duration_hours >= 0
        assert 0.0 <= metrics.task_completion_rate <= 1.0
        assert metrics.error_frequency >= 0
        assert metrics.decision_latency_ms > 0
        assert 0.0 <= metrics.context_retention_score <= 1.0
        assert 0.0 <= metrics.complexity_handling_score <= 1.0

    @pytest.mark.asyncio
    async def test_baseline_establishment(
        self, cognitive_load_manager, sample_performance_data
    ):
        """Test baseline metrics establishment."""
        assert cognitive_load_manager.baseline_metrics is None

        metrics = await cognitive_load_manager.assess_cognitive_load(
            sample_performance_data
        )

        assert cognitive_load_manager.baseline_metrics is not None
        assert cognitive_load_manager.baseline_metrics == metrics

    @pytest.mark.asyncio
    async def test_session_mode_transitions(
        self, cognitive_load_manager, sample_performance_data
    ):
        """Test session mode transitions based on duration."""
        # Simulate different session durations
        test_cases = [
            (2.0, SessionMode.NORMAL),
            (5.0, SessionMode.MAINTENANCE),
            (10.0, SessionMode.CONSERVATIVE),
            (18.0, SessionMode.ULTRA_CONSERVATIVE),
        ]

        for duration, expected_mode in test_cases:
            # Mock session start time
            cognitive_load_manager.session_start_time = time.time() - (duration * 3600)

            await cognitive_load_manager.assess_cognitive_load(sample_performance_data)

            assert cognitive_load_manager.current_mode == expected_mode

    @pytest.mark.asyncio
    async def test_performance_degradation_mode_override(
        self, cognitive_load_manager, sample_performance_data
    ):
        """Test mode override due to performance degradation."""
        # Establish baseline
        await cognitive_load_manager.assess_cognitive_load(sample_performance_data)

        # Simulate performance degradation
        degraded_performance = sample_performance_data.copy()
        degraded_performance["task_completion_rate"] = 0.3  # 30% of baseline (0.8)

        await cognitive_load_manager.assess_cognitive_load(degraded_performance)

        # Should move to more conservative mode
        assert cognitive_load_manager.current_mode in [
            SessionMode.MAINTENANCE,
            SessionMode.CONSERVATIVE,
            SessionMode.ULTRA_CONSERVATIVE,
        ]

    @pytest.mark.asyncio
    async def test_extended_session_optimization(
        self, cognitive_load_manager, sample_context, sample_tasks
    ):
        """Test extended session optimization."""
        # Set conservative mode
        cognitive_load_manager.current_mode = SessionMode.CONSERVATIVE

        optimization_result = (
            await cognitive_load_manager.optimize_for_extended_session(
                sample_context, sample_tasks
            )
        )

        assert "optimized_context" in optimization_result
        assert "suitable_tasks" in optimization_result
        assert "additional_constraints" in optimization_result
        assert "optimization_actions" in optimization_result

        # Should have context compression in conservative mode
        assert "context_compression" in optimization_result["optimization_actions"]

    def test_mode_constraints(self, cognitive_load_manager):
        """Test mode-specific constraints."""
        constraints = cognitive_load_manager._get_mode_constraints()

        assert "max_concurrent_tasks" in constraints
        assert "test_coverage_threshold" in constraints
        assert "rollback_on_single_failure" in constraints

    @pytest.mark.asyncio
    async def test_fatigue_indicators_update(
        self, cognitive_load_manager, sample_performance_data
    ):
        """Test fatigue indicators update."""
        # Establish baseline
        await cognitive_load_manager.assess_cognitive_load(sample_performance_data)

        # Simulate degraded performance
        degraded_performance = sample_performance_data.copy()
        degraded_performance["task_completion_rate"] = 0.4
        degraded_performance["error_frequency"] = 0.3
        degraded_performance["context_retention_score"] = 0.5

        await cognitive_load_manager.assess_cognitive_load(degraded_performance)

        # Check fatigue indicators
        fatigue = cognitive_load_manager.fatigue_indicators
        assert (
            fatigue.longer_task_completion_times or fatigue.increased_rollback_frequency
        )

    @pytest.mark.asyncio
    async def test_complexity_handling_score_calculation(self, cognitive_load_manager):
        """Test complexity handling score calculation."""
        performance_data = {
            "recent_tasks": [
                {"success": True, "complexity": 0.8},
                {"success": True, "complexity": 0.6},
                {"success": False, "complexity": 0.9},
                {"success": True, "complexity": 0.4},
            ]
        }

        score = await cognitive_load_manager._calculate_complexity_handling_score(
            performance_data
        )

        assert 0.0 <= score <= 1.0
        # Should be weighted toward more complex tasks
        assert score > 0.5  # Given mostly successful complex tasks

    def test_statistics_generation(self, cognitive_load_manager):
        """Test cognitive load statistics generation."""
        # Add some mock history
        mock_metrics = CognitiveLoadMetrics(
            session_duration_hours=4.0,
            task_completion_rate=0.8,
            error_frequency=0.1,
            decision_latency_ms=1200,
            context_retention_score=0.9,
            complexity_handling_score=0.7,
        )
        cognitive_load_manager.cognitive_history.append(mock_metrics)

        stats = cognitive_load_manager.get_cognitive_load_statistics()

        assert "session_duration_hours" in stats
        assert "current_mode" in stats
        assert "task_completion_rate" in stats
        assert "fatigue_indicators" in stats
        assert "optimization_history" in stats


@pytest.mark.asyncio
async def test_integration_cognitive_load_workflow(
    cognitive_load_manager, sample_performance_data, sample_context, sample_tasks
):
    """Test complete cognitive load management workflow."""
    # 1. Initial assessment
    metrics = await cognitive_load_manager.assess_cognitive_load(
        sample_performance_data
    )
    assert metrics.session_duration_hours >= 0

    # 2. Extended session optimization
    optimization = await cognitive_load_manager.optimize_for_extended_session(
        sample_context, sample_tasks
    )
    assert "optimized_context" in optimization
    assert "suitable_tasks" in optimization

    # 3. Statistics retrieval
    stats = cognitive_load_manager.get_cognitive_load_statistics()
    assert stats["session_duration_hours"] >= 0
    assert "current_mode" in stats


@pytest.mark.asyncio
async def test_error_handling_robustness(cognitive_load_manager):
    """Test error handling in cognitive load manager."""
    # Test with invalid performance data
    invalid_data = {"invalid": "data"}

    metrics = await cognitive_load_manager.assess_cognitive_load(invalid_data)

    # Should return safe defaults
    assert isinstance(metrics, CognitiveLoadMetrics)
    assert metrics.task_completion_rate >= 0
    assert metrics.error_frequency >= 0

    # Test with invalid context/tasks
    optimization = await cognitive_load_manager.optimize_for_extended_session({}, [])

    # Should not crash
    assert "optimized_context" in optimization
