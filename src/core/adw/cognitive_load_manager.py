"""Cognitive Load Manager for Extended Autonomous Development Sessions.

This module manages agent cognitive load and performance during extended
development sessions (16+ hours) to prevent degradation and maintain quality.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class SessionMode(Enum):
    """Operating modes based on session duration and cognitive load."""

    NORMAL = "normal"  # 0-4 hours
    MAINTENANCE = "maintenance"  # 4-8 hours
    CONSERVATIVE = "conservative"  # 8-16 hours
    ULTRA_CONSERVATIVE = "ultra_conservative"  # 16+ hours

    # Activity-specific modes
    FOCUS = "focus"  # Deep work periods with minimal distractions
    REST = "rest"  # Break periods for cognitive recovery
    EXPLORATION = "exploration"  # Learning and discovery phases


@dataclass
class CognitiveLoadMetrics:
    """Metrics for tracking cognitive load and performance."""

    session_duration_hours: float
    task_completion_rate: float
    error_frequency: float
    decision_latency_ms: float
    context_retention_score: float
    complexity_handling_score: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class SessionFatigueIndicators:
    """Indicators of session fatigue and cognitive overload."""

    repetitive_error_patterns: list[str] = field(default_factory=list)
    increased_rollback_frequency: bool = False
    declining_test_coverage: bool = False
    longer_task_completion_times: bool = False
    degraded_code_quality: bool = False
    context_confusion_events: int = 0


class ContextCompressionEngine:
    """Compresses and manages context for memory efficiency."""

    def __init__(self):
        self.compression_history: list[dict[str, Any]] = []
        self.critical_context_keys = [
            "current_task_objectives",
            "recent_decisions_rationale",
            "active_code_patterns",
            "test_strategies",
            "known_issues",
        ]

    async def compress_session_context(
        self, full_context: dict[str, Any], target_compression_ratio: float = 0.3
    ) -> dict[str, Any]:
        """Compress session context while preserving critical information."""
        try:
            start_size = len(str(full_context))
            compressed_context = {}

            # Always preserve critical context
            for key in self.critical_context_keys:
                if key in full_context:
                    compressed_context[key] = full_context[key]

            # Compress other context based on importance scoring
            remaining_context = {
                k: v
                for k, v in full_context.items()
                if k not in self.critical_context_keys
            }

            # Simple importance scoring based on recency and frequency
            importance_scores = await self._calculate_importance_scores(
                remaining_context
            )

            # Sort by importance and keep top items to meet compression ratio
            sorted_items = sorted(
                importance_scores.items(), key=lambda x: x[1], reverse=True
            )

            items_to_keep = int(len(sorted_items) * target_compression_ratio)
            for key, score in sorted_items[:items_to_keep]:
                compressed_context[key] = remaining_context[key]

            end_size = len(str(compressed_context))
            compression_ratio = 1 - (end_size / start_size) if start_size > 0 else 0

            compression_result = {
                "original_size": start_size,
                "compressed_size": end_size,
                "compression_ratio": compression_ratio,
                "critical_keys_preserved": len(self.critical_context_keys),
                "timestamp": time.time(),
            }

            self.compression_history.append(compression_result)

            logger.info("Context compression completed", **compression_result)
            return compressed_context

        except Exception as e:
            logger.error("Context compression failed", error=str(e))
            return full_context  # Return original on failure

    async def _calculate_importance_scores(
        self, context: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate importance scores for context items."""
        scores = {}
        current_time = time.time()

        for key, value in context.items():
            score = 0.0

            # Recency scoring
            if isinstance(value, dict) and "timestamp" in value:
                age_hours = (current_time - value["timestamp"]) / 3600
                recency_score = max(0, 1 - (age_hours / 24))  # Decay over 24 hours
                score += recency_score * 0.4

            # Size scoring (smaller items are more important for compression)
            size_score = 1 / (len(str(value)) + 1)
            score += size_score * 0.2

            # Keyword importance scoring
            important_keywords = [
                "error",
                "failure",
                "success",
                "test",
                "bug",
                "performance",
                "optimization",
                "critical",
            ]

            value_str = str(value).lower()
            keyword_score = sum(
                1 for keyword in important_keywords if keyword in value_str
            )
            score += min(keyword_score / len(important_keywords), 1.0) * 0.4

            scores[key] = score

        return scores


class TaskComplexityAdapter:
    """Adapts task complexity based on cognitive load and session duration."""

    def __init__(self):
        self.complexity_history: list[dict[str, Any]] = []
        self.base_complexity_scores = {
            "refactoring": 0.3,
            "bug_fixing": 0.4,
            "testing": 0.5,
            "documentation": 0.2,
            "new_feature": 0.8,
            "architecture": 0.9,
            "optimization": 0.7,
        }

    async def adapt_task_complexity(
        self,
        available_tasks: list[dict[str, Any]],
        current_mode: SessionMode,
        cognitive_metrics: CognitiveLoadMetrics,
    ) -> list[dict[str, Any]]:
        """Adapt task selection based on current cognitive state."""
        try:
            # Define complexity limits per mode
            complexity_limits = {
                SessionMode.NORMAL: 1.0,
                SessionMode.MAINTENANCE: 0.7,
                SessionMode.CONSERVATIVE: 0.5,
                SessionMode.ULTRA_CONSERVATIVE: 0.3,
                SessionMode.FOCUS: 1.2,  # Enhanced performance during focus
                SessionMode.REST: 0.1,  # Minimal activity during rest
                SessionMode.EXPLORATION: 0.8,  # Moderate complexity for learning
            }

            max_complexity = complexity_limits[current_mode]

            # Adjust based on performance metrics
            performance_multiplier = self._calculate_performance_multiplier(
                cognitive_metrics
            )
            effective_max_complexity = max_complexity * performance_multiplier

            # Filter and rank tasks
            suitable_tasks = []
            for task in available_tasks:
                task_complexity = self._calculate_task_complexity(task)

                if task_complexity <= effective_max_complexity:
                    # Add complexity score to task
                    task_copy = task.copy()
                    task_copy["calculated_complexity"] = task_complexity
                    task_copy["suitability_score"] = (
                        effective_max_complexity - task_complexity
                    )
                    suitable_tasks.append(task_copy)

            # Sort by suitability (lower complexity first in conservative modes)
            suitable_tasks.sort(key=lambda t: t["suitability_score"], reverse=True)

            adaptation_result = {
                "mode": current_mode.value,
                "max_complexity": max_complexity,
                "performance_multiplier": performance_multiplier,
                "effective_max_complexity": effective_max_complexity,
                "original_task_count": len(available_tasks),
                "suitable_task_count": len(suitable_tasks),
                "timestamp": time.time(),
            }

            self.complexity_history.append(adaptation_result)
            logger.info("Task complexity adaptation completed", **adaptation_result)

            return suitable_tasks

        except Exception as e:
            logger.error("Task complexity adaptation failed", error=str(e))
            return available_tasks  # Return original on failure

    def _calculate_performance_multiplier(self, metrics: CognitiveLoadMetrics) -> float:
        """Calculate performance-based complexity multiplier."""
        # Higher completion rate and retention = higher multiplier
        completion_factor = metrics.task_completion_rate
        retention_factor = metrics.context_retention_score

        # Lower error frequency = higher multiplier
        error_factor = max(0, 1 - metrics.error_frequency)

        # Combine factors
        multiplier = (completion_factor + retention_factor + error_factor) / 3

        # Clamp between 0.1 and 1.0
        return max(0.1, min(1.0, multiplier))

    def _calculate_task_complexity(self, task: dict[str, Any]) -> float:
        """Calculate complexity score for a task."""
        task_type = task.get("type", "unknown").lower()
        base_score = self.base_complexity_scores.get(task_type, 0.5)

        # Adjust based on task properties
        size_factor = 1.0
        if "estimated_hours" in task:
            # Scale complexity with time estimate
            size_factor = min(2.0, task["estimated_hours"] / 2.0)

        dependency_factor = 1.0
        if "dependencies" in task and task["dependencies"]:
            # More dependencies = higher complexity
            dependency_factor = 1.0 + (len(task["dependencies"]) * 0.1)

        # Risk factor based on task properties
        risk_factor = 1.0
        risk_keywords = ["database", "migration", "security", "api", "breaking"]
        task_description = task.get("description", "").lower()
        for keyword in risk_keywords:
            if keyword in task_description:
                risk_factor += 0.1

        complexity = base_score * size_factor * dependency_factor * risk_factor
        return min(1.0, complexity)  # Cap at 1.0


class CognitiveLoadManager:
    """Main cognitive load manager for extended autonomous sessions."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.session_start_time = time.time()
        self.current_mode = SessionMode.NORMAL

        self.context_compressor = ContextCompressionEngine()
        self.task_adapter = TaskComplexityAdapter()

        self.cognitive_history: list[CognitiveLoadMetrics] = []
        self.fatigue_indicators = SessionFatigueIndicators()
        self.mode_transitions: list[dict[str, Any]] = []

        # Performance baselines
        self.baseline_metrics: CognitiveLoadMetrics | None = None

    async def assess_cognitive_load(
        self, recent_performance_data: dict[str, Any]
    ) -> CognitiveLoadMetrics:
        """Assess current cognitive load based on performance indicators."""
        try:
            session_duration = (time.time() - self.session_start_time) / 3600

            # Extract metrics from performance data
            task_completion_rate = recent_performance_data.get(
                "task_completion_rate", 0.8
            )
            error_frequency = recent_performance_data.get("error_frequency", 0.1)
            decision_latency = recent_performance_data.get("avg_decision_time_ms", 1000)
            context_retention = recent_performance_data.get(
                "context_retention_score", 0.9
            )

            # Calculate complexity handling score based on recent task success
            complexity_handling = await self._calculate_complexity_handling_score(
                recent_performance_data
            )

            metrics = CognitiveLoadMetrics(
                session_duration_hours=session_duration,
                task_completion_rate=task_completion_rate,
                error_frequency=error_frequency,
                decision_latency_ms=decision_latency,
                context_retention_score=context_retention,
                complexity_handling_score=complexity_handling,
            )

            self.cognitive_history.append(metrics)

            # Set baseline if this is the first measurement
            if self.baseline_metrics is None:
                self.baseline_metrics = metrics
                logger.info(
                    "Cognitive load baseline established",
                    session_duration=session_duration,
                )

            # Detect fatigue indicators
            await self._update_fatigue_indicators(metrics, recent_performance_data)

            # Update session mode based on metrics
            await self._update_session_mode(metrics)

            logger.info(
                "Cognitive load assessment completed",
                duration_hours=session_duration,
                mode=self.current_mode.value,
                completion_rate=task_completion_rate,
                error_frequency=error_frequency,
            )

            return metrics

        except Exception as e:
            logger.error("Cognitive load assessment failed", error=str(e))
            # Return safe defaults
            return CognitiveLoadMetrics(
                session_duration_hours=(time.time() - self.session_start_time) / 3600,
                task_completion_rate=0.5,
                error_frequency=0.2,
                decision_latency_ms=2000,
                context_retention_score=0.7,
                complexity_handling_score=0.6,
            )

    async def optimize_for_extended_session(
        self, session_context: dict[str, Any], available_tasks: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Optimize operations for extended session duration."""
        try:
            optimization_actions = []

            # Compress context if in conservative modes
            if self.current_mode in [
                SessionMode.CONSERVATIVE,
                SessionMode.ULTRA_CONSERVATIVE,
            ]:
                compression_ratio = (
                    0.3 if self.current_mode == SessionMode.CONSERVATIVE else 0.2
                )
                compressed_context = (
                    await self.context_compressor.compress_session_context(
                        session_context, compression_ratio
                    )
                )
                optimization_actions.append("context_compression")
            else:
                compressed_context = session_context

            # Adapt task complexity
            suitable_tasks = await self.task_adapter.adapt_task_complexity(
                available_tasks, self.current_mode, self.cognitive_history[-1]
            )
            optimization_actions.append("task_complexity_adaptation")

            # Additional optimizations based on mode
            additional_constraints = self._get_mode_constraints()
            optimization_actions.extend(additional_constraints.keys())

            optimization_result = {
                "optimized_context": compressed_context,
                "suitable_tasks": suitable_tasks,
                "additional_constraints": additional_constraints,
                "optimization_actions": optimization_actions,
                "current_mode": self.current_mode.value,
                "session_duration_hours": (time.time() - self.session_start_time)
                / 3600,
                "timestamp": time.time(),
            }

            logger.info(
                "Extended session optimization completed",
                mode=self.current_mode.value,
                actions=optimization_actions,
                task_count=len(suitable_tasks),
            )

            return optimization_result

        except Exception as e:
            logger.error("Extended session optimization failed", error=str(e))
            return {
                "optimized_context": session_context,
                "suitable_tasks": available_tasks,
                "additional_constraints": {},
                "optimization_actions": [],
                "error": str(e),
            }

    async def _calculate_complexity_handling_score(
        self, performance_data: dict[str, Any]
    ) -> float:
        """Calculate how well the agent is handling task complexity."""
        # Base score on recent task outcomes
        recent_tasks = performance_data.get("recent_tasks", [])
        if not recent_tasks:
            return 0.8  # Default reasonable score

        success_rate = sum(
            1 for task in recent_tasks if task.get("success", False)
        ) / len(recent_tasks)

        # Adjust based on task complexity
        complexity_weighted_score = 0.0
        total_weight = 0.0

        for task in recent_tasks:
            complexity = task.get("complexity", 0.5)
            success = 1.0 if task.get("success", False) else 0.0
            weight = complexity + 0.1  # Give more weight to complex tasks

            complexity_weighted_score += success * weight
            total_weight += weight

        if total_weight > 0:
            return complexity_weighted_score / total_weight
        else:
            return success_rate

    async def _update_fatigue_indicators(
        self, current_metrics: CognitiveLoadMetrics, performance_data: dict[str, Any]
    ) -> None:
        """Update fatigue indicators based on current performance."""
        if self.baseline_metrics is None:
            return

        # Check for declining performance indicators
        completion_decline = (
            self.baseline_metrics.task_completion_rate
            - current_metrics.task_completion_rate
        ) > 0.2

        error_increase = (
            current_metrics.error_frequency - self.baseline_metrics.error_frequency
        ) > 0.1

        context_decline = (
            self.baseline_metrics.context_retention_score
            - current_metrics.context_retention_score
        ) > 0.2

        # Update indicators
        if completion_decline:
            self.fatigue_indicators.longer_task_completion_times = True

        if error_increase:
            self.fatigue_indicators.increased_rollback_frequency = True

        if context_decline:
            self.fatigue_indicators.context_confusion_events += 1

        # Check for repetitive errors
        recent_errors = performance_data.get("recent_errors", [])
        error_types = [error.get("type", "") for error in recent_errors]
        if (
            len(set(error_types)) < len(error_types) * 0.7
        ):  # 70% threshold for repetition
            self.fatigue_indicators.repetitive_error_patterns = list(set(error_types))

    async def _update_session_mode(self, metrics: CognitiveLoadMetrics) -> None:
        """Update session mode based on duration and performance."""
        old_mode = self.current_mode
        duration_hours = metrics.session_duration_hours

        # Determine mode based on duration
        if duration_hours >= 16:
            new_mode = SessionMode.ULTRA_CONSERVATIVE
        elif duration_hours >= 8:
            new_mode = SessionMode.CONSERVATIVE
        elif duration_hours >= 4:
            new_mode = SessionMode.MAINTENANCE
        else:
            new_mode = SessionMode.NORMAL

        # Override to more conservative mode if performance is declining
        if self.baseline_metrics:
            performance_ratio = (
                metrics.task_completion_rate
                / self.baseline_metrics.task_completion_rate
            )

            if performance_ratio < 0.7:  # 30% decline
                if new_mode == SessionMode.NORMAL:
                    new_mode = SessionMode.MAINTENANCE
                elif new_mode == SessionMode.MAINTENANCE:
                    new_mode = SessionMode.CONSERVATIVE
                elif new_mode == SessionMode.CONSERVATIVE:
                    new_mode = SessionMode.ULTRA_CONSERVATIVE

        if new_mode != old_mode:
            transition = {
                "timestamp": time.time(),
                "from_mode": old_mode.value,
                "to_mode": new_mode.value,
                "duration_hours": duration_hours,
                "trigger": "duration" if duration_hours >= 16 else "performance",
                "metrics": metrics,
            }

            self.mode_transitions.append(transition)
            self.current_mode = new_mode

            logger.info(
                "Session mode transition",
                from_mode=old_mode.value,
                to_mode=new_mode.value,
                duration_hours=duration_hours,
            )

    def _get_mode_constraints(self) -> dict[str, Any]:
        """Get operational constraints for current mode."""
        constraints = {
            SessionMode.NORMAL: {
                "max_concurrent_tasks": 3,
                "test_coverage_threshold": 0.8,
                "rollback_on_single_failure": False,
            },
            SessionMode.MAINTENANCE: {
                "max_concurrent_tasks": 2,
                "test_coverage_threshold": 0.85,
                "rollback_on_single_failure": False,
                "prefer_safe_operations": True,
            },
            SessionMode.CONSERVATIVE: {
                "max_concurrent_tasks": 1,
                "test_coverage_threshold": 0.9,
                "rollback_on_single_failure": True,
                "prefer_safe_operations": True,
                "require_checkpoint_before_changes": True,
            },
            SessionMode.ULTRA_CONSERVATIVE: {
                "max_concurrent_tasks": 1,
                "test_coverage_threshold": 0.95,
                "rollback_on_single_failure": True,
                "prefer_safe_operations": True,
                "require_checkpoint_before_changes": True,
                "read_only_analysis_mode": True,
            },
            SessionMode.FOCUS: {
                "max_concurrent_tasks": 1,
                "test_coverage_threshold": 0.85,
                "rollback_on_single_failure": False,
                "deep_work_mode": True,
                "minimize_interruptions": True,
            },
            SessionMode.REST: {
                "max_concurrent_tasks": 0,
                "test_coverage_threshold": 0.9,
                "rollback_on_single_failure": True,
                "read_only_mode": True,
                "passive_monitoring_only": True,
            },
            SessionMode.EXPLORATION: {
                "max_concurrent_tasks": 2,
                "test_coverage_threshold": 0.7,
                "rollback_on_single_failure": False,
                "experimental_mode": True,
                "allow_exploratory_changes": True,
            },
        }

        return constraints.get(self.current_mode, constraints[SessionMode.NORMAL])

    def get_cognitive_load_statistics(self) -> dict[str, Any]:
        """Get cognitive load and performance statistics."""
        if not self.cognitive_history:
            return {"session_duration_hours": 0, "mode": "normal"}

        latest_metrics = self.cognitive_history[-1]

        # Calculate trends
        if len(self.cognitive_history) >= 2:
            previous_metrics = self.cognitive_history[-2]
            completion_trend = (
                latest_metrics.task_completion_rate
                - previous_metrics.task_completion_rate
            )
            error_trend = (
                latest_metrics.error_frequency - previous_metrics.error_frequency
            )
        else:
            completion_trend = 0.0
            error_trend = 0.0

        return {
            "session_duration_hours": latest_metrics.session_duration_hours,
            "current_mode": self.current_mode.value,
            "task_completion_rate": latest_metrics.task_completion_rate,
            "error_frequency": latest_metrics.error_frequency,
            "context_retention_score": latest_metrics.context_retention_score,
            "completion_rate_trend": completion_trend,
            "error_frequency_trend": error_trend,
            "mode_transitions": len(self.mode_transitions),
            "fatigue_indicators": {
                "repetitive_errors": len(
                    self.fatigue_indicators.repetitive_error_patterns
                ),
                "increased_rollbacks": self.fatigue_indicators.increased_rollback_frequency,
                "context_confusion_events": self.fatigue_indicators.context_confusion_events,
            },
            "optimization_history": {
                "context_compressions": len(
                    self.context_compressor.compression_history
                ),
                "complexity_adaptations": len(self.task_adapter.complexity_history),
            },
        }
