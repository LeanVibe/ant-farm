"""Predictive Failure Prevention for Autonomous Development Sessions.

This module analyzes historical patterns to predict and prevent failures
before they occur during extended autonomous development workflows.
"""

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


class FailureRiskLevel(Enum):
    """Risk levels for predicted failures."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FailureCategory(Enum):
    """Categories of potential failures."""

    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    QUALITY_REGRESSION = "quality_regression"
    COGNITIVE_OVERLOAD = "cognitive_overload"
    SYSTEM_INSTABILITY = "system_instability"


@dataclass
class FailurePattern:
    """Represents a learned failure pattern."""

    category: FailureCategory
    precursor_indicators: list[str]
    failure_probability: float
    time_to_failure_hours: float
    confidence_score: float
    historical_occurrences: int
    mitigation_strategies: list[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Current risk assessment for potential failures."""

    timestamp: float = field(default_factory=time.time)
    overall_risk_level: FailureRiskLevel = FailureRiskLevel.LOW
    category_risks: dict[FailureCategory, float] = field(default_factory=dict)
    active_patterns: list[FailurePattern] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    confidence_score: float = 0.0


@dataclass
class HistoricalFailure:
    """Record of a historical failure event."""

    timestamp: float
    category: FailureCategory
    severity: int  # 1-5 scale
    precursor_metrics: dict[str, float]
    resolution_time_minutes: float
    mitigation_used: str
    context: dict[str, Any] = field(default_factory=dict)


class PatternAnalyzer:
    """Analyzes historical data to identify failure patterns."""

    def __init__(self):
        self.learned_patterns: list[FailurePattern] = []
        self.pattern_accuracy_history: dict[str, list[float]] = {}

    async def analyze_failure_patterns(
        self,
        historical_failures: list[HistoricalFailure],
        historical_metrics: list[dict[str, Any]],
    ) -> list[FailurePattern]:
        """Analyze historical data to identify failure patterns."""
        try:
            patterns = []

            # Group failures by category
            failures_by_category = {}
            for failure in historical_failures:
                category = failure.category
                if category not in failures_by_category:
                    failures_by_category[category] = []
                failures_by_category[category].append(failure)

            # Analyze patterns for each category
            for category, failures in failures_by_category.items():
                if len(failures) >= 2:  # Need at least 2 failures to identify pattern
                    category_patterns = await self._analyze_category_patterns(
                        category, failures, historical_metrics
                    )
                    patterns.extend(category_patterns)

            # Update learned patterns
            self.learned_patterns = patterns

            logger.info(
                "Failure pattern analysis completed",
                patterns_found=len(patterns),
                categories_analyzed=len(failures_by_category),
            )

            return patterns

        except Exception as e:
            logger.error("Failure pattern analysis failed", error=str(e))
            return []

    async def _analyze_category_patterns(
        self,
        category: FailureCategory,
        failures: list[HistoricalFailure],
        historical_metrics: list[dict[str, Any]],
    ) -> list[FailurePattern]:
        """Analyze patterns for a specific failure category."""
        patterns = []

        try:
            # Identify common precursor indicators
            precursor_indicators = await self._find_precursor_indicators(
                failures, historical_metrics
            )

            if precursor_indicators:
                # Calculate pattern statistics
                failure_count = len(failures)
                avg_time_to_failure = (
                    sum(
                        self._calculate_time_to_failure(f, historical_metrics)
                        for f in failures
                    )
                    / failure_count
                )

                # Calculate probability based on historical occurrence rate
                failure_probability = self._calculate_failure_probability(
                    failures, historical_metrics
                )

                # Generate mitigation strategies
                mitigation_strategies = self._generate_mitigation_strategies(category)

                pattern = FailurePattern(
                    category=category,
                    precursor_indicators=precursor_indicators,
                    failure_probability=failure_probability,
                    time_to_failure_hours=avg_time_to_failure,
                    confidence_score=self._calculate_pattern_confidence(failures),
                    historical_occurrences=failure_count,
                    mitigation_strategies=mitigation_strategies,
                )

                patterns.append(pattern)

        except Exception as e:
            logger.error(
                "Category pattern analysis failed",
                category=category.value,
                error=str(e),
            )

        return patterns

    async def _find_precursor_indicators(
        self,
        failures: list[HistoricalFailure],
        historical_metrics: list[dict[str, Any]],
    ) -> list[str]:
        """Find common indicators that precede failures."""
        indicators = []

        # Analyze metrics before each failure
        for failure in failures:
            # Look at metrics 1-4 hours before failure
            pre_failure_metrics = [
                m
                for m in historical_metrics
                if abs(m.get("timestamp", 0) - failure.timestamp) <= 4 * 3600
                and m.get("timestamp", 0) < failure.timestamp
            ]

            if pre_failure_metrics:
                # Find anomalous metric values
                anomalies = self._detect_metric_anomalies(pre_failure_metrics)
                indicators.extend(anomalies)

        # Find common indicators across failures
        indicator_counts = {}
        for indicator in indicators:
            indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

        # Return indicators that appear in at least 50% of failures
        threshold = len(failures) * 0.5
        common_indicators = [
            indicator
            for indicator, count in indicator_counts.items()
            if count >= threshold
        ]

        return common_indicators

    def _detect_metric_anomalies(self, metrics: list[dict[str, Any]]) -> list[str]:
        """Detect anomalous metric values that might indicate upcoming failure."""
        anomalies = []

        # Simple anomaly detection based on thresholds
        for metric in metrics:
            # Memory issues
            if metric.get("memory_percent", 0) > 80:
                anomalies.append("high_memory_usage")

            # CPU issues
            if metric.get("cpu_percent", 0) > 90:
                anomalies.append("high_cpu_usage")

            # Error rate issues
            if metric.get("error_frequency", 0) > 0.2:
                anomalies.append("high_error_rate")

            # Performance issues
            if metric.get("task_completion_rate", 1.0) < 0.5:
                anomalies.append("low_completion_rate")

            # Test coverage decline
            if metric.get("test_coverage", 1.0) < 0.8:
                anomalies.append("declining_test_coverage")

            # Context retention issues
            if metric.get("context_retention_score", 1.0) < 0.7:
                anomalies.append("context_degradation")

        return list(set(anomalies))  # Remove duplicates

    def _calculate_time_to_failure(
        self, failure: HistoricalFailure, historical_metrics: list[dict[str, Any]]
    ) -> float:
        """Calculate average time from precursor indicators to failure."""
        # Find the earliest precursor indicator before this failure
        earliest_indicator_time = failure.timestamp

        # Look back up to 8 hours
        lookback_window = 8 * 3600
        start_time = failure.timestamp - lookback_window

        relevant_metrics = [
            m
            for m in historical_metrics
            if start_time <= m.get("timestamp", 0) < failure.timestamp
        ]

        for metric in relevant_metrics:
            if self._has_failure_indicators(metric):
                earliest_indicator_time = min(
                    earliest_indicator_time, metric["timestamp"]
                )

        time_to_failure = (failure.timestamp - earliest_indicator_time) / 3600
        return max(0.1, time_to_failure)  # Minimum 0.1 hours

    def _has_failure_indicators(self, metric: dict[str, Any]) -> bool:
        """Check if a metric contains failure indicators."""
        indicators = [
            metric.get("memory_percent", 0) > 75,
            metric.get("cpu_percent", 0) > 85,
            metric.get("error_frequency", 0) > 0.15,
            metric.get("task_completion_rate", 1.0) < 0.6,
        ]
        return any(indicators)

    def _calculate_failure_probability(
        self,
        failures: list[HistoricalFailure],
        historical_metrics: list[dict[str, Any]],
    ) -> float:
        """Calculate the probability of failure given precursor conditions."""
        if not historical_metrics:
            return 0.1

        # Count how many times precursor conditions occurred
        precursor_occurrences = 0
        actual_failures = len(failures)

        for metric in historical_metrics:
            if self._has_failure_indicators(metric):
                precursor_occurrences += 1

        if precursor_occurrences == 0:
            return 0.1

        # Probability = actual failures / precursor occurrences
        probability = actual_failures / precursor_occurrences
        return min(0.9, max(0.1, probability))  # Clamp between 0.1 and 0.9

    def _calculate_pattern_confidence(self, failures: list[HistoricalFailure]) -> float:
        """Calculate confidence score for the pattern."""
        # Higher confidence with more failures and recent occurrences
        failure_count_factor = min(
            1.0, len(failures) / 5.0
        )  # Max confidence at 5+ failures

        # Recency factor
        current_time = time.time()
        most_recent_failure = max(f.timestamp for f in failures)
        days_since_recent = (current_time - most_recent_failure) / (24 * 3600)
        recency_factor = max(0.3, 1.0 - (days_since_recent / 30))  # Decay over 30 days

        confidence = (failure_count_factor + recency_factor) / 2
        return min(0.95, max(0.1, confidence))

    def _generate_mitigation_strategies(self, category: FailureCategory) -> list[str]:
        """Generate mitigation strategies for a failure category."""
        strategies = {
            FailureCategory.RESOURCE_EXHAUSTION: [
                "trigger_memory_optimization",
                "activate_cpu_throttling",
                "cleanup_temporary_files",
                "reduce_concurrent_operations",
            ],
            FailureCategory.PERFORMANCE_DEGRADATION: [
                "rollback_to_stable_checkpoint",
                "optimize_test_suite",
                "reduce_task_complexity",
                "restart_development_session",
            ],
            FailureCategory.QUALITY_REGRESSION: [
                "enable_quality_gates",
                "increase_test_coverage_requirement",
                "activate_code_review_mode",
                "rollback_problematic_changes",
            ],
            FailureCategory.COGNITIVE_OVERLOAD: [
                "switch_to_conservative_mode",
                "compress_session_context",
                "reduce_task_complexity",
                "take_session_break",
            ],
            FailureCategory.SYSTEM_INSTABILITY: [
                "create_safety_checkpoint",
                "run_system_diagnostics",
                "restart_system_services",
                "activate_safe_mode",
            ],
        }

        return strategies.get(category, ["create_safety_checkpoint"])


class RiskMonitor:
    """Monitors current conditions and assesses failure risk."""

    def __init__(self):
        self.risk_thresholds = {
            FailureRiskLevel.LOW: 0.2,
            FailureRiskLevel.MEDIUM: 0.4,
            FailureRiskLevel.HIGH: 0.7,
            FailureRiskLevel.CRITICAL: 0.9,
        }

    async def assess_current_risk(
        self,
        current_metrics: dict[str, Any],
        learned_patterns: list[FailurePattern],
        session_context: dict[str, Any],
    ) -> RiskAssessment:
        """Assess current failure risk based on patterns and metrics."""
        try:
            assessment = RiskAssessment()
            category_risks = {}
            active_patterns = []
            recommendations = []

            # Evaluate each learned pattern
            for pattern in learned_patterns:
                risk_score = await self._evaluate_pattern_risk(
                    pattern, current_metrics, session_context
                )

                category_risks[pattern.category] = max(
                    category_risks.get(pattern.category, 0.0), risk_score
                )

                if risk_score > self.risk_thresholds[FailureRiskLevel.MEDIUM]:
                    active_patterns.append(pattern)
                    recommendations.extend(pattern.mitigation_strategies)

            # Calculate overall risk
            if category_risks:
                overall_risk_score = max(category_risks.values())
            else:
                overall_risk_score = 0.1

            # Determine risk level
            overall_risk_level = FailureRiskLevel.LOW
            for level in [
                FailureRiskLevel.CRITICAL,
                FailureRiskLevel.HIGH,
                FailureRiskLevel.MEDIUM,
                FailureRiskLevel.LOW,
            ]:
                if overall_risk_score >= self.risk_thresholds[level]:
                    overall_risk_level = level
                    break

            # Calculate confidence
            confidence = self._calculate_assessment_confidence(
                active_patterns, current_metrics
            )

            assessment.overall_risk_level = overall_risk_level
            assessment.category_risks = category_risks
            assessment.active_patterns = active_patterns
            assessment.recommended_actions = list(
                set(recommendations)
            )  # Remove duplicates
            assessment.confidence_score = confidence

            logger.info(
                "Risk assessment completed",
                overall_risk=overall_risk_level.value,
                active_patterns=len(active_patterns),
                confidence=confidence,
            )

            return assessment

        except Exception as e:
            logger.error("Risk assessment failed", error=str(e))
            return RiskAssessment()

    async def _evaluate_pattern_risk(
        self,
        pattern: FailurePattern,
        current_metrics: dict[str, Any],
        session_context: dict[str, Any],
    ) -> float:
        """Evaluate risk score for a specific pattern."""
        risk_score = 0.0

        try:
            # Check for precursor indicators
            indicators_present = 0
            total_indicators = len(pattern.precursor_indicators)

            for indicator in pattern.precursor_indicators:
                if self._check_indicator_present(
                    indicator, current_metrics, session_context
                ):
                    indicators_present += 1

            if total_indicators > 0:
                indicator_ratio = indicators_present / total_indicators

                # Base risk from indicator presence
                base_risk = indicator_ratio * pattern.failure_probability

                # Adjust for pattern confidence
                confidence_adjusted_risk = base_risk * pattern.confidence_score

                # Time factor - higher risk if we're in the typical time window
                session_duration = session_context.get("session_duration_hours", 0)
                time_factor = self._calculate_time_factor(
                    session_duration, pattern.time_to_failure_hours
                )

                risk_score = confidence_adjusted_risk * time_factor

            return min(1.0, max(0.0, risk_score))

        except Exception as e:
            logger.error(
                "Pattern risk evaluation failed",
                pattern_category=pattern.category.value,
                error=str(e),
            )
            return 0.0

    def _check_indicator_present(
        self,
        indicator: str,
        current_metrics: dict[str, Any],
        session_context: dict[str, Any],
    ) -> bool:
        """Check if a specific indicator is present in current conditions."""
        # Map indicators to current metrics
        indicator_checks = {
            "high_memory_usage": current_metrics.get("memory_percent", 0) > 80,
            "high_cpu_usage": current_metrics.get("cpu_percent", 0) > 90,
            "high_error_rate": current_metrics.get("error_frequency", 0) > 0.2,
            "low_completion_rate": current_metrics.get("task_completion_rate", 1.0)
            < 0.5,
            "declining_test_coverage": current_metrics.get("test_coverage", 1.0) < 0.8,
            "context_degradation": current_metrics.get("context_retention_score", 1.0)
            < 0.7,
            "extended_session": session_context.get("session_duration_hours", 0) > 8,
            "cognitive_fatigue": session_context.get("cognitive_load_score", 0) > 0.7,
        }

        return indicator_checks.get(indicator, False)

    def _calculate_time_factor(
        self, current_session_duration: float, pattern_time_to_failure: float
    ) -> float:
        """Calculate time-based risk factor."""
        if current_session_duration >= pattern_time_to_failure:
            # We're in or past the typical failure window
            return 1.0
        elif current_session_duration >= pattern_time_to_failure * 0.7:
            # Approaching the failure window
            progress = current_session_duration / pattern_time_to_failure
            return 0.5 + (progress - 0.7) / 0.3 * 0.5  # Scale from 0.5 to 1.0
        else:
            # Still early, lower risk
            return 0.3

    def _calculate_assessment_confidence(
        self, active_patterns: list[FailurePattern], current_metrics: dict[str, Any]
    ) -> float:
        """Calculate confidence in the risk assessment."""
        if not active_patterns:
            return 0.5  # Moderate confidence when no patterns active

        # Higher confidence with more patterns and better pattern confidence
        pattern_confidence = sum(p.confidence_score for p in active_patterns) / len(
            active_patterns
        )

        # Data quality factor
        metric_completeness = sum(
            1 for v in current_metrics.values() if v is not None
        ) / max(1, len(current_metrics))

        overall_confidence = (pattern_confidence + metric_completeness) / 2
        return min(0.95, max(0.1, overall_confidence))


class FailurePredictionSystem:
    """Main failure prediction and prevention system."""

    def __init__(self, project_path: Path, history_file: Path | None = None):
        self.project_path = project_path

        if history_file is None:
            history_file = project_path / ".adw" / "failure_history.json"
        self.history_file = history_file

        self.pattern_analyzer = PatternAnalyzer()
        self.risk_monitor = RiskMonitor()

        self.failure_history: list[HistoricalFailure] = []
        self.prediction_accuracy_history: list[dict[str, Any]] = []

        # Load historical data
        self._load_failure_history()

    async def initialize_prediction_system(
        self, historical_metrics: list[dict[str, Any]]
    ) -> None:
        """Initialize the prediction system with historical data."""
        try:
            # Analyze failure patterns from history
            await self.pattern_analyzer.analyze_failure_patterns(
                self.failure_history, historical_metrics
            )

            logger.info(
                "Prediction system initialized",
                historical_failures=len(self.failure_history),
                learned_patterns=len(self.pattern_analyzer.learned_patterns),
            )

        except Exception as e:
            logger.error("Failed to initialize prediction system", error=str(e))

    async def predict_and_prevent_failures(
        self, current_metrics: dict[str, Any], session_context: dict[str, Any]
    ) -> tuple[RiskAssessment, list[str]]:
        """Predict potential failures and return prevention actions."""
        try:
            # Assess current risk
            risk_assessment = await self.risk_monitor.assess_current_risk(
                current_metrics, self.pattern_analyzer.learned_patterns, session_context
            )

            # Determine immediate actions based on risk level
            immediate_actions = await self._determine_immediate_actions(
                risk_assessment, current_metrics, session_context
            )

            # Log prediction
            self._log_prediction_event(risk_assessment, immediate_actions)

            return risk_assessment, immediate_actions

        except Exception as e:
            logger.error("Failure prediction failed", error=str(e))
            return RiskAssessment(), []

    async def record_failure_event(
        self,
        category: FailureCategory,
        severity: int,
        context: dict[str, Any],
        resolution_time_minutes: float,
        mitigation_used: str,
    ) -> None:
        """Record a failure event for future pattern analysis."""
        try:
            failure = HistoricalFailure(
                timestamp=time.time(),
                category=category,
                severity=severity,
                precursor_metrics=context.get("precursor_metrics", {}),
                resolution_time_minutes=resolution_time_minutes,
                mitigation_used=mitigation_used,
                context=context,
            )

            self.failure_history.append(failure)

            # Save updated history
            self._save_failure_history()

            # Re-analyze patterns with new data
            historical_metrics = context.get("historical_metrics", [])
            await self.pattern_analyzer.analyze_failure_patterns(
                self.failure_history, historical_metrics
            )

            logger.info(
                "Failure event recorded",
                category=category.value,
                severity=severity,
                total_failures=len(self.failure_history),
            )

        except Exception as e:
            logger.error("Failed to record failure event", error=str(e))

    async def _determine_immediate_actions(
        self,
        risk_assessment: RiskAssessment,
        current_metrics: dict[str, Any],
        session_context: dict[str, Any],
    ) -> list[str]:
        """Determine immediate actions based on risk assessment."""
        actions = []

        try:
            risk_level = risk_assessment.overall_risk_level

            # Add actions based on risk level
            if risk_level == FailureRiskLevel.CRITICAL:
                actions.extend(
                    [
                        "create_emergency_checkpoint",
                        "activate_safe_mode",
                        "reduce_to_read_only_operations",
                    ]
                )
            elif risk_level == FailureRiskLevel.HIGH:
                actions.extend(
                    [
                        "create_safety_checkpoint",
                        "switch_to_conservative_mode",
                        "increase_monitoring_frequency",
                    ]
                )
            elif risk_level == FailureRiskLevel.MEDIUM:
                actions.extend(
                    ["increase_checkpoint_frequency", "monitor_resource_usage_closely"]
                )

            # Add pattern-specific recommendations
            actions.extend(risk_assessment.recommended_actions)

            # Add context-specific actions
            session_duration = session_context.get("session_duration_hours", 0)
            if session_duration > 16:
                actions.append("consider_session_break")
            elif session_duration > 8:
                actions.append("enable_ultra_conservative_mode")

            # Remove duplicates and prioritize
            unique_actions = list(set(actions))
            prioritized_actions = self._prioritize_actions(
                unique_actions, risk_assessment
            )

            return prioritized_actions

        except Exception as e:
            logger.error("Failed to determine immediate actions", error=str(e))
            return []

    def _prioritize_actions(
        self, actions: list[str], risk_assessment: RiskAssessment
    ) -> list[str]:
        """Prioritize actions based on urgency and effectiveness."""
        # Define action priorities (higher number = higher priority)
        action_priorities = {
            "create_emergency_checkpoint": 10,
            "activate_safe_mode": 9,
            "reduce_to_read_only_operations": 9,
            "create_safety_checkpoint": 8,
            "switch_to_conservative_mode": 7,
            "enable_ultra_conservative_mode": 6,
            "trigger_memory_optimization": 5,
            "activate_cpu_throttling": 5,
            "increase_monitoring_frequency": 4,
            "consider_session_break": 3,
            "monitor_resource_usage_closely": 2,
        }

        # Sort actions by priority
        prioritized = sorted(
            actions, key=lambda a: action_priorities.get(a, 1), reverse=True
        )

        return prioritized

    def _log_prediction_event(
        self, risk_assessment: RiskAssessment, actions: list[str]
    ) -> None:
        """Log prediction event for accuracy tracking."""
        prediction_event = {
            "timestamp": time.time(),
            "predicted_risk_level": risk_assessment.overall_risk_level.value,
            "confidence": risk_assessment.confidence_score,
            "active_patterns": len(risk_assessment.active_patterns),
            "recommended_actions": actions,
            "category_risks": {
                cat.value: risk for cat, risk in risk_assessment.category_risks.items()
            },
        }

        self.prediction_accuracy_history.append(prediction_event)

        # Keep only last 1000 prediction events
        if len(self.prediction_accuracy_history) > 1000:
            self.prediction_accuracy_history = self.prediction_accuracy_history[-1000:]

    def _load_failure_history(self) -> None:
        """Load failure history from file."""
        try:
            if self.history_file.exists():
                with open(self.history_file) as f:
                    data = json.load(f)

                # Convert back to HistoricalFailure objects
                self.failure_history = []
                for item in data.get("failure_history", []):
                    failure = HistoricalFailure(
                        timestamp=item["timestamp"],
                        category=FailureCategory(item["category"]),
                        severity=item["severity"],
                        precursor_metrics=item["precursor_metrics"],
                        resolution_time_minutes=item["resolution_time_minutes"],
                        mitigation_used=item["mitigation_used"],
                        context=item.get("context", {}),
                    )
                    self.failure_history.append(failure)

                logger.info("Loaded failure history", count=len(self.failure_history))

        except Exception as e:
            logger.error("Failed to load failure history", error=str(e))
            self.failure_history = []

    def _save_failure_history(self) -> None:
        """Save failure history to file."""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                "failure_history": [
                    asdict(failure) for failure in self.failure_history
                ],
                "last_saved": datetime.now(UTC).isoformat(),
            }

            # Convert enums to strings
            for failure_data in data["failure_history"]:
                failure_data["category"] = failure_data["category"].value

            with open(self.history_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved failure history", count=len(self.failure_history))

        except Exception as e:
            logger.error("Failed to save failure history", error=str(e))

    def get_prediction_statistics(self) -> dict[str, Any]:
        """Get prediction system statistics."""
        return {
            "total_failures_in_history": len(self.failure_history),
            "learned_patterns": len(self.pattern_analyzer.learned_patterns),
            "pattern_categories": list(
                set(
                    pattern.category.value
                    for pattern in self.pattern_analyzer.learned_patterns
                )
            ),
            "recent_predictions": len(self.prediction_accuracy_history),
            "failure_categories_seen": list(
                set(failure.category.value for failure in self.failure_history)
            ),
            "average_resolution_time_minutes": (
                sum(f.resolution_time_minutes for f in self.failure_history)
                / len(self.failure_history)
            )
            if self.failure_history
            else 0.0,
        }
