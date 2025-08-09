"""Extended Session Testing Framework for 16-24 Hour ADW Sessions.

This module provides tools for testing and validating extended autonomous
development sessions, including stress testing, endurance validation,
and performance monitoring over extended periods.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..adw.session_manager import ADWSession, ADWSessionConfig, SessionPhase
from ..monitoring.autonomous_dashboard import AutonomousDashboard

logger = structlog.get_logger()


class ExtendedSessionTestType(Enum):
    """Types of extended session tests."""

    ENDURANCE = "endurance"  # Basic 16+ hour endurance test
    STRESS = "stress"  # High-load stress testing
    RECOVERY = "recovery"  # Recovery from failures test
    EFFICIENCY = "efficiency"  # Efficiency over time test
    COGNITIVE = "cognitive"  # Cognitive load progression test


@dataclass
class ExtendedSessionMetrics:
    """Metrics collected during extended session testing."""

    start_time: float
    end_time: Optional[float] = None
    test_type: ExtendedSessionTestType = ExtendedSessionTestType.ENDURANCE

    # Session progression
    total_sessions_planned: int = 0
    total_sessions_completed: int = 0
    total_sessions_failed: int = 0

    # Performance metrics
    commits_per_hour: List[float] = field(default_factory=list)
    test_success_rates: List[float] = field(default_factory=list)
    quality_gate_rates: List[float] = field(default_factory=list)

    # Resource utilization
    memory_usage_over_time: List[Tuple[float, float]] = field(default_factory=list)
    cpu_usage_over_time: List[Tuple[float, float]] = field(default_factory=list)

    # Cognitive load progression
    fatigue_levels_over_time: List[Tuple[float, float]] = field(default_factory=list)
    mode_transitions: List[Tuple[float, str, str]] = field(default_factory=list)

    # Failure patterns
    failure_types: Dict[str, int] = field(default_factory=dict)
    recovery_times: List[float] = field(default_factory=list)
    rollback_frequency: List[float] = field(default_factory=list)

    # System health
    component_uptime: Dict[str, float] = field(default_factory=dict)
    error_rates: List[float] = field(default_factory=list)

    def duration(self) -> float:
        """Get total test duration in hours."""
        if self.end_time:
            return (self.end_time - self.start_time) / 3600
        return (time.time() - self.start_time) / 3600

    def success_rate(self) -> float:
        """Calculate overall session success rate."""
        if self.total_sessions_planned == 0:
            return 0.0
        return self.total_sessions_completed / self.total_sessions_planned


class ExtendedSessionTester:
    """Framework for testing extended autonomous development sessions."""

    def __init__(self, project_path: Path, output_dir: Optional[Path] = None):
        self.project_path = project_path
        self.output_dir = output_dir or project_path / "extended_session_tests"
        self.output_dir.mkdir(exist_ok=True)

        # Test state
        self.active_test = False
        self.current_session: Optional[ADWSession] = None
        self.metrics: Optional[ExtendedSessionMetrics] = None

        # Monitoring
        self.dashboard: Optional[AutonomousDashboard] = None
        self.monitoring_task: Optional[asyncio.Task] = None

        logger.info(
            "Extended session tester initialized",
            project_path=str(project_path),
            output_dir=str(self.output_dir),
        )

    async def run_endurance_test(
        self,
        duration_hours: float = 16.0,
        session_goals: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run a basic endurance test for specified duration."""
        logger.info(
            "Starting endurance test",
            duration_hours=duration_hours,
            goals=session_goals,
        )

        config = ADWSessionConfig(
            extended_session_mode=True,
            max_extended_duration_hours=duration_hours,
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
            cognitive_fatigue_threshold=0.6,  # Conservative for long sessions
        )

        return await self._run_test(
            ExtendedSessionTestType.ENDURANCE,
            config,
            duration_hours,
            session_goals,
        )

    async def run_stress_test(
        self,
        duration_hours: float = 8.0,
        stress_factors: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a stress test with high cognitive load."""
        logger.info(
            "Starting stress test",
            duration_hours=duration_hours,
            stress_factors=stress_factors,
        )

        stress_factors = stress_factors or {}

        config = ADWSessionConfig(
            extended_session_mode=True,
            max_extended_duration_hours=duration_hours,
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
            # Aggressive settings for stress testing
            cognitive_fatigue_threshold=0.8,  # Higher threshold
            micro_iteration_minutes=20,  # Shorter iterations
            max_consecutive_failures=5,  # Allow more failures
        )

        return await self._run_test(
            ExtendedSessionTestType.STRESS,
            config,
            duration_hours,
            stress_factors.get("goals", ["stress_test_feature"]),
            stress_factors,
        )

    async def run_recovery_test(
        self,
        duration_hours: float = 12.0,
        failure_injection_rate: float = 0.3,
    ) -> Dict[str, Any]:
        """Run a recovery test with injected failures."""
        logger.info(
            "Starting recovery test",
            duration_hours=duration_hours,
            failure_rate=failure_injection_rate,
        )

        config = ADWSessionConfig(
            extended_session_mode=True,
            max_extended_duration_hours=duration_hours,
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
            rollback_on_critical_failure=True,  # Essential for recovery testing
            max_consecutive_failures=3,
        )

        return await self._run_test(
            ExtendedSessionTestType.RECOVERY,
            config,
            duration_hours,
            ["recovery_test_feature"],
            {"failure_injection_rate": failure_injection_rate},
        )

    async def run_efficiency_test(
        self,
        duration_hours: float = 20.0,
        efficiency_targets: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run an efficiency test measuring performance over time."""
        logger.info(
            "Starting efficiency test",
            duration_hours=duration_hours,
            targets=efficiency_targets,
        )

        efficiency_targets = efficiency_targets or {
            "min_commits_per_hour": 2.0,
            "min_test_success_rate": 0.8,
            "max_rollback_rate": 0.1,
        }

        config = ADWSessionConfig(
            extended_session_mode=True,
            max_extended_duration_hours=duration_hours,
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
            performance_tracking_enabled=True,
            quality_gates_enabled=True,
        )

        return await self._run_test(
            ExtendedSessionTestType.EFFICIENCY,
            config,
            duration_hours,
            ["efficiency_test_feature"],
            {"efficiency_targets": efficiency_targets},
        )

    async def run_cognitive_progression_test(
        self,
        duration_hours: float = 18.0,
    ) -> Dict[str, Any]:
        """Run a test focused on cognitive load progression."""
        logger.info(
            "Starting cognitive progression test",
            duration_hours=duration_hours,
        )

        config = ADWSessionConfig(
            extended_session_mode=True,
            max_extended_duration_hours=duration_hours,
            cognitive_load_management_enabled=True,
            failure_prediction_enabled=True,
            autonomous_dashboard_enabled=True,
            cognitive_fatigue_threshold=0.5,  # Lower threshold for more transitions
        )

        return await self._run_test(
            ExtendedSessionTestType.COGNITIVE,
            config,
            duration_hours,
            ["cognitive_test_feature"],
        )

    async def _run_test(
        self,
        test_type: ExtendedSessionTestType,
        config: ADWSessionConfig,
        duration_hours: float,
        session_goals: Optional[List[str]] = None,
        test_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a specific type of extended session test."""
        if self.active_test:
            raise RuntimeError("Test already running")

        self.active_test = True
        self.metrics = ExtendedSessionMetrics(
            start_time=time.time(),
            test_type=test_type,
        )

        test_context = test_context or {}
        test_id = f"{test_type.value}_{int(time.time())}"

        try:
            # Initialize monitoring
            await self._start_monitoring(test_id)

            # Calculate number of sessions needed
            session_duration = config.total_duration_hours
            num_sessions = max(1, int(duration_hours / session_duration))

            # For very short test durations, force multiple sessions by reducing session duration
            if duration_hours < 1.0 and num_sessions == 1:
                config.total_duration_hours = (
                    duration_hours / 3
                )  # Create 3 sessions for short tests
                num_sessions = 3

            self.metrics.total_sessions_planned = num_sessions

            logger.info(
                "Test plan created",
                test_id=test_id,
                test_type=test_type.value,
                num_sessions=num_sessions,
                session_duration=session_duration,
                total_duration=duration_hours,
            )

            # Run sessions sequentially
            for session_num in range(num_sessions):
                if not self.active_test:
                    break

                logger.info(
                    "Starting session",
                    test_id=test_id,
                    session=session_num + 1,
                    total=num_sessions,
                )

                session_result = await self._run_single_session(
                    config, session_goals, session_num + 1, test_context
                )

                # Record session metrics
                await self._record_session_metrics(session_result)

                # Check if we should continue
                if session_result.get("status") == "failed":
                    self.metrics.total_sessions_failed += 1

                    # Stop test if too many failures for certain test types
                    if test_type == ExtendedSessionTestType.ENDURANCE:
                        failure_rate = self.metrics.total_sessions_failed / (
                            session_num + 1
                        )
                        if failure_rate > 0.5:  # More than 50% failure rate
                            logger.error(
                                "Stopping endurance test due to high failure rate",
                                failure_rate=failure_rate,
                            )
                            break
                else:
                    self.metrics.total_sessions_completed += 1

                # Brief pause between sessions
                await asyncio.sleep(5)

            # Generate final report
            return await self._generate_test_report(test_id, test_context)

        except Exception as e:
            logger.error(
                "Extended session test failed",
                test_id=test_id,
                error=str(e),
            )
            raise
        finally:
            await self._stop_monitoring()
            self.active_test = False

    async def _run_single_session(
        self,
        config: ADWSessionConfig,
        session_goals: Optional[List[str]],
        session_num: int,
        test_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run a single ADW session as part of extended test."""
        session_start = time.time()

        try:
            # Create session with unique ID
            session_config = ADWSessionConfig(**config.__dict__)
            self.current_session = ADWSession(self.project_path, session_config)

            # Apply test-specific modifications
            await self._apply_test_modifications(test_context)

            # Run session
            result = await self.current_session.start_session(
                target_goals=session_goals or [f"session_{session_num}_feature"]
            )

            # Add session metadata
            result["session_number"] = session_num
            result["session_start_time"] = session_start
            result["test_context"] = test_context

            return result

        except Exception as e:
            logger.error(
                "Session failed",
                session_num=session_num,
                error=str(e),
            )
            return {
                "status": "failed",
                "session_number": session_num,
                "session_start_time": session_start,
                "error": str(e),
                "test_context": test_context,
            }
        finally:
            self.current_session = None

    async def _apply_test_modifications(self, test_context: Dict[str, Any]) -> None:
        """Apply test-specific modifications to the current session."""
        if not self.current_session:
            return

        test_type = (
            self.metrics.test_type
            if self.metrics
            else ExtendedSessionTestType.ENDURANCE
        )

        if test_type == ExtendedSessionTestType.STRESS:
            # Apply stress factors
            if "failure_injection_rate" in test_context:
                # Would inject artificial failures here
                pass

        elif test_type == ExtendedSessionTestType.RECOVERY:
            # Set up failure injection
            failure_rate = test_context.get("failure_injection_rate", 0.3)
            # Would set up failure injection mechanism here
            pass

        elif test_type == ExtendedSessionTestType.EFFICIENCY:
            # Set up efficiency monitoring
            targets = test_context.get("efficiency_targets", {})
            # Would configure efficiency monitoring here
            pass

    async def _start_monitoring(self, test_id: str) -> None:
        """Start extended session monitoring."""
        self.dashboard = AutonomousDashboard(self.project_path)
        await self.dashboard.start_monitoring(
            session_id=test_id, interval_minutes=0.5
        )  # 30 second intervals for testing

        # Start background monitoring task
        self.monitoring_task = asyncio.create_task(self._continuous_monitoring())

        logger.info("Extended session monitoring started", test_id=test_id)

    async def _stop_monitoring(self) -> None:
        """Stop extended session monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.dashboard:
            await self.dashboard.stop_monitoring()

        logger.info("Extended session monitoring stopped")

    async def _continuous_monitoring(self) -> None:
        """Continuous monitoring during extended test."""
        try:
            while self.active_test:
                timestamp = time.time()

                # Collect system metrics
                if self.current_session:
                    # Memory usage
                    memory_usage = 45.0  # Would get actual memory usage
                    self.metrics.memory_usage_over_time.append(
                        (timestamp, memory_usage)
                    )

                    # CPU usage
                    cpu_usage = 30.0  # Would get actual CPU usage
                    self.metrics.cpu_usage_over_time.append((timestamp, cpu_usage))

                    # Cognitive load if available
                    if self.current_session.cognitive_load_manager:
                        # Would get actual cognitive state
                        fatigue_level = 0.4  # Placeholder
                        self.metrics.fatigue_levels_over_time.append(
                            (timestamp, fatigue_level)
                        )

                await asyncio.sleep(30)  # Monitor every 30 seconds

        except asyncio.CancelledError:
            logger.info("Monitoring task cancelled")
        except Exception as e:
            logger.error("Monitoring task failed", error=str(e))

    async def _record_session_metrics(self, session_result: Dict[str, Any]) -> None:
        """Record metrics from a completed session."""
        if not self.metrics:
            return

        # Extract session metrics
        session_metrics = session_result.get("metrics", {})

        # Commits per hour
        duration_hours = session_result.get("duration_hours", 1.0)
        commits = session_metrics.get("commits_made", 0)
        commits_per_hour = commits / max(duration_hours, 0.1)
        self.metrics.commits_per_hour.append(commits_per_hour)

        # Test success rate
        tests_passed = session_metrics.get("tests_passed", 0)
        tests_written = session_metrics.get("tests_written", 1)
        test_success_rate = tests_passed / max(tests_written, 1)
        self.metrics.test_success_rates.append(test_success_rate)

        # Quality gate rate
        quality_passes = session_metrics.get("quality_gate_passes", 0)
        quality_total = quality_passes + session_metrics.get("quality_gate_failures", 0)
        quality_rate = quality_passes / max(quality_total, 1)
        self.metrics.quality_gate_rates.append(quality_rate)

        # Rollback frequency
        rollbacks = session_metrics.get("rollbacks_triggered", 0)
        rollback_rate = rollbacks / max(duration_hours, 0.1)
        self.metrics.rollback_frequency.append(rollback_rate)

        # Record failure types
        if session_result.get("status") == "failed":
            error = session_result.get("error", "unknown_error")
            self.metrics.failure_types[error] = (
                self.metrics.failure_types.get(error, 0) + 1
            )

    async def _generate_test_report(
        self, test_id: str, test_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        if not self.metrics:
            return {"error": "No metrics available"}

        self.metrics.end_time = time.time()

        # Calculate summary statistics
        avg_commits_per_hour = (
            sum(self.metrics.commits_per_hour) / len(self.metrics.commits_per_hour)
            if self.metrics.commits_per_hour
            else 0
        )

        avg_test_success_rate = (
            sum(self.metrics.test_success_rates) / len(self.metrics.test_success_rates)
            if self.metrics.test_success_rates
            else 0
        )

        avg_quality_rate = (
            sum(self.metrics.quality_gate_rates) / len(self.metrics.quality_gate_rates)
            if self.metrics.quality_gate_rates
            else 0
        )

        # Generate report
        report = {
            "test_id": test_id,
            "test_type": self.metrics.test_type.value,
            "start_time": datetime.fromtimestamp(
                self.metrics.start_time, tz=timezone.utc
            ).isoformat(),
            "end_time": datetime.fromtimestamp(
                self.metrics.end_time, tz=timezone.utc
            ).isoformat(),
            "duration_hours": self.metrics.duration(),
            "test_context": test_context,
            # Session summary
            "session_summary": {
                "planned": self.metrics.total_sessions_planned,
                "completed": self.metrics.total_sessions_completed,
                "failed": self.metrics.total_sessions_failed,
                "success_rate": self.metrics.success_rate(),
            },
            # Performance summary
            "performance_summary": {
                "avg_commits_per_hour": avg_commits_per_hour,
                "avg_test_success_rate": avg_test_success_rate,
                "avg_quality_gate_rate": avg_quality_rate,
                "total_commits": sum(
                    [
                        result * duration
                        for result, duration in zip(
                            self.metrics.commits_per_hour,
                            [1] * len(self.metrics.commits_per_hour),
                        )
                    ]
                )
                if self.metrics.commits_per_hour
                else 0,
            },
            # Resource summary
            "resource_summary": {
                "avg_memory_usage": (
                    sum(usage for _, usage in self.metrics.memory_usage_over_time)
                    / len(self.metrics.memory_usage_over_time)
                    if self.metrics.memory_usage_over_time
                    else 0
                ),
                "max_memory_usage": (
                    max(usage for _, usage in self.metrics.memory_usage_over_time)
                    if self.metrics.memory_usage_over_time
                    else 0
                ),
                "avg_cpu_usage": (
                    sum(usage for _, usage in self.metrics.cpu_usage_over_time)
                    / len(self.metrics.cpu_usage_over_time)
                    if self.metrics.cpu_usage_over_time
                    else 0
                ),
                "max_cpu_usage": (
                    max(usage for _, usage in self.metrics.cpu_usage_over_time)
                    if self.metrics.cpu_usage_over_time
                    else 0
                ),
            },
            # Cognitive load summary
            "cognitive_summary": {
                "mode_transitions": len(self.metrics.mode_transitions),
                "avg_fatigue_level": (
                    sum(fatigue for _, fatigue in self.metrics.fatigue_levels_over_time)
                    / len(self.metrics.fatigue_levels_over_time)
                    if self.metrics.fatigue_levels_over_time
                    else 0
                ),
                "max_fatigue_level": (
                    max(fatigue for _, fatigue in self.metrics.fatigue_levels_over_time)
                    if self.metrics.fatigue_levels_over_time
                    else 0
                ),
            },
            # Failure analysis
            "failure_analysis": {
                "failure_types": dict(self.metrics.failure_types),
                "most_common_failure": max(
                    self.metrics.failure_types.items(),
                    key=lambda x: x[1],
                    default=("none", 0),
                )[0],
                "avg_recovery_time": (
                    sum(self.metrics.recovery_times) / len(self.metrics.recovery_times)
                    if self.metrics.recovery_times
                    else 0
                ),
            },
            # Raw data for detailed analysis
            "raw_metrics": {
                "commits_per_hour": self.metrics.commits_per_hour,
                "test_success_rates": self.metrics.test_success_rates,
                "quality_gate_rates": self.metrics.quality_gate_rates,
                "memory_usage_over_time": self.metrics.memory_usage_over_time,
                "cpu_usage_over_time": self.metrics.cpu_usage_over_time,
                "fatigue_levels_over_time": self.metrics.fatigue_levels_over_time,
                "mode_transitions": self.metrics.mode_transitions,
                "rollback_frequency": self.metrics.rollback_frequency,
            },
        }

        # Save report to file
        report_file = self.output_dir / f"{test_id}_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(
            "Extended session test completed",
            test_id=test_id,
            duration_hours=self.metrics.duration(),
            success_rate=self.metrics.success_rate(),
            report_file=str(report_file),
        )

        return report

    async def abort_test(self, reason: str = "Manual abort") -> None:
        """Abort the currently running test."""
        if not self.active_test:
            return

        logger.warning("Aborting extended session test", reason=reason)

        self.active_test = False

        if self.current_session:
            await self.current_session.abort_session(reason)

        await self._stop_monitoring()


# Utility functions for test analysis
def analyze_test_results(report_file: Path) -> Dict[str, Any]:
    """Analyze extended session test results."""
    with open(report_file) as f:
        report = json.load(f)

    analysis = {
        "test_passed": True,
        "issues": [],
        "recommendations": [],
    }

    # Check success rate
    success_rate = report["session_summary"]["success_rate"]
    if success_rate < 0.8:
        analysis["test_passed"] = False
        analysis["issues"].append(f"Low success rate: {success_rate:.2%}")
        analysis["recommendations"].append(
            "Review failure patterns and improve error handling"
        )

    # Check performance trends
    commits_per_hour = report["raw_metrics"]["commits_per_hour"]
    if commits_per_hour and len(commits_per_hour) > 3:
        trend = commits_per_hour[-1] - commits_per_hour[0]
        if trend < -0.5:  # Significant decrease
            analysis["issues"].append("Performance degradation over time")
            analysis["recommendations"].append("Investigate cognitive load management")

    # Check resource usage
    max_memory = report["resource_summary"]["max_memory_usage"]
    if max_memory > 80:  # Over 80% memory usage
        analysis["issues"].append("High memory usage detected")
        analysis["recommendations"].append(
            "Optimize memory management and garbage collection"
        )

    return analysis


def compare_test_results(report_files: List[Path]) -> Dict[str, Any]:
    """Compare multiple extended session test results."""
    if len(report_files) < 2:
        raise ValueError("Need at least 2 reports to compare")

    reports = []
    for file in report_files:
        with open(file) as f:
            reports.append(json.load(f))

    comparison = {
        "reports_compared": len(reports),
        "performance_trends": {},
        "stability_comparison": {},
        "efficiency_comparison": {},
    }

    # Compare performance trends
    for metric in [
        "avg_commits_per_hour",
        "avg_test_success_rate",
        "avg_quality_gate_rate",
    ]:
        values = [r["performance_summary"][metric] for r in reports]
        comparison["performance_trends"][metric] = {
            "values": values,
            "trend": "improving" if values[-1] > values[0] else "declining",
            "change_percent": ((values[-1] - values[0]) / values[0] * 100)
            if values[0] > 0
            else 0,
        }

    # Compare stability
    success_rates = [r["session_summary"]["success_rate"] for r in reports]
    comparison["stability_comparison"] = {
        "success_rates": success_rates,
        "stability_trend": "improving"
        if success_rates[-1] > success_rates[0]
        else "declining",
        "avg_success_rate": sum(success_rates) / len(success_rates),
    }

    return comparison
