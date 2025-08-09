"""Autonomous Monitoring Dashboard for Extended Development Sessions.

This module provides real-time monitoring and metrics collection for
autonomous development workflows without requiring human oversight.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()


@dataclass
class VelocityMetrics:
    """Code velocity and quality metrics."""

    commits_per_hour: float = 0.0
    tests_per_hour: float = 0.0
    lines_of_code_per_hour: float = 0.0
    bugs_introduced_per_hour: float = 0.0
    quality_score: float = 0.0  # Composite quality metric
    technical_debt_trend: float = 0.0  # Positive = increasing debt


@dataclass
class AutonomyScore:
    """Composite score measuring autonomous operation effectiveness."""

    overall_score: float = 0.0  # 0-100 scale
    reliability_component: float = 0.0  # Time since last rollback
    quality_component: float = 0.0  # Code quality maintenance
    velocity_component: float = 0.0  # Sustainable development speed
    efficiency_component: float = 0.0  # Resource utilization
    learning_component: float = 0.0  # Improvement over time


@dataclass
class AutonomousMetrics:
    """Complete autonomous development metrics snapshot."""

    timestamp: float = field(default_factory=time.time)
    session_duration_hours: float = 0.0
    velocity: VelocityMetrics = field(default_factory=VelocityMetrics)
    autonomy_score: AutonomyScore = field(default_factory=AutonomyScore)

    # Performance indicators
    task_completion_rate: float = 0.0
    error_recovery_rate: float = 0.0
    test_coverage_percentage: float = 0.0
    performance_regression_count: int = 0

    # Resource utilization
    memory_efficiency_score: float = 0.0
    cpu_efficiency_score: float = 0.0
    disk_usage_trend: float = 0.0

    # Safety metrics
    rollback_events_count: int = 0
    quality_gate_failures: int = 0
    time_since_last_rollback_hours: float = 0.0


class MetricsCollector:
    """Collects and aggregates metrics from various system components."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.collection_history: List[Dict[str, Any]] = []
        self.baselines: Dict[str, float] = {}

    async def collect_velocity_metrics(
        self, timeframe_hours: float = 1.0
    ) -> VelocityMetrics:
        """Collect code velocity metrics for specified timeframe."""
        try:
            metrics = VelocityMetrics()

            # Git-based velocity metrics
            git_metrics = await self._collect_git_velocity_metrics(timeframe_hours)
            metrics.commits_per_hour = git_metrics.get("commits_per_hour", 0.0)
            metrics.lines_of_code_per_hour = git_metrics.get("lines_per_hour", 0.0)

            # Test metrics
            test_metrics = await self._collect_test_velocity_metrics(timeframe_hours)
            metrics.tests_per_hour = test_metrics.get("tests_per_hour", 0.0)

            # Quality metrics
            quality_metrics = await self._collect_quality_metrics()
            metrics.quality_score = quality_metrics.get("composite_score", 0.0)
            metrics.technical_debt_trend = quality_metrics.get("debt_trend", 0.0)

            # Bug introduction rate (simplified)
            bug_metrics = await self._estimate_bug_introduction_rate(timeframe_hours)
            metrics.bugs_introduced_per_hour = bug_metrics

            return metrics

        except Exception as e:
            logger.error("Failed to collect velocity metrics", error=str(e))
            return VelocityMetrics()

    async def collect_performance_metrics(self) -> Dict[str, float]:
        """Collect system performance metrics."""
        try:
            performance = {}

            # Test execution performance
            test_perf = await self._measure_test_performance()
            performance.update(test_perf)

            # Memory efficiency
            memory_metrics = await self._collect_memory_efficiency()
            performance["memory_efficiency"] = memory_metrics

            # CPU efficiency
            cpu_metrics = await self._collect_cpu_efficiency()
            performance["cpu_efficiency"] = cpu_metrics

            # Disk usage trends
            disk_metrics = await self._collect_disk_trends()
            performance["disk_trend"] = disk_metrics

            return performance

        except Exception as e:
            logger.error("Failed to collect performance metrics", error=str(e))
            return {}

    async def _collect_git_velocity_metrics(
        self, timeframe_hours: float
    ) -> Dict[str, float]:
        """Collect git-based velocity metrics."""
        try:
            since_time = time.time() - (timeframe_hours * 3600)
            since_date = datetime.fromtimestamp(since_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            )

            # Count commits
            process = await asyncio.create_subprocess_exec(
                "git",
                "rev-list",
                "--count",
                f"--since={since_date}",
                "HEAD",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            commit_count = (
                int(stdout.decode().strip()) if stdout.decode().strip() else 0
            )
            commits_per_hour = (
                commit_count / timeframe_hours if timeframe_hours > 0 else 0
            )

            # Count lines changed
            process = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--shortstat",
                f"HEAD~{commit_count}",
                "HEAD",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            lines_changed = 0
            if stdout.decode().strip():
                # Parse shortstat output: "X files changed, Y insertions(+), Z deletions(-)"
                parts = stdout.decode().strip().split(", ")
                for part in parts:
                    if "insertion" in part:
                        lines_changed += int(part.split()[0])
                    elif "deletion" in part:
                        lines_changed += int(part.split()[0])

            lines_per_hour = (
                lines_changed / timeframe_hours if timeframe_hours > 0 else 0
            )

            return {
                "commits_per_hour": commits_per_hour,
                "lines_per_hour": lines_per_hour,
            }

        except Exception as e:
            logger.error("Failed to collect git velocity metrics", error=str(e))
            return {"commits_per_hour": 0.0, "lines_per_hour": 0.0}

    async def _collect_test_velocity_metrics(
        self, timeframe_hours: float
    ) -> Dict[str, float]:
        """Collect test-related velocity metrics."""
        try:
            # Count test files and functions
            test_files = list(self.project_path.rglob("test_*.py"))
            test_files.extend(list(self.project_path.rglob("*_test.py")))

            total_tests = 0
            for test_file in test_files:
                try:
                    content = test_file.read_text()
                    # Simple count of test functions
                    total_tests += content.count("def test_")
                except Exception:
                    continue

            # Estimate tests added in timeframe (simplified)
            # This is a rough approximation - in practice would need git blame analysis
            recent_test_ratio = 0.1  # Assume 10% of tests are recent
            recent_tests = total_tests * recent_test_ratio
            tests_per_hour = (
                recent_tests / timeframe_hours if timeframe_hours > 0 else 0
            )

            return {"tests_per_hour": tests_per_hour}

        except Exception as e:
            logger.error("Failed to collect test velocity metrics", error=str(e))
            return {"tests_per_hour": 0.0}

    async def _collect_quality_metrics(self) -> Dict[str, float]:
        """Collect code quality metrics."""
        try:
            quality_score = 0.0
            debt_trend = 0.0

            # Run basic quality checks
            # Test coverage
            try:
                process = await asyncio.create_subprocess_exec(
                    "pytest",
                    "--cov",
                    "--cov-report=term-missing",
                    "--tb=no",
                    "-q",
                    cwd=self.project_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await process.communicate()

                # Parse coverage percentage
                coverage_lines = [
                    line for line in stdout.decode().split("\n") if "TOTAL" in line
                ]
                if coverage_lines:
                    # Extract percentage from lines like "TOTAL    100    50    50%"
                    parts = coverage_lines[0].split()
                    if len(parts) >= 4 and "%" in parts[-1]:
                        coverage = float(parts[-1].rstrip("%")) / 100
                        quality_score += coverage * 0.4  # 40% weight for coverage

            except Exception:
                pass

            # Code complexity (simplified)
            python_files = list(self.project_path.rglob("*.py"))
            if python_files:
                # Simple complexity estimation based on file sizes
                total_lines = 0
                total_files = 0
                for py_file in python_files:
                    try:
                        lines = len(py_file.read_text().splitlines())
                        total_lines += lines
                        total_files += 1
                    except Exception:
                        continue

                if total_files > 0:
                    avg_file_size = total_lines / total_files
                    # Lower average file size = better modularity
                    modularity_score = max(
                        0, 1 - (avg_file_size / 500)
                    )  # 500 lines baseline
                    quality_score += modularity_score * 0.3  # 30% weight

            # Documentation coverage (simplified)
            doc_score = 0.5  # Placeholder - would need proper docstring analysis
            quality_score += doc_score * 0.3  # 30% weight

            return {
                "composite_score": min(1.0, quality_score),
                "debt_trend": debt_trend,
            }

        except Exception as e:
            logger.error("Failed to collect quality metrics", error=str(e))
            return {"composite_score": 0.5, "debt_trend": 0.0}

    async def _estimate_bug_introduction_rate(self, timeframe_hours: float) -> float:
        """Estimate bug introduction rate based on recent activity."""
        try:
            # Look for rollback events and test failures as bug indicators
            recent_rollbacks = 0  # Would integrate with rollback system
            recent_test_failures = 0  # Would integrate with test results

            # Simple estimation
            bug_indicators = recent_rollbacks + (recent_test_failures * 0.3)
            bugs_per_hour = (
                bug_indicators / timeframe_hours if timeframe_hours > 0 else 0
            )

            return bugs_per_hour

        except Exception as e:
            logger.error("Failed to estimate bug introduction rate", error=str(e))
            return 0.0

    async def _measure_test_performance(self) -> Dict[str, float]:
        """Measure test execution performance."""
        try:
            start_time = time.time()

            process = await asyncio.create_subprocess_exec(
                "pytest",
                "tests/",
                "--tb=no",
                "-q",
                "--durations=0",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await process.communicate()

            test_runtime = time.time() - start_time

            # Parse test results
            test_count = 0
            if "passed" in stdout.decode():
                # Extract test count from output like "X passed in Y seconds"
                lines = stdout.decode().split("\n")
                for line in lines:
                    if "passed" in line and "in" in line:
                        parts = line.split()
                        if parts and parts[0].isdigit():
                            test_count = int(parts[0])
                        break

            tests_per_second = test_count / test_runtime if test_runtime > 0 else 0

            return {
                "test_runtime_seconds": test_runtime,
                "tests_per_second": tests_per_second,
                "test_count": test_count,
            }

        except Exception as e:
            logger.error("Failed to measure test performance", error=str(e))
            return {}

    async def _collect_memory_efficiency(self) -> float:
        """Collect memory efficiency metrics."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            # Simple efficiency metric: higher available memory = higher efficiency
            efficiency = memory.available / memory.total
            return efficiency

        except Exception as e:
            logger.error("Failed to collect memory efficiency", error=str(e))
            return 0.5

    async def _collect_cpu_efficiency(self) -> float:
        """Collect CPU efficiency metrics."""
        try:
            import psutil

            # Measure CPU usage over a short period
            cpu_percent = psutil.cpu_percent(interval=1)
            # Efficiency is inverse of utilization (lower usage = higher efficiency for development)
            efficiency = max(0, 1 - (cpu_percent / 100))
            return efficiency

        except Exception as e:
            logger.error("Failed to collect CPU efficiency", error=str(e))
            return 0.5

    async def _collect_disk_trends(self) -> float:
        """Collect disk usage trend metrics."""
        try:
            import shutil

            disk_usage = shutil.disk_usage(self.project_path)
            # Simple trend metric: more free space = better trend
            free_ratio = disk_usage.free / disk_usage.total
            return free_ratio

        except Exception as e:
            logger.error("Failed to collect disk trends", error=str(e))
            return 0.5


class AutonomyScoreCalculator:
    """Calculates composite autonomy effectiveness scores."""

    def __init__(self):
        self.score_history: List[AutonomyScore] = []
        self.component_weights = {
            "reliability": 0.3,
            "quality": 0.25,
            "velocity": 0.2,
            "efficiency": 0.15,
            "learning": 0.1,
        }

    def calculate_autonomy_score(
        self, metrics: AutonomousMetrics, historical_data: List[AutonomousMetrics]
    ) -> AutonomyScore:
        """Calculate comprehensive autonomy score."""
        try:
            score = AutonomyScore()

            # Reliability component (based on time since last rollback)
            score.reliability_component = self._calculate_reliability_score(
                metrics.time_since_last_rollback_hours
            )

            # Quality component (based on test coverage and quality metrics)
            score.quality_component = self._calculate_quality_score(
                metrics.test_coverage_percentage,
                metrics.velocity.quality_score,
                metrics.quality_gate_failures,
            )

            # Velocity component (sustainable development speed)
            score.velocity_component = self._calculate_velocity_score(
                metrics.velocity, metrics.task_completion_rate
            )

            # Efficiency component (resource utilization)
            score.efficiency_component = self._calculate_efficiency_score(
                metrics.memory_efficiency_score, metrics.cpu_efficiency_score
            )

            # Learning component (improvement over time)
            score.learning_component = self._calculate_learning_score(
                metrics, historical_data
            )

            # Calculate overall score
            score.overall_score = (
                score.reliability_component * self.component_weights["reliability"]
                + score.quality_component * self.component_weights["quality"]
                + score.velocity_component * self.component_weights["velocity"]
                + score.efficiency_component * self.component_weights["efficiency"]
                + score.learning_component * self.component_weights["learning"]
            ) * 100  # Scale to 0-100

            self.score_history.append(score)

            # Keep only last 100 scores
            if len(self.score_history) > 100:
                self.score_history = self.score_history[-100:]

            return score

        except Exception as e:
            logger.error("Failed to calculate autonomy score", error=str(e))
            return AutonomyScore()

    def _calculate_reliability_score(self, hours_since_rollback: float) -> float:
        """Calculate reliability score based on time since last rollback."""
        if hours_since_rollback >= 24:
            return 1.0  # Perfect score after 24 hours
        elif hours_since_rollback >= 12:
            return 0.8 + (hours_since_rollback - 12) / 12 * 0.2
        elif hours_since_rollback >= 4:
            return 0.5 + (hours_since_rollback - 4) / 8 * 0.3
        else:
            return max(0.1, hours_since_rollback / 4 * 0.5)

    def _calculate_quality_score(
        self, test_coverage: float, quality_metric: float, quality_gate_failures: int
    ) -> float:
        """Calculate quality score."""
        # Base score from coverage and quality metrics
        base_score = (test_coverage + quality_metric) / 2

        # Penalty for quality gate failures
        failure_penalty = min(0.5, quality_gate_failures * 0.1)

        return max(0.0, base_score - failure_penalty)

    def _calculate_velocity_score(
        self, velocity: VelocityMetrics, completion_rate: float
    ) -> float:
        """Calculate velocity score."""
        # Normalize velocity metrics (these would need proper baselines)
        normalized_velocity = min(
            1.0, velocity.commits_per_hour / 2.0
        )  # 2 commits/hour baseline

        # Combine with completion rate
        velocity_score = (normalized_velocity + completion_rate) / 2

        # Penalty for bug introduction
        bug_penalty = min(0.3, velocity.bugs_introduced_per_hour * 0.5)

        return max(0.0, velocity_score - bug_penalty)

    def _calculate_efficiency_score(
        self, memory_efficiency: float, cpu_efficiency: float
    ) -> float:
        """Calculate resource efficiency score."""
        return (memory_efficiency + cpu_efficiency) / 2

    def _calculate_learning_score(
        self,
        current_metrics: AutonomousMetrics,
        historical_data: List[AutonomousMetrics],
    ) -> float:
        """Calculate learning/improvement score."""
        if len(historical_data) < 2:
            return 0.5  # Neutral score without enough history

        try:
            # Compare recent performance to earlier performance
            recent_period = (
                historical_data[-5:] if len(historical_data) >= 5 else historical_data
            )
            earlier_period = (
                historical_data[-10:-5]
                if len(historical_data) >= 10
                else historical_data[:-5]
            )

            if not earlier_period:
                return 0.5

            # Calculate average completion rates
            recent_avg = sum(m.task_completion_rate for m in recent_period) / len(
                recent_period
            )
            earlier_avg = sum(m.task_completion_rate for m in earlier_period) / len(
                earlier_period
            )

            # Improvement ratio
            if earlier_avg > 0:
                improvement = (recent_avg - earlier_avg) / earlier_avg
                # Convert to 0-1 score (50% improvement = 1.0)
                return max(0.0, min(1.0, 0.5 + improvement))
            else:
                return 0.5

        except Exception as e:
            logger.error("Failed to calculate learning score", error=str(e))
            return 0.5


class AutonomousDashboard:
    """Main autonomous monitoring dashboard."""

    def __init__(self, project_path: Path, metrics_file: Optional[Path] = None):
        self.project_path = project_path

        if metrics_file is None:
            metrics_file = project_path / ".adw" / "autonomous_metrics.json"
        self.metrics_file = metrics_file

        self.collector = MetricsCollector(project_path)
        self.score_calculator = AutonomyScoreCalculator()

        self.metrics_history: List[AutonomousMetrics] = []
        self.monitoring_active = False

        # Load historical data
        self._load_metrics_history()

    async def start_monitoring(
        self, session_id: str = None, interval_minutes: float = 5.0
    ) -> None:
        """Start autonomous monitoring."""
        self.monitoring_active = True
        interval_seconds = interval_minutes * 60

        logger.info(
            "Autonomous monitoring started",
            session_id=session_id,
            interval_minutes=interval_minutes,
        )

        while self.monitoring_active:
            try:
                await self.collect_and_record_metrics()
                await asyncio.sleep(interval_seconds)

            except Exception as e:
                logger.error("Monitoring cycle error", error=str(e))
                await asyncio.sleep(interval_seconds)

    async def stop_monitoring(self) -> None:
        """Stop autonomous monitoring."""
        self.monitoring_active = False
        self._save_metrics_history()
        logger.info("Autonomous monitoring stopped")

    async def collect_and_record_metrics(self) -> AutonomousMetrics:
        """Collect and record current metrics."""
        try:
            # Collect all metrics
            velocity = await self.collector.collect_velocity_metrics()
            performance = await self.collector.collect_performance_metrics()

            # Create metrics snapshot
            metrics = AutonomousMetrics(
                velocity=velocity,
                memory_efficiency_score=performance.get("memory_efficiency", 0.5),
                cpu_efficiency_score=performance.get("cpu_efficiency", 0.5),
                disk_usage_trend=performance.get("disk_trend", 0.5),
            )

            # Calculate session duration
            if self.metrics_history:
                session_start = self.metrics_history[0].timestamp
                metrics.session_duration_hours = (time.time() - session_start) / 3600

            # Calculate autonomy score
            metrics.autonomy_score = self.score_calculator.calculate_autonomy_score(
                metrics, self.metrics_history
            )

            # Add to history
            self.metrics_history.append(metrics)

            # Keep reasonable history size
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]

            # Periodic save
            if len(self.metrics_history) % 10 == 0:
                self._save_metrics_history()

            logger.info(
                "Metrics collected",
                session_hours=metrics.session_duration_hours,
                autonomy_score=metrics.autonomy_score.overall_score,
                velocity_commits=velocity.commits_per_hour,
            )

            return metrics

        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
            return AutonomousMetrics()

    def get_current_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for display."""
        if not self.metrics_history:
            return {"status": "no_data"}

        latest = self.metrics_history[-1]

        # Calculate trends
        trends = {}
        if len(self.metrics_history) >= 2:
            previous = self.metrics_history[-2]
            trends = {
                "velocity_trend": latest.velocity.commits_per_hour
                - previous.velocity.commits_per_hour,
                "quality_trend": latest.velocity.quality_score
                - previous.velocity.quality_score,
                "autonomy_trend": latest.autonomy_score.overall_score
                - previous.autonomy_score.overall_score,
            }

        # Get recent statistics
        recent_metrics = (
            self.metrics_history[-10:]
            if len(self.metrics_history) >= 10
            else self.metrics_history
        )

        avg_autonomy = sum(
            m.autonomy_score.overall_score for m in recent_metrics
        ) / len(recent_metrics)
        avg_velocity = sum(m.velocity.commits_per_hour for m in recent_metrics) / len(
            recent_metrics
        )

        return {
            "status": "active",
            "session_duration_hours": latest.session_duration_hours,
            "current_autonomy_score": latest.autonomy_score.overall_score,
            "recent_avg_autonomy": avg_autonomy,
            "current_velocity": latest.velocity.commits_per_hour,
            "recent_avg_velocity": avg_velocity,
            "quality_score": latest.velocity.quality_score,
            "test_coverage": latest.test_coverage_percentage,
            "resource_efficiency": {
                "memory": latest.memory_efficiency_score,
                "cpu": latest.cpu_efficiency_score,
                "disk": latest.disk_usage_trend,
            },
            "safety_metrics": {
                "rollback_events": latest.rollback_events_count,
                "hours_since_rollback": latest.time_since_last_rollback_hours,
                "quality_gate_failures": latest.quality_gate_failures,
            },
            "trends": trends,
            "autonomy_components": {
                "reliability": latest.autonomy_score.reliability_component,
                "quality": latest.autonomy_score.quality_component,
                "velocity": latest.autonomy_score.velocity_component,
                "efficiency": latest.autonomy_score.efficiency_component,
                "learning": latest.autonomy_score.learning_component,
            },
            "total_metrics_collected": len(self.metrics_history),
            "last_updated": datetime.fromtimestamp(latest.timestamp).isoformat(),
        }

    def _load_metrics_history(self) -> None:
        """Load metrics history from file."""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, "r") as f:
                    data = json.load(f)

                # Convert back to AutonomousMetrics objects
                self.metrics_history = []
                for item in data.get("metrics_history", []):
                    metrics = AutonomousMetrics(
                        timestamp=item["timestamp"],
                        session_duration_hours=item["session_duration_hours"],
                        velocity=VelocityMetrics(**item["velocity"]),
                        autonomy_score=AutonomyScore(**item["autonomy_score"]),
                        task_completion_rate=item.get("task_completion_rate", 0.0),
                        error_recovery_rate=item.get("error_recovery_rate", 0.0),
                        test_coverage_percentage=item.get(
                            "test_coverage_percentage", 0.0
                        ),
                        performance_regression_count=item.get(
                            "performance_regression_count", 0
                        ),
                        memory_efficiency_score=item.get(
                            "memory_efficiency_score", 0.0
                        ),
                        cpu_efficiency_score=item.get("cpu_efficiency_score", 0.0),
                        disk_usage_trend=item.get("disk_usage_trend", 0.0),
                        rollback_events_count=item.get("rollback_events_count", 0),
                        quality_gate_failures=item.get("quality_gate_failures", 0),
                        time_since_last_rollback_hours=item.get(
                            "time_since_last_rollback_hours", 0.0
                        ),
                    )
                    self.metrics_history.append(metrics)

                logger.info("Loaded metrics history", count=len(self.metrics_history))

        except Exception as e:
            logger.error("Failed to load metrics history", error=str(e))
            self.metrics_history = []

    def _save_metrics_history(self) -> None:
        """Save metrics history to file."""
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # Convert to serializable format
            data = {
                "metrics_history": [
                    asdict(metrics) for metrics in self.metrics_history
                ],
                "last_saved": datetime.now(timezone.utc).isoformat(),
            }

            with open(self.metrics_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Saved metrics history", count=len(self.metrics_history))

        except Exception as e:
            logger.error("Failed to save metrics history", error=str(e))
