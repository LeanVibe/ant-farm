"""
Meta-learning phase implementation for ADW sessions.
Analyzes session data and improves system performance.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class LearningInsight:
    """A single learning insight from session analysis."""

    type: str  # "pattern", "improvement", "failure_mode", "optimization"
    description: str
    confidence: float  # 0.0 to 1.0
    impact: str  # "low", "medium", "high"
    actionable: bool
    metadata: dict[str, Any]


@dataclass
class MetaLearningReport:
    """Complete meta-learning analysis report."""

    timestamp: float
    session_id: str
    insights: list[LearningInsight]
    performance_improvements: dict[str, float]
    knowledge_updates: list[str]
    next_priorities: list[str]
    system_adaptations: list[str]


class MetaLearningEngine:
    """Analyzes session data and generates learning insights."""

    def __init__(self, project_path: Path, session_id: str):
        self.project_path = project_path
        self.session_id = session_id
        self.knowledge_base_path = project_path / ".adw" / "knowledge_base.json"
        self.patterns_path = project_path / ".adw" / "patterns.json"

        # Ensure knowledge directories exist
        self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)

    async def analyze_session_and_learn(
        self,
        session_metrics: dict[str, Any],
        rollback_stats: dict[str, Any],
        quality_stats: dict[str, Any],
        resource_stats: dict[str, Any],
        validation_history: list[Any],
        iteration_history: list[Any],
    ) -> MetaLearningReport:
        """Perform comprehensive meta-learning analysis."""
        start_time = time.time()

        logger.info("Starting meta-learning analysis", session_id=self.session_id)

        # Run analysis tasks in parallel
        analysis_tasks = [
            self._analyze_code_patterns(),
            self._analyze_performance_trends(session_metrics, resource_stats),
            self._analyze_failure_modes(rollback_stats, validation_history),
            self._analyze_quality_trends(quality_stats),
            self._analyze_development_velocity(iteration_history),
        ]

        analysis_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        # Combine insights from all analyses
        all_insights = []
        for result in analysis_results:
            if isinstance(result, list):
                all_insights.extend(result)
            elif isinstance(result, Exception):
                logger.warning("Analysis task failed", error=str(result))

        # Generate performance improvements
        performance_improvements = self._calculate_performance_improvements(
            session_metrics, resource_stats
        )

        # Update knowledge base
        knowledge_updates = await self._update_knowledge_base(
            all_insights, session_metrics
        )

        # Generate next priorities
        next_priorities = self._generate_next_priorities(all_insights, session_metrics)

        # Generate system adaptations
        system_adaptations = self._generate_system_adaptations(all_insights)

        report = MetaLearningReport(
            timestamp=start_time,
            session_id=self.session_id,
            insights=all_insights,
            performance_improvements=performance_improvements,
            knowledge_updates=knowledge_updates,
            next_priorities=next_priorities,
            system_adaptations=system_adaptations,
        )

        # Persist learning for future sessions
        await self._persist_learning(report)

        logger.info(
            "Meta-learning analysis completed",
            session_id=self.session_id,
            insights_count=len(all_insights),
            duration=time.time() - start_time,
        )

        return report

    async def _analyze_code_patterns(self) -> list[LearningInsight]:
        """Analyze code patterns and identify best practices."""
        insights = []

        try:
            # Analyze git commits from this session
            process = await asyncio.create_subprocess_exec(
                "git",
                "log",
                "--oneline",
                "--since=1 hour ago",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                commits = stdout.decode().strip().split("\n")
                commit_count = len([c for c in commits if c.strip()])

                if commit_count > 0:
                    # Analyze commit frequency and patterns
                    if commit_count >= 5:
                        insights.append(
                            LearningInsight(
                                type="pattern",
                                description="Frequent micro-commits pattern observed - good for safety",
                                confidence=0.8,
                                impact="medium",
                                actionable=True,
                                metadata={
                                    "commit_count": commit_count,
                                    "pattern": "micro_commits",
                                },
                            )
                        )

                    # Analyze commit messages for patterns
                    test_commits = len([c for c in commits if "test" in c.lower()])
                    if test_commits > commit_count * 0.4:  # >40% test-related commits
                        insights.append(
                            LearningInsight(
                                type="pattern",
                                description="Strong test-first development pattern observed",
                                confidence=0.9,
                                impact="high",
                                actionable=True,
                                metadata={
                                    "test_commit_ratio": test_commits / commit_count
                                },
                            )
                        )

            # Analyze code complexity trends
            complexity_insight = await self._analyze_complexity_trends()
            if complexity_insight:
                insights.append(complexity_insight)

        except Exception as e:
            logger.warning("Code pattern analysis failed", error=str(e))

        return insights

    async def _analyze_complexity_trends(self) -> LearningInsight | None:
        """Analyze code complexity trends."""
        try:
            from ..safety.quality_gates import CodeComplexityAnalyzer

            analyzer = CodeComplexityAnalyzer()
            analysis = analyzer.analyze_project(self.project_path)

            max_complexity = analysis.get("max_function_complexity", 0)
            avg_complexity = analysis.get("avg_complexity_per_file", 0)
            problematic_files = analysis.get("problematic_files", [])

            if max_complexity > 20:
                return LearningInsight(
                    type="improvement",
                    description=f"High complexity detected (max: {max_complexity}) - refactoring needed",
                    confidence=0.9,
                    impact="high",
                    actionable=True,
                    metadata={
                        "max_complexity": max_complexity,
                        "avg_complexity": avg_complexity,
                        "problematic_files_count": len(problematic_files),
                    },
                )
            elif avg_complexity < 5.0:
                return LearningInsight(
                    type="pattern",
                    description="Low average complexity maintained - good code quality",
                    confidence=0.8,
                    impact="medium",
                    actionable=False,
                    metadata={
                        "avg_complexity": avg_complexity,
                        "status": "good",
                    },
                )

        except Exception as e:
            logger.debug("Complexity analysis failed", error=str(e))

        return None

    async def _analyze_performance_trends(
        self, session_metrics: dict[str, Any], resource_stats: dict[str, Any]
    ) -> list[LearningInsight]:
        """Analyze performance trends and bottlenecks."""
        insights = []

        try:
            # Analyze memory usage trends
            avg_memory = resource_stats.get("recent_avg_memory_percent", 0)
            if avg_memory > 80:
                insights.append(
                    LearningInsight(
                        type="optimization",
                        description=f"High memory usage detected ({avg_memory:.1f}%) - optimization needed",
                        confidence=0.9,
                        impact="high",
                        actionable=True,
                        metadata={"avg_memory_percent": avg_memory, "threshold": 80},
                    )
                )

            # Analyze test performance
            test_runtime = session_metrics.get("test_performance", {}).get(
                "current_runtime", 0
            )
            if test_runtime > 60:  # More than 1 minute
                insights.append(
                    LearningInsight(
                        type="optimization",
                        description=f"Slow test suite ({test_runtime:.1f}s) - performance optimization needed",
                        confidence=0.8,
                        impact="medium",
                        actionable=True,
                        metadata={"test_runtime": test_runtime, "threshold": 60},
                    )
                )

            # Analyze development velocity
            duration_hours = session_metrics.get("duration_hours", 0)
            commits_made = session_metrics.get("metrics", {}).get("commits_made", 0)

            if duration_hours > 0:
                commits_per_hour = commits_made / duration_hours

                if commits_per_hour > 3:
                    insights.append(
                        LearningInsight(
                            type="pattern",
                            description=f"High development velocity ({commits_per_hour:.1f} commits/hour)",
                            confidence=0.8,
                            impact="high",
                            actionable=False,
                            metadata={
                                "commits_per_hour": commits_per_hour,
                                "status": "excellent",
                            },
                        )
                    )
                elif commits_per_hour < 1:
                    insights.append(
                        LearningInsight(
                            type="improvement",
                            description=f"Low development velocity ({commits_per_hour:.1f} commits/hour)",
                            confidence=0.7,
                            impact="medium",
                            actionable=True,
                            metadata={
                                "commits_per_hour": commits_per_hour,
                                "target": 2.0,
                            },
                        )
                    )

        except Exception as e:
            logger.warning("Performance analysis failed", error=str(e))

        return insights

    async def _analyze_failure_modes(
        self, rollback_stats: dict[str, Any], validation_history: list[Any]
    ) -> list[LearningInsight]:
        """Analyze failure modes and patterns."""
        insights = []

        try:
            # Analyze rollback patterns
            total_rollbacks = rollback_stats.get("total_attempts", 0)
            success_rate = rollback_stats.get("success_rate", 1.0)

            if total_rollbacks > 0:
                if success_rate < 0.8:
                    insights.append(
                        LearningInsight(
                            type="failure_mode",
                            description=f"Low rollback success rate ({success_rate:.1%}) - rollback system needs improvement",
                            confidence=0.9,
                            impact="high",
                            actionable=True,
                            metadata={
                                "success_rate": success_rate,
                                "total_rollbacks": total_rollbacks,
                            },
                        )
                    )

                # Analyze failure type distribution
                failure_distribution = rollback_stats.get(
                    "failure_type_distribution", {}
                )
                if failure_distribution:
                    most_common_failure = max(
                        failure_distribution.items(), key=lambda x: x[1]
                    )
                    failure_type, count = most_common_failure

                    if count > 1:
                        insights.append(
                            LearningInsight(
                                type="failure_mode",
                                description=f"Recurring {failure_type} failures ({count} times) - pattern needs attention",
                                confidence=0.8,
                                impact="medium",
                                actionable=True,
                                metadata={"failure_type": failure_type, "count": count},
                            )
                        )

            # Analyze validation patterns if available
            if validation_history:
                failed_validations = []
                for validation in validation_history[-5:]:  # Last 5 validations
                    if hasattr(validation, "validation_results"):
                        failed_validations.extend(
                            [
                                v.name
                                for v in validation.validation_results
                                if not v.passed
                            ]
                        )

                if failed_validations:
                    # Find most common validation failures
                    failure_counts = {}
                    for failure in failed_validations:
                        failure_counts[failure] = failure_counts.get(failure, 0) + 1

                    most_common = max(failure_counts.items(), key=lambda x: x[1])
                    validation_type, count = most_common

                    if count > 2:
                        insights.append(
                            LearningInsight(
                                type="failure_mode",
                                description=f"Recurring {validation_type} validation failures - needs investigation",
                                confidence=0.8,
                                impact="medium",
                                actionable=True,
                                metadata={
                                    "validation_type": validation_type,
                                    "failure_count": count,
                                },
                            )
                        )

        except Exception as e:
            logger.warning("Failure mode analysis failed", error=str(e))

        return insights

    async def _analyze_quality_trends(
        self, quality_stats: dict[str, Any]
    ) -> list[LearningInsight]:
        """Analyze code quality trends."""
        insights = []

        try:
            success_rate = quality_stats.get("overall_success_rate", 1.0)
            gate_success_rates = quality_stats.get("gate_success_rates", {})

            if success_rate < 0.8:
                insights.append(
                    LearningInsight(
                        type="improvement",
                        description=f"Low quality gate success rate ({success_rate:.1%}) - quality practices need improvement",
                        confidence=0.9,
                        impact="high",
                        actionable=True,
                        metadata={"success_rate": success_rate, "target": 0.9},
                    )
                )

            # Analyze individual gate performance
            for gate_name, gate_stats in gate_success_rates.items():
                gate_success_rate = gate_stats.get("success_rate", 1.0)
                if gate_success_rate < 0.7:
                    insights.append(
                        LearningInsight(
                            type="improvement",
                            description=f"Poor {gate_name} gate performance ({gate_success_rate:.1%}) - specific improvement needed",
                            confidence=0.8,
                            impact="medium",
                            actionable=True,
                            metadata={
                                "gate_name": gate_name,
                                "success_rate": gate_success_rate,
                            },
                        )
                    )

        except Exception as e:
            logger.warning("Quality trend analysis failed", error=str(e))

        return insights

    async def _analyze_development_velocity(
        self, iteration_history: list[Any]
    ) -> list[LearningInsight]:
        """Analyze development velocity and iteration effectiveness."""
        insights = []

        try:
            if not iteration_history:
                return insights

            # Calculate iteration statistics
            total_iterations = len(iteration_history)
            successful_iterations = len(
                [i for i in iteration_history if getattr(i, "status", "") == "success"]
            )

            if total_iterations > 0:
                success_rate = successful_iterations / total_iterations

                if success_rate > 0.8:
                    insights.append(
                        LearningInsight(
                            type="pattern",
                            description=f"High iteration success rate ({success_rate:.1%}) - effective development pattern",
                            confidence=0.9,
                            impact="high",
                            actionable=False,
                            metadata={
                                "success_rate": success_rate,
                                "total_iterations": total_iterations,
                            },
                        )
                    )
                elif success_rate < 0.5:
                    insights.append(
                        LearningInsight(
                            type="improvement",
                            description=f"Low iteration success rate ({success_rate:.1%}) - development process needs refinement",
                            confidence=0.8,
                            impact="high",
                            actionable=True,
                            metadata={
                                "success_rate": success_rate,
                                "total_iterations": total_iterations,
                            },
                        )
                    )

                # Analyze iteration timing
                avg_duration = (
                    sum(
                        getattr(i, "end_time", 0) - getattr(i, "start_time", 0)
                        for i in iteration_history
                    )
                    / total_iterations
                )

                if avg_duration > 35 * 60:  # More than 35 minutes
                    insights.append(
                        LearningInsight(
                            type="optimization",
                            description=f"Long iteration duration ({avg_duration / 60:.1f} min) - scope reduction needed",
                            confidence=0.7,
                            impact="medium",
                            actionable=True,
                            metadata={
                                "avg_duration_minutes": avg_duration / 60,
                                "target": 30,
                            },
                        )
                    )

        except Exception as e:
            logger.warning("Velocity analysis failed", error=str(e))

        return insights

    def _calculate_performance_improvements(
        self, session_metrics: dict[str, Any], resource_stats: dict[str, Any]
    ) -> dict[str, float]:
        """Calculate quantifiable performance improvements."""
        improvements = {}

        try:
            # Development velocity
            duration_hours = session_metrics.get("duration_hours", 0)
            commits_made = session_metrics.get("metrics", {}).get("commits_made", 0)

            if duration_hours > 0:
                improvements["commits_per_hour"] = commits_made / duration_hours

            # Quality metrics
            quality_passes = session_metrics.get("metrics", {}).get(
                "quality_gate_passes", 0
            )
            quality_failures = session_metrics.get("metrics", {}).get(
                "quality_gate_failures", 0
            )

            if quality_passes + quality_failures > 0:
                improvements["quality_success_rate"] = quality_passes / (
                    quality_passes + quality_failures
                )

            # Test metrics
            tests_passed = session_metrics.get("metrics", {}).get("tests_passed", 0)
            tests_written = session_metrics.get("metrics", {}).get(
                "tests_written", 1
            )  # Avoid division by zero

            improvements["test_success_rate"] = tests_passed / max(tests_written, 1)

            # Resource efficiency
            improvements["avg_memory_usage"] = (
                resource_stats.get("recent_avg_memory_percent", 0) / 100
            )
            improvements["avg_cpu_usage"] = (
                resource_stats.get("recent_avg_cpu_percent", 0) / 100
            )

            # Safety metrics
            rollbacks = session_metrics.get("metrics", {}).get("rollbacks_triggered", 0)
            improvements["rollback_rate"] = rollbacks / max(duration_hours, 1)

        except Exception as e:
            logger.warning("Performance calculation failed", error=str(e))

        return improvements

    async def _update_knowledge_base(
        self, insights: list[LearningInsight], session_metrics: dict[str, Any]
    ) -> list[str]:
        """Update the persistent knowledge base."""
        updates = []

        try:
            # Load existing knowledge base
            knowledge_base = {}
            if self.knowledge_base_path.exists():
                with open(self.knowledge_base_path) as f:
                    knowledge_base = json.load(f)

            # Update with new insights
            if "sessions" not in knowledge_base:
                knowledge_base["sessions"] = []

            session_summary = {
                "session_id": self.session_id,
                "timestamp": time.time(),
                "duration_hours": session_metrics.get("duration_hours", 0),
                "commits_made": session_metrics.get("metrics", {}).get(
                    "commits_made", 0
                ),
                "insights_count": len(insights),
                "key_insights": [
                    {
                        "type": insight.type,
                        "description": insight.description,
                        "confidence": insight.confidence,
                    }
                    for insight in insights
                    if insight.confidence > 0.8
                ],
            }

            knowledge_base["sessions"].append(session_summary)

            # Keep only last 50 sessions
            if len(knowledge_base["sessions"]) > 50:
                knowledge_base["sessions"] = knowledge_base["sessions"][-50:]

            # Update patterns
            if "patterns" not in knowledge_base:
                knowledge_base["patterns"] = {}

            for insight in insights:
                if insight.type == "pattern" and insight.confidence > 0.7:
                    pattern_key = insight.description[:50]  # Use first 50 chars as key
                    if pattern_key not in knowledge_base["patterns"]:
                        knowledge_base["patterns"][pattern_key] = {
                            "count": 0,
                            "first_seen": time.time(),
                            "last_seen": time.time(),
                            "confidence": insight.confidence,
                        }

                    knowledge_base["patterns"][pattern_key]["count"] += 1
                    knowledge_base["patterns"][pattern_key]["last_seen"] = time.time()
                    knowledge_base["patterns"][pattern_key]["confidence"] = max(
                        knowledge_base["patterns"][pattern_key]["confidence"],
                        insight.confidence,
                    )

            # Save updated knowledge base
            with open(self.knowledge_base_path, "w") as f:
                json.dump(knowledge_base, f, indent=2)

            updates.append(f"Updated knowledge base with {len(insights)} insights")
            updates.append(f"Session {self.session_id} added to knowledge base")

            if len(insights) > 0:
                high_confidence_insights = [i for i in insights if i.confidence > 0.8]
                updates.append(
                    f"Added {len(high_confidence_insights)} high-confidence insights"
                )

        except Exception as e:
            logger.warning("Knowledge base update failed", error=str(e))
            updates.append(f"Knowledge base update failed: {str(e)}")

        return updates

    def _generate_next_priorities(
        self, insights: list[LearningInsight], session_metrics: dict[str, Any]
    ) -> list[str]:
        """Generate prioritized list of next actions."""
        priorities = []

        # Extract actionable insights with high impact
        actionable_insights = [
            insight
            for insight in insights
            if insight.actionable and insight.impact in ["high", "medium"]
        ]

        # Sort by impact and confidence
        actionable_insights.sort(
            key=lambda x: ({"high": 3, "medium": 2, "low": 1}[x.impact], x.confidence),
            reverse=True,
        )

        # Convert top insights to priorities
        for insight in actionable_insights[:5]:  # Top 5 priorities
            if insight.type == "improvement":
                priorities.append(f"Improve: {insight.description}")
            elif insight.type == "optimization":
                priorities.append(f"Optimize: {insight.description}")
            elif insight.type == "failure_mode":
                priorities.append(f"Fix: {insight.description}")
            else:
                priorities.append(f"Address: {insight.description}")

        # Add general priorities based on session performance
        commits_made = session_metrics.get("metrics", {}).get("commits_made", 0)
        if commits_made == 0:
            priorities.append(
                "Focus on making incremental progress with regular commits"
            )

        # Default priorities if none generated
        if not priorities:
            priorities = [
                "Continue current development workflow",
                "Maintain code quality standards",
                "Monitor system performance",
            ]

        return priorities

    def _generate_system_adaptations(
        self, insights: list[LearningInsight]
    ) -> list[str]:
        """Generate system adaptations based on insights."""
        adaptations = []

        # Analyze insights for system-level adaptations
        optimization_insights = [i for i in insights if i.type == "optimization"]
        pattern_insights = [i for i in insights if i.type == "pattern"]
        failure_insights = [i for i in insights if i.type == "failure_mode"]

        # Memory optimization adaptations
        memory_issues = [
            i for i in optimization_insights if "memory" in i.description.lower()
        ]
        if memory_issues:
            adaptations.append("Increase memory cleanup frequency during iterations")
            adaptations.append("Implement more aggressive garbage collection")

        # Performance adaptations
        performance_issues = [
            i
            for i in optimization_insights
            if "performance" in i.description.lower() or "slow" in i.description.lower()
        ]
        if performance_issues:
            adaptations.append("Reduce test suite scope for faster iteration cycles")
            adaptations.append("Implement parallel test execution")

        # Quality adaptations
        quality_issues = [
            i for i in failure_insights if "quality" in i.description.lower()
        ]
        if quality_issues:
            adaptations.append("Increase quality gate strictness")
            adaptations.append("Add more comprehensive validation checks")

        # Pattern-based adaptations
        successful_patterns = [i for i in pattern_insights if i.confidence > 0.8]
        if len(successful_patterns) >= 2:
            adaptations.append("Reinforce successful development patterns")
            adaptations.append("Document and template successful workflows")

        return adaptations

    async def _persist_learning(self, report: MetaLearningReport) -> None:
        """Persist learning report for future reference."""
        try:
            reports_dir = self.knowledge_base_path.parent / "learning_reports"
            reports_dir.mkdir(exist_ok=True)

            report_file = (
                reports_dir / f"session_{self.session_id}_{int(time.time())}.json"
            )

            # Convert report to JSON-serializable format
            report_data = {
                "timestamp": report.timestamp,
                "session_id": report.session_id,
                "insights": [
                    {
                        "type": insight.type,
                        "description": insight.description,
                        "confidence": insight.confidence,
                        "impact": insight.impact,
                        "actionable": insight.actionable,
                        "metadata": insight.metadata,
                    }
                    for insight in report.insights
                ],
                "performance_improvements": report.performance_improvements,
                "knowledge_updates": report.knowledge_updates,
                "next_priorities": report.next_priorities,
                "system_adaptations": report.system_adaptations,
            }

            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)

            logger.info("Learning report persisted", report_file=str(report_file))

        except Exception as e:
            logger.warning("Failed to persist learning report", error=str(e))
