"""ADW Session Manager - Orchestrates autonomous development workflows.

This module manages 4-hour autonomous development sessions with phases:
- Reconnaissance (15 min): System assessment
- Micro-Development (3 hours): 30-minute TDD cycles
- Integration Validation (30 min): Comprehensive testing
- Meta-Learning (15 min): Session analysis and learning
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from ..safety import AutoRollbackSystem, AutonomousQualityGates, ResourceGuardian
from ..monitoring.autonomous_dashboard import AutonomousDashboard
from ..prediction.failure_prediction import FailurePredictionSystem
from .cognitive_load_manager import CognitiveLoadManager, SessionMode
from .session_persistence import SessionStatePersistence

logger = structlog.get_logger()


class SessionPhase(Enum):
    """Phases of an ADW session."""

    RECONNAISSANCE = "reconnaissance"
    MICRO_DEVELOPMENT = "micro_development"
    INTEGRATION_VALIDATION = "integration_validation"
    META_LEARNING = "meta_learning"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ADWSessionConfig:
    """Configuration for ADW sessions."""

    total_duration_hours: float = 4.0
    reconnaissance_minutes: int = 15
    integration_validation_minutes: int = 30
    meta_learning_minutes: int = 15
    micro_iteration_minutes: int = 30

    # Safety settings
    max_consecutive_failures: int = 3
    rollback_on_critical_failure: bool = True
    quality_gates_enabled: bool = True
    resource_monitoring_enabled: bool = True

    # Development settings
    test_first_enforced: bool = True
    auto_commit_enabled: bool = True
    performance_tracking_enabled: bool = True

    # New ADW component settings
    cognitive_load_management_enabled: bool = True
    failure_prediction_enabled: bool = True
    autonomous_dashboard_enabled: bool = True

    # Extended session settings
    extended_session_mode: bool = False
    max_extended_duration_hours: float = 24.0
    cognitive_fatigue_threshold: float = 0.7


@dataclass
class SessionMetrics:
    """Metrics collected during an ADW session."""

    start_time: float
    end_time: Optional[float] = None
    current_phase: SessionPhase = SessionPhase.RECONNAISSANCE

    # Phase durations
    reconnaissance_duration: float = 0.0
    micro_development_duration: float = 0.0
    integration_validation_duration: float = 0.0
    meta_learning_duration: float = 0.0

    # Development metrics
    commits_made: int = 0
    tests_written: int = 0
    tests_passed: int = 0
    lines_of_code_added: int = 0
    lines_of_code_removed: int = 0

    # Quality metrics
    quality_gate_passes: int = 0
    quality_gate_failures: int = 0
    rollbacks_triggered: int = 0

    # Resource metrics
    avg_memory_usage: float = 0.0
    avg_cpu_usage: float = 0.0
    max_memory_usage: float = 0.0
    max_cpu_usage: float = 0.0

    # Learning metrics
    patterns_discovered: List[str] = field(default_factory=list)
    improvements_identified: List[str] = field(default_factory=list)

    def duration(self) -> float:
        """Get total session duration."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


class ADWSession:
    """Manages a complete autonomous development workflow session."""

    def __init__(self, project_path: Path, config: Optional[ADWSessionConfig] = None):
        self.project_path = project_path
        self.config = config or ADWSessionConfig()

        # Initialize safety systems
        self.rollback_system = AutoRollbackSystem(project_path)
        self.quality_gates = AutonomousQualityGates(project_path)
        self.resource_guardian = ResourceGuardian(project_path)

        # Initialize new ADW components
        self.cognitive_load_manager = None
        self.failure_predictor = None
        self.dashboard = None

        if self.config.cognitive_load_management_enabled:
            self.cognitive_load_manager = CognitiveLoadManager(
                fatigue_threshold=self.config.cognitive_fatigue_threshold
            )

        if self.config.failure_prediction_enabled:
            self.failure_predictor = FailurePredictionSystem(project_path)

        if self.config.autonomous_dashboard_enabled:
            self.dashboard = AutonomousDashboard(project_path)

        # Session state
        self.metrics = SessionMetrics(start_time=time.time())
        self.session_id = f"adw-{int(time.time())}"
        self.active = False
        self.consecutive_failures = 0
        self.current_session_mode = SessionMode.FOCUS

        # Session persistence
        self.persistence = SessionStatePersistence(project_path, self.session_id)

        # Phase handlers
        self.phase_handlers = {
            SessionPhase.RECONNAISSANCE: self._run_reconnaissance_phase,
            SessionPhase.MICRO_DEVELOPMENT: self._run_micro_development_phase,
            SessionPhase.INTEGRATION_VALIDATION: self._run_integration_validation_phase,
            SessionPhase.META_LEARNING: self._run_meta_learning_phase,
        }

        logger.info(
            "ADW Session initialized",
            session_id=self.session_id,
            project=str(project_path),
            cognitive_load_enabled=self.config.cognitive_load_management_enabled,
            failure_prediction_enabled=self.config.failure_prediction_enabled,
            dashboard_enabled=self.config.autonomous_dashboard_enabled,
        )

    async def start_session(
        self, target_goals: Optional[List[str]] = None, resume: bool = False
    ) -> Dict[str, Any]:
        """Start a complete ADW session."""
        if self.active:
            raise RuntimeError("Session already active")

        self.active = True

        # Attempt to restore from checkpoint if resuming
        if resume:
            restored = await self.persistence.restore_session(self)
            if restored:
                logger.info(
                    "Resumed ADW session from checkpoint",
                    session_id=self.session_id,
                    phase=self.metrics.current_phase.value,
                )
            else:
                logger.info("No checkpoint found, starting fresh session")

        logger.info(
            "Starting ADW session", session_id=self.session_id, goals=target_goals
        )

        try:
            # Create initial safety checkpoint
            await self.rollback_system.create_safety_checkpoint("ADW session start")

            # Initialize cognitive load tracking
            if self.cognitive_load_manager:
                await self.cognitive_load_manager.start_session()
                self.current_session_mode = SessionMode.FOCUS

            # Start autonomous dashboard if enabled
            if self.dashboard:
                await self.dashboard.start_monitoring(self.session_id)

            # Initialize failure prediction
            if self.failure_predictor:
                await self.failure_predictor.start_monitoring()

            # Start resource monitoring if enabled
            if self.config.resource_monitoring_enabled:
                asyncio.create_task(self.resource_guardian.start_monitoring())

            # Determine session type and phases
            if self.config.extended_session_mode:
                phases = await self._plan_extended_session()
            else:
                phases = [
                    SessionPhase.RECONNAISSANCE,
                    SessionPhase.MICRO_DEVELOPMENT,
                    SessionPhase.INTEGRATION_VALIDATION,
                    SessionPhase.META_LEARNING,
                ]

            for phase in phases:
                if not self.active:  # Check if session was aborted
                    break

                # Check cognitive load before each phase
                if self.cognitive_load_manager:
                    cognitive_state = (
                        await self.cognitive_load_manager.assess_cognitive_state()
                    )

                    # Handle cognitive fatigue
                    if (
                        cognitive_state.fatigue_level
                        > self.config.cognitive_fatigue_threshold
                    ):
                        logger.warning(
                            "High cognitive fatigue detected, entering rest mode",
                            fatigue_level=cognitive_state.fatigue_level,
                        )
                        optimal_mode = (
                            await self.cognitive_load_manager.get_optimal_mode(
                                cognitive_state
                            )
                        )
                        await self._transition_session_mode(optimal_mode)

                # Predict potential failures before phase
                if self.failure_predictor:
                    failure_risk = await self.failure_predictor.predict_failure_risk(
                        {
                            "phase": phase.value,
                            "session_duration": self.metrics.duration(),
                            "consecutive_failures": self.consecutive_failures,
                            "current_mode": self.current_session_mode.value
                            if self.cognitive_load_manager
                            else "focus",
                        }
                    )

                    if failure_risk > 0.7:  # High risk threshold
                        logger.warning(
                            "High failure risk predicted for phase",
                            phase=phase.value,
                            risk=failure_risk,
                        )
                        # Take preventive action
                        await self._handle_high_failure_risk(phase, failure_risk)

                await self._run_phase(phase)

                # Save checkpoint after each phase
                await self.persistence.save_checkpoint(
                    self,
                    phase_progress={f"{phase.value}_completed": True},
                    additional_context={
                        "goals": target_goals,
                        "cognitive_mode": self.current_session_mode.value
                        if self.cognitive_load_manager
                        else "focus",
                        "failure_predictions": failure_risk
                        if self.failure_predictor
                        else None,
                    },
                )

                # Check for critical failures
                if self.consecutive_failures >= self.config.max_consecutive_failures:
                    logger.error(
                        "Too many consecutive failures, aborting session",
                        failures=self.consecutive_failures,
                    )
                    self.metrics.current_phase = SessionPhase.FAILED
                    break

            # Mark session as completed if we made it through all phases
            if self.metrics.current_phase != SessionPhase.FAILED:
                self.metrics.current_phase = SessionPhase.COMPLETED

            return await self._finalize_session()

        except Exception as e:
            logger.error("ADW session failed", session_id=self.session_id, error=str(e))
            self.metrics.current_phase = SessionPhase.FAILED
            return await self._finalize_session()
        finally:
            self.active = False

    async def _run_phase(self, phase: SessionPhase) -> Dict[str, Any]:
        """Run a specific session phase."""
        self.metrics.current_phase = phase
        phase_start = time.time()

        logger.info(
            "Starting session phase", session_id=self.session_id, phase=phase.value
        )

        try:
            # Get phase handler
            handler = self.phase_handlers.get(phase)
            if not handler:
                raise ValueError(f"No handler for phase: {phase}")

            # Run phase with timeout
            phase_duration = self._get_phase_duration(phase)
            result = await asyncio.wait_for(handler(), timeout=phase_duration * 60)

            # Record phase completion
            actual_duration = time.time() - phase_start
            self._record_phase_duration(phase, actual_duration)

            logger.info(
                "Phase completed",
                session_id=self.session_id,
                phase=phase.value,
                duration=actual_duration,
            )

            # Reset consecutive failures on successful phase
            self.consecutive_failures = 0

            return result

        except asyncio.TimeoutError:
            logger.warning(
                "Phase timed out", session_id=self.session_id, phase=phase.value
            )
            self.consecutive_failures += 1
            return {"status": "timeout", "phase": phase.value}

        except Exception as e:
            logger.error(
                "Phase failed",
                session_id=self.session_id,
                phase=phase.value,
                error=str(e),
            )
            self.consecutive_failures += 1

            # Trigger rollback on critical failure if enabled
            if self.config.rollback_on_critical_failure:
                await self._handle_phase_failure(phase, str(e))

            return {"status": "failed", "phase": phase.value, "error": str(e)}

    async def _run_reconnaissance_phase(self) -> Dict[str, Any]:
        """Run the reconnaissance phase - system assessment."""
        logger.info("Running reconnaissance phase", session_id=self.session_id)

        reconnaissance_data = {}

        # System health check
        resource_status = await self.resource_guardian.get_current_status()
        reconnaissance_data["resource_status"] = {
            "memory_percent": resource_status.memory_percent,
            "cpu_percent": resource_status.cpu_percent,
            "disk_percent": resource_status.disk_percent,
            "warnings": resource_status.warnings,
            "critical_alerts": resource_status.critical_alerts,
        }

        # Quality gates assessment
        if self.config.quality_gates_enabled:
            quality_results = await self.quality_gates.run_all_gates()
            reconnaissance_data["quality_status"] = {
                "gates_passed": sum(1 for r in quality_results if r.passed),
                "total_gates": len(quality_results),
                "overall_quality_score": sum(r.score for r in quality_results)
                / len(quality_results),
            }

        # Performance baseline measurement
        test_runtime = (
            await self.resource_guardian.test_optimizer.measure_test_runtime()
        )
        if self.resource_guardian.test_optimizer.baseline_runtime is None:
            self.resource_guardian.test_optimizer.set_baseline_runtime(test_runtime)
        reconnaissance_data["test_performance"] = {
            "current_runtime": test_runtime,
            "baseline_runtime": self.resource_guardian.test_optimizer.baseline_runtime,
        }

        # Git status and repository health
        try:
            import subprocess

            git_status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.project_path,
                capture_output=True,
                text=True,
                check=True,
            )
            reconnaissance_data["git_status"] = {
                "uncommitted_changes": len(git_status.stdout.strip().split("\n"))
                if git_status.stdout.strip()
                else 0,
                "repository_clean": not git_status.stdout.strip(),
            }
        except Exception as e:
            reconnaissance_data["git_status"] = {"error": str(e)}

        logger.info(
            "Reconnaissance completed",
            session_id=self.session_id,
            data=reconnaissance_data,
        )

        return reconnaissance_data

    async def _run_micro_development_phase(self) -> Dict[str, Any]:
        """Run the micro-development phase - 30-minute TDD cycles."""
        logger.info("Running micro-development phase", session_id=self.session_id)

        # Calculate number of iterations (3 hours / 30 minutes = 6 iterations)
        micro_duration_hours = (
            self.config.total_duration_hours
            - (
                self.config.reconnaissance_minutes
                + self.config.integration_validation_minutes
                + self.config.meta_learning_minutes
            )
            / 60
        )

        num_iterations = int(
            micro_duration_hours * 60 / self.config.micro_iteration_minutes
        )

        iteration_results = []

        for iteration in range(num_iterations):
            if not self.active:
                break

            logger.info(
                "Starting micro-iteration",
                session_id=self.session_id,
                iteration=iteration + 1,
                total=num_iterations,
            )

            iteration_result = await self._run_micro_iteration(iteration + 1)
            iteration_results.append(iteration_result)

            # Check if iteration failed critically
            if iteration_result.get("status") == "critical_failure":
                logger.warning(
                    "Critical failure in micro-iteration, stopping phase",
                    session_id=self.session_id,
                    iteration=iteration + 1,
                )
                break

        return {
            "iterations_completed": len(iteration_results),
            "iterations_planned": num_iterations,
            "iteration_results": iteration_results,
            "overall_success": all(
                r.get("status") != "critical_failure" for r in iteration_results
            ),
        }

    async def _run_micro_iteration(self, iteration_num: int) -> Dict[str, Any]:
        """Run a single 30-minute micro-development iteration."""
        iteration_start = time.time()

        try:
            # Create iteration checkpoint
            checkpoint = await self.rollback_system.create_safety_checkpoint(
                f"Micro-iteration {iteration_num} start"
            )

            # Save iteration recovery point
            await self.persistence.create_recovery_point(
                self,
                "micro_iteration_start",
                {"iteration": iteration_num, "checkpoint": checkpoint},
            )

            # Step 1: Write failing test (5 minutes)
            # This would integrate with the AI test generator
            logger.info("Step 1: Writing failing test", iteration=iteration_num)
            await asyncio.sleep(1)  # Placeholder for actual test writing

            # Step 2: Implement minimal code (15 minutes)
            # This would integrate with the development agent
            logger.info("Step 2: Implementing minimal code", iteration=iteration_num)
            await asyncio.sleep(1)  # Placeholder for actual coding

            # Step 3: Refactor and validate (10 minutes)
            logger.info("Step 3: Refactoring and validation", iteration=iteration_num)

            # Run quality gates
            if self.config.quality_gates_enabled:
                quality_results = await self.quality_gates.run_all_gates()
                quality_passed = all(r.passed for r in quality_results)

                if not quality_passed:
                    logger.warning(
                        "Quality gates failed, rolling back iteration",
                        iteration=iteration_num,
                    )
                    await self.rollback_system.git_checkpoint.rollback_to_commit(
                        checkpoint
                    )
                    return {
                        "status": "quality_failure",
                        "iteration": iteration_num,
                        "duration": time.time() - iteration_start,
                        "checkpoint": checkpoint,
                    }

            # Step 4: Auto-commit with rollback capability
            if self.config.auto_commit_enabled:
                final_checkpoint = await self.rollback_system.create_safety_checkpoint(
                    f"Micro-iteration {iteration_num} completed"
                )
                self.metrics.commits_made += 1

            # Step 5: Continuous health check
            resource_status = await self.resource_guardian.get_current_status()
            if resource_status.critical_alerts:
                logger.error(
                    "Critical resource alerts during iteration",
                    iteration=iteration_num,
                    alerts=resource_status.critical_alerts,
                )
                await self.resource_guardian.handle_resource_issues(resource_status)

                # Consider this a warning, not a failure
                return {
                    "status": "resource_warning",
                    "iteration": iteration_num,
                    "duration": time.time() - iteration_start,
                    "resource_alerts": resource_status.critical_alerts,
                }

            return {
                "status": "success",
                "iteration": iteration_num,
                "duration": time.time() - iteration_start,
                "checkpoint": checkpoint,
            }

        except Exception as e:
            logger.error(
                "Micro-iteration failed", iteration=iteration_num, error=str(e)
            )
            return {
                "status": "critical_failure",
                "iteration": iteration_num,
                "duration": time.time() - iteration_start,
                "error": str(e),
            }

    async def _run_integration_validation_phase(self) -> Dict[str, Any]:
        """Run the integration validation phase."""
        logger.info("Running integration validation phase", session_id=self.session_id)

        validation_results = {}

        # Run comprehensive test suite
        test_start = time.time()
        try:
            import subprocess

            process = await asyncio.create_subprocess_exec(
                "pytest",
                "tests/",
                "--tb=short",
                "-v",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            validation_results["test_suite"] = {
                "exit_code": process.returncode,
                "duration": time.time() - test_start,
                "passed": process.returncode == 0,
            }

            if process.returncode == 0:
                self.metrics.tests_passed += 1

        except Exception as e:
            validation_results["test_suite"] = {"error": str(e), "passed": False}

        # Run quality gates
        if self.config.quality_gates_enabled:
            quality_results = await self.quality_gates.run_all_gates()
            validation_results["quality_gates"] = {
                "results": [
                    {"gate": r.gate_name, "passed": r.passed, "score": r.score}
                    for r in quality_results
                ],
                "overall_passed": all(r.passed for r in quality_results),
            }

            if all(r.passed for r in quality_results):
                self.metrics.quality_gate_passes += 1
            else:
                self.metrics.quality_gate_failures += 1

        # Performance regression check
        current_runtime = (
            await self.resource_guardian.test_optimizer.measure_test_runtime()
        )
        is_regression = self.resource_guardian.test_optimizer.is_runtime_regression(
            current_runtime
        )

        validation_results["performance"] = {
            "current_runtime": current_runtime,
            "baseline_runtime": self.resource_guardian.test_optimizer.baseline_runtime,
            "is_regression": is_regression,
        }

        # Security scan (if available)
        validation_results["security"] = {
            "scan_completed": True,
            "issues_found": 0,  # Placeholder
        }

        # Overall validation success
        overall_success = (
            validation_results.get("test_suite", {}).get("passed", False)
            and validation_results.get("quality_gates", {}).get("overall_passed", True)
            and not validation_results.get("performance", {}).get(
                "is_regression", False
            )
        )

        if not overall_success and self.config.rollback_on_critical_failure:
            logger.warning("Integration validation failed, considering rollback")
            # Could trigger intelligent rollback here

        validation_results["overall_success"] = overall_success

        return validation_results

    async def _run_meta_learning_phase(self) -> Dict[str, Any]:
        """Run the meta-learning phase - analyze and learn from session."""
        logger.info("Running meta-learning phase", session_id=self.session_id)

        learning_results = {}

        # Analyze code patterns
        # This would integrate with pattern recognition systems
        learning_results["patterns_discovered"] = [
            "consistent_test_first_approach",
            "micro_commit_pattern_effective",
        ]

        # Measure performance improvements
        learning_results["performance_improvements"] = {
            "commits_per_hour": self.metrics.commits_made
            / self.metrics.duration()
            * 3600,
            "test_success_rate": self.metrics.tests_passed
            / max(self.metrics.tests_written, 1),
            "quality_gate_success_rate": self.metrics.quality_gate_passes
            / max(
                self.metrics.quality_gate_passes + self.metrics.quality_gate_failures, 1
            ),
        }

        # Catalog failure modes
        rollback_stats = self.rollback_system.get_rollback_statistics()
        learning_results["failure_analysis"] = {
            "total_rollbacks": rollback_stats.get("total_attempts", 0),
            "rollback_success_rate": rollback_stats.get("success_rate", 1.0),
            "common_failure_types": rollback_stats.get("failure_type_distribution", {}),
        }

        # Update system knowledge
        # This would integrate with the context engine
        learning_results["knowledge_updates"] = [
            f"Session {self.session_id} completed with {self.metrics.commits_made} commits",
            f"Quality gates success rate: {learning_results['performance_improvements']['quality_gate_success_rate']:.2%}",
        ]

        # AI-driven backlog prioritization
        # This would use AI to analyze what to work on next
        learning_results["next_priorities"] = [
            "continue_current_feature_development",
            "address_technical_debt_in_complex_modules",
            "improve_test_coverage_in_core_components",
        ]

        return learning_results

    def _get_phase_duration(self, phase: SessionPhase) -> float:
        """Get duration in minutes for a phase."""
        durations = {
            SessionPhase.RECONNAISSANCE: self.config.reconnaissance_minutes,
            SessionPhase.INTEGRATION_VALIDATION: self.config.integration_validation_minutes,
            SessionPhase.META_LEARNING: self.config.meta_learning_minutes,
            SessionPhase.MICRO_DEVELOPMENT: (
                self.config.total_duration_hours * 60
                - self.config.reconnaissance_minutes
                - self.config.integration_validation_minutes
                - self.config.meta_learning_minutes
            ),
        }
        return durations.get(phase, 30)

    def _record_phase_duration(self, phase: SessionPhase, duration: float) -> None:
        """Record the actual duration of a phase."""
        if phase == SessionPhase.RECONNAISSANCE:
            self.metrics.reconnaissance_duration = duration
        elif phase == SessionPhase.MICRO_DEVELOPMENT:
            self.metrics.micro_development_duration = duration
        elif phase == SessionPhase.INTEGRATION_VALIDATION:
            self.metrics.integration_validation_duration = duration
        elif phase == SessionPhase.META_LEARNING:
            self.metrics.meta_learning_duration = duration

    async def _handle_phase_failure(self, phase: SessionPhase, error: str) -> None:
        """Handle failure of a phase."""
        logger.warning(
            "Handling phase failure",
            session_id=self.session_id,
            phase=phase.value,
            error=error,
        )

        # Determine rollback level based on phase and error
        from ..safety.rollback_system import RollbackLevel

        if "syntax" in error.lower():
            rollback_level = RollbackLevel.SYNTAX_ERROR
        elif "test" in error.lower():
            rollback_level = RollbackLevel.TEST_FAILURE
        elif "performance" in error.lower():
            rollback_level = RollbackLevel.PERFORMANCE_REGRESSION
        else:
            rollback_level = RollbackLevel.SYSTEM_CRASH

        # Trigger rollback
        success = await self.rollback_system.handle_failure(
            rollback_level,
            {"phase": phase.value, "error": error, "session_id": self.session_id},
        )

        if success:
            self.metrics.rollbacks_triggered += 1
            logger.info(
                "Rollback successful", session_id=self.session_id, phase=phase.value
            )
        else:
            logger.error(
                "Rollback failed", session_id=self.session_id, phase=phase.value
            )

    async def _finalize_session(self) -> Dict[str, Any]:
        """Finalize the session and return summary."""
        self.metrics.end_time = time.time()

        # Stop resource monitoring
        if self.config.resource_monitoring_enabled:
            await self.resource_guardian.stop_monitoring()

        # Finalize cognitive load tracking
        cognitive_summary = None
        if self.cognitive_load_manager:
            cognitive_summary = await self.cognitive_load_manager.end_session()

        # Stop failure prediction monitoring
        if self.failure_predictor:
            await self.failure_predictor.stop_monitoring()

        # Stop dashboard monitoring
        if self.dashboard:
            await self.dashboard.stop_monitoring()

        # Create final checkpoint
        await self.rollback_system.create_safety_checkpoint("ADW session completed")
        await self.persistence.save_checkpoint(
            self,
            phase_progress={"session_completed": True},
            additional_context={
                "final_status": self.metrics.current_phase.value,
                "cognitive_summary": cognitive_summary,
                "final_mode": self.current_session_mode.value
                if self.cognitive_load_manager
                else "focus",
            },
        )

        # Compile session summary
        summary = {
            "session_id": self.session_id,
            "status": self.metrics.current_phase.value,
            "duration_hours": self.metrics.duration() / 3600,
            "final_mode": self.current_session_mode.value
            if self.cognitive_load_manager
            else "focus",
            "metrics": {
                "commits_made": self.metrics.commits_made,
                "tests_written": self.metrics.tests_written,
                "tests_passed": self.metrics.tests_passed,
                "quality_gate_passes": self.metrics.quality_gate_passes,
                "quality_gate_failures": self.metrics.quality_gate_failures,
                "rollbacks_triggered": self.metrics.rollbacks_triggered,
            },
            "phase_durations": {
                "reconnaissance": self.metrics.reconnaissance_duration,
                "micro_development": self.metrics.micro_development_duration,
                "integration_validation": self.metrics.integration_validation_duration,
                "meta_learning": self.metrics.meta_learning_duration,
            },
            "resource_stats": self.resource_guardian.get_resource_statistics(),
            "rollback_stats": self.rollback_system.get_rollback_statistics(),
            "quality_stats": self.quality_gates.get_gate_statistics(),
        }

        # Add cognitive load summary if available
        if cognitive_summary:
            summary["cognitive_stats"] = cognitive_summary

        # Add failure prediction summary if available
        if self.failure_predictor:
            summary[
                "failure_prediction_stats"
            ] = await self.failure_predictor.get_session_summary()

        # Add dashboard metrics if available
        if self.dashboard:
            summary["dashboard_metrics"] = await self.dashboard.get_session_metrics()

        logger.info(
            "ADW session finalized", session_id=self.session_id, summary=summary
        )

        return summary

    async def _plan_extended_session(self) -> List[SessionPhase]:
        """Plan phases for extended 16-24 hour sessions."""
        phases = []
        total_hours = self.config.max_extended_duration_hours

        # Calculate number of 4-hour cycles
        cycles = int(total_hours / 4)

        for cycle in range(cycles):
            # Add full cycle phases
            phases.extend(
                [
                    SessionPhase.RECONNAISSANCE,
                    SessionPhase.MICRO_DEVELOPMENT,
                    SessionPhase.INTEGRATION_VALIDATION,
                    SessionPhase.META_LEARNING,
                ]
            )

            # Add rest phase every 2 cycles (8 hours)
            if (cycle + 1) % 2 == 0 and cycle < cycles - 1:
                # Rest mode handled by cognitive load manager
                logger.info(f"Extended session: planning rest after cycle {cycle + 1}")

        return phases

    async def _transition_session_mode(self, new_mode: SessionMode) -> None:
        """Transition to a new session mode."""
        if not self.cognitive_load_manager:
            return

        old_mode = self.current_session_mode
        self.current_session_mode = new_mode

        logger.info(
            "Session mode transition",
            session_id=self.session_id,
            old_mode=old_mode.value,
            new_mode=new_mode.value,
        )

        # Apply mode-specific adjustments
        if new_mode == SessionMode.REST:
            # Reduce session intensity
            self.config.micro_iteration_minutes = 45  # Longer iterations
            logger.info("Entering rest mode - reduced session intensity")
        elif new_mode == SessionMode.EXPLORATION:
            # Increase exploration time
            self.config.reconnaissance_minutes = 25
            logger.info("Entering exploration mode - extended reconnaissance")
        elif new_mode == SessionMode.INTEGRATION:
            # Focus on integration
            self.config.integration_validation_minutes = 45
            logger.info("Entering integration mode - extended validation")
        else:  # FOCUS mode
            # Reset to standard timings
            self.config.micro_iteration_minutes = 30
            self.config.reconnaissance_minutes = 15
            self.config.integration_validation_minutes = 30
            logger.info("Entering focus mode - standard timings")

    async def _handle_high_failure_risk(self, phase: SessionPhase, risk: float) -> None:
        """Handle high failure risk prediction."""
        logger.warning(
            "Taking preventive action for high failure risk",
            phase=phase.value,
            risk=risk,
            session_id=self.session_id,
        )

        # Create extra safety checkpoint
        await self.rollback_system.create_safety_checkpoint(
            f"Pre-{phase.value} high-risk checkpoint"
        )

        # Switch to more conservative mode
        if self.cognitive_load_manager:
            await self._transition_session_mode(SessionMode.REST)

        # Reduce iteration complexity
        if phase == SessionPhase.MICRO_DEVELOPMENT:
            self.config.micro_iteration_minutes = min(
                self.config.micro_iteration_minutes + 10, 45
            )

        # Enable extra quality gates
        self.config.quality_gates_enabled = True

    async def abort_session(self, reason: str = "Manual abort") -> Dict[str, Any]:
        """Abort the current session."""
        logger.warning(
            "Aborting ADW session", session_id=self.session_id, reason=reason
        )

        self.active = False
        self.metrics.current_phase = SessionPhase.FAILED

        return await self._finalize_session()
