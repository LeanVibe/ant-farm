"""
Micro-development cycle implementation for ADW sessions.
Implements 30-minute TDD iterations with automatic breaks and safety checks.
"""

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class MicroIterationResult:
    """Result of a single micro-development iteration."""

    iteration_number: int
    start_time: float
    end_time: float
    status: str  # "success", "timeout", "quality_failure", "critical_failure"
    tests_written: int
    code_changes: Dict[str, Any]
    quality_score: float
    checkpoint_hash: Optional[str] = None
    error_message: Optional[str] = None


class MicroDevelopmentEngine:
    """Manages 30-minute TDD micro-iterations with automatic breaks."""

    def __init__(self, project_path: Path, rollback_system, quality_gates):
        self.project_path = project_path
        self.rollback_system = rollback_system
        self.quality_gates = quality_gates
        self.iteration_history: List[MicroIterationResult] = []

    async def run_tdd_iteration(
        self,
        iteration_number: int,
        target_feature: Optional[str] = None,
        time_limit_minutes: int = 30,
    ) -> MicroIterationResult:
        """Run a single TDD micro-iteration."""
        start_time = time.time()
        timeout_seconds = time_limit_minutes * 60

        logger.info(
            "Starting micro-iteration",
            iteration=iteration_number,
            target_feature=target_feature,
            time_limit_minutes=time_limit_minutes,
        )

        try:
            # Create iteration checkpoint
            checkpoint_hash = await self.rollback_system.create_safety_checkpoint(
                f"Micro-iteration {iteration_number} start"
            )

            # Run TDD cycle with timeout
            result = await asyncio.wait_for(
                self._execute_tdd_cycle(
                    iteration_number, target_feature, checkpoint_hash
                ),
                timeout=timeout_seconds,
            )

            # Record successful iteration
            self.iteration_history.append(result)

            logger.info(
                "Micro-iteration completed",
                iteration=iteration_number,
                status=result.status,
                duration=result.end_time - result.start_time,
            )

            return result

        except asyncio.TimeoutError:
            # Handle timeout
            result = MicroIterationResult(
                iteration_number=iteration_number,
                start_time=start_time,
                end_time=time.time(),
                status="timeout",
                tests_written=0,
                code_changes={},
                quality_score=0.0,
                error_message=f"Iteration timed out after {time_limit_minutes} minutes",
            )

            logger.warning(
                "Micro-iteration timed out",
                iteration=iteration_number,
                time_limit_minutes=time_limit_minutes,
            )

            self.iteration_history.append(result)
            return result

        except Exception as e:
            # Handle critical failure
            result = MicroIterationResult(
                iteration_number=iteration_number,
                start_time=start_time,
                end_time=time.time(),
                status="critical_failure",
                tests_written=0,
                code_changes={},
                quality_score=0.0,
                error_message=str(e),
            )

            logger.error(
                "Micro-iteration failed critically",
                iteration=iteration_number,
                error=str(e),
            )

            self.iteration_history.append(result)
            return result

    async def _execute_tdd_cycle(
        self, iteration_number: int, target_feature: Optional[str], checkpoint_hash: str
    ) -> MicroIterationResult:
        """Execute the core TDD cycle: Red → Green → Refactor."""
        start_time = time.time()
        tests_written = 0
        code_changes = {}

        # Phase 1: Red - Write failing test (5-7 minutes)
        logger.info("TDD Phase 1: Writing failing test", iteration=iteration_number)
        red_result = await self._red_phase(iteration_number, target_feature)
        tests_written += red_result.get("tests_written", 0)
        code_changes.update(red_result.get("code_changes", {}))

        # Phase 2: Green - Make test pass (15-20 minutes)
        logger.info("TDD Phase 2: Making test pass", iteration=iteration_number)
        green_result = await self._green_phase(iteration_number, target_feature)
        code_changes.update(green_result.get("code_changes", {}))

        # Phase 3: Refactor - Improve code quality (5-8 minutes)
        logger.info("TDD Phase 3: Refactoring", iteration=iteration_number)
        refactor_result = await self._refactor_phase(iteration_number)
        code_changes.update(refactor_result.get("code_changes", {}))

        # Quality validation
        quality_score = await self._validate_iteration_quality()

        # Auto-commit if quality passes
        if quality_score >= 0.7:  # 70% quality threshold
            final_checkpoint = await self.rollback_system.create_safety_checkpoint(
                f"Micro-iteration {iteration_number} completed"
            )

            result = MicroIterationResult(
                iteration_number=iteration_number,
                start_time=start_time,
                end_time=time.time(),
                status="success",
                tests_written=tests_written,
                code_changes=code_changes,
                quality_score=quality_score,
                checkpoint_hash=final_checkpoint,
            )
        else:
            # Quality failed - rollback
            await self.rollback_system.git_checkpoint.rollback_to_commit(
                checkpoint_hash
            )

            result = MicroIterationResult(
                iteration_number=iteration_number,
                start_time=start_time,
                end_time=time.time(),
                status="quality_failure",
                tests_written=tests_written,
                code_changes=code_changes,
                quality_score=quality_score,
                error_message=f"Quality score {quality_score:.2f} below threshold 0.7",
            )

        return result

    async def _red_phase(
        self, iteration_number: int, target_feature: Optional[str]
    ) -> Dict[str, Any]:
        """Red phase: Write a failing test."""
        # This would integrate with AI test generation
        # For now, we'll simulate the process

        await asyncio.sleep(0.1)  # Simulate test writing time

        # Placeholder for actual test generation logic
        test_files_created = []

        # Check if we're in a Python project and can analyze existing patterns
        try:
            test_files = list(self.project_path.rglob("test_*.py"))
            existing_tests = len(test_files)

            # Simulate creating a new test
            if target_feature:
                test_name = f"test_{target_feature.lower().replace(' ', '_')}.py"
            else:
                test_name = f"test_iteration_{iteration_number}.py"

            test_files_created.append(test_name)

        except Exception as e:
            logger.warning("Failed to analyze test patterns", error=str(e))

        return {
            "tests_written": len(test_files_created),
            "code_changes": {
                "test_files_created": test_files_created,
                "phase": "red",
            },
            "duration": 5.0,  # Estimated 5 minutes
        }

    async def _green_phase(
        self, iteration_number: int, target_feature: Optional[str]
    ) -> Dict[str, Any]:
        """Green phase: Write minimal code to make tests pass."""
        # This would integrate with AI code generation
        # For now, we'll simulate the process

        await asyncio.sleep(0.1)  # Simulate coding time

        # Placeholder for actual code generation logic
        source_files_modified = []

        try:
            # Find source files that might need modification
            source_files = list(self.project_path.rglob("src/**/*.py"))

            # Simulate modifying relevant files
            if source_files:
                # Pick one or two files to modify
                files_to_modify = source_files[: min(2, len(source_files))]
                source_files_modified = [
                    str(f.relative_to(self.project_path)) for f in files_to_modify
                ]

        except Exception as e:
            logger.warning("Failed to analyze source patterns", error=str(e))

        return {
            "code_changes": {
                "source_files_modified": source_files_modified,
                "phase": "green",
                "lines_added": 20,  # Estimated
                "lines_modified": 5,  # Estimated
            },
            "duration": 15.0,  # Estimated 15 minutes
        }

    async def _refactor_phase(self, iteration_number: int) -> Dict[str, Any]:
        """Refactor phase: Improve code quality without changing behavior."""
        # This would integrate with automated refactoring tools
        # For now, we'll simulate the process

        await asyncio.sleep(0.1)  # Simulate refactoring time

        refactoring_actions = []

        try:
            # Simulate common refactoring actions
            refactoring_actions = [
                "extract_method",
                "improve_variable_names",
                "remove_code_duplication",
                "optimize_imports",
            ]

        except Exception as e:
            logger.warning("Failed to perform refactoring", error=str(e))

        return {
            "code_changes": {
                "refactoring_actions": refactoring_actions,
                "phase": "refactor",
                "complexity_improvement": 0.1,  # Estimated improvement
            },
            "duration": 7.0,  # Estimated 7 minutes
        }

    async def _validate_iteration_quality(self) -> float:
        """Validate the quality of the iteration."""
        try:
            # Run quality gates
            quality_results = await self.quality_gates.run_all_gates()

            if not quality_results:
                return 0.5  # Default score if no gates run

            # Calculate overall quality score
            total_score = sum(result.score for result in quality_results)
            avg_score = total_score / len(quality_results)

            # Bonus for all gates passing
            all_passed = all(result.passed for result in quality_results)
            if all_passed:
                avg_score = min(1.0, avg_score * 1.1)  # 10% bonus, capped at 1.0

            return avg_score

        except Exception as e:
            logger.error("Failed to validate iteration quality", error=str(e))
            return 0.0

    async def enforce_test_first_development(self) -> bool:
        """Enforce that tests are written before implementation code."""
        try:
            # Check git diff to see if tests were added before implementation
            import subprocess

            # Get staged changes
            process = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "--cached",
                "--name-only",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                changed_files = stdout.decode().strip().split("\n")
                test_files = [f for f in changed_files if "test" in f.lower()]
                source_files = [
                    f
                    for f in changed_files
                    if f.startswith("src/") and "test" not in f.lower()
                ]

                # Test-first: tests should be present when source files are changed
                if source_files and not test_files:
                    logger.warning(
                        "Test-first violation: source files changed without corresponding tests",
                        source_files=source_files,
                    )
                    return False

                return True

        except Exception as e:
            logger.error("Failed to enforce test-first development", error=str(e))

        return True  # Default to allowing if check fails

    async def take_micro_break(self, duration_minutes: int = 5) -> None:
        """Take a micro-break between iterations."""
        logger.info(f"Taking {duration_minutes}-minute micro-break")

        # Perform maintenance tasks during break
        tasks = [
            self._cleanup_temporary_files(),
            self._optimize_memory_usage(),
            self._update_progress_metrics(),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # Wait for remaining break time
        await asyncio.sleep(duration_minutes * 60)

        logger.info("Micro-break completed")

    async def _cleanup_temporary_files(self) -> None:
        """Clean up temporary files during break."""
        try:
            import shutil

            # Common temporary file patterns
            temp_patterns = ["*.tmp", "*.temp", "*.pyc", "__pycache__"]

            for pattern in temp_patterns:
                temp_files = list(self.project_path.rglob(pattern))
                for temp_file in temp_files:
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                    except Exception:
                        pass  # Ignore errors for individual files

        except Exception as e:
            logger.debug("Minor cleanup error during break", error=str(e))

    async def _optimize_memory_usage(self) -> None:
        """Optimize memory usage during break."""
        try:
            import gc

            gc.collect()

        except Exception as e:
            logger.debug("Minor memory optimization error during break", error=str(e))

    async def _update_progress_metrics(self) -> None:
        """Update progress metrics during break."""
        try:
            total_iterations = len(self.iteration_history)
            successful_iterations = len(
                [i for i in self.iteration_history if i.status == "success"]
            )

            if total_iterations > 0:
                success_rate = successful_iterations / total_iterations
                avg_quality_score = (
                    sum(i.quality_score for i in self.iteration_history)
                    / total_iterations
                )

                logger.info(
                    "Iteration progress update",
                    total_iterations=total_iterations,
                    success_rate=success_rate,
                    avg_quality_score=avg_quality_score,
                )

        except Exception as e:
            logger.debug("Minor progress update error during break", error=str(e))

    def get_iteration_statistics(self) -> Dict[str, Any]:
        """Get statistics for all completed iterations."""
        if not self.iteration_history:
            return {"total_iterations": 0}

        total_iterations = len(self.iteration_history)
        successful_iterations = len(
            [i for i in self.iteration_history if i.status == "success"]
        )
        failed_iterations = len(
            [
                i
                for i in self.iteration_history
                if i.status in ["quality_failure", "critical_failure"]
            ]
        )
        timeout_iterations = len(
            [i for i in self.iteration_history if i.status == "timeout"]
        )

        total_tests_written = sum(i.tests_written for i in self.iteration_history)
        avg_quality_score = (
            sum(i.quality_score for i in self.iteration_history) / total_iterations
        )
        avg_duration = (
            sum(i.end_time - i.start_time for i in self.iteration_history)
            / total_iterations
        )

        return {
            "total_iterations": total_iterations,
            "successful_iterations": successful_iterations,
            "failed_iterations": failed_iterations,
            "timeout_iterations": timeout_iterations,
            "success_rate": successful_iterations / total_iterations,
            "total_tests_written": total_tests_written,
            "avg_quality_score": avg_quality_score,
            "avg_duration_minutes": avg_duration / 60,
            "recent_iterations": self.iteration_history[-5:]
            if len(self.iteration_history) > 5
            else self.iteration_history,
        }
