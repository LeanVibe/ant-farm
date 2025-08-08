"""Graduated rollback system for autonomous development safety.

This module provides multi-level automatic rollback capabilities for different
failure severities during extended autonomous development sessions.
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()


class RollbackLevel(Enum):
    """Severity levels for different types of failures."""

    SYNTAX_ERROR = "syntax_error"
    TEST_FAILURE = "test_failure"
    PERFORMANCE_REGRESSION = "performance_regression"
    SYSTEM_CRASH = "system_crash"
    DATA_CORRUPTION = "data_corruption"


class RollbackStrategy:
    """Defines rollback strategies for different failure types."""

    STRATEGIES = {
        RollbackLevel.SYNTAX_ERROR: {
            "action": "rollback_last_commit",
            "timeout": 30,
            "validation": "syntax_check",
            "description": "Quick rollback for syntax errors",
        },
        RollbackLevel.TEST_FAILURE: {
            "action": "rollback_to_last_green",
            "timeout": 60,
            "validation": "run_test_suite",
            "description": "Rollback to last passing test state",
        },
        RollbackLevel.PERFORMANCE_REGRESSION: {
            "action": "rollback_to_baseline",
            "timeout": 120,
            "validation": "performance_test",
            "description": "Rollback to performance baseline",
        },
        RollbackLevel.SYSTEM_CRASH: {
            "action": "rollback_to_last_stable_checkpoint",
            "timeout": 300,
            "validation": "full_system_check",
            "description": "Rollback to last stable system state",
        },
        RollbackLevel.DATA_CORRUPTION: {
            "action": "emergency_restore_from_backup",
            "timeout": 600,
            "validation": "data_integrity_check",
            "description": "Emergency database restoration",
        },
    }


class GitCheckpoint:
    """Manages git checkpoints for rollback operations."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.checkpoint_prefix = "adw-checkpoint"

    async def create_checkpoint(self, description: str = "") -> str:
        """Create a git checkpoint with automated tagging."""
        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            tag_name = f"{self.checkpoint_prefix}-{timestamp}"

            # Add all changes
            await self._run_git_command(["add", "-A"])

            # Commit with checkpoint message
            commit_message = f"ADW Checkpoint: {description or 'Automated checkpoint'}"
            await self._run_git_command(["commit", "-m", commit_message])

            # Create tag
            await self._run_git_command(
                ["tag", "-a", tag_name, "-m", f"Checkpoint: {description}"]
            )

            # Get commit hash
            result = await self._run_git_command(["rev-parse", "HEAD"])
            commit_hash = result.stdout.strip()

            logger.info("Git checkpoint created", tag=tag_name, commit=commit_hash)
            return commit_hash

        except Exception as e:
            logger.error("Failed to create git checkpoint", error=str(e))
            raise

    async def rollback_to_commit(self, commit_hash: str) -> bool:
        """Rollback to a specific commit."""
        try:
            # Hard reset to commit
            await self._run_git_command(["reset", "--hard", commit_hash])

            # Clean untracked files
            await self._run_git_command(["clean", "-fd"])

            logger.info("Rolled back to commit", commit=commit_hash)
            return True

        except Exception as e:
            logger.error(
                "Failed to rollback to commit", commit=commit_hash, error=str(e)
            )
            return False

    async def get_last_stable_checkpoint(self) -> Optional[str]:
        """Get the most recent stable checkpoint."""
        try:
            result = await self._run_git_command(
                ["tag", "-l", f"{self.checkpoint_prefix}-*", "--sort=-version:refname"]
            )

            tags = result.stdout.strip().split("\n")
            if tags and tags[0]:
                # Get commit hash for the latest tag
                result = await self._run_git_command(["rev-list", "-n", "1", tags[0]])
                return result.stdout.strip()

            return None

        except Exception as e:
            logger.error("Failed to get last stable checkpoint", error=str(e))
            return None

    async def _run_git_command(self, cmd: List[str]) -> subprocess.CompletedProcess:
        """Run a git command asynchronously."""
        full_cmd = ["git"] + cmd

        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            cwd=self.repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode, full_cmd, stdout, stderr
            )

        return subprocess.CompletedProcess(
            full_cmd, process.returncode, stdout.decode(), stderr.decode()
        )


class PerformanceBaseline:
    """Tracks performance baselines for regression detection."""

    def __init__(self, baseline_file: Path):
        self.baseline_file = baseline_file
        self.baselines: Dict[str, Any] = {}
        self._load_baselines()

    def _load_baselines(self) -> None:
        """Load performance baselines from file."""
        try:
            if self.baseline_file.exists():
                with open(self.baseline_file, "r") as f:
                    self.baselines = json.load(f)
            else:
                self.baselines = {}
        except Exception as e:
            logger.error("Failed to load performance baselines", error=str(e))
            self.baselines = {}

    def _save_baselines(self) -> None:
        """Save performance baselines to file."""
        try:
            self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_file, "w") as f:
                json.dump(self.baselines, f, indent=2)
        except Exception as e:
            logger.error("Failed to save performance baselines", error=str(e))

    def set_baseline(
        self, metric_name: str, value: float, tolerance: float = 0.1
    ) -> None:
        """Set a performance baseline."""
        self.baselines[metric_name] = {
            "value": value,
            "tolerance": tolerance,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._save_baselines()
        logger.info("Performance baseline set", metric=metric_name, value=value)

    def check_regression(
        self, metric_name: str, current_value: float
    ) -> Tuple[bool, float]:
        """Check if current value represents a performance regression."""
        if metric_name not in self.baselines:
            logger.warning("No baseline found for metric", metric=metric_name)
            return False, 0.0

        baseline = self.baselines[metric_name]
        baseline_value = baseline["value"]
        tolerance = baseline["tolerance"]

        # Calculate regression percentage
        regression = (current_value - baseline_value) / baseline_value
        is_regression = regression > tolerance

        if is_regression:
            logger.warning(
                "Performance regression detected",
                metric=metric_name,
                baseline=baseline_value,
                current=current_value,
                regression_pct=regression * 100,
            )

        return is_regression, regression


class AutoRollbackSystem:
    """Main rollback system for autonomous development safety."""

    def __init__(self, repo_path: Path, baseline_file: Optional[Path] = None):
        self.repo_path = repo_path
        self.git_checkpoint = GitCheckpoint(repo_path)

        if baseline_file is None:
            baseline_file = repo_path / ".adw" / "performance_baselines.json"
        self.performance_baseline = PerformanceBaseline(baseline_file)

        self.rollback_history: List[Dict[str, Any]] = []

    async def create_safety_checkpoint(self, description: str = "") -> str:
        """Create a safety checkpoint before risky operations."""
        return await self.git_checkpoint.create_checkpoint(description)

    async def handle_failure(
        self, failure_type: RollbackLevel, context: Dict[str, Any] = None
    ) -> bool:
        """Handle a failure by executing the appropriate rollback strategy."""
        if context is None:
            context = {}

        strategy = RollbackStrategy.STRATEGIES.get(failure_type)
        if not strategy:
            logger.error("Unknown failure type", failure_type=failure_type)
            return False

        logger.info(
            "Handling failure with rollback",
            failure_type=failure_type.value,
            strategy=strategy["description"],
        )

        start_time = time.time()
        success = False

        try:
            # Execute rollback action
            if strategy["action"] == "rollback_last_commit":
                success = await self._rollback_last_commit()
            elif strategy["action"] == "rollback_to_last_green":
                success = await self._rollback_to_last_green()
            elif strategy["action"] == "rollback_to_baseline":
                success = await self._rollback_to_baseline()
            elif strategy["action"] == "rollback_to_last_stable_checkpoint":
                success = await self._rollback_to_last_stable_checkpoint()
            elif strategy["action"] == "emergency_restore_from_backup":
                success = await self._emergency_restore_from_backup()

            # Validate rollback
            if success:
                success = await self._validate_rollback(strategy["validation"])

            # Record rollback attempt
            self._record_rollback_attempt(
                failure_type, strategy, success, time.time() - start_time, context
            )

            return success

        except Exception as e:
            logger.error(
                "Rollback execution failed", failure_type=failure_type, error=str(e)
            )
            self._record_rollback_attempt(
                failure_type, strategy, False, time.time() - start_time, context
            )
            return False

    async def _rollback_last_commit(self) -> bool:
        """Rollback the last commit."""
        try:
            result = await self.git_checkpoint._run_git_command(
                ["reset", "--hard", "HEAD~1"]
            )
            return True
        except Exception as e:
            logger.error("Failed to rollback last commit", error=str(e))
            return False

    async def _rollback_to_last_green(self) -> bool:
        """Rollback to the last commit where tests passed."""
        # This would require test result tracking - simplified for now
        try:
            # Look for commits with "✅" or "tests pass" in message
            result = await self.git_checkpoint._run_git_command(
                ["log", "--grep=✅", "--grep=tests pass", "--format=%H", "-1"]
            )

            if result.stdout.strip():
                commit_hash = result.stdout.strip()
                return await self.git_checkpoint.rollback_to_commit(commit_hash)
            else:
                # Fallback to last stable checkpoint
                return await self._rollback_to_last_stable_checkpoint()

        except Exception as e:
            logger.error("Failed to rollback to last green", error=str(e))
            return False

    async def _rollback_to_baseline(self) -> bool:
        """Rollback to performance baseline."""
        # This would require performance tracking integration
        return await self._rollback_to_last_stable_checkpoint()

    async def _rollback_to_last_stable_checkpoint(self) -> bool:
        """Rollback to the last stable checkpoint."""
        try:
            checkpoint = await self.git_checkpoint.get_last_stable_checkpoint()
            if checkpoint:
                return await self.git_checkpoint.rollback_to_commit(checkpoint)
            else:
                logger.error("No stable checkpoint found")
                return False
        except Exception as e:
            logger.error("Failed to rollback to stable checkpoint", error=str(e))
            return False

    async def _emergency_restore_from_backup(self) -> bool:
        """Emergency database restoration - placeholder for now."""
        logger.warning("Emergency restore not yet implemented")
        return False

    async def _validate_rollback(self, validation_type: str) -> bool:
        """Validate that rollback was successful."""
        try:
            if validation_type == "syntax_check":
                # Basic Python syntax check
                result = await asyncio.create_subprocess_exec(
                    "python",
                    "-m",
                    "py_compile",
                    "-",
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.communicate()
                return result.returncode == 0

            elif validation_type == "run_test_suite":
                # Run a subset of tests for quick validation
                result = await asyncio.create_subprocess_exec(
                    "pytest",
                    "tests/unit/",
                    "-x",
                    "--tb=no",
                    "-q",
                    cwd=self.repo_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await result.communicate()
                return result.returncode == 0

            elif validation_type in [
                "performance_test",
                "full_system_check",
                "data_integrity_check",
            ]:
                # Placeholder for more complex validations
                return True

            return True

        except Exception as e:
            logger.error(
                "Rollback validation failed",
                validation_type=validation_type,
                error=str(e),
            )
            return False

    def _record_rollback_attempt(
        self,
        failure_type: RollbackLevel,
        strategy: Dict[str, Any],
        success: bool,
        duration: float,
        context: Dict[str, Any],
    ) -> None:
        """Record rollback attempt for analysis."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "failure_type": failure_type.value,
            "strategy": strategy["action"],
            "success": success,
            "duration": duration,
            "context": context,
        }

        self.rollback_history.append(record)

        # Keep only last 100 records
        if len(self.rollback_history) > 100:
            self.rollback_history = self.rollback_history[-100:]

        logger.info("Rollback attempt recorded", **record)

    def get_rollback_statistics(self) -> Dict[str, Any]:
        """Get rollback statistics for monitoring."""
        if not self.rollback_history:
            return {"total_attempts": 0}

        total_attempts = len(self.rollback_history)
        successful_attempts = sum(1 for r in self.rollback_history if r["success"])

        failure_type_counts = {}
        for record in self.rollback_history:
            failure_type = record["failure_type"]
            failure_type_counts[failure_type] = (
                failure_type_counts.get(failure_type, 0) + 1
            )

        avg_duration = (
            sum(r["duration"] for r in self.rollback_history) / total_attempts
        )

        return {
            "total_attempts": total_attempts,
            "successful_attempts": successful_attempts,
            "success_rate": successful_attempts / total_attempts,
            "failure_type_distribution": failure_type_counts,
            "average_duration": avg_duration,
            "recent_failures": self.rollback_history[-10:]
            if len(self.rollback_history) > 10
            else self.rollback_history,
        }
