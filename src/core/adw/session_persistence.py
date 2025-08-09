"""
Session state persistence for ADW sessions.
Enables sessions to survive interruptions and resume from where they left off.
"""

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class SessionCheckpoint:
    """A checkpoint of session state."""

    session_id: str
    timestamp: float
    current_phase: str
    phase_progress: dict[str, Any]
    metrics: dict[str, Any]
    git_commit_hash: str | None
    iteration_count: int
    consecutive_failures: int
    context_data: dict[str, Any]


class SessionStatePersistence:
    """Manages persistence and recovery of ADW session state."""

    def __init__(self, project_path: Path, session_id: str):
        self.project_path = project_path
        self.session_id = session_id
        self.state_dir = project_path / ".adw" / "sessions"
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_file = self.state_dir / f"{session_id}.json"
        self.recovery_file = self.state_dir / f"{session_id}_recovery.json"

    async def save_checkpoint(
        self,
        session: "ADWSession",
        phase_progress: dict[str, Any] = None,
        additional_context: dict[str, Any] = None,
    ) -> bool:
        """Save a session checkpoint."""
        try:
            # Get current git commit
            git_commit = await self._get_current_git_commit()

            # Create checkpoint data
            checkpoint = SessionCheckpoint(
                session_id=self.session_id,
                timestamp=time.time(),
                current_phase=session.metrics.current_phase.value,
                phase_progress=phase_progress or {},
                metrics={
                    "start_time": session.metrics.start_time,
                    "commits_made": session.metrics.commits_made,
                    "tests_written": session.metrics.tests_written,
                    "tests_passed": session.metrics.tests_passed,
                    "quality_gate_passes": session.metrics.quality_gate_passes,
                    "quality_gate_failures": session.metrics.quality_gate_failures,
                    "rollbacks_triggered": session.metrics.rollbacks_triggered,
                    "reconnaissance_duration": session.metrics.reconnaissance_duration,
                    "micro_development_duration": session.metrics.micro_development_duration,
                    "integration_validation_duration": session.metrics.integration_validation_duration,
                    "meta_learning_duration": session.metrics.meta_learning_duration,
                },
                git_commit_hash=git_commit,
                iteration_count=len(getattr(session, "iteration_history", [])),
                consecutive_failures=session.consecutive_failures,
                context_data=additional_context or {},
            )

            # Save checkpoint to file
            checkpoint_data = asdict(checkpoint)
            with open(self.checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2)

            logger.info(
                "Session checkpoint saved",
                session_id=self.session_id,
                phase=checkpoint.current_phase,
                timestamp=checkpoint.timestamp,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to save session checkpoint",
                session_id=self.session_id,
                error=str(e),
            )
            return False

    async def load_checkpoint(self) -> SessionCheckpoint | None:
        """Load the most recent session checkpoint."""
        try:
            if not self.checkpoint_file.exists():
                return None

            with open(self.checkpoint_file) as f:
                checkpoint_data = json.load(f)

            checkpoint = SessionCheckpoint(**checkpoint_data)

            logger.info(
                "Session checkpoint loaded",
                session_id=self.session_id,
                phase=checkpoint.current_phase,
                timestamp=checkpoint.timestamp,
            )

            return checkpoint

        except Exception as e:
            logger.error(
                "Failed to load session checkpoint",
                session_id=self.session_id,
                error=str(e),
            )
            return None

    async def restore_session(self, session: "ADWSession") -> bool:
        """Restore session state from checkpoint."""
        try:
            checkpoint = await self.load_checkpoint()
            if not checkpoint:
                logger.info(
                    "No checkpoint found for session", session_id=self.session_id
                )
                return False

            # Restore session metrics
            session.metrics.start_time = checkpoint.metrics.get(
                "start_time", time.time()
            )
            session.metrics.commits_made = checkpoint.metrics.get("commits_made", 0)
            session.metrics.tests_written = checkpoint.metrics.get("tests_written", 0)
            session.metrics.tests_passed = checkpoint.metrics.get("tests_passed", 0)
            session.metrics.quality_gate_passes = checkpoint.metrics.get(
                "quality_gate_passes", 0
            )
            session.metrics.quality_gate_failures = checkpoint.metrics.get(
                "quality_gate_failures", 0
            )
            session.metrics.rollbacks_triggered = checkpoint.metrics.get(
                "rollbacks_triggered", 0
            )
            session.metrics.reconnaissance_duration = checkpoint.metrics.get(
                "reconnaissance_duration", 0.0
            )
            session.metrics.micro_development_duration = checkpoint.metrics.get(
                "micro_development_duration", 0.0
            )
            session.metrics.integration_validation_duration = checkpoint.metrics.get(
                "integration_validation_duration", 0.0
            )
            session.metrics.meta_learning_duration = checkpoint.metrics.get(
                "meta_learning_duration", 0.0
            )

            # Restore session state
            from .session_manager import SessionPhase

            session.metrics.current_phase = SessionPhase(checkpoint.current_phase)
            session.consecutive_failures = checkpoint.consecutive_failures

            # Verify git state matches checkpoint
            current_commit = await self._get_current_git_commit()
            if current_commit and checkpoint.git_commit_hash:
                if current_commit != checkpoint.git_commit_hash:
                    logger.warning(
                        "Git state mismatch during restore",
                        current=current_commit,
                        checkpoint=checkpoint.git_commit_hash,
                    )

            logger.info(
                "Session state restored from checkpoint",
                session_id=self.session_id,
                phase=checkpoint.current_phase,
                age_minutes=(time.time() - checkpoint.timestamp) / 60,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to restore session state",
                session_id=self.session_id,
                error=str(e),
            )
            return False

    async def create_recovery_point(
        self,
        session: "ADWSession",
        recovery_type: str,
        recovery_data: dict[str, Any],
    ) -> bool:
        """Create a recovery point for emergency restoration."""
        try:
            recovery_point = {
                "session_id": self.session_id,
                "timestamp": time.time(),
                "recovery_type": recovery_type,
                "git_commit_hash": await self._get_current_git_commit(),
                "session_metrics": {
                    "current_phase": session.metrics.current_phase.value,
                    "commits_made": session.metrics.commits_made,
                    "consecutive_failures": session.consecutive_failures,
                },
                "recovery_data": recovery_data,
            }

            with open(self.recovery_file, "w") as f:
                json.dump(recovery_point, f, indent=2)

            logger.info(
                "Recovery point created",
                session_id=self.session_id,
                recovery_type=recovery_type,
            )

            return True

        except Exception as e:
            logger.error(
                "Failed to create recovery point",
                session_id=self.session_id,
                error=str(e),
            )
            return False

    async def cleanup_old_checkpoints(self, max_age_hours: int = 48) -> int:
        """Clean up old session checkpoints."""
        cleaned_count = 0

        try:
            cutoff_time = time.time() - (max_age_hours * 3600)

            for checkpoint_file in self.state_dir.glob("*.json"):
                if checkpoint_file.name.endswith("_recovery.json"):
                    continue  # Skip recovery files

                try:
                    with open(checkpoint_file) as f:
                        data = json.load(f)

                    checkpoint_time = data.get("timestamp", 0)
                    if checkpoint_time < cutoff_time:
                        checkpoint_file.unlink()

                        # Also remove corresponding recovery file if it exists
                        recovery_file = checkpoint_file.with_name(
                            checkpoint_file.stem + "_recovery.json"
                        )
                        if recovery_file.exists():
                            recovery_file.unlink()

                        cleaned_count += 1

                except Exception as e:
                    logger.debug(
                        "Error cleaning checkpoint file",
                        file=str(checkpoint_file),
                        error=str(e),
                    )

            if cleaned_count > 0:
                logger.info(
                    "Cleaned up old checkpoints",
                    count=cleaned_count,
                    max_age_hours=max_age_hours,
                )

        except Exception as e:
            logger.error("Failed to cleanup old checkpoints", error=str(e))

        return cleaned_count

    async def _get_current_git_commit(self) -> str | None:
        """Get the current git commit hash."""
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                return stdout.decode().strip()

        except Exception as e:
            logger.debug("Failed to get git commit", error=str(e))

        return None

    def get_checkpoint_info(self) -> dict[str, Any]:
        """Get information about existing checkpoints."""
        info = {
            "checkpoint_exists": self.checkpoint_file.exists(),
            "recovery_point_exists": self.recovery_file.exists(),
            "checkpoint_age_minutes": None,
            "session_id": self.session_id,
        }

        try:
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file) as f:
                    data = json.load(f)

                checkpoint_time = data.get("timestamp", 0)
                info["checkpoint_age_minutes"] = (time.time() - checkpoint_time) / 60
                info["checkpoint_phase"] = data.get("current_phase")
                info["checkpoint_git_hash"] = data.get("git_commit_hash")

        except Exception as e:
            logger.debug("Error reading checkpoint info", error=str(e))

        return info


class SessionStateManager:
    """High-level manager for session state operations."""

    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.state_dir = project_path / ".adw" / "sessions"
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def get_persistence_manager(self, session_id: str) -> SessionStatePersistence:
        """Get a persistence manager for a specific session."""
        return SessionStatePersistence(self.project_path, session_id)

    async def list_active_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions with checkpoint data."""
        sessions = []

        try:
            for checkpoint_file in self.state_dir.glob("*.json"):
                if checkpoint_file.name.endswith("_recovery.json"):
                    continue

                try:
                    with open(checkpoint_file) as f:
                        data = json.load(f)

                    session_info = {
                        "session_id": data.get("session_id"),
                        "current_phase": data.get("current_phase"),
                        "timestamp": data.get("timestamp"),
                        "age_minutes": (time.time() - data.get("timestamp", 0)) / 60,
                        "git_commit_hash": data.get("git_commit_hash"),
                        "iteration_count": data.get("iteration_count", 0),
                        "consecutive_failures": data.get("consecutive_failures", 0),
                    }

                    sessions.append(session_info)

                except Exception as e:
                    logger.debug(
                        "Error reading session checkpoint",
                        file=str(checkpoint_file),
                        error=str(e),
                    )

            # Sort by timestamp (most recent first)
            sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        except Exception as e:
            logger.error("Failed to list active sessions", error=str(e))

        return sessions

    async def resume_latest_session(self) -> str | None:
        """Find and return the session ID of the most recent session."""
        try:
            sessions = await self.list_active_sessions()

            if sessions:
                latest_session = sessions[0]
                session_id = latest_session["session_id"]

                # Check if session is recent enough to resume (within 24 hours)
                age_hours = latest_session["age_minutes"] / 60
                if age_hours <= 24:
                    logger.info(
                        "Found resumable session",
                        session_id=session_id,
                        age_hours=age_hours,
                        phase=latest_session["current_phase"],
                    )
                    return session_id
                else:
                    logger.info(
                        "Latest session too old to resume",
                        session_id=session_id,
                        age_hours=age_hours,
                    )

        except Exception as e:
            logger.error("Failed to find resumable session", error=str(e))

        return None

    async def cleanup_all_old_checkpoints(self, max_age_hours: int = 48) -> int:
        """Clean up all old session checkpoints."""
        total_cleaned = 0

        try:
            sessions = await self.list_active_sessions()

            for session_info in sessions:
                session_id = session_info["session_id"]
                persistence = self.get_persistence_manager(session_id)
                cleaned = await persistence.cleanup_old_checkpoints(max_age_hours)
                total_cleaned += cleaned

        except Exception as e:
            logger.error("Failed to cleanup all checkpoints", error=str(e))

        return total_cleaned
