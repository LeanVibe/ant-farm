"""
Unit tests for ADW Session Persistence.
"""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.adw.session_persistence import (
    SessionStatePersistence,
    SessionStateManager,
    SessionCheckpoint,
)
from src.core.adw.session_manager import ADWSession, SessionPhase


@pytest.mark.asyncio
async def test_session_checkpoint_save_load():
    """Test saving and loading session checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session_id = "test-session-123"

        # Create persistence manager
        persistence = SessionStatePersistence(project_path, session_id)

        # Create mock session
        session = MagicMock()
        session.metrics.current_phase = SessionPhase.MICRO_DEVELOPMENT
        session.metrics.start_time = 1000.0
        session.metrics.commits_made = 5
        session.metrics.tests_written = 10
        session.metrics.tests_passed = 9
        session.metrics.quality_gate_passes = 3
        session.metrics.quality_gate_failures = 1
        session.metrics.rollbacks_triggered = 0
        session.metrics.reconnaissance_duration = 900.0
        session.metrics.micro_development_duration = 3600.0
        session.metrics.integration_validation_duration = 0.0
        session.metrics.meta_learning_duration = 0.0
        session.consecutive_failures = 0

        # Mock git command
        with patch.object(
            persistence, "_get_current_git_commit", return_value="abc123"
        ):
            # Save checkpoint
            success = await persistence.save_checkpoint(
                session,
                phase_progress={"current_iteration": 2},
                additional_context={"test": "data"},
            )

            assert success is True
            assert persistence.checkpoint_file.exists()

        # Load checkpoint
        checkpoint = await persistence.load_checkpoint()

        assert checkpoint is not None
        assert checkpoint.session_id == session_id
        assert checkpoint.current_phase == SessionPhase.MICRO_DEVELOPMENT.value
        assert checkpoint.metrics["commits_made"] == 5
        assert checkpoint.metrics["tests_written"] == 10
        assert checkpoint.git_commit_hash == "abc123"
        assert checkpoint.phase_progress["current_iteration"] == 2
        assert checkpoint.context_data["test"] == "data"


@pytest.mark.asyncio
async def test_session_restore():
    """Test restoring session state from checkpoint."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session_id = "test-session-456"

        persistence = SessionStatePersistence(project_path, session_id)

        # Create checkpoint file manually
        checkpoint_data = {
            "session_id": session_id,
            "timestamp": 1000.0,
            "current_phase": SessionPhase.INTEGRATION_VALIDATION.value,
            "phase_progress": {"validation_step": 3},
            "metrics": {
                "start_time": 500.0,
                "commits_made": 8,
                "tests_written": 15,
                "tests_passed": 14,
                "quality_gate_passes": 5,
                "quality_gate_failures": 0,
                "rollbacks_triggered": 1,
                "reconnaissance_duration": 900.0,
                "micro_development_duration": 3600.0,
                "integration_validation_duration": 1800.0,
                "meta_learning_duration": 0.0,
            },
            "git_commit_hash": "def456",
            "iteration_count": 6,
            "consecutive_failures": 0,
            "context_data": {"restored": True},
        }

        with open(persistence.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Create mock session to restore into
        session = MagicMock()
        session.metrics = MagicMock()
        session.consecutive_failures = 0

        # Mock git command
        with patch.object(
            persistence, "_get_current_git_commit", return_value="def456"
        ):
            # Restore session
            success = await persistence.restore_session(session)

            assert success is True
            assert session.metrics.commits_made == 8
            assert session.metrics.tests_written == 15
            assert session.consecutive_failures == 0


@pytest.mark.asyncio
async def test_recovery_point():
    """Test creating recovery points."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session_id = "test-session-789"

        persistence = SessionStatePersistence(project_path, session_id)

        # Create mock session
        session = MagicMock()
        session.metrics.current_phase = SessionPhase.MICRO_DEVELOPMENT
        session.metrics.commits_made = 3
        session.consecutive_failures = 0

        # Mock git command
        with patch.object(
            persistence, "_get_current_git_commit", return_value="recovery123"
        ):
            # Create recovery point
            success = await persistence.create_recovery_point(
                session, "iteration_failure", {"iteration": 2, "error": "test error"}
            )

            assert success is True
            assert persistence.recovery_file.exists()

        # Verify recovery point content
        with open(persistence.recovery_file, "r") as f:
            recovery_data = json.load(f)

        assert recovery_data["session_id"] == session_id
        assert recovery_data["recovery_type"] == "iteration_failure"
        assert recovery_data["git_commit_hash"] == "recovery123"
        assert recovery_data["recovery_data"]["iteration"] == 2


@pytest.mark.asyncio
async def test_session_state_manager():
    """Test session state manager functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)

        manager = SessionStateManager(project_path)

        # Create some test session checkpoints
        sessions_data = [
            {
                "session_id": "session-1",
                "current_phase": "micro_development",
                "timestamp": 1000.0,
                "git_commit_hash": "abc123",
                "iteration_count": 3,
                "consecutive_failures": 0,
            },
            {
                "session_id": "session-2",
                "current_phase": "integration_validation",
                "timestamp": 2000.0,
                "git_commit_hash": "def456",
                "iteration_count": 5,
                "consecutive_failures": 1,
            },
        ]

        # Create checkpoint files
        for session_data in sessions_data:
            checkpoint_file = manager.state_dir / f"{session_data['session_id']}.json"
            with open(checkpoint_file, "w") as f:
                json.dump(session_data, f)

        # List active sessions
        sessions = await manager.list_active_sessions()

        assert len(sessions) == 2
        # Should be sorted by timestamp (most recent first)
        assert sessions[0]["session_id"] == "session-2"
        assert sessions[1]["session_id"] == "session-1"

        # Test resuming latest session
        with patch("time.time", return_value=2100.0):  # 100 seconds after session-2
            latest_session_id = await manager.resume_latest_session()
            assert latest_session_id == "session-2"


@pytest.mark.asyncio
async def test_cleanup_old_checkpoints():
    """Test cleaning up old session checkpoints."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session_id = "test-cleanup"

        persistence = SessionStatePersistence(project_path, session_id)

        # Create old checkpoint (49 hours ago)
        old_timestamp = 1000.0 - (49 * 3600)
        old_checkpoint = {
            "session_id": session_id,
            "timestamp": old_timestamp,
            "current_phase": "completed",
        }

        with open(persistence.checkpoint_file, "w") as f:
            json.dump(old_checkpoint, f)

        # Create recovery file too
        with open(persistence.recovery_file, "w") as f:
            json.dump({"timestamp": old_timestamp}, f)

        # Mock current time to be much later
        with patch("time.time", return_value=1000.0):
            cleaned_count = await persistence.cleanup_old_checkpoints(max_age_hours=48)

        assert cleaned_count == 1
        assert not persistence.checkpoint_file.exists()
        assert not persistence.recovery_file.exists()


def test_checkpoint_info():
    """Test getting checkpoint information."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session_id = "test-info"

        persistence = SessionStatePersistence(project_path, session_id)

        # Test when no checkpoint exists
        info = persistence.get_checkpoint_info()
        assert info["checkpoint_exists"] is False
        assert info["session_id"] == session_id

        # Create checkpoint
        checkpoint_data = {
            "session_id": session_id,
            "timestamp": 1000.0,
            "current_phase": "micro_development",
            "git_commit_hash": "test123",
        }

        with open(persistence.checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)

        # Test when checkpoint exists
        with patch("time.time", return_value=1060.0):  # 1 minute later
            info = persistence.get_checkpoint_info()
            assert info["checkpoint_exists"] is True
            assert info["checkpoint_age_minutes"] == 1.0
            assert info["checkpoint_phase"] == "micro_development"
            assert info["checkpoint_git_hash"] == "test123"


@pytest.mark.asyncio
async def test_git_commit_retrieval():
    """Test git commit hash retrieval."""
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        session_id = "test-git"

        persistence = SessionStatePersistence(project_path, session_id)

        # Test successful git command
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"abcdef123456\n", b"")
            mock_process.returncode = 0
            mock_subprocess.return_value = mock_process

            commit_hash = await persistence._get_current_git_commit()
            assert commit_hash == "abcdef123456"

        # Test failed git command
        with patch("asyncio.create_subprocess_exec") as mock_subprocess:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"fatal: not a git repo")
            mock_process.returncode = 128
            mock_subprocess.return_value = mock_process

            commit_hash = await persistence._get_current_git_commit()
            assert commit_hash is None
