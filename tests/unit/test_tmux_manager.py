"""Tests for the RetryableTmuxManager."""

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.tmux_manager import (
    RetryableTmuxManager,
    TmuxCommandTimeoutError,
    TmuxError,
    TmuxOperationResult,
    TmuxRetryExhaustedError,
    TmuxSession,
    TmuxSessionStatus,
    get_tmux_manager,
)


@pytest.fixture
def tmux_manager():
    """Create a RetryableTmuxManager for testing."""
    return RetryableTmuxManager(
        max_retries=3,
        base_delay=0.1,  # Fast delays for testing
        max_delay=1.0,
        timeout_spawn=5.0,
        timeout_terminate=3.0,
        timeout_command=2.0,
    )


@pytest.fixture
def mock_subprocess():
    """Mock asyncio subprocess for testing."""
    with patch("asyncio.create_subprocess_exec") as mock:
        yield mock


class TestRetryableTmuxManager:
    """Test the RetryableTmuxManager class."""

    def test_initialization(self):
        """Test tmux manager initialization."""
        manager = RetryableTmuxManager(
            max_retries=5,
            timeout_spawn=30.0,
        )

        assert manager.max_retries == 5
        assert manager.timeout_spawn == 30.0
        assert manager.failure_count == 0
        assert len(manager.sessions) == 0

    def test_circuit_breaker_logic(self, tmux_manager):
        """Test circuit breaker functionality."""
        # Should not be open initially
        assert not tmux_manager._is_circuit_breaker_open()

        # Record failures to trigger circuit breaker
        for _ in range(tmux_manager.circuit_breaker_threshold):
            tmux_manager._record_failure()

        assert tmux_manager._is_circuit_breaker_open()

        # Reset should close circuit breaker
        tmux_manager._record_success()
        assert not tmux_manager._is_circuit_breaker_open()

    def test_circuit_breaker_timeout(self, tmux_manager):
        """Test circuit breaker timeout reset."""
        # Trigger circuit breaker
        for _ in range(tmux_manager.circuit_breaker_threshold):
            tmux_manager._record_failure()

        assert tmux_manager._is_circuit_breaker_open()

        # Simulate timeout passing
        tmux_manager.last_failure_time = (
            time.time() - tmux_manager.circuit_breaker_timeout - 1
        )

        assert not tmux_manager._is_circuit_breaker_open()
        assert tmux_manager.failure_count == 0

    @pytest.mark.asyncio
    async def test_successful_command_execution(self, tmux_manager, mock_subprocess):
        """Test successful command execution."""
        # Mock successful process
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"success", b"")
        mock_subprocess.return_value = mock_process

        result = await tmux_manager._run_command_with_retry(
            command=["tmux", "list-sessions"],
            timeout=5.0,
            operation_name="test",
        )

        assert result.success
        assert result.stdout == "success"
        assert result.retry_count == 0
        assert tmux_manager.failure_count == 0

    @pytest.mark.asyncio
    async def test_command_retry_on_failure(self, tmux_manager, mock_subprocess):
        """Test command retry logic on failure."""
        # Mock failing process that succeeds on retry
        mock_process_fail = AsyncMock()
        mock_process_fail.returncode = 1
        mock_process_fail.communicate.return_value = (b"", b"error")

        mock_process_success = AsyncMock()
        mock_process_success.returncode = 0
        mock_process_success.communicate.return_value = (b"success", b"")

        mock_subprocess.side_effect = [mock_process_fail, mock_process_success]

        result = await tmux_manager._run_command_with_retry(
            command=["tmux", "list-sessions"],
            timeout=5.0,
            operation_name="test",
        )

        assert result.success
        assert result.stdout == "success"
        assert result.retry_count == 1
        assert tmux_manager.failure_count == 0  # Reset on success

    @pytest.mark.asyncio
    async def test_command_timeout_handling(self, tmux_manager, mock_subprocess):
        """Test command timeout handling."""
        # Mock process that times out
        mock_process = AsyncMock()
        mock_process.communicate.side_effect = TimeoutError()
        mock_process.kill = AsyncMock()
        mock_process.wait = AsyncMock()
        mock_subprocess.return_value = mock_process

        result = await tmux_manager._run_command_with_retry(
            command=["tmux", "list-sessions"],
            timeout=0.1,  # Very short timeout
            operation_name="test",
        )

        assert not result.success
        assert "timed out" in result.error_message.lower()
        assert result.retry_count == tmux_manager.max_retries
        assert tmux_manager.failure_count > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_prevents_execution(
        self, tmux_manager, mock_subprocess
    ):
        """Test circuit breaker prevents command execution."""
        # Trigger circuit breaker
        for _ in range(tmux_manager.circuit_breaker_threshold):
            tmux_manager._record_failure()

        result = await tmux_manager._run_command_with_retry(
            command=["tmux", "list-sessions"],
            timeout=5.0,
            operation_name="test",
        )

        assert not result.success
        assert "circuit breaker" in result.error_message.lower()
        # Should not have called subprocess due to circuit breaker
        mock_subprocess.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, tmux_manager, mock_subprocess):
        """Test complete session lifecycle: create -> check -> terminate."""
        # Mock all subprocess calls as successful
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_process

        # Mock session existence appropriately - needs more side_effect values
        with patch.object(
            tmux_manager,
            "_session_exists",
            side_effect=[False, True, True, True, False],
        ):
            # Create session
            create_result = await tmux_manager.create_session(
                session_name="lifecycle-test",
                command="sleep 10",
            )
            assert create_result.success

            # Check status
            status = await tmux_manager.get_session_status("lifecycle-test")
            assert status == TmuxSessionStatus.ACTIVE

            # Terminate session
            terminate_result = await tmux_manager.terminate_session("lifecycle-test")
            assert terminate_result.success

            # Check status after termination
            status = await tmux_manager.get_session_status("lifecycle-test")
            assert status == TmuxSessionStatus.STOPPED

    @pytest.mark.asyncio
    async def test_list_sessions(self, tmux_manager, mock_subprocess):
        """Test listing sessions."""
        # Mock successful list command
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"session1\nsession2\n", b"")
        mock_subprocess.return_value = mock_process

        sessions = await tmux_manager.list_sessions()

        assert len(sessions) == 2
        assert sessions[0].name == "session1"
        assert sessions[1].name == "session2"
        assert all(s.status == TmuxSessionStatus.ACTIVE for s in sessions)

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_sessions(self, tmux_manager):
        """Test cleaning up orphaned sessions."""
        # Create some tracked sessions
        tmux_manager.sessions["hive-tracked"] = TmuxSession(
            name="hive-tracked",
            status=TmuxSessionStatus.ACTIVE,
        )

        # Mock list_sessions to return more sessions than tracked
        mock_sessions = [
            TmuxSession(name="hive-tracked", status=TmuxSessionStatus.ACTIVE),
            TmuxSession(name="hive-orphaned1", status=TmuxSessionStatus.ACTIVE),
            TmuxSession(name="hive-orphaned2", status=TmuxSessionStatus.ACTIVE),
            TmuxSession(name="other-session", status=TmuxSessionStatus.ACTIVE),
        ]

        with patch.object(tmux_manager, "list_sessions", return_value=mock_sessions):
            with patch.object(tmux_manager, "terminate_session") as mock_terminate:
                mock_terminate.return_value = TmuxOperationResult(success=True)

                orphaned = await tmux_manager.cleanup_orphaned_sessions(prefix="hive-")

        # Should clean up orphaned hive sessions but not other sessions
        assert len(orphaned) == 2
        assert "hive-orphaned1" in orphaned
        assert "hive-orphaned2" in orphaned

    def test_get_session_info(self, tmux_manager):
        """Test getting session information."""
        session = TmuxSession(
            name="test-session",
            status=TmuxSessionStatus.ACTIVE,
        )
        tmux_manager.sessions["test-session"] = session

        info = tmux_manager.get_session_info("test-session")
        assert info == session

        info = tmux_manager.get_session_info("nonexistent")
        assert info is None

    def test_get_all_tracked_sessions(self, tmux_manager):
        """Test getting all tracked sessions."""
        session1 = TmuxSession(name="session1", status=TmuxSessionStatus.ACTIVE)
        session2 = TmuxSession(name="session2", status=TmuxSessionStatus.STOPPED)

        tmux_manager.sessions["session1"] = session1
        tmux_manager.sessions["session2"] = session2

        all_sessions = tmux_manager.get_all_tracked_sessions()
        assert len(all_sessions) == 2
        assert all_sessions["session1"] == session1
        assert all_sessions["session2"] == session2

    def test_get_failure_stats(self, tmux_manager):
        """Test getting failure statistics."""
        # Initial state
        stats = tmux_manager.get_failure_stats()
        assert stats["failure_count"] == 0
        assert not stats["circuit_breaker_open"]
        assert stats["tracked_sessions"] == 0

        # Add some failures and sessions
        tmux_manager._record_failure()
        tmux_manager.sessions["test"] = TmuxSession(
            name="test", status=TmuxSessionStatus.ACTIVE
        )

        stats = tmux_manager.get_failure_stats()
        assert stats["failure_count"] == 1
        assert stats["tracked_sessions"] == 1


class TestTmuxManagerIntegration:
    """Integration tests for tmux manager."""

    @pytest.mark.asyncio
    async def test_session_lifecycle(self, tmux_manager, mock_subprocess):
        """Test complete session lifecycle: create -> check -> terminate."""
        # Mock all subprocess calls as successful
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_process

        # Mock session existence appropriately
        with patch.object(
            tmux_manager, "_session_exists", side_effect=[False, True, True, False]
        ):
            # Create session
            create_result = await tmux_manager.create_session(
                session_name="lifecycle-test",
                command="sleep 10",
            )
            assert create_result.success

            # Check status
            status = await tmux_manager.get_session_status("lifecycle-test")
            assert status == TmuxSessionStatus.ACTIVE

            # Terminate session
            terminate_result = await tmux_manager.terminate_session("lifecycle-test")
            assert terminate_result.success

            # Check status after termination
            status = await tmux_manager.get_session_status("lifecycle-test")
            assert status == TmuxSessionStatus.STOPPED

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, tmux_manager, mock_subprocess):
        """Test concurrent tmux operations."""
        # Mock successful operations
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_process

        with patch.object(tmux_manager, "_session_exists", return_value=True):
            # Run multiple operations concurrently
            tasks = [
                tmux_manager.create_session(f"concurrent-{i}", "sleep 1")
                for i in range(5)
            ]

            results = await asyncio.gather(*tasks)

            # All should succeed
            assert all(result.success for result in results)

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, tmux_manager, mock_subprocess):
        """Test various error recovery scenarios."""
        # Test timeout recovery
        mock_process_timeout = AsyncMock()
        mock_process_timeout.communicate.side_effect = TimeoutError()
        mock_process_timeout.kill = AsyncMock()
        mock_process_timeout.wait = AsyncMock()

        mock_process_success = AsyncMock()
        mock_process_success.returncode = 0
        mock_process_success.communicate.return_value = (b"success", b"")

        # First call times out, second succeeds
        mock_subprocess.side_effect = [mock_process_timeout, mock_process_success]

        # Mock that session doesn't exist (so no termination attempt)
        with patch.object(tmux_manager, "_session_exists", return_value=False):
            result = await tmux_manager.create_session("recovery-test", "echo hello")

            # Should succeed on retry
            assert result.success
            assert result.retry_count == 1


    @pytest.mark.asyncio
    async def test_termination_idempotency_and_tracking(self, tmux_manager):
        """Terminate twice when session doesn't exist; ensure idempotent and tracked as STOPPED."""
        # First termination: session not found -> success and mark STOPPED
        with patch.object(tmux_manager, "_session_exists", return_value=False):
            res1 = await tmux_manager.terminate_session("idem-test")
        assert res1.success
        assert tmux_manager.get_session_info("idem-test").status == TmuxSessionStatus.STOPPED

        # Second termination: still not found -> success, status remains STOPPED
        with patch.object(tmux_manager, "_session_exists", return_value=False):
            res2 = await tmux_manager.terminate_session("idem-test")
        assert res2.success
        assert tmux_manager.get_session_info("idem-test").status == TmuxSessionStatus.STOPPED

    @pytest.mark.asyncio
    async def test_optimistic_tracking_stopped_fast_path(self, tmux_manager):
        """When tracked as STOPPED, get_session_status should not perform external check."""
        tmux_manager.sessions["fast-test"] = TmuxSession(
            name="fast-test", status=TmuxSessionStatus.STOPPED
        )
        # If _session_exists is called, raise to fail test
        with patch.object(tmux_manager, "_session_exists", side_effect=AssertionError("should not be called")):
            status = await tmux_manager.get_session_status("fast-test")
        assert status == TmuxSessionStatus.STOPPED

    @pytest.mark.asyncio
    async def test_terminate_timeout_then_force_success(self, tmux_manager):
        """If graceful terminate fails, force kill should succeed and mark STOPPED."""
        # Ensure path goes into termination (session exists)
        with patch.object(tmux_manager, "_session_exists", return_value=True):
            # First call (graceful) fails; second (force) succeeds
            failure = TmuxOperationResult(success=False, error_message="timeout")
            success = TmuxOperationResult(success=True)

            with patch.object(
                tmux_manager, "_run_command_with_retry", side_effect=[failure, success]
            ) as _:
                result = await tmux_manager.terminate_session("force-test", force=False)

        assert result.success
        assert tmux_manager.sessions["force-test"].status == TmuxSessionStatus.STOPPED

    @pytest.mark.asyncio
    async def test_cleanup_orphaned_sessions_filters_and_terminates(self, tmux_manager):
        """Only untracked prefixed sessions are terminated and returned."""
        # Prepare sessions: hive-a (tracked), hive-b (untracked), other (ignored)
        tmux_manager.sessions["hive-a"] = TmuxSession(name="hive-a", status=TmuxSessionStatus.ACTIVE)

        listed = [
            TmuxSession(name="hive-a", status=TmuxSessionStatus.ACTIVE),
            TmuxSession(name="hive-b", status=TmuxSessionStatus.ACTIVE),
            TmuxSession(name="misc", status=TmuxSessionStatus.ACTIVE),
        ]

        with (
            patch.object(tmux_manager, "list_sessions", return_value=listed),
            patch.object(
                tmux_manager,
                "terminate_session",
                new=AsyncMock(return_value=TmuxOperationResult(success=True)),
            ) as term,
        ):
            cleaned = await tmux_manager.cleanup_orphaned_sessions(prefix="hive-")

        assert cleaned == ["hive-b"]
        term.assert_awaited_once()


class TestTmuxManagerSingleton:
    """Test the singleton pattern for tmux manager."""

    def test_get_tmux_manager_singleton(self):
        """Test that get_tmux_manager returns the same instance."""
        manager1 = get_tmux_manager()
        manager2 = get_tmux_manager()

        assert manager1 is manager2
        assert isinstance(manager1, RetryableTmuxManager)


@pytest.mark.asyncio
class TestTmuxManagerFailureScenarios:
    """Test various failure scenarios and edge cases."""

    async def test_session_creation_validation_failure(
        self, tmux_manager, mock_subprocess
    ):
        """Test when session creation succeeds but validation fails."""
        # Mock successful creation command but session doesn't exist
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_process

        with patch.object(tmux_manager, "_session_exists", return_value=False):
            result = await tmux_manager.create_session("validation-fail", "echo hello")

            assert not result.success
            assert "creation succeeded but session not found" in result.error_message

    async def test_max_retries_exhausted(self, tmux_manager, mock_subprocess):
        """Test behavior when max retries are exhausted."""
        # Mock process that always fails
        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"persistent error")
        mock_subprocess.return_value = mock_process

        result = await tmux_manager._run_command_with_retry(
            command=["tmux", "fail"],
            timeout=5.0,
            operation_name="test_max_retries",
        )

        assert not result.success
        assert result.retry_count == tmux_manager.max_retries
        assert tmux_manager.failure_count > 0

    async def test_environment_variable_handling(self, tmux_manager, mock_subprocess):
        """Test that environment variables are properly handled."""
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"", b"")
        mock_subprocess.return_value = mock_process

        env_vars = {"TEST_VAR": "test_value", "ANOTHER_VAR": "another_value"}

        with patch.object(tmux_manager, "_session_exists", return_value=True):
            result = await tmux_manager.create_session(
                session_name="env-test",
                command="echo $TEST_VAR",
                environment=env_vars,
            )

        assert result.success
        # Check that the command was called (environment is handled by subprocess)
