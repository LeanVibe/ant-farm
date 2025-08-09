"""Resilient tmux session management with retry logic and failure recovery."""

import asyncio
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog

logger = structlog.get_logger()


class TmuxSessionStatus(Enum):
    """Status of a tmux session."""

    ACTIVE = "active"
    STARTING = "starting"
    STOPPED = "stopped"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class TmuxSession:
    """Represents a tmux session."""

    name: str
    status: TmuxSessionStatus
    pid: Optional[int] = None
    created_at: float = 0.0
    last_checked: float = 0.0
    command: Optional[str] = None
    working_directory: Optional[str] = None


@dataclass
class TmuxOperationResult:
    """Result of a tmux operation."""

    success: bool
    session_name: Optional[str] = None
    error_message: Optional[str] = None
    stdout: Optional[str] = None
    stderr: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0


class TmuxError(Exception):
    """Base exception for tmux operations."""

    pass


class TmuxSessionNotFoundError(TmuxError):
    """Raised when a tmux session is not found."""

    pass


class TmuxCommandTimeoutError(TmuxError):
    """Raised when a tmux command times out."""

    pass


class TmuxRetryExhaustedError(TmuxError):
    """Raised when all retry attempts are exhausted."""

    pass


class RetryableTmuxManager:
    """
    Resilient tmux session manager with retry logic, timeout handling,
    and comprehensive error recovery for production environments.

    Key features:
    - Exponential backoff retry logic for failed operations
    - Timeout handling for long-running operations
    - Session health validation and monitoring
    - Automatic cleanup on failures
    - Circuit breaker pattern for repeated failures
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        timeout_spawn: float = 30.0,
        timeout_terminate: float = 10.0,
        timeout_command: float = 5.0,
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout_spawn = timeout_spawn
        self.timeout_terminate = timeout_terminate
        self.timeout_command = timeout_command

        # Session tracking
        self.sessions: dict[str, TmuxSession] = {}

        # Circuit breaker for repeated failures
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 300.0  # 5 minutes

        logger.info(
            "RetryableTmuxManager initialized",
            max_retries=max_retries,
            timeout_spawn=timeout_spawn,
            timeout_terminate=timeout_terminate,
        )

    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open (too many recent failures)."""
        if self.failure_count < self.circuit_breaker_threshold:
            return False

        time_since_failure = time.time() - self.last_failure_time
        if time_since_failure > self.circuit_breaker_timeout:
            # Reset circuit breaker
            self.failure_count = 0
            return False

        return True

    def _record_failure(self):
        """Record a failure for circuit breaker tracking."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        logger.warning(
            "Tmux operation failure recorded",
            failure_count=self.failure_count,
            circuit_breaker_open=self._is_circuit_breaker_open(),
        )

    def _record_success(self):
        """Record a success (resets failure count)."""
        if self.failure_count > 0:
            logger.info("Tmux operation success, resetting failure count")
            self.failure_count = 0

    async def _run_command_with_retry(
        self,
        command: list[str],
        timeout: float,
        operation_name: str,
        cwd: Optional[Path] = None,
    ) -> TmuxOperationResult:
        """
        Run a tmux command with retry logic and timeout handling.

        Args:
            command: Command to execute
            timeout: Timeout in seconds
            operation_name: Name of operation for logging
            cwd: Working directory for command

        Returns:
            TmuxOperationResult with success status and details
        """
        if self._is_circuit_breaker_open():
            logger.error(
                "Circuit breaker open, refusing tmux operation",
                operation=operation_name,
                failure_count=self.failure_count,
            )
            return TmuxOperationResult(
                success=False,
                error_message="Circuit breaker open due to repeated failures",
            )

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries):
            try:
                logger.debug(
                    "Executing tmux command",
                    command=command,
                    attempt=attempt + 1,
                    operation=operation_name,
                )

                # Execute command with timeout
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=cwd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(), timeout=timeout
                    )
                except asyncio.TimeoutError:
                    # Kill the process if it times out
                    try:
                        process.kill()
                        await process.wait()
                    except ProcessLookupError:
                        pass

                    error_msg = f"Command timed out after {timeout}s"
                    logger.warning(
                        "Tmux command timeout",
                        command=command,
                        timeout=timeout,
                        attempt=attempt + 1,
                    )
                    last_error = TmuxCommandTimeoutError(error_msg)
                    continue

                stdout_str = stdout.decode() if stdout else ""
                stderr_str = stderr.decode() if stderr else ""
                execution_time = time.time() - start_time

                if process.returncode == 0:
                    # Success
                    self._record_success()
                    logger.debug(
                        "Tmux command succeeded",
                        operation=operation_name,
                        attempt=attempt + 1,
                        execution_time=execution_time,
                    )

                    return TmuxOperationResult(
                        success=True,
                        stdout=stdout_str,
                        stderr=stderr_str,
                        execution_time=execution_time,
                        retry_count=attempt,
                    )
                else:
                    # Command failed
                    error_msg = f"Command failed with exit code {process.returncode}: {stderr_str}"
                    logger.warning(
                        "Tmux command failed",
                        command=command,
                        exit_code=process.returncode,
                        stderr=stderr_str,
                        attempt=attempt + 1,
                    )
                    last_error = TmuxError(error_msg)

            except Exception as e:
                error_msg = f"Unexpected error executing command: {str(e)}"
                logger.warning(
                    "Tmux command exception",
                    command=command,
                    error=str(e),
                    attempt=attempt + 1,
                )
                last_error = TmuxError(error_msg)

            # Wait before retry (exponential backoff)
            if attempt < self.max_retries - 1:
                delay = min(self.base_delay * (2**attempt), self.max_delay)
                logger.debug(
                    "Waiting before retry",
                    delay=delay,
                    attempt=attempt + 1,
                    operation=operation_name,
                )
                await asyncio.sleep(delay)

        # All retries exhausted
        self._record_failure()
        execution_time = time.time() - start_time
        error_message = str(last_error) if last_error else "Unknown error"

        logger.error(
            "Tmux command failed after all retries",
            operation=operation_name,
            max_retries=self.max_retries,
            execution_time=execution_time,
            error=error_message,
        )

        return TmuxOperationResult(
            success=False,
            error_message=error_message,
            execution_time=execution_time,
            retry_count=self.max_retries,
        )

    async def create_session(
        self,
        session_name: str,
        command: str,
        working_directory: Optional[Path] = None,
        environment: Optional[dict[str, str]] = None,
    ) -> TmuxOperationResult:
        """
        Create a new tmux session with the given command.

        Args:
            session_name: Name of the tmux session
            command: Command to run in the session
            working_directory: Working directory for the command
            environment: Environment variables for the session

        Returns:
            TmuxOperationResult indicating success or failure
        """
        logger.info(
            "Creating tmux session",
            session_name=session_name,
            command=command[:100] + "..." if len(command) > 100 else command,
            working_directory=str(working_directory) if working_directory else None,
        )

        # Check if session already exists
        if await self._session_exists(session_name):
            logger.warning(
                "Session already exists, terminating first",
                session_name=session_name,
            )
            await self.terminate_session(session_name)

        # Prepare command
        tmux_cmd = [
            "tmux",
            "new-session",
            "-d",  # detached
            "-s",
            session_name,  # session name
        ]

        # Add working directory if specified
        if working_directory:
            tmux_cmd.extend(["-c", str(working_directory)])

        # Add the command to run
        tmux_cmd.append(command)

        # Set environment variables if provided
        env = None
        if environment:
            import os

            env = os.environ.copy()
            env.update(environment)

        # Execute with retry logic
        result = await self._run_command_with_retry(
            command=tmux_cmd,
            timeout=self.timeout_spawn,
            operation_name="create_session",
            cwd=working_directory,
        )

        if result.success:
            # Validate session was created successfully
            if await self._session_exists(session_name):
                # Track the session
                session = TmuxSession(
                    name=session_name,
                    status=TmuxSessionStatus.ACTIVE,
                    created_at=time.time(),
                    last_checked=time.time(),
                    command=command,
                    working_directory=str(working_directory)
                    if working_directory
                    else None,
                )
                self.sessions[session_name] = session

                logger.info(
                    "Tmux session created successfully",
                    session_name=session_name,
                    execution_time=result.execution_time,
                    retry_count=result.retry_count,
                )

                result.session_name = session_name
            else:
                # Session creation command succeeded but session doesn't exist
                error_msg = (
                    f"Session {session_name} creation succeeded but session not found"
                )
                logger.error("Session validation failed", session_name=session_name)
                result.success = False
                result.error_message = error_msg
                self._record_failure()

        return result

    async def terminate_session(
        self, session_name: str, force: bool = False
    ) -> TmuxOperationResult:
        """
        Terminate a tmux session gracefully or forcefully.

        Args:
            session_name: Name of the session to terminate
            force: If True, use kill-session instead of graceful termination

        Returns:
            TmuxOperationResult indicating success or failure
        """
        logger.info(
            "Terminating tmux session",
            session_name=session_name,
            force=force,
        )

        # Check if session exists
        if not await self._session_exists(session_name):
            logger.info(
                "Session does not exist, considering termination successful",
                session_name=session_name,
            )

            # Remove from tracking if it was there
            if session_name in self.sessions:
                del self.sessions[session_name]

            return TmuxOperationResult(
                success=True,
                session_name=session_name,
            )

        if not force:
            # Try graceful termination first (send Ctrl+C)
            logger.debug("Attempting graceful termination", session_name=session_name)
            graceful_result = await self._run_command_with_retry(
                command=["tmux", "send-keys", "-t", session_name, "C-c", "Enter"],
                timeout=self.timeout_command,
                operation_name="graceful_terminate",
            )

            if graceful_result.success:
                # Wait a bit for graceful shutdown
                await asyncio.sleep(2.0)

                # Check if session is gone
                if not await self._session_exists(session_name):
                    logger.info(
                        "Session terminated gracefully",
                        session_name=session_name,
                    )

                    if session_name in self.sessions:
                        self.sessions[session_name].status = TmuxSessionStatus.STOPPED
                        del self.sessions[session_name]

                    return TmuxOperationResult(
                        success=True,
                        session_name=session_name,
                    )

        # Force termination
        logger.debug("Force terminating session", session_name=session_name)
        result = await self._run_command_with_retry(
            command=["tmux", "kill-session", "-t", session_name],
            timeout=self.timeout_terminate,
            operation_name="force_terminate",
        )

        if result.success:
            # Remove from tracking
            if session_name in self.sessions:
                self.sessions[session_name].status = TmuxSessionStatus.STOPPED
                del self.sessions[session_name]

            logger.info(
                "Session terminated forcefully",
                session_name=session_name,
                execution_time=result.execution_time,
                retry_count=result.retry_count,
            )

        result.session_name = session_name
        return result

    async def get_session_status(self, session_name: str) -> TmuxSessionStatus:
        """
        Get the current status of a tmux session.

        Args:
            session_name: Name of the session to check

        Returns:
            TmuxSessionStatus indicating the current status
        """
        if await self._session_exists(session_name):
            # Update tracking if we have it
            if session_name in self.sessions:
                self.sessions[session_name].status = TmuxSessionStatus.ACTIVE
                self.sessions[session_name].last_checked = time.time()

            return TmuxSessionStatus.ACTIVE
        else:
            # Update tracking if we have it
            if session_name in self.sessions:
                self.sessions[session_name].status = TmuxSessionStatus.STOPPED
                self.sessions[session_name].last_checked = time.time()

            return TmuxSessionStatus.STOPPED

    async def list_sessions(self) -> list[TmuxSession]:
        """
        List all active tmux sessions.

        Returns:
            List of TmuxSession objects
        """
        result = await self._run_command_with_retry(
            command=["tmux", "list-sessions", "-F", "#{session_name}"],
            timeout=self.timeout_command,
            operation_name="list_sessions",
        )

        sessions = []
        if result.success and result.stdout:
            session_names = result.stdout.strip().split("\n")
            for name in session_names:
                if name:  # Skip empty lines
                    if name in self.sessions:
                        session = self.sessions[name]
                        session.status = TmuxSessionStatus.ACTIVE
                        session.last_checked = time.time()
                    else:
                        session = TmuxSession(
                            name=name,
                            status=TmuxSessionStatus.ACTIVE,
                            created_at=0.0,
                            last_checked=time.time(),
                        )
                        self.sessions[name] = session

                    sessions.append(session)

        return sessions

    async def cleanup_orphaned_sessions(self, prefix: str = "hive-") -> list[str]:
        """
        Clean up tmux sessions that match the given prefix but are not tracked.

        Args:
            prefix: Prefix to match for cleanup

        Returns:
            List of session names that were cleaned up
        """
        logger.info("Cleaning up orphaned tmux sessions", prefix=prefix)

        # Get all active sessions
        all_sessions = await self.list_sessions()
        orphaned = []

        for session in all_sessions:
            if session.name.startswith(prefix) and session.name not in self.sessions:
                logger.info(
                    "Found orphaned session",
                    session_name=session.name,
                )

                result = await self.terminate_session(session.name, force=True)
                if result.success:
                    orphaned.append(session.name)
                    logger.info(
                        "Cleaned up orphaned session", session_name=session.name
                    )
                else:
                    logger.warning(
                        "Failed to clean up orphaned session",
                        session_name=session.name,
                        error=result.error_message,
                    )

        logger.info(
            "Orphaned session cleanup completed",
            cleaned_count=len(orphaned),
            orphaned_sessions=orphaned,
        )

        return orphaned

    async def _session_exists(self, session_name: str) -> bool:
        """Check if a tmux session exists."""
        result = await self._run_command_with_retry(
            command=["tmux", "has-session", "-t", session_name],
            timeout=self.timeout_command,
            operation_name="check_session",
        )

        return result.success

    def get_session_info(self, session_name: str) -> Optional[TmuxSession]:
        """Get information about a tracked session."""
        return self.sessions.get(session_name)

    def get_all_tracked_sessions(self) -> dict[str, TmuxSession]:
        """Get all tracked sessions."""
        return self.sessions.copy()

    def get_failure_stats(self) -> dict[str, Any]:
        """Get failure statistics for monitoring."""
        return {
            "failure_count": self.failure_count,
            "last_failure_time": self.last_failure_time,
            "circuit_breaker_open": self._is_circuit_breaker_open(),
            "tracked_sessions": len(self.sessions),
        }


# Global instance
_tmux_manager: Optional[RetryableTmuxManager] = None


def get_tmux_manager() -> RetryableTmuxManager:
    """Get or create the global tmux manager instance."""
    global _tmux_manager
    if _tmux_manager is None:
        _tmux_manager = RetryableTmuxManager()
    return _tmux_manager
