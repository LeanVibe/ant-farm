"""Tmux backend protocol and implementations for injection into AgentSpawner.

Provides two backends:
- SubprocessTmuxBackend: direct tmux subprocess calls (test-aligned)
- TmuxManagerBackend: uses RetryableTmuxManager for resilience (prod default)
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .tmux_manager import RetryableTmuxManager, get_tmux_manager


class TmuxBackendProtocol:
    async def create_session(self, session_name: str, command: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    async def terminate_session(self, session_name: str) -> bool:  # pragma: no cover - interface
        raise NotImplementedError


class SubprocessTmuxBackend(TmuxBackendProtocol):
    """Direct tmux subprocess backend to preserve existing test behavior."""

    def __init__(self, project_root: Path):
        self.project_root = project_root

    async def create_session(self, session_name: str, command: str) -> bool:
        subprocess.run(
            [
                "tmux",
                "new-session",
                "-d",
                "-s",
                session_name,
                "-c",
                str(self.project_root),
                command,
            ],
            check=True,
        )
        check = subprocess.run(["tmux", "has-session", "-t", session_name], capture_output=True)
        return check.returncode == 0

    async def terminate_session(self, session_name: str) -> bool:
        # Graceful then force
        subprocess.run(["tmux", "send-keys", "-t", session_name, "C-c", "Enter"], check=False)
        result = subprocess.run(["tmux", "kill-session", "-t", session_name], capture_output=True)
        return result.returncode == 0


class TmuxManagerBackend(TmuxBackendProtocol):
    """Resilient backend using RetryableTmuxManager."""

    def __init__(self, manager: RetryableTmuxManager | None = None, project_root: Path | None = None):
        self.manager = manager or get_tmux_manager()
        self.project_root = project_root

    async def create_session(self, session_name: str, command: str) -> bool:
        result = await self.manager.create_session(
            session_name=session_name,
            command=command,
            working_directory=self.project_root,
        )
        return result.success

    async def terminate_session(self, session_name: str) -> bool:
        result = await self.manager.terminate_session(session_name, force=True)
        return result.success


def select_default_tmux_backend(project_root: Path) -> TmuxBackendProtocol:
    """Select default backend based on env flag.

    If HIVE_TMUX_BACKEND=subprocess, use SubprocessTmuxBackend for test-aligned behavior.
    Otherwise default to TmuxManagerBackend for production resilience.
    """
    backend = os.getenv("HIVE_TMUX_BACKEND")
    if backend:
        backend = backend.lower()
    else:
        # Default to subprocess during tests to preserve existing behavior
        # pytest sets PYTEST_CURRENT_TEST; use manager in normal runtime
        backend = "subprocess" if os.getenv("PYTEST_CURRENT_TEST") else "manager"
    if backend == "subprocess":
        return SubprocessTmuxBackend(project_root)
    return TmuxManagerBackend(project_root=project_root)
