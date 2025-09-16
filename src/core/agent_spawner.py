# Agent Spawner - Handles agent process creation and management

import asyncio
import subprocess
import time
from pathlib import Path
from typing import List, Optional

import structlog

from .interfaces import AgentInfo, AgentStatus, IAgentFactory

logger = structlog.get_logger()


class AgentSpawner:
    """Handles spawning and terminating agent processes using tmux sessions."""

    def __init__(
        self,
        project_root: Path,
        agent_factory: IAgentFactory,
        max_concurrent_spawns: int = 5
    ):
        self.project_root = project_root
        self.agent_factory = agent_factory
        self.max_concurrent_spawns = max_concurrent_spawns
        self._spawn_semaphore = asyncio.Semaphore(max_concurrent_spawns)

    async def spawn_agent(
        self,
        agent_type: str,
        agent_name: str,
        capabilities: Optional[List[str]] = None
    ) -> Optional[str]:
        """Spawn a new agent using tmux session."""
        async with self._spawn_semaphore:
            if capabilities is None:
                capabilities = []

            session_name = f"hive-{agent_name}"

            logger.info(
                "Spawning agent",
                agent_type=agent_type,
                agent_name=agent_name,
                session_name=session_name,
                capabilities=capabilities
            )

            try:
                # Build command to run in tmux session
                command = self._build_agent_command(agent_type, agent_name)

                # Create tmux session
                success = await self._create_tmux_session(session_name, command)
                if not success:
                    logger.error("Failed to create tmux session", session_name=session_name)
                    return None

                # Wait for agent to initialize
                await asyncio.sleep(2)  # Give agent time to start

                # Verify session is running
                if await self._verify_session_running(session_name):
                    logger.info("Agent spawned successfully", agent_name=agent_name, session_name=session_name)
                    return session_name
                else:
                    logger.error("Agent session failed to start", agent_name=agent_name, session_name=session_name)
                    # Clean up failed session
                    await self._cleanup_failed_session(session_name)
                    return None

            except Exception as e:
                logger.error(
                    "Failed to spawn agent",
                    agent_type=agent_type,
                    agent_name=agent_name,
                    error=str(e)
                )
                # Clean up on error
                await self._cleanup_failed_session(session_name)
                return None

    async def terminate_agent(self, agent_name: str, session_name: str) -> bool:
        """Terminate an agent tmux session."""
        try:
            logger.info("Terminating agent", agent_name=agent_name, session_name=session_name)

            # Terminate tmux session
            success = await self._terminate_tmux_session(session_name)

            if success:
                logger.info("Agent terminated successfully", agent_name=agent_name, session_name=session_name)
                return True
            else:
                logger.error("Failed to terminate agent session", agent_name=agent_name, session_name=session_name)
                return False

        except Exception as e:
            logger.error(
                "Error terminating agent",
                agent_name=agent_name,
                session_name=session_name,
                error=str(e)
            )
            return False

    async def get_session_status(self, session_name: str) -> Optional[str]:
        """Get the status of a tmux session."""
        try:
            result = await asyncio.create_subprocess_exec(
                "tmux", "has-session", "-t", session_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            if result.returncode == 0:
                return "running"
            else:
                return "stopped"

        except Exception as e:
            logger.error("Failed to get session status", session_name=session_name, error=str(e))
            return None

    async def list_active_sessions(self, prefix: str = "hive-") -> List[str]:
        """List all active tmux sessions with the given prefix."""
        try:
            result = await asyncio.create_subprocess_exec(
                "tmux", "list-sessions", "-F", "#{session_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()

            if result.returncode == 0:
                sessions = stdout.decode().strip().split('\n')
                return [s for s in sessions if s.startswith(prefix)]
            else:
                logger.warning("Failed to list tmux sessions", error=stderr.decode())
                return []

        except Exception as e:
            logger.error("Error listing tmux sessions", error=str(e))
            return []

    async def cleanup_orphaned_sessions(self, prefix: str = "hive-") -> List[str]:
        """Clean up tmux sessions that don't have corresponding agents."""
        try:
            active_sessions = await self.list_active_sessions(prefix)

            # For now, return all sessions - in a real implementation,
            # this would check against the agent registry
            orphaned = []
            for session in active_sessions:
                # Check if session is actually orphaned
                # This is a simplified implementation
                if await self._is_session_orphaned(session):
                    await self._terminate_tmux_session(session)
                    orphaned.append(session)

            if orphaned:
                logger.info("Cleaned up orphaned sessions", count=len(orphaned), sessions=orphaned)

            return orphaned

        except Exception as e:
            logger.error("Error cleaning up orphaned sessions", error=str(e))
            return []

    def _build_agent_command(self, agent_type: str, agent_name: str) -> str:
        """Build the command to run the agent."""
        # Use uv to run the agent
        return f"cd {self.project_root} && uv run python -m src.agents.runner --type {agent_type} --name {agent_name}"

    async def _create_tmux_session(self, session_name: str, command: str) -> bool:
        """Create a new tmux session with the given command."""
        try:
            # Create detached tmux session
            result = await asyncio.create_subprocess_exec(
                "tmux", "new-session", "-d", "-s", session_name, command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            return result.returncode == 0

        except Exception as e:
            logger.error("Failed to create tmux session", session_name=session_name, error=str(e))
            return False

    async def _terminate_tmux_session(self, session_name: str) -> bool:
        """Terminate a tmux session."""
        try:
            result = await asyncio.create_subprocess_exec(
                "tmux", "kill-session", "-t", session_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.wait()

            return result.returncode == 0

        except Exception as e:
            logger.error("Failed to terminate tmux session", session_name=session_name, error=str(e))
            return False

    async def _verify_session_running(self, session_name: str) -> bool:
        """Verify that a tmux session is running and healthy."""
        try:
            # Check if session exists
            status = await self.get_session_status(session_name)
            if status != "running":
                return False

            # Additional health checks could be added here
            # For example, checking if the process is actually running
            return True

        except Exception as e:
            logger.error("Error verifying session", session_name=session_name, error=str(e))
            return False

    async def _cleanup_failed_session(self, session_name: str) -> None:
        """Clean up a failed tmux session."""
        try:
            await self._terminate_tmux_session(session_name)
        except Exception as e:
            # Ignore cleanup errors
            pass

    async def _is_session_orphaned(self, session_name: str) -> bool:
        """Check if a session is orphaned (no corresponding agent)."""
        # This is a simplified implementation
        # In a real system, this would check against the agent registry
        # For now, we'll consider sessions older than a certain threshold as potentially orphaned

        try:
            # Get session creation time (simplified)
            # This would need more sophisticated logic in a real implementation
            return False  # Placeholder - don't orphan sessions automatically

        except Exception:
            return False