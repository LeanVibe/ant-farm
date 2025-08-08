"""Persistent CLI session manager for maintaining long-running coding sessions."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class CLISession:
    """Represents a persistent CLI session."""

    session_id: str
    tool_type: str  # opencode, claude, gemini
    process: Optional[asyncio.subprocess.Process] = None
    input_file: Optional[str] = None
    output_file: Optional[str] = None
    status: str = "inactive"  # inactive, starting, active, error
    created_at: float = 0.0
    last_activity: float = 0.0
    conversation_history: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.conversation_history is None:
            self.conversation_history = []
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.last_activity == 0.0:
            self.last_activity = time.time()


class PersistentCLIManager:
    """Manages persistent CLI sessions for agents."""

    def __init__(self, workspace_dir: str = None):
        self.workspace_dir = workspace_dir or tempfile.mkdtemp(prefix="hive_cli_")
        self.sessions: Dict[str, CLISession] = {}
        self.session_files_dir = os.path.join(self.workspace_dir, "sessions")
        os.makedirs(self.session_files_dir, exist_ok=True)

        # Session management
        self.max_sessions_per_tool = 3
        self.session_timeout = 3600  # 1 hour
        self.heartbeat_interval = 30  # 30 seconds

        logger.info("Persistent CLI Manager initialized", workspace=self.workspace_dir)

    async def create_session(
        self, session_id: str, tool_type: str = "opencode", initial_prompt: str = None
    ) -> CLISession:
        """Create a new persistent CLI session."""

        if session_id in self.sessions:
            logger.warning("Session already exists", session_id=session_id)
            return self.sessions[session_id]

        # Create session workspace
        session_workspace = os.path.join(self.session_files_dir, session_id)
        os.makedirs(session_workspace, exist_ok=True)

        # Create input/output files for session communication
        input_file = os.path.join(session_workspace, "input.txt")
        output_file = os.path.join(session_workspace, "output.txt")
        history_file = os.path.join(session_workspace, "history.json")

        # Create session object
        session = CLISession(
            session_id=session_id,
            tool_type=tool_type,
            input_file=input_file,
            output_file=output_file,
            status="starting",
        )

        self.sessions[session_id] = session

        try:
            # Start the CLI tool process
            if tool_type == "opencode":
                await self._start_opencode_session(
                    session, session_workspace, initial_prompt
                )
            elif tool_type == "claude":
                await self._start_claude_session(
                    session, session_workspace, initial_prompt
                )
            elif tool_type == "gemini":
                await self._start_gemini_session(
                    session, session_workspace, initial_prompt
                )
            else:
                raise ValueError(f"Unsupported tool type: {tool_type}")

            session.status = "active"
            logger.info(
                "CLI session created", session_id=session_id, tool_type=tool_type
            )

        except Exception as e:
            session.status = "error"
            logger.error(
                "Failed to create CLI session", session_id=session_id, error=str(e)
            )
            raise

        return session

    async def _start_opencode_session(
        self, session: CLISession, workspace: str, initial_prompt: str = None
    ):
        """Start an OpenCode session in the workspace."""

        # Create a startup script that keeps OpenCode running
        startup_script = os.path.join(workspace, "startup.sh")

        with open(startup_script, "w") as f:
            f.write(f"""#!/bin/bash
cd "{workspace}"

# Start opencode in interactive mode
# We'll pipe commands through named pipes
mkfifo input_pipe output_pipe 2>/dev/null || true

# Start opencode with workspace context
opencode --workspace="{workspace}" < input_pipe > output_pipe 2>&1 &
OPENCODE_PID=$!

# Keep the session alive and monitor for commands
while kill -0 $OPENCODE_PID 2>/dev/null; do
    if [ -f "input.txt" ] && [ -s "input.txt" ]; then
        # Send the command to opencode
        cat "input.txt" > input_pipe
        # Wait for response and capture it
        timeout 30 cat output_pipe > "output.txt" 2>&1 || echo "Command timed out" > "output.txt"
        # Clear the input file
        > "input.txt"
        # Update last activity
        echo "$(date +%s)" > "last_activity.txt"
    fi
    sleep 1
done
""")

        os.chmod(startup_script, 0o755)

        # Start the session script
        session.process = await asyncio.create_subprocess_exec(
            "bash",
            startup_script,
            cwd=workspace,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait a moment for startup
        await asyncio.sleep(2)

        # Send initial prompt if provided
        if initial_prompt:
            await self.send_command(session.session_id, initial_prompt)

    async def _start_claude_session(
        self, session: CLISession, workspace: str, initial_prompt: str = None
    ):
        """Start a Claude CLI session."""

        # Claude CLI doesn't support persistent sessions, so we'll simulate it
        # by maintaining conversation history and context
        session.conversation_history = []

        if initial_prompt:
            # Store the initial prompt in history
            session.conversation_history.append(
                {"role": "user", "content": initial_prompt, "timestamp": time.time()}
            )

            # Execute the initial prompt
            await self._execute_claude_command(session, initial_prompt)

    async def _start_gemini_session(
        self, session: CLISession, workspace: str, initial_prompt: str = None
    ):
        """Start a Gemini CLI session."""

        # Similar to Claude, Gemini doesn't support persistent sessions
        session.conversation_history = []

        if initial_prompt:
            session.conversation_history.append(
                {"role": "user", "content": initial_prompt, "timestamp": time.time()}
            )

            await self._execute_gemini_command(session, initial_prompt)

    async def send_command(self, session_id: str, command: str) -> str:
        """Send a command to a persistent CLI session."""

        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.sessions[session_id]
        session.last_activity = time.time()

        try:
            if session.tool_type == "opencode":
                return await self._send_opencode_command(session, command)
            elif session.tool_type == "claude":
                return await self._execute_claude_command(session, command)
            elif session.tool_type == "gemini":
                return await self._execute_gemini_command(session, command)
            else:
                raise ValueError(f"Unsupported tool type: {session.tool_type}")

        except Exception as e:
            logger.error("Failed to send command", session_id=session_id, error=str(e))
            session.status = "error"
            raise

    async def _send_opencode_command(self, session: CLISession, command: str) -> str:
        """Send command to OpenCode session."""

        # Add command to conversation history
        session.conversation_history.append(
            {"role": "user", "content": command, "timestamp": time.time()}
        )

        # Write command to input file
        with open(session.input_file, "w") as f:
            f.write(command + "\n")

        # Wait for response with timeout
        response_timeout = 60  # 1 minute
        start_time = time.time()

        while time.time() - start_time < response_timeout:
            if (
                os.path.exists(session.output_file)
                and os.path.getsize(session.output_file) > 0
            ):
                with open(session.output_file, "r") as f:
                    response = f.read().strip()

                if response:
                    # Add response to conversation history
                    session.conversation_history.append(
                        {
                            "role": "assistant",
                            "content": response,
                            "timestamp": time.time(),
                        }
                    )

                    # Clear output file for next command
                    open(session.output_file, "w").close()

                    return response

            await asyncio.sleep(0.5)

        # Timeout occurred
        timeout_msg = "Command timed out - no response received"
        session.conversation_history.append(
            {"role": "system", "content": timeout_msg, "timestamp": time.time()}
        )
        return timeout_msg

    async def _execute_claude_command(self, session: CLISession, command: str) -> str:
        """Execute command using Claude CLI with conversation history."""

        # Build context from conversation history
        context_parts = []
        for msg in session.conversation_history[-10:]:  # Last 10 messages for context
            role = msg["role"]
            content = msg["content"]
            context_parts.append(f"{role}: {content}")

        # Prepare full prompt with context
        full_prompt = f"""
Previous conversation:
{chr(10).join(context_parts)}

Current request: {command}

Please provide a response that takes into account the previous conversation context.
"""

        try:
            # Execute Claude CLI command
            process = await asyncio.create_subprocess_exec(
                "claude",
                "--no-interactive",
                full_prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

            if process.returncode == 0:
                response = stdout.decode().strip()
            else:
                response = f"Error: {stderr.decode().strip()}"

            # Add to conversation history
            session.conversation_history.append(
                {"role": "user", "content": command, "timestamp": time.time()}
            )
            session.conversation_history.append(
                {"role": "assistant", "content": response, "timestamp": time.time()}
            )

            return response

        except asyncio.TimeoutError:
            timeout_msg = "Claude command timed out"
            session.conversation_history.append(
                {"role": "system", "content": timeout_msg, "timestamp": time.time()}
            )
            return timeout_msg

    async def _execute_gemini_command(self, session: CLISession, command: str) -> str:
        """Execute command using Gemini CLI with conversation history."""

        # Similar implementation to Claude but for Gemini
        context_parts = []
        for msg in session.conversation_history[-10:]:
            role = msg["role"]
            content = msg["content"]
            context_parts.append(f"{role}: {content}")

        full_prompt = f"""
Previous conversation:
{chr(10).join(context_parts)}

Current request: {command}

Please provide a response that maintains conversation context.
"""

        try:
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "code",
                full_prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)

            if process.returncode == 0:
                response = stdout.decode().strip()
            else:
                response = f"Error: {stderr.decode().strip()}"

            # Add to conversation history
            session.conversation_history.append(
                {"role": "user", "content": command, "timestamp": time.time()}
            )
            session.conversation_history.append(
                {"role": "assistant", "content": response, "timestamp": time.time()}
            )

            return response

        except asyncio.TimeoutError:
            timeout_msg = "Gemini command timed out"
            session.conversation_history.append(
                {"role": "system", "content": timeout_msg, "timestamp": time.time()}
            )
            return timeout_msg

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a CLI session."""

        if session_id not in self.sessions:
            return {"status": "not_found"}

        session = self.sessions[session_id]

        status = {
            "session_id": session_id,
            "tool_type": session.tool_type,
            "status": session.status,
            "created_at": session.created_at,
            "last_activity": session.last_activity,
            "uptime": time.time() - session.created_at,
            "idle_time": time.time() - session.last_activity,
            "conversation_length": len(session.conversation_history),
            "process_alive": False,
        }

        # Check if process is still alive
        if session.process:
            try:
                status["process_alive"] = session.process.returncode is None
            except:
                status["process_alive"] = False

        return status

    async def close_session(self, session_id: str) -> bool:
        """Close a CLI session."""

        if session_id not in self.sessions:
            logger.warning(
                "Attempted to close non-existent session", session_id=session_id
            )
            return False

        session = self.sessions[session_id]

        try:
            # Terminate the process if it exists
            if session.process:
                try:
                    session.process.terminate()
                    await asyncio.wait_for(session.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    logger.warning(
                        "Process didn't terminate gracefully, killing",
                        session_id=session_id,
                    )
                    session.process.kill()
                    await session.process.wait()

            # Save conversation history
            session_workspace = os.path.join(self.session_files_dir, session_id)
            history_file = os.path.join(session_workspace, "history.json")

            with open(history_file, "w") as f:
                json.dump(
                    {
                        "session_id": session_id,
                        "tool_type": session.tool_type,
                        "created_at": session.created_at,
                        "closed_at": time.time(),
                        "conversation_history": session.conversation_history,
                    },
                    f,
                    indent=2,
                )

            # Remove from active sessions
            del self.sessions[session_id]

            logger.info("CLI session closed", session_id=session_id)
            return True

        except Exception as e:
            logger.error("Failed to close session", session_id=session_id, error=str(e))
            return False

    async def cleanup_idle_sessions(self):
        """Clean up idle sessions that have exceeded timeout."""

        current_time = time.time()
        idle_sessions = []

        for session_id, session in self.sessions.items():
            idle_time = current_time - session.last_activity
            if idle_time > self.session_timeout:
                idle_sessions.append(session_id)

        for session_id in idle_sessions:
            logger.info("Cleaning up idle session", session_id=session_id)
            await self.close_session(session_id)

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""

        sessions = []
        for session_id in self.sessions:
            status = await self.get_session_status(session_id)
            sessions.append(status)

        return sessions

    async def start_heartbeat_monitor(self):
        """Start monitoring sessions for heartbeat and cleanup."""

        async def monitor():
            while True:
                try:
                    await self.cleanup_idle_sessions()
                    await asyncio.sleep(self.heartbeat_interval)
                except Exception as e:
                    logger.error("Error in session monitor", error=str(e))
                    await asyncio.sleep(5)

        # Start monitoring task
        asyncio.create_task(monitor())
        logger.info("Session heartbeat monitor started")


# Global persistent CLI manager instance
_persistent_cli_manager: Optional[PersistentCLIManager] = None


def get_persistent_cli_manager(workspace_dir: str = None) -> PersistentCLIManager:
    """Get the global persistent CLI manager instance."""
    global _persistent_cli_manager

    if _persistent_cli_manager is None:
        _persistent_cli_manager = PersistentCLIManager(workspace_dir)

    return _persistent_cli_manager
