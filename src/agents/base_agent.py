"""Base agent class for LeanVibe Agent Hive 2.0 with multi-CLI tool support."""

import asyncio
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import random
from pathlib import Path
from typing import Any

import structlog

# Handle both module and direct execution imports
try:
    from ..core.async_db import get_async_database_manager
    from ..core.config import CLIToolType, settings
    from ..core.constants import Intervals
    from ..core.context_engine import ContextSearchResult, get_context_engine
    from ..core.message_broker import (
        Message,
        MessageHandler,
        MessageType,
        message_broker,
    )
    from ..core.persistent_cli import get_persistent_cli_manager
    from ..core.task_queue import Task, task_queue
except ImportError:
    # Direct execution - add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from core.async_db import get_async_database_manager
    from core.config import CLIToolType, settings
    from core.context_engine import ContextSearchResult, get_context_engine
    from core.message_broker import Message, MessageHandler, MessageType, message_broker
    from core.persistent_cli import get_persistent_cli_manager
    from core.task_queue import Task, task_queue

logger = structlog.get_logger()


class HealthStatus(Enum):
    """Agent health status."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ToolResult:
    """Result from CLI tool execution."""

    success: bool
    output: str
    error: str | None = None
    tool_used: str | None = None
    execution_time: float = 0.0
    error_category: str | None = None


class ErrorCategory(str, Enum):
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    OOM = "oom"
    OTHER = "other"


@dataclass
class TaskResult:
    """Result from task processing."""

    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None
    metrics: dict[str, float] | None = None


class CLIToolManager:
    """Manages multiple CLI agentic coding tools."""

    def __init__(self):
        self.available_tools = self._detect_available_tools()
        self.preferred_tool = self._select_preferred_tool()
        self.tool_configs = self._get_tool_configs()
        # Counters and budgets
        self._counters: dict[str, dict[str, Any]] = {}
        self._tool_window_start: dict[str, float] = {}
        self._tool_attempts_in_window: dict[str, int] = {}
        self.per_tool_budget_per_minute: int = 30

    def _detect_available_tools(self) -> dict[str, dict[str, Any]]:
        """Detect available CLI tools."""
        tools = {}

        # Check opencode
        if self._check_tool_available("opencode", ["--version"]):
            tools[CLIToolType.OPENCODE] = {
                "name": "OpenCode",
                "command": "opencode",
                "execute_pattern": self._opencode_execute,
                "supports_interactive": True,
            }

        # Check Claude CLI
        if self._check_tool_available("claude", ["--version"]):
            tools[CLIToolType.CLAUDE] = {
                "name": "Claude Code CLI",
                "command": "claude",
                "execute_pattern": self._claude_execute,
                "supports_interactive": False,
            }

        # Check Gemini CLI
        if self._check_tool_available(
            "gemini", ["--version"]
        ) or self._check_tool_available("gcloud", ["ai", "--version"]):
            tools[CLIToolType.GEMINI] = {
                "name": "Gemini CLI",
                "command": "gemini",
                "execute_pattern": self._gemini_execute,
                "supports_interactive": False,
            }

        return tools

    def _check_tool_available(self, command: str, args: list[str]) -> bool:
        """Check if a CLI tool is available."""
        try:
            result = subprocess.run([command] + args, capture_output=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def _select_preferred_tool(self) -> CLIToolType | None:
        """Select preferred tool based on configuration and availability."""
        # Use configured preference if available
        if (
            settings.preferred_cli_tool
            and settings.preferred_cli_tool in self.available_tools
        ):
            return settings.preferred_cli_tool

        # Default priority order
        priority_order = [CLIToolType.OPENCODE, CLIToolType.CLAUDE, CLIToolType.GEMINI]

        for tool in priority_order:
            if tool in self.available_tools:
                return tool

        return None

    def _get_tool_configs(self) -> dict[CLIToolType, dict[str, Any]]:
        """Get configuration for each tool."""
        return {
            CLIToolType.OPENCODE: {
                "timeout": settings.cli_tool_timeout,
                "max_context_length": 8000,
                "supports_files": True,
            },
            CLIToolType.CLAUDE: {
                "timeout": settings.cli_tool_timeout,
                "max_context_length": 100000,
                "supports_files": True,
            },
            CLIToolType.GEMINI: {
                "timeout": settings.cli_tool_timeout,
                "max_context_length": 30000,
                "supports_files": False,
            },
        }

    @staticmethod
    def classify_error_text(text: str) -> str:
        """Classify stderr/stdout text into an ErrorCategory string.

        Lightweight heuristics to surface actionable categories.
        """
        if not text:
            return ErrorCategory.OTHER.value
        lower = text.lower()
        if "rate limit" in lower or "too many requests" in lower or "429" in lower:
            return ErrorCategory.RATE_LIMIT.value
        if "timed out" in lower or "timeout" in lower:
            return ErrorCategory.TIMEOUT.value
        if "unauthorized" in lower or "forbidden" in lower or "401" in lower or "403" in lower:
            return ErrorCategory.AUTH.value
        if "out of memory" in lower or "oom" in lower or "memory limit" in lower:
            return ErrorCategory.OOM.value
        return ErrorCategory.OTHER.value

    async def execute_prompt(
        self, prompt: str, tool_override: CLIToolType | None = None
    ) -> ToolResult:
        """Execute a prompt using available CLI tools."""
        start_time = time.time()

        # Select tool to use
        tool_to_use = tool_override or self.preferred_tool

        if not tool_to_use or tool_to_use not in self.available_tools:
            return ToolResult(
                success=False,
                output="",
                error="No suitable CLI tool available",
                execution_time=time.time() - start_time,
            )

        tool_config = self.available_tools[tool_to_use]

        # Budget enforcement per tool (rolling 60s window)
        tool_key = tool_to_use.value
        now = time.time()
        window_start = self._tool_window_start.get(tool_key, now)
        if now - window_start >= 60:
            self._tool_window_start[tool_key] = now
            self._tool_attempts_in_window[tool_key] = 0
        attempts = self._tool_attempts_in_window.get(tool_key, 0)
        if attempts >= self.per_tool_budget_per_minute:
            self._increment_counters(tool_key, success=False, category=ErrorCategory.RATE_LIMIT.value)
            return ToolResult(
                success=False,
                output="",
                error=f"Per-minute budget exceeded for {tool_key}",
                execution_time=time.time() - start_time,
                error_category=ErrorCategory.RATE_LIMIT.value,
            )
        # Count attempted call in window
        self._tool_attempts_in_window[tool_key] = attempts + 1

        # Try primary tool once, with a single timed backoff retry on timeout/rate limit
        for attempt in range(2):
            try:
                result = await tool_config["execute_pattern"](prompt)
                result.tool_used = tool_to_use.value
                result.execution_time = time.time() - start_time
                if result.success:
                    logger.info(
                        "CLI tool execution successful",
                        tool=tool_to_use.value,
                        execution_time=result.execution_time,
                    )
                    self._increment_counters(tool_key, success=True)
                    return result
                # If timeout or rate limit, optionally backoff once
                if (
                    attempt == 0
                    and result.error_category in {ErrorCategory.TIMEOUT.value, ErrorCategory.RATE_LIMIT.value}
                ):
                    delay = min(1.0 * (2 ** attempt), 3.0) + random.uniform(0, 0.2)
                    logger.warning(
                        "Primary tool transient failure; backing off",
                        tool=tool_to_use.value,
                        category=result.error_category,
                        delay=delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                self._increment_counters(tool_key, success=False, category=result.error_category)
                break
            except Exception as e:
                logger.error(
                    "CLI tool execution error", tool=tool_to_use.value, error=str(e)
                )
                self._increment_counters(tool_key, success=False, category=ErrorCategory.OTHER.value)
                break

        # Try fallback tools
        for fallback_tool in self.available_tools:
            if fallback_tool == tool_to_use:
                continue

            try:
                tool_config = self.available_tools[fallback_tool]
                result = await tool_config["execute_pattern"](prompt)
                result.tool_used = fallback_tool.value
                result.execution_time = time.time() - start_time

                if result.success:
                    logger.info(
                        "Fallback CLI tool execution successful",
                        tool=fallback_tool.value,
                        execution_time=result.execution_time,
                    )
                    self._increment_counters(fallback_tool.value, success=True)
                    return result

            except Exception as e:
                logger.error(
                    "Fallback CLI tool execution error",
                    tool=fallback_tool.value,
                    error=str(e),
                )
                self._increment_counters(fallback_tool.value, success=False, category=ErrorCategory.OTHER.value)

        # All tools failed
        return ToolResult(
            success=False,
            output="",
            error="All CLI tools failed",
            execution_time=time.time() - start_time,
            error_category=ErrorCategory.OTHER.value,
        )

    def _increment_counters(self, tool_key: str, success: bool, category: str | None = None) -> None:
        """Increment per-tool counters with minimal overhead."""
        c = self._counters.setdefault(tool_key, {"calls": 0, "success": 0, "failure": 0, "by_category": {}})
        c["calls"] += 1
        if success:
            c["success"] += 1
        else:
            c["failure"] += 1
            if category:
                c["by_category"][category] = c["by_category"].get(category, 0) + 1

    def get_counters(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of CLI counters per tool."""
        # Return a shallow copy to avoid external mutation
        return {k: {**v, "by_category": dict(v.get("by_category", {}))} for k, v in self._counters.items()}

    async def _opencode_execute(self, prompt: str) -> ToolResult:
        """Execute prompt using opencode."""
        try:
            # opencode works best in interactive mode, but we'll use it in batch mode
            process = await asyncio.create_subprocess_exec(
                "opencode",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode()),
                timeout=settings.cli_tool_timeout,
            )

            if process.returncode == 0:
                return ToolResult(success=True, output=stdout.decode(), error=None)
            else:
                stderr_text = stderr.decode() if stderr else ""
                return ToolResult(
                    success=False,
                    output=stdout.decode(),
                    error=stderr_text,
                    error_category=self.classify_error_text(stderr_text),
                )

        except TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error="opencode execution timed out",
                error_category=ErrorCategory.TIMEOUT.value,
            )

    async def _claude_execute(self, prompt: str) -> ToolResult:
        """Execute prompt using Claude CLI."""
        try:
            process = await asyncio.create_subprocess_exec(
                "claude",
                "--no-interactive",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=settings.cli_tool_timeout
            )

            if process.returncode == 0:
                return ToolResult(success=True, output=stdout.decode(), error=None)
            else:
                stderr_text = stderr.decode() if stderr else ""
                return ToolResult(
                    success=False,
                    output=stdout.decode(),
                    error=stderr_text,
                    error_category=self.classify_error_text(stderr_text),
                )

        except TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error="Claude CLI execution timed out",
                error_category=ErrorCategory.TIMEOUT.value,
            )

    async def _gemini_execute(self, prompt: str) -> ToolResult:
        """Execute prompt using Gemini CLI."""
        try:
            process = await asyncio.create_subprocess_exec(
                "gemini",
                "code",
                prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=settings.cli_tool_timeout
            )

            if process.returncode == 0:
                return ToolResult(success=True, output=stdout.decode(), error=None)
            else:
                stderr_text = stderr.decode() if stderr else ""
                return ToolResult(
                    success=False,
                    output=stdout.decode(),
                    error=stderr_text,
                    error_category=self.classify_error_text(stderr_text),
                )

        except TimeoutError:
            return ToolResult(
                success=False,
                output="",
                error="Gemini CLI execution timed out",
                error_category=ErrorCategory.TIMEOUT.value,
            )


class BaseAgent(ABC):
    """Abstract base class for all agents with multi-CLI tool support."""

    def __init__(
        self, name: str, agent_type: str, role: str, enhanced_communication: bool = True
    ):
        self.name = name
        self.agent_type = agent_type
        self.role = role
        self.status = "inactive"
        self.current_task_id: str | None = None
        self.start_time = time.time()
        self.agent_uuid: str | None = None  # Database UUID for this agent
        self.enhanced_communication = enhanced_communication

        # Initialize components
        self.cli_tools = CLIToolManager()
        self.message_handler = MessageHandler(name)
        self.async_db_manager = None  # Will be initialized in start()
        self.persistent_cli = get_persistent_cli_manager()
        self.cli_session_id = f"{name}_{int(time.time())}"
        self.cli_session = None

        # Enhanced communication components (initialized by default)
        if enhanced_communication:
            try:
                from ..core.enhanced_message_broker import get_enhanced_message_broker
                from ..core.realtime_collaboration import get_collaboration_sync

                self.enhanced_broker = get_enhanced_message_broker()
                self.collaboration_sync = get_collaboration_sync(self.enhanced_broker)

                logger.info(
                    "Enhanced communication initialized",
                    agent=self.name,
                    has_enhanced_broker=True,
                    has_collaboration_sync=True,
                )
            except ImportError as e:
                logger.warning(
                    "Enhanced communication not available, falling back to basic messaging",
                    agent=self.name,
                    error=str(e),
                )
                self.enhanced_broker = None
                self.collaboration_sync = None
        else:
            self.enhanced_broker = None
            self.collaboration_sync = None

        # Performance tracking
        self.tasks_completed = 0
        self.tasks_failed = 0
        self.total_execution_time = 0.0

        # Rate limiting
        self._last_cli_call = 0.0
        self._cli_call_count = 0
        self._rate_limit_window = 60.0  # 1 minute
        self._max_calls_per_window = 30

        # Setup message handlers
        self._setup_message_handlers()

        logger.info(
            "Base agent initialized",
            agent=self.name,
            type=self.agent_type,
            available_cli_tools=list(self.cli_tools.available_tools.keys()),
        )

    def _setup_message_handlers(self):
        """Setup message handlers for common topics."""
        self.message_handler.register_handler("ping", self._handle_ping)
        self.message_handler.register_handler(
            "task_assignment", self._handle_task_assignment
        )
        self.message_handler.register_handler("health_check", self._handle_health_check)
        self.message_handler.register_handler("shutdown", self._handle_shutdown)

        # Collaboration message handlers
        self.message_handler.register_handler(
            "collaboration_invitation", self._handle_collaboration_invitation
        )
        self.message_handler.register_handler("task_ready", self._handle_task_ready)
        self.message_handler.register_handler(
            "task_reassigned", self._handle_task_reassigned
        )
        self.message_handler.register_handler(
            "collaboration_completed", self._handle_collaboration_completed
        )
        self.message_handler.register_handler(
            "collaboration_failed", self._handle_collaboration_failed
        )

    async def start(self) -> None:
        """Start the agent."""
        self.status = "active"
        logger.info("Agent starting", agent=self.name)

        try:
            # Create persistent CLI session
            logger.info("Creating persistent CLI session", agent=self.name)
            try:
                preferred_tool = self.cli_tools.preferred_tool
                if preferred_tool:
                    tool_type = preferred_tool.value
                    self.cli_session = await asyncio.wait_for(
                        self.persistent_cli.create_session(
                            session_id=self.cli_session_id,
                            tool_type=tool_type,
                            initial_prompt=f"Hello! I'm {self.name}, a {self.agent_type} agent. I'm ready to work on coding tasks.",
                        ),
                        timeout=30.0,
                    )
                    logger.info(
                        "Persistent CLI session created",
                        agent=self.name,
                        session_id=self.cli_session_id,
                        tool_type=tool_type,
                    )
                else:
                    logger.warning(
                        "No CLI tools available for persistent session", agent=self.name
                    )
            except TimeoutError:
                logger.warning(
                    "CLI session creation timeout - continuing without persistent session",
                    agent=self.name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to create persistent CLI session - continuing",
                    agent=self.name,
                    error=str(e),
                )

            # Initialize async database manager
            logger.info("Initializing database manager", agent=self.name)
            try:
                self.async_db_manager = await asyncio.wait_for(
                    get_async_database_manager(settings.database_url), timeout=10.0
                )
                logger.info("Database manager initialized", agent=self.name)
            except TimeoutError:
                logger.warning(
                    "Database manager initialization timeout - continuing without DB",
                    agent=self.name,
                )
                self.async_db_manager = None
            except Exception as e:
                logger.warning(
                    "Database manager initialization failed - continuing",
                    agent=self.name,
                    error=str(e),
                )
                self.async_db_manager = None

            # Initialize context engine with timeout (optional)
            logger.info("Initializing context engine", agent=self.name)
            try:
                self.context_engine = await asyncio.wait_for(
                    get_context_engine(settings.database_url), timeout=15.0
                )
                logger.info("Context engine initialized", agent=self.name)
            except TimeoutError:
                logger.warning(
                    "Context engine initialization timeout - continuing without context engine",
                    agent=self.name,
                )
                self.context_engine = None
            except Exception as e:
                logger.warning(
                    "Context engine initialization failed - continuing",
                    agent=self.name,
                    error=str(e),
                )
                self.context_engine = None

            # Initialize task queue with timeout (optional)
            logger.info("Initializing task queue", agent=self.name)
            try:
                await asyncio.wait_for(task_queue.initialize(), timeout=10.0)
                logger.info("Task queue initialized", agent=self.name)
            except TimeoutError:
                logger.warning(
                    "Task queue initialization timeout - continuing", agent=self.name
                )
            except Exception as e:
                logger.warning(
                    "Task queue initialization failed - continuing",
                    agent=self.name,
                    error=str(e),
                )

            # Register with message broker with timeout (optional)
            logger.info("Initializing message broker", agent=self.name)
            try:
                await asyncio.wait_for(message_broker.initialize(), timeout=5.0)
                await asyncio.wait_for(
                    message_broker.start_listening(self.name, self.message_handler),
                    timeout=5.0,
                )
                logger.info("Message broker initialized", agent=self.name)
            except TimeoutError:
                logger.warning(
                    "Message broker initialization timeout - continuing",
                    agent=self.name,
                )
            except Exception as e:
                logger.warning(
                    "Message broker initialization failed - continuing",
                    agent=self.name,
                    error=str(e),
                )

            # Register agent in database (non-blocking, best effort)
            logger.info("Registering agent in database", agent=self.name)
            try:
                await asyncio.wait_for(self._register_agent(), timeout=5.0)
            except TimeoutError:
                logger.warning(
                    "Database registration timeout - continuing without DB registration",
                    agent=self.name,
                )
            except Exception as e:
                logger.warning(
                    "Database registration failed - continuing",
                    agent=self.name,
                    error=str(e),
                )

            logger.info(
                "Agent initialization complete - starting main loop", agent=self.name
            )
            # Start main execution loop
            await self.run()

        except Exception as e:
            logger.error("Agent error", agent=self.name, error=str(e))
            self.status = "error"
            raise
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        """Cleanup resources."""
        self.status = "inactive"

        # Close persistent CLI session
        if self.cli_session:
            try:
                await self.persistent_cli.close_session(self.cli_session_id)
                logger.info("Persistent CLI session closed", agent=self.name)
            except Exception as e:
                logger.warning(
                    "Failed to close CLI session", agent=self.name, error=str(e)
                )

        await message_broker.stop_listening(self.name)
        logger.info("Agent stopped", agent=self.name)

    @abstractmethod
    async def run(self) -> None:
        """Main agent execution loop - must be implemented by subclasses."""
        pass

    async def execute_with_cli_tool(
        self, prompt: str, tool_override: CLIToolType | None = None, prompt_file: str | None = None
    ) -> ToolResult:
        """Execute prompt using CLI tools with persistent session support."""

        # Rate limiting
        current_time = time.time()
        if current_time - self._last_cli_call < self._rate_limit_window:
            if self._cli_call_count >= self._max_calls_per_window:
                await asyncio.sleep(Intervals.AGENT_BRIEF_DELAY)  # Brief delay
                self._cli_call_count = 0
        else:
            self._cli_call_count = 0

        self._last_cli_call = current_time
        self._cli_call_count += 1

        # If a prompt file is provided, prefer loading its contents to reduce round-trips
        if prompt_file:
            try:
                prompt_path = Path(prompt_file)
                if prompt_path.exists():
                    prompt = prompt_path.read_text()
            except Exception as e:
                logger.warning("Failed to read prompt file", path=prompt_file, error=str(e))

        # Try persistent session first
        if self.cli_session and self.cli_session.status == "active":
            try:
                logger.info(
                    "Using persistent CLI session",
                    agent=self.name,
                    session_id=self.cli_session_id,
                )

                start_time = time.time()
                response = await self.persistent_cli.send_command(
                    self.cli_session_id, prompt
                )
                execution_time = time.time() - start_time

                result = ToolResult(
                    success=True,
                    output=response,
                    tool_used=self.cli_session.tool_type,
                    execution_time=execution_time,
                )

                logger.info(
                    "Persistent CLI execution successful",
                    agent=self.name,
                    execution_time=execution_time,
                )

            except Exception as e:
                logger.warning(
                    "Persistent CLI session failed, falling back to standard execution",
                    agent=self.name,
                    error=str(e),
                )
                # Fall back to standard CLI execution
                result = await self.cli_tools.execute_prompt(prompt, tool_override)
        else:
            # Use standard CLI execution
            result = await self.cli_tools.execute_prompt(prompt, tool_override)

        # Store execution context
        if result.success and self.context_engine and self.agent_uuid:
            try:
                await self.store_context(
                    content=f"CLI Execution:\nPrompt: {prompt[:200]}...\nResult: {result.output[:500]}...",
                    importance_score=0.6,
                    category="cli_execution",
                    metadata={
                        "tool_used": result.tool_used,
                        "execution_time": result.execution_time,
                        "prompt_length": len(prompt),
                        "output_length": len(result.output),
                        "used_persistent_session": self.cli_session is not None
                        and self.cli_session.status == "active",
                    },
                )
            except Exception as e:
                logger.warning("Failed to store CLI execution context", error=str(e))

        return result

    async def process_task(self, task: Task) -> TaskResult:
        """Process a task - can be overridden by subclasses."""
        start_time = time.time()
        self.current_task_id = task.id

        try:
            # Update task status
            await task_queue.start_task(task.id)

            # Store task context
            if self.context_engine and self.agent_uuid:
                try:
                    await self.store_context(
                        content=f"Processing task: {task.title}\n{task.description}",
                        importance_score=0.7,
                        category="task_processing",
                        metadata={
                            "task_id": task.id,
                            "task_type": task.task_type,
                            "priority": task.priority,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to store task processing context", error=str(e)
                    )

            # Process the task
            result = await self._process_task_implementation(task)

            # Update metrics
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            if result.success:
                self.tasks_completed += 1
                await task_queue.complete_task(task.id, result.data)

                # Store success context
                if self.context_engine and self.agent_uuid:
                    try:
                        await self.store_context(
                            content=f"Task completed successfully: {task.title}",
                            importance_score=0.8,
                            category="task_completion",
                            metadata={
                                "task_id": task.id,
                                "execution_time": execution_time,
                                "result_data": result.data,
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to store task completion context", error=str(e)
                        )
            else:
                self.tasks_failed += 1
                await task_queue.fail_task(task.id, result.error or "Unknown error")

                # Store failure context
                if self.context_engine and self.agent_uuid:
                    try:
                        await self.store_context(
                            content=f"Task failed: {task.title}\nError: {result.error}",
                            importance_score=0.9,  # Failures are important to remember
                            category="task_failure",
                            metadata={
                                "task_id": task.id,
                                "error": result.error,
                                "execution_time": execution_time,
                            },
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to store task failure context", error=str(e)
                        )

            # Record metrics
            await self._record_task_metrics(task, result, execution_time)

            return result

        except Exception as e:
            self.tasks_failed += 1
            error_msg = f"Task processing exception: {str(e)}"
            await task_queue.fail_task(task.id, error_msg)

            logger.error(
                "Task processing failed", agent=self.name, task_id=task.id, error=str(e)
            )

            return TaskResult(success=False, error=error_msg)
        finally:
            self.current_task_id = None

    async def _process_task_implementation(self, task: Task) -> TaskResult:
        """Default task processing implementation."""
        # This is a basic implementation - subclasses should override

        # Retrieve relevant context
        context_results = []
        if self.context_engine and self.agent_uuid:
            try:
                context_results = await self.retrieve_context(
                    f"task {task.task_type} {task.title}", limit=5
                )
            except Exception as e:
                logger.warning("Failed to retrieve context", error=str(e))

        # Build prompt with context
        context_text = "\n".join(
            [f"Context: {r.context.content[:200]}..." for r in context_results]
        )

        prompt = f"""
        Task: {task.title}
        Description: {task.description}
        Type: {task.task_type}
        Priority: {task.priority}

        Relevant Context:
        {context_text}

        Please process this task and provide a detailed response.
        """

        # Execute with CLI tool
        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            return TaskResult(
                success=True,
                data={
                    "output": result.output,
                    "tool_used": result.tool_used,
                    "execution_time": result.execution_time,
                },
                metrics={
                    "execution_time": result.execution_time,
                    "output_length": len(result.output),
                },
            )
        else:
            return TaskResult(
                success=False,
                error=result.error,
                metrics={"execution_time": result.execution_time},
            )

    async def send_message(
        self,
        to_agent: str,
        topic: str,
        content: dict[str, Any],
        message_type: MessageType = MessageType.DIRECT,
    ) -> str:
        """Send message to another agent."""

        message_id = await message_broker.send_message(
            from_agent=self.name,
            to_agent=to_agent,
            topic=topic,
            payload=content,
            message_type=message_type,
        )

        logger.info(
            "Message sent",
            from_agent=self.name,
            to_agent=to_agent,
            topic=topic,
            message_id=message_id,
        )

        return message_id

    # Enhanced Communication Methods
    async def create_shared_work_session(
        self,
        title: str,
        participants: list[str] = None,
        task: str = None,
        shared_resources: dict[str, Any] = None,
    ) -> str | None:
        """Create a shared work session for collaboration."""
        if not self.collaboration_sync:
            logger.warning("Enhanced communication not available", agent=self.name)
            return None

        try:
            session_id = await self.collaboration_sync.start_collaboration_session(
                title=title,
                coordinator=self.name,
                initial_participants=set(participants or [self.name]),
                shared_resources=shared_resources or {},
            )

            logger.info(
                "Shared work session created",
                agent=self.name,
                session_id=session_id,
                title=title,
                participants=len(participants or []),
            )

            return session_id
        except Exception as e:
            logger.error(
                "Failed to create shared work session", agent=self.name, error=str(e)
            )
            return None

    async def join_shared_work_session(self, session_id: str) -> bool:
        """Join an existing shared work session."""
        if not self.collaboration_sync:
            logger.warning("Enhanced communication not available", agent=self.name)
            return False

        try:
            success = await self.collaboration_sync.join_collaboration_session(
                session_id, self.name
            )

            if success:
                logger.info(
                    "Joined shared work session", agent=self.name, session_id=session_id
                )

            return success
        except Exception as e:
            logger.error(
                "Failed to join shared work session",
                agent=self.name,
                session_id=session_id,
                error=str(e),
            )
            return False

    async def share_work_context(
        self,
        context_type: str,
        data: dict[str, Any],
        participants: set[str] = None,
    ) -> str | None:
        """Share work context with other agents."""
        if not self.enhanced_broker:
            logger.warning("Enhanced communication not available", agent=self.name)
            return None

        try:
            # Map context type string to enum
            from ..core.enhanced_message_broker import ContextShareType

            context_type_mapping = {
                "work_session": ContextShareType.WORK_SESSION,
                "knowledge_base": ContextShareType.KNOWLEDGE_BASE,
                "task_state": ContextShareType.TASK_STATE,
                "performance_metrics": ContextShareType.PERFORMANCE_METRICS,
                "error_patterns": ContextShareType.ERROR_PATTERNS,
                "decision_history": ContextShareType.DECISION_HISTORY,
            }

            context_enum = context_type_mapping.get(
                context_type, ContextShareType.WORK_SESSION
            )

            context_id = await self.enhanced_broker.create_shared_context(
                context_type=context_enum,
                owner_agent=self.name,
                initial_data=data,
                participants=participants or {self.name},
            )

            logger.info(
                "Work context shared",
                agent=self.name,
                context_id=context_id,
                context_type=context_type,
            )

            return context_id
        except Exception as e:
            logger.error("Failed to share work context", agent=self.name, error=str(e))
            return None

    async def send_enhanced_message(
        self,
        to_agent: str,
        topic: str,
        content: dict[str, Any],
        context_ids: list[str] = None,
        priority: str = "normal",
        include_context: bool = True,
    ) -> bool:
        """Send enhanced message with context awareness."""
        if not self.enhanced_broker:
            logger.warning(
                "Enhanced communication not available, using basic messaging",
                agent=self.name,
            )
            await self.send_message(to_agent, topic, content)
            return True

        try:
            # Map priority string to enum
            from ..core.enhanced_message_broker import MessagePriority

            priority_mapping = {
                "critical": MessagePriority.CRITICAL,
                "high": MessagePriority.HIGH,
                "normal": MessagePriority.NORMAL,
                "low": MessagePriority.LOW,
            }

            message_priority = priority_mapping.get(priority, MessagePriority.NORMAL)

            if include_context and context_ids:
                success = await self.enhanced_broker.send_context_aware_message(
                    from_agent=self.name,
                    to_agent=to_agent,
                    topic=topic,
                    payload=content,
                    context_ids=context_ids,
                    include_relevant_context=True,
                )
            else:
                message_id = await self.enhanced_broker.send_priority_message(
                    from_agent=self.name,
                    to_agent=to_agent,
                    topic=topic,
                    payload=content,
                    priority=message_priority,
                )
                success = message_id is not None

            logger.info(
                "Enhanced message sent",
                agent=self.name,
                to_agent=to_agent,
                topic=topic,
                priority=priority,
                has_context=bool(context_ids),
            )

            return success
        except Exception as e:
            logger.error(
                "Failed to send enhanced message",
                agent=self.name,
                to_agent=to_agent,
                error=str(e),
            )
            return False

    async def update_agent_status(
        self,
        status: str,
        current_task: str = None,
        capabilities: list[str] = None,
        performance_metrics: dict[str, float] = None,
    ) -> None:
        """Update agent status for enhanced coordination."""
        if not self.enhanced_broker:
            return

        try:
            state_updates = {"status": status}

            if current_task:
                state_updates["current_task"] = current_task
            if capabilities:
                state_updates["capabilities"] = capabilities
            if performance_metrics:
                state_updates["performance_metrics"] = performance_metrics

            await self.enhanced_broker.update_agent_state(
                agent_name=self.name, state_updates=state_updates
            )

            logger.debug(
                "Agent status updated",
                agent=self.name,
                status=status,
                task=current_task,
            )
        except Exception as e:
            logger.error("Failed to update agent status", agent=self.name, error=str(e))

    async def get_collaboration_opportunities(self) -> list[dict[str, Any]]:
        """Get available collaboration opportunities."""
        if not self.enhanced_broker:
            return []

        try:
            # Get current agent states to find collaboration opportunities
            agent_states = await self.enhanced_broker.get_agent_states(self.name)

            opportunities = []
            for agent_name, state in agent_states.items():
                if agent_name != self.name and state.get("status") != "inactive":
                    # Look for agents with complementary capabilities
                    agent_capabilities = state.get("capabilities", [])
                    if agent_capabilities:
                        opportunities.append(
                            {
                                "agent_name": agent_name,
                                "status": state.get("status"),
                                "capabilities": agent_capabilities,
                                "current_task": state.get("current_task"),
                                "shared_contexts": state.get("shared_contexts", []),
                            }
                        )

            return opportunities
        except Exception as e:
            logger.error(
                "Failed to get collaboration opportunities",
                agent=self.name,
                error=str(e),
            )
            return []

    async def store_context(
        self,
        content: str,
        importance_score: float = 0.5,
        category: str = "general",
        topic: str = None,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Store context in semantic memory."""
        if not self.async_db_manager or not self.agent_uuid:
            logger.warning(
                "Cannot store context - database manager or agent UUID not available"
            )
            return ""

        return await self.async_db_manager.store_context(
            agent_id=self.agent_uuid,
            content=content,
            importance_score=importance_score,
            category=category,
            topic=topic,
            metadata=metadata,
        )

    async def retrieve_context(
        self,
        query: str,
        limit: int = 10,
        category_filter: str = None,
        min_importance: float = 0.0,
    ) -> list[ContextSearchResult]:
        """Retrieve relevant context from semantic memory."""
        if not self.context_engine or not self.agent_uuid:
            logger.warning(
                "Cannot retrieve context - context engine or agent UUID not available"
            )
            return []

        return await self.context_engine.retrieve_context(
            query=query,
            agent_id=self.agent_uuid,
            limit=limit,
            category_filter=category_filter,
            min_importance=min_importance,
        )

    async def health_check(self) -> HealthStatus:
        """Check agent health."""
        try:
            # Check CLI tools availability
            if not self.cli_tools.available_tools:
                return HealthStatus.UNHEALTHY

            # Check if preferred tool is available
            if not self.cli_tools.preferred_tool:
                return HealthStatus.DEGRADED

            # Check CLI session health
            if self.cli_session:
                try:
                    session_status = await self.persistent_cli.get_session_status(
                        self.cli_session_id
                    )
                    if session_status.get("status") != "active":
                        return HealthStatus.DEGRADED
                except Exception as e:
                    logger.warning("CLI session health check failed", error=str(e))
                    return HealthStatus.DEGRADED

            # Check database connection
            if self.async_db_manager:
                try:
                    db_healthy = await self.async_db_manager.health_check()
                    if not db_healthy:
                        return HealthStatus.UNHEALTHY
                except Exception as e:
                    logger.warning("Database health check failed", error=str(e))
                    return HealthStatus.DEGRADED

            # Check recent performance
            if (
                self.tasks_failed > self.tasks_completed * 0.5
            ):  # More than 50% failure rate
                return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.UNKNOWN

    async def _register_agent(self):
        """Register agent in database."""
        if not self.async_db_manager:
            logger.warning(
                "Cannot register agent - async database manager not available"
            )
            return

        try:
            self.agent_uuid = await self.async_db_manager.register_agent(
                name=self.name,
                agent_type=self.agent_type,
                role=self.role,
                capabilities=self._get_capabilities(),
                tmux_session=f"hive-{self.name}",
            )
            logger.info(
                "Agent registered in database", agent=self.name, uuid=self.agent_uuid
            )

        except Exception as e:
            logger.warning(
                "Failed to register agent in database", agent=self.name, error=str(e)
            )
            # Continue without database registration - agent can still function

    def _get_capabilities(self) -> dict[str, Any]:
        """Get agent capabilities."""
        # Normalize preferred_tool to string for tests where enum may be stubbed as str
        preferred_tool = (
            self.cli_tools.preferred_tool.value
            if getattr(self.cli_tools.preferred_tool, "value", None) is not None
            else self.cli_tools.preferred_tool
        )

        capabilities = {
            "cli_tools": list(self.cli_tools.available_tools.keys()),
            "preferred_tool": preferred_tool,
            "supports_context": True,
            "supports_messaging": True,
            "supports_persistent_sessions": True,
            "enhanced_communication": self.enhanced_communication,
            "supports_collaboration": self.collaboration_sync is not None,
            "supports_shared_contexts": self.enhanced_broker is not None,
            "supports_real_time_sync": self.collaboration_sync is not None,
            "agent_version": "2.0.0",
        }

        # Add CLI session info if available
        if self.cli_session:
            capabilities["cli_session"] = {
                "session_id": self.cli_session_id,
                "tool_type": self.cli_session.tool_type,
                "status": self.cli_session.status,
                "created_at": self.cli_session.created_at,
            }

        # Add enhanced communication features
        if self.enhanced_communication:
            capabilities["enhanced_features"] = {
                "shared_contexts": self.enhanced_broker is not None,
                "real_time_collaboration": self.collaboration_sync is not None,
                "priority_messaging": self.enhanced_broker is not None,
                "context_aware_messaging": self.enhanced_broker is not None,
                "agent_state_sync": self.enhanced_broker is not None,
            }

        return capabilities

    @property
    def capabilities(self) -> list[str]:
        """Get agent capabilities as a list for collaboration matching."""
        caps = self._get_capabilities()

        # Extract capability names for matching
        capability_list = []

        # Add basic capabilities
        if caps.get("supports_context"):
            capability_list.append("context_management")
        if caps.get("supports_messaging"):
            capability_list.append("messaging")
        if caps.get("supports_persistent_sessions"):
            capability_list.append("persistent_sessions")

        # Add CLI tool capabilities
        for tool in caps.get("cli_tools", []):
            capability_list.append(f"cli_{tool}")

        # Add enhanced communication capabilities
        if caps.get("enhanced_communication"):
            capability_list.append("enhanced_communication")
        if caps.get("supports_collaboration"):
            capability_list.append("real_time_collaboration")
        if caps.get("supports_shared_contexts"):
            capability_list.append("shared_contexts")

        # Add agent type as capability
        capability_list.append(self.agent_type)
        capability_list.append(self.role)

        return capability_list

    async def _record_task_metrics(
        self, task: Task, result: TaskResult, execution_time: float
    ):
        """Record task execution metrics."""
        if not self.async_db_manager or not self.agent_uuid:
            logger.warning(
                "Cannot record metrics - database manager or agent UUID not available"
            )
            return

        try:
            # Record execution time metric
            await self.async_db_manager.record_system_metric(
                metric_name="task_execution_time",
                metric_type="histogram",
                value=execution_time,
                unit="seconds",
                agent_id=self.agent_uuid,
                task_id=task.id,
                labels={
                    "task_type": task.task_type,
                    "agent_type": self.agent_type,
                    "success": str(result.success).lower(),
                },
            )

            # Record success/failure metric
            await self.async_db_manager.record_system_metric(
                metric_name="task_completion",
                metric_type="counter",
                value=1.0 if result.success else 0.0,
                unit="count",
                agent_id=self.agent_uuid,
                task_id=task.id,
                labels={
                    "task_type": task.task_type,
                    "agent_type": self.agent_type,
                    "result": "success" if result.success else "failure",
                },
            )

        except Exception as e:
            logger.error("Failed to record metrics", agent=self.name, error=str(e))

    # Collaboration support
    async def initiate_collaboration(
        self,
        title: str,
        description: str,
        collaboration_type: str,
        required_capabilities: list[str] = None,
        deadline: datetime = None,
        priority: int = 5,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Initiate a new collaboration with other agents."""

        # Import here to avoid circular imports
        try:
            from core.agent_coordination import CollaborationType, coordination_system
        except ImportError:
            from ..core.agent_coordination import CollaborationType, coordination_system

        collaboration_id = await coordination_system.start_collaboration(
            title=title,
            description=description,
            collaboration_type=CollaborationType(collaboration_type),
            coordinator_agent=self.name,
            required_capabilities=required_capabilities,
            deadline=deadline,
            priority=priority,
            metadata=metadata or {},
        )

        logger.info(
            "Collaboration initiated",
            agent=self.name,
            collaboration_id=collaboration_id,
            type=collaboration_type,
        )

        return collaboration_id

    async def request_collaboration(
        self,
        title: str,
        description: str,
        collaboration_type: str,
        required_capabilities: list[str] = None,
        deadline: datetime = None,
        priority: int = 5,
    ) -> dict[str, Any]:
        """Request a collaboration through the coordination system."""

        message_id = await message_broker.send_message(
            from_agent=self.name,
            to_agent="coordination_system",
            topic="collaboration_request",
            payload={
                "title": title,
                "description": description,
                "collaboration_type": collaboration_type,
                "required_capabilities": required_capabilities,
                "deadline": deadline.isoformat() if deadline else None,
                "priority": priority,
                "metadata": {"requester": self.name},
            },
            message_type=MessageType.REQUEST,
        )

        # In a real implementation, we would wait for the response
        return {"request_id": message_id, "status": "requested"}

    async def complete_collaborative_task(
        self, collaboration_id: str, task_id: str, result: dict[str, Any]
    ) -> None:
        """Complete a collaborative task and notify the coordination system."""

        await message_broker.send_message(
            from_agent=self.name,
            to_agent="coordination_system",
            topic="sub_task_completed",
            payload={
                "collaboration_id": collaboration_id,
                "task_id": task_id,
                "result": result,
                "completed_at": time.time(),
            },
            message_type=MessageType.NOTIFICATION,
        )

        logger.info(
            "Collaborative task completed",
            agent=self.name,
            collaboration_id=collaboration_id,
            task_id=task_id,
        )

    # Message handlers
    async def _handle_ping(self, message: Message) -> dict[str, Any]:
        """Handle ping message."""
        return {
            "pong": True,
            "timestamp": time.time(),
            "status": self.status,
            "uptime": time.time() - self.start_time,
        }

    async def _handle_task_assignment(self, message: Message) -> dict[str, Any]:
        """Handle task assignment message."""
        task_data = message.payload.get("task")
        if task_data:
            task = Task(**task_data)
            result = await self.process_task(task)

            return {
                "task_id": task.id,
                "success": result.success,
                "error": result.error,
            }
        else:
            return {"error": "No task data provided"}

    async def _handle_collaboration_invitation(
        self, message: Message
    ) -> dict[str, Any]:
        """Handle collaboration invitation."""
        payload = message.payload
        collaboration_id = payload["collaboration_id"]

        logger.info(
            "Received collaboration invitation",
            agent=self.name,
            collaboration_id=collaboration_id,
            title=payload["title"],
            type=payload["collaboration_type"],
        )

        # Accept the invitation (subclasses can override this logic)
        accepted = await self._evaluate_collaboration_invitation(payload)

        if accepted:
            # Process assigned tasks
            your_tasks = payload.get("your_tasks", [])
            for task in your_tasks:
                await self._handle_collaborative_task(collaboration_id, task)

        return {"accepted": accepted}

    async def _evaluate_collaboration_invitation(
        self, invitation: dict[str, Any]
    ) -> bool:
        """Evaluate whether to accept a collaboration invitation."""
        # Default implementation accepts all invitations
        # Subclasses can override with more sophisticated logic

        required_capabilities = invitation.get("required_capabilities", [])
        my_capabilities = self.capabilities

        # Check if we have required capabilities
        if required_capabilities:
            has_capabilities = any(
                cap in my_capabilities for cap in required_capabilities
            )
            if not has_capabilities:
                logger.warning(
                    "Declining collaboration - missing capabilities",
                    agent=self.name,
                    required=required_capabilities,
                    available=my_capabilities,
                )
                return False

        # Check current load
        if self.status == "busy":
            logger.info("Declining collaboration - currently busy", agent=self.name)
            return False

        return True

    async def _handle_collaborative_task(
        self, collaboration_id: str, task: dict[str, Any]
    ) -> None:
        """Handle a collaborative task assignment."""

        task_id = f"{collaboration_id}_{task.get('description', 'task')}"

        logger.info(
            "Processing collaborative task",
            agent=self.name,
            collaboration_id=collaboration_id,
            task_description=task.get("description", ""),
        )

        try:
            # Process the collaborative task
            result = await self._process_collaborative_task(task)

            # Report completion
            await self.complete_collaborative_task(collaboration_id, task_id, result)

        except Exception as e:
            logger.error(
                "Collaborative task failed",
                agent=self.name,
                collaboration_id=collaboration_id,
                error=str(e),
            )

            # Report failure
            await self.complete_collaborative_task(
                collaboration_id, task_id, {"success": False, "error": str(e)}
            )

    async def _process_collaborative_task(self, task: dict[str, Any]) -> dict[str, Any]:
        """Process a collaborative task."""
        # Default implementation - subclasses should override
        dependencies = task.get("depends_on", [])

        # Wait for dependencies if any
        if dependencies:
            logger.info(
                "Waiting for task dependencies",
                agent=self.name,
                dependencies=dependencies,
            )
            # In a real implementation, we would wait for dependency completion

        # Simulate task processing
        await asyncio.sleep(Intervals.AGENT_STARTUP_DELAY)

        return {
            "success": True,
            "output": f"Collaborative task completed by {self.name}",
            "agent": self.name,
            "timestamp": time.time(),
        }

    async def _handle_task_ready(self, message: Message) -> dict[str, Any]:
        """Handle notification that a task is ready (dependencies met)."""
        payload = message.payload
        collaboration_id = payload["collaboration_id"]
        task_id = payload["task_id"]
        task = payload["task"]

        logger.info(
            "Task ready for execution",
            agent=self.name,
            collaboration_id=collaboration_id,
            task_id=task_id,
        )

        # Start processing the task
        await self._handle_collaborative_task(collaboration_id, task)

        return {"status": "processing"}

    async def _handle_task_reassigned(self, message: Message) -> dict[str, Any]:
        """Handle task reassignment notification."""
        payload = message.payload
        collaboration_id = payload["collaboration_id"]
        task_id = payload["task_id"]
        task = payload["task"]
        reason = payload.get("reason", "unknown")

        logger.info(
            "Task reassigned to agent",
            agent=self.name,
            collaboration_id=collaboration_id,
            task_id=task_id,
            reason=reason,
        )

        # Accept the reassigned task
        await self._handle_collaborative_task(collaboration_id, task)

        return {"status": "accepted"}

    async def _handle_collaboration_completed(self, message: Message) -> dict[str, Any]:
        """Handle collaboration completion notification."""
        payload = message.payload
        collaboration_id = payload["collaboration_id"]

        logger.info(
            "Collaboration completed",
            agent=self.name,
            collaboration_id=collaboration_id,
            duration=payload.get("duration", 0),
            success=payload.get("success", False),
        )

        # Subclasses can override to perform cleanup or learning
        await self._on_collaboration_completed(payload)

        return {"status": "acknowledged"}

    async def _handle_collaboration_failed(self, message: Message) -> dict[str, Any]:
        """Handle collaboration failure notification."""
        payload = message.payload
        collaboration_id = payload["collaboration_id"]
        reason = payload.get("reason", "unknown")

        logger.warning(
            "Collaboration failed",
            agent=self.name,
            collaboration_id=collaboration_id,
            reason=reason,
        )

        # Subclasses can override to handle failure recovery
        await self._on_collaboration_failed(payload)

        return {"status": "acknowledged"}

    async def _on_collaboration_completed(self, result: dict[str, Any]) -> None:
        """Called when a collaboration is completed."""
        # Default implementation does nothing
        # Subclasses can override for learning or cleanup
        pass

    async def _on_collaboration_failed(self, failure_info: dict[str, Any]) -> None:
        """Called when a collaboration fails."""
        # Default implementation does nothing
        # Subclasses can override for failure handling
        pass

    async def _handle_health_check(self, message: Message) -> dict[str, Any]:
        """Handle health check message."""
        health = await self.health_check()

        response = {
            "health_status": health.value,
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "status": self.status,
            "uptime": time.time() - self.start_time,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "cli_tools_available": list(self.cli_tools.available_tools.keys()),
            "preferred_cli_tool": self.cli_tools.preferred_tool.value
            if self.cli_tools.preferred_tool
            else None,
        }

        # Add CLI session info if available
        if self.cli_session:
            try:
                session_status = await self.persistent_cli.get_session_status(
                    self.cli_session_id
                )
                response["cli_session"] = session_status
            except Exception as e:
                response["cli_session"] = {"error": str(e)}

        return response

    async def _handle_shutdown(self, message: Message) -> dict[str, Any]:
        """Handle shutdown message."""
        logger.info("Received shutdown message", agent=self.name)
        self.status = "stopping"

        # Graceful shutdown
        await asyncio.sleep(
            Intervals.AGENT_SHUTDOWN_GRACE
        )  # Allow current operations to complete

        return {"shutdown": True, "timestamp": time.time()}

    async def discover_collaboration_opportunities(self) -> list[dict[str, Any]]:
        """Discover available agents for collaboration."""
        try:
            # Get all active agents from the communication system
            active_agents = []

            # In a real implementation, this would query the agent registry
            # For now, return mock data based on current system state
            opportunities = [
                {
                    "agent_name": "qa_agent",
                    "agent_type": "qa",
                    "status": "busy",
                    "capabilities": ["code_review", "testing", "security_analysis"],
                    "availability": "high",
                },
                {
                    "agent_name": "architect_agent",
                    "agent_type": "architect",
                    "status": "busy",
                    "capabilities": [
                        "system_design",
                        "architecture_review",
                        "scalability_analysis",
                    ],
                    "availability": "medium",
                },
            ]

            logger.info(
                "Discovered collaboration opportunities",
                agent=self.name,
                opportunities=len(opportunities),
            )
            return opportunities

        except Exception as e:
            logger.error(
                "Failed to discover collaboration opportunities",
                agent=self.name,
                error=str(e),
            )
            return []
