"""Base agent class for LeanVibe Agent Hive 2.0 with multi-CLI tool support."""

import asyncio
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

# Handle both module and direct execution imports
try:
    from ..core.config import CLIToolType, settings
    from ..core.context_engine import ContextSearchResult, get_context_engine
    from ..core.message_broker import (
        Message,
        MessageHandler,
        MessageType,
        message_broker,
    )
    from ..core.models import Agent as AgentModel
    from ..core.models import SystemMetric, get_database_manager
    from ..core.task_queue import Task, task_queue
except ImportError:
    # Direct execution - add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from core.config import CLIToolType, settings
    from core.context_engine import ContextSearchResult, get_context_engine
    from core.message_broker import Message, MessageHandler, MessageType, message_broker
    from core.models import Agent as AgentModel
    from core.models import SystemMetric, get_database_manager
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

        try:
            # Execute with selected tool
            result = await tool_config["execute_pattern"](prompt)
            result.tool_used = tool_to_use.value
            result.execution_time = time.time() - start_time

            if result.success:
                logger.info(
                    "CLI tool execution successful",
                    tool=tool_to_use.value,
                    execution_time=result.execution_time,
                )
                return result
            else:
                logger.warning(
                    "CLI tool execution failed, trying fallback",
                    tool=tool_to_use.value,
                    error=result.error,
                )

        except Exception as e:
            logger.error(
                "CLI tool execution error", tool=tool_to_use.value, error=str(e)
            )

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
                    return result

            except Exception as e:
                logger.error(
                    "Fallback CLI tool execution error",
                    tool=fallback_tool.value,
                    error=str(e),
                )

        # All tools failed
        return ToolResult(
            success=False,
            output="",
            error="All CLI tools failed",
            execution_time=time.time() - start_time,
        )

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
                return ToolResult(
                    success=False, output=stdout.decode(), error=stderr.decode()
                )

        except TimeoutError:
            return ToolResult(
                success=False, output="", error="opencode execution timed out"
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
                return ToolResult(
                    success=False, output=stdout.decode(), error=stderr.decode()
                )

        except TimeoutError:
            return ToolResult(
                success=False, output="", error="Claude CLI execution timed out"
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
                return ToolResult(
                    success=False, output=stdout.decode(), error=stderr.decode()
                )

        except TimeoutError:
            return ToolResult(
                success=False, output="", error="Gemini CLI execution timed out"
            )


class BaseAgent(ABC):
    """Abstract base class for all agents with multi-CLI tool support."""

    def __init__(self, name: str, agent_type: str, role: str):
        self.name = name
        self.agent_type = agent_type
        self.role = role
        self.status = "inactive"
        self.current_task_id: str | None = None
        self.start_time = time.time()
        self.agent_uuid: str | None = None  # Database UUID for this agent

        # Initialize components
        self.cli_tools = CLIToolManager()
        self.message_handler = MessageHandler(name)
        self.db_manager = get_database_manager(settings.database_url)

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

    async def start(self) -> None:
        """Start the agent."""
        self.status = "active"
        logger.info("Agent starting", agent=self.name)

        try:
            # Initialize context engine
            self.context_engine = await get_context_engine(settings.database_url)

            # Initialize task queue
            await task_queue.initialize()

            # Register with message broker
            await message_broker.initialize()
            await message_broker.start_listening(self.name, self.message_handler)

            # Register agent in database
            await self._register_agent()

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
        await message_broker.stop_listening(self.name)
        logger.info("Agent stopped", agent=self.name)

    @abstractmethod
    async def run(self) -> None:
        """Main agent execution loop - must be implemented by subclasses."""
        pass

    async def execute_with_cli_tool(
        self, prompt: str, tool_override: CLIToolType | None = None
    ) -> ToolResult:
        """Execute prompt using CLI tools with rate limiting."""

        # Rate limiting
        current_time = time.time()
        if current_time - self._last_cli_call < self._rate_limit_window:
            if self._cli_call_count >= self._max_calls_per_window:
                await asyncio.sleep(1)  # Brief delay
                self._cli_call_count = 0
        else:
            self._cli_call_count = 0

        self._last_cli_call = current_time
        self._cli_call_count += 1

        # Execute with CLI tools
        result = await self.cli_tools.execute_prompt(prompt, tool_override)

        # Store execution context
        if result.success:
            await self.store_context(
                content=f"CLI Execution:\nPrompt: {prompt[:200]}...\nResult: {result.output[:500]}...",
                importance_score=0.6,
                category="cli_execution",
                metadata={
                    "tool_used": result.tool_used,
                    "execution_time": result.execution_time,
                    "prompt_length": len(prompt),
                    "output_length": len(result.output),
                },
            )

        return result

    async def process_task(self, task: Task) -> TaskResult:
        """Process a task - can be overridden by subclasses."""
        start_time = time.time()
        self.current_task_id = task.id

        try:
            # Update task status
            await task_queue.start_task(task.id)

            # Store task context
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

            # Process the task
            result = await self._process_task_implementation(task)

            # Update metrics
            execution_time = time.time() - start_time
            self.total_execution_time += execution_time

            if result.success:
                self.tasks_completed += 1
                await task_queue.complete_task(task.id, result.data)

                # Store success context
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
            else:
                self.tasks_failed += 1
                await task_queue.fail_task(task.id, result.error or "Unknown error")

                # Store failure context
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
        context_results = await self.retrieve_context(
            f"task {task.task_type} {task.title}", limit=5
        )

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

    async def store_context(
        self,
        content: str,
        importance_score: float = 0.5,
        category: str = "general",
        topic: str = None,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Store context in semantic memory."""

        context_id = await self.context_engine.store_context(
            agent_id=self.agent_uuid,
            content=content,
            importance_score=importance_score,
            category=category,
            topic=topic,
            metadata=metadata,
        )

        return context_id

    async def retrieve_context(
        self,
        query: str,
        limit: int = 10,
        category_filter: str = None,
        min_importance: float = 0.0,
    ) -> list[ContextSearchResult]:
        """Retrieve relevant context from semantic memory."""

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

            # Check database connection
            db_session = self.db_manager.get_session()
            try:
                db_session.execute("SELECT 1")
                db_session.close()
            except Exception:
                return HealthStatus.UNHEALTHY

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
        db_session = self.db_manager.get_session()
        try:
            # Check if agent exists
            existing_agent = (
                db_session.query(AgentModel).filter_by(name=self.name).first()
            )

            if existing_agent:
                # Update existing agent
                existing_agent.status = "active"
                existing_agent.last_heartbeat = datetime.fromtimestamp(time.time())
                self.agent_uuid = str(existing_agent.id)  # Store the UUID
            else:
                # Create new agent
                agent = AgentModel(
                    name=self.name,
                    type=self.agent_type,
                    role=self.role,
                    capabilities=self._get_capabilities(),
                    status="active",
                )
                db_session.add(agent)
                db_session.flush()  # Flush to get the ID
                self.agent_uuid = str(agent.id)  # Store the UUID

            db_session.commit()

        except Exception as e:
            db_session.rollback()
            logger.error("Failed to register agent", agent=self.name, error=str(e))
            raise
        finally:
            db_session.close()

    def _get_capabilities(self) -> dict[str, Any]:
        """Get agent capabilities."""
        return {
            "cli_tools": list(self.cli_tools.available_tools.keys()),
            "preferred_tool": self.cli_tools.preferred_tool.value
            if self.cli_tools.preferred_tool
            else None,
            "supports_context": True,
            "supports_messaging": True,
            "agent_version": "2.0.0",
        }

    async def _record_task_metrics(
        self, task: Task, result: TaskResult, execution_time: float
    ):
        """Record task execution metrics."""
        db_session = self.db_manager.get_session()
        try:
            # Record execution time metric
            metric = SystemMetric(
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
            db_session.add(metric)

            # Record success/failure metric
            success_metric = SystemMetric(
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
            db_session.add(success_metric)

            db_session.commit()

        except Exception as e:
            db_session.rollback()
            logger.error("Failed to record metrics", agent=self.name, error=str(e))
        finally:
            db_session.close()

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

    async def _handle_health_check(self, message: Message) -> dict[str, Any]:
        """Handle health check message."""
        health = await self.health_check()

        return {
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

    async def _handle_shutdown(self, message: Message) -> dict[str, Any]:
        """Handle shutdown message."""
        logger.info("Received shutdown message", agent=self.name)
        self.status = "stopping"

        # Graceful shutdown
        await asyncio.sleep(1)  # Allow current operations to complete

        return {"shutdown": True, "timestamp": time.time()}
