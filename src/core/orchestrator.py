"""Agent orchestrator for lifecycle management and coordination."""

import asyncio
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog

from .async_db import get_async_database_manager
from .constants import Intervals
from .enums import AgentStatus
from .message_broker import message_broker
from .models import Agent
from .short_id import ShortIDGenerator
from .task_queue import Task, task_queue
from .tmux_manager import get_tmux_manager, TmuxOperationResult

logger = structlog.get_logger()


@dataclass
class AgentInfo:
    """Agent information structure."""

    id: str
    name: str
    type: str
    role: str
    status: AgentStatus
    capabilities: list[str]
    tmux_session: str | None
    last_heartbeat: float
    created_at: float
    tasks_completed: int
    tasks_failed: int
    current_task_id: str | None = None
    load_factor: float = 0.0
    short_id: str | None = None


@dataclass
class SystemHealth:
    """System health metrics."""

    total_agents: int
    active_agents: int
    idle_agents: int
    busy_agents: int
    error_agents: int
    avg_load_factor: float
    queue_size: int
    tasks_per_minute: float


class AgentRegistry:
    """Manages agent registration and tracking using SQLAlchemy."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.db_manager = None  # Will be initialized async
        self.agents: dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the registry and load existing agents from database."""
        self.db_manager = await get_async_database_manager(self.database_url)
        await self.load_agents_from_database()

    async def load_agents_from_database(self) -> None:
        """Load all agents from database into memory registry."""
        async with self._lock:
            try:
                # Get all active agents from database
                db_agents = await self.db_manager.get_active_agents()

                # Convert database results to AgentInfo objects
                loaded_count = 0
                for agent in db_agents:
                    # Convert string status to AgentStatus enum
                    try:
                        agent_status = AgentStatus(agent.status)
                    except ValueError:
                        agent_status = AgentStatus.STOPPED  # Default for invalid status

                    agent_info = AgentInfo(
                        id=str(agent.id),
                        name=agent.name,
                        type=agent.type,
                        role=agent.role,
                        status=agent_status,
                        capabilities=agent.capabilities or {},
                        tmux_session=agent.tmux_session,
                        last_heartbeat=agent.last_heartbeat.timestamp()
                        if agent.last_heartbeat
                        else time.time(),
                        created_at=agent.created_at.timestamp()
                        if agent.created_at
                        else time.time(),
                        tasks_completed=agent.tasks_completed or 0,
                        tasks_failed=agent.tasks_failed or 0,
                    )

                    self.agents[agent.name] = agent_info
                    loaded_count += 1

                logger.info(f"Loaded {loaded_count} agents from database")

            except Exception as e:
                logger.error("Failed to load agents from database", error=str(e))

    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register a new agent."""
        async with self._lock:
            try:
                # Store in database first (async implementation)
                await self._store_agent_in_db(agent_info)

                # Store in memory only if database operation succeeds
                self.agents[agent_info.name] = agent_info

                logger.info("Agent registered", agent_name=agent_info.name)
                return True

            except Exception as e:
                logger.error(
                    "Failed to register agent", agent_name=agent_info.name, error=str(e)
                )
                # Remove from memory if it was added
                if agent_info.name in self.agents:
                    del self.agents[agent_info.name]
                return False

    async def _store_agent_in_db(self, agent_info: AgentInfo) -> None:
        """Store agent in database asynchronously."""
        try:
            await self.db_manager.register_agent(
                name=agent_info.name,
                agent_type=agent_info.type,
                role=agent_info.role,
                capabilities=agent_info.capabilities,
                tmux_session=agent_info.tmux_session,
            )
        except Exception as e:
            logger.error(
                "Failed to store agent in database", agent=agent_info.name, error=str(e)
            )
            raise

    async def update_agent_status(
        self,
        agent_name: str,
        status: AgentStatus,
        current_task_id: str | None = None,
    ) -> bool:
        """Update agent status."""
        async with self._lock:
            if agent_name not in self.agents:
                return False

            try:
                # Update database first
                await self._update_agent_in_db(agent_name, status)

                # Update memory only if database operation succeeds
                self.agents[agent_name].status = status
                self.agents[agent_name].last_heartbeat = time.time()
                if current_task_id is not None:
                    self.agents[agent_name].current_task_id = current_task_id

                return True
            except Exception as e:
                logger.error(
                    "Failed to update agent status", agent_name=agent_name, error=str(e)
                )
                return False

    async def _update_agent_in_db(self, agent_name: str, status: AgentStatus) -> None:
        """Update agent in database asynchronously."""
        import asyncio

        def sync_update():
            session = self.db_manager.get_session()
            try:
                agent = session.query(Agent).filter_by(name=agent_name).first()
                if agent:
                    agent.status = status.value
                    agent.last_heartbeat = datetime.now(UTC)
                    agent.updated_at = datetime.now(UTC)
                    session.commit()
            finally:
                session.close()

        # Run in executor to avoid blocking async loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sync_update)

    async def get_agent(self, agent_name: str) -> AgentInfo | None:
        """Get agent information."""
        return self.agents.get(agent_name)

    async def list_agents(
        self, status_filter: AgentStatus | None = None
    ) -> list[AgentInfo]:
        """List all agents, optionally filtered by status."""
        agents = list(self.agents.values())
        if status_filter:
            agents = [a for a in agents if a.status == status_filter]
        return agents

    async def remove_agent(self, agent_name: str) -> bool:
        """Remove agent from registry."""
        async with self._lock:
            if agent_name not in self.agents:
                return False

            try:
                # Remove from database first
                await self._remove_agent_from_db(agent_name)

                # Remove from memory only if database operation succeeds
                del self.agents[agent_name]
                return True
            except Exception as e:
                logger.error(
                    "Failed to remove agent", agent_name=agent_name, error=str(e)
                )
                return False

    async def _remove_agent_from_db(self, agent_name: str) -> None:
        """Remove agent from database asynchronously."""
        import asyncio

        def sync_remove():
            session = self.db_manager.get_session()
            try:
                agent = session.query(Agent).filter_by(name=agent_name).first()
                if agent:
                    session.delete(agent)
                    session.commit()
            finally:
                session.close()

        # Run in executor to avoid blocking async loop
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, sync_remove)


class AgentSpawner:
    """Handles spawning new agent instances using resilient tmux management."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tmux_manager = get_tmux_manager()

    async def spawn_agent(
        self, agent_type: str, agent_name: str, capabilities: list[str] = None
    ) -> str | None:
        """Spawn a new agent in a tmux session with retry logic and validation."""
        if capabilities is None:
            capabilities = []

        session_name = f"hive-{agent_name}"

        try:
            logger.info(
                "Spawning agent with resilient tmux manager",
                agent_type=agent_type,
                agent_name=agent_name,
                session_name=session_name,
            )

            # Get current environment variables
            import os

            current_env = os.environ.copy()

            # Build command to run in tmux session
            command = f"cd {self.project_root} && uv run python -m src.agents.runner --type {agent_type} --name {agent_name}"

            # Create session using resilient tmux manager
            result: TmuxOperationResult = await self.tmux_manager.create_session(
                session_name=session_name,
                command=command,
                working_directory=self.project_root,
                environment=current_env,
            )

            if result.success:
                # Wait for agent to initialize (uv needs time to set up environment)
                await asyncio.sleep(Intervals.ORCHESTRATOR_STARTUP_DELAY)

                # Validate session is still active and agent is responsive
                session_status = await self.tmux_manager.get_session_status(
                    session_name
                )
                if session_status.name == "ACTIVE":
                    logger.info(
                        "Agent spawned successfully with resilient tmux",
                        agent_name=agent_name,
                        session_name=session_name,
                        execution_time=result.execution_time,
                        retry_count=result.retry_count,
                    )
                    return session_name
                else:
                    logger.error(
                        "Agent session became inactive after spawn",
                        agent_name=agent_name,
                        session_name=session_name,
                        session_status=session_status.name,
                    )
                    return None
            else:
                logger.error(
                    "Failed to spawn agent with resilient tmux",
                    agent_name=agent_name,
                    session_name=session_name,
                    error=result.error_message,
                    retry_count=result.retry_count,
                )
                return None

        except Exception as e:
            logger.error(
                "Unexpected error during agent spawn",
                agent_name=agent_name,
                session_name=session_name,
                error=str(e),
            )
            return None

    async def terminate_agent(self, agent_name: str, session_name: str) -> bool:
        """Terminate an agent and its tmux session using resilient tmux management."""
        try:
            logger.info(
                "Terminating agent with resilient tmux manager",
                agent_name=agent_name,
                session_name=session_name,
            )

            # Use resilient tmux manager for termination
            result: TmuxOperationResult = await self.tmux_manager.terminate_session(
                session_name=session_name,
                force=False,  # Try graceful first
            )

            if result.success:
                logger.info(
                    "Agent terminated successfully with resilient tmux",
                    agent_name=agent_name,
                    session_name=session_name,
                    execution_time=result.execution_time,
                    retry_count=result.retry_count,
                )
                return True
            else:
                logger.error(
                    "Failed to terminate agent with resilient tmux",
                    agent_name=agent_name,
                    session_name=session_name,
                    error=result.error_message,
                )

                # Try force termination as fallback
                logger.info("Attempting force termination", agent_name=agent_name)
                force_result = await self.tmux_manager.terminate_session(
                    session_name=session_name,
                    force=True,
                )

                if force_result.success:
                    logger.info(
                        "Agent force-terminated successfully",
                        agent_name=agent_name,
                        session_name=session_name,
                    )
                    return True
                else:
                    logger.error(
                        "Force termination also failed",
                        agent_name=agent_name,
                        session_name=session_name,
                        error=force_result.error_message,
                    )
                    return False

        except Exception as e:
            logger.error(
                "Unexpected error during agent termination",
                agent_name=agent_name,
                session_name=session_name,
                error=str(e),
            )
            return False


class HealthMonitor:
    """Monitors agent health and system metrics."""

    def __init__(self, registry: AgentRegistry, heartbeat_interval: int = 30):
        self.registry = registry
        self.heartbeat_interval = heartbeat_interval
        self.running = False
        self.last_cleanup = time.time()
        self.cleanup_interval = 300  # 5 minutes between database cleanups

    async def start_monitoring(self) -> None:
        """Start health monitoring loop."""
        self.running = True

        while self.running:
            try:
                await self._check_agent_health()
                await self._periodic_database_cleanup()
                await asyncio.sleep(self.heartbeat_interval)
            except Exception as e:
                logger.error("Health monitor error", error=str(e))
                await asyncio.sleep(5)

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.running = False

    async def _periodic_database_cleanup(self) -> None:
        """Perform periodic database cleanup of stale agents."""
        current_time = time.time()

        if current_time - self.last_cleanup > self.cleanup_interval:
            try:
                from .async_db import get_async_database_manager

                db_manager = await get_async_database_manager()
                cleanup_stats = await db_manager.cleanup_stale_agents(
                    threshold_minutes=10
                )

                if isinstance(cleanup_stats, dict) and "error" not in cleanup_stats:
                    total_cleaned = sum(cleanup_stats.values())
                    if total_cleaned > 0:
                        logger.info(
                            "Database cleanup completed",
                            **cleanup_stats,
                            total_cleaned=total_cleaned,
                        )

                self.last_cleanup = current_time

            except Exception as e:
                logger.error("Failed periodic database cleanup", error=str(e))

    async def _check_agent_health(self) -> None:
        """Check health of all agents."""
        current_time = time.time()
        dead_agents = []

        for agent_name, agent_info in self.registry.agents.items():
            # For agents in STARTING status, check if their tmux session is actually running
            if agent_info.status == AgentStatus.STARTING:
                await self._check_starting_agent(agent_name, agent_info)
                continue

            # Check heartbeat timeout for active agents
            time_since_heartbeat = current_time - agent_info.last_heartbeat

            if time_since_heartbeat > self.heartbeat_interval * 2:
                if agent_info.status != AgentStatus.STOPPED:
                    logger.warning(
                        "Agent appears unresponsive",
                        agent_name=agent_name,
                        time_since_heartbeat=time_since_heartbeat,
                    )

                    # Try to ping the agent
                    if not await self._ping_agent(agent_name):
                        dead_agents.append(agent_name)

        # Handle dead agents
        for agent_name in dead_agents:
            await self._handle_dead_agent(agent_name)

    async def _check_starting_agent(
        self, agent_name: str, agent_info: AgentInfo
    ) -> None:
        """Check if a starting agent has actually become active."""
        import subprocess

        try:
            # Check if tmux session exists and is running
            session_name = agent_info.tmux_session
            if session_name:
                # Check if session exists
                check_cmd = ["tmux", "has-session", "-t", session_name]
                result = subprocess.run(check_cmd, capture_output=True)

                if result.returncode == 0:
                    # Session exists, check if agent process is running (basic check)
                    # If it's been more than 10 seconds since spawn, assume it should be active
                    time_since_start = time.time() - agent_info.created_at
                    if time_since_start > 10:  # Give agent 10 seconds to start
                        logger.info(
                            "Marking long-running agent as active",
                            agent_name=agent_name,
                            time_since_start=time_since_start,
                        )
                        await self.registry.update_agent_status(
                            agent_name, AgentStatus.ACTIVE
                        )
                else:
                    # Session doesn't exist, agent failed to start
                    logger.warning(
                        "Agent tmux session not found, marking as stopped",
                        agent_name=agent_name,
                        session_name=session_name,
                    )
                    await self.registry.update_agent_status(
                        agent_name, AgentStatus.STOPPED
                    )
        except Exception as e:
            logger.warning(
                "Failed to check starting agent status",
                agent_name=agent_name,
                error=str(e),
            )

    async def _ping_agent(self, agent_name: str) -> bool:
        """Ping an agent to check if it's responsive."""
        try:
            # Send ping message
            await message_broker.send_message(
                from_agent="orchestrator",
                to_agent=agent_name,
                topic="ping",
                payload={"timestamp": time.time()},
            )

            # Wait for pong response (simplified - would need proper message handling)
            # This is a basic implementation
            return True

        except Exception as e:
            logger.error("Failed to ping agent", agent_name=agent_name, error=str(e))
            return False

    async def _handle_dead_agent(self, agent_name: str) -> None:
        """Handle a dead/unresponsive agent."""
        agent_info = await self.registry.get_agent(agent_name)
        if not agent_info:
            return

        # Mark as error state
        await self.registry.update_agent_status(agent_name, AgentStatus.ERROR)

        # If agent had a current task, mark it as failed
        if agent_info.current_task_id:
            await task_queue.fail_task(
                agent_info.current_task_id, f"Agent {agent_name} became unresponsive"
            )

        # Clean up tmux session if it exists
        if agent_info.tmux_session:
            try:
                subprocess.run(
                    ["tmux", "kill-session", "-t", agent_info.tmux_session], check=False
                )  # Don't fail if session doesn't exist
            except Exception:
                pass

        logger.error("Agent marked as dead", agent_name=agent_name)


class AgentOrchestrator:
    """Main orchestrator for managing agent lifecycle and task distribution."""

    def __init__(
        self,
        db_url: str,
        project_root: Path,
        max_agents: int = 50,
        heartbeat_interval: int = 30,
    ):
        self.db_url = db_url
        self.project_root = project_root
        self.max_agents = max_agents

        self.registry = AgentRegistry(db_url)
        self.spawner = AgentSpawner(project_root)
        self.health_monitor = HealthMonitor(self.registry, heartbeat_interval)

        self.running = False
        self.task_assignment_running = False

    async def start(self) -> None:
        """Start the orchestrator."""
        self.running = True
        logger.info("Agent orchestrator starting")

        # Initialize agent registry (load agents from database)
        await self.registry.initialize()

        # Initialize task queue
        await task_queue.initialize()

        # Start background tasks as fire-and-forget
        asyncio.create_task(self.health_monitor.start_monitoring())
        asyncio.create_task(self._task_assignment_loop())
        asyncio.create_task(self._cleanup_loop())

        logger.info("Agent orchestrator background tasks started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        self.running = False
        self.task_assignment_running = False
        await self.health_monitor.stop_monitoring()
        logger.info("Agent orchestrator stopped")

    async def spawn_agent(
        self, agent_type: str, agent_name: str = None, capabilities: list[str] = None
    ) -> str | None:
        """Spawn a new agent."""
        if len(self.registry.agents) >= self.max_agents:
            logger.warning("Maximum agent limit reached", max_agents=self.max_agents)
            return None

        if not agent_name:
            agent_name = f"{agent_type}-{uuid.uuid4().hex[:8]}"

        if capabilities is None:
            capabilities = self._get_default_capabilities(agent_type)

        # Check if agent name already exists
        if agent_name in self.registry.agents:
            logger.warning("Agent name already exists", agent_name=agent_name)
            return None

        # Spawn the agent
        session_name = await self.spawner.spawn_agent(
            agent_type, agent_name, capabilities
        )
        if not session_name:
            return None

        # Register the agent
        agent_short_id = ShortIDGenerator.generate_agent_short_id(
            agent_name, agent_name
        )
        agent_info = AgentInfo(
            id=str(uuid.uuid4()),
            name=agent_name,
            type=agent_type,
            role=agent_type,  # Default role same as type
            status=AgentStatus.STARTING,
            capabilities=capabilities,
            tmux_session=session_name,
            last_heartbeat=time.time(),
            created_at=time.time(),
            tasks_completed=0,
            tasks_failed=0,
            short_id=agent_short_id,
        )

        if await self.registry.register_agent(agent_info):
            logger.info(
                "Agent successfully spawned and registered", agent_name=agent_name
            )
            return agent_name
        else:
            # Registration failed, clean up
            await self.spawner.terminate_agent(agent_name, session_name)
            return None

    async def terminate_agent(self, agent_name: str) -> bool:
        """Terminate an agent."""
        agent_info = await self.registry.get_agent(agent_name)
        if not agent_info:
            return False

        # Mark as stopping
        await self.registry.update_agent_status(agent_name, AgentStatus.STOPPING)

        # Terminate the session
        success = await self.spawner.terminate_agent(
            agent_name, agent_info.tmux_session
        )

        if success:
            # Remove from registry
            await self.registry.remove_agent(agent_name)

        return success

    async def get_active_agent_count(self) -> int:
        """Get count of active agents."""
        agents = await self.registry.list_agents()
        return sum(1 for agent in agents if agent.status == AgentStatus.ACTIVE)

    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health metrics."""
        agents = await self.registry.list_agents()

        status_counts = dict.fromkeys(AgentStatus, 0)
        total_load = 0.0

        for agent in agents:
            status_counts[agent.status] += 1
            total_load += agent.load_factor

        # Get queue stats
        queue_stats = await task_queue.get_queue_stats()

        # Calculate tasks per minute (simplified)
        tasks_per_minute = queue_stats.completed_tasks / max(
            1, (time.time() - 3600) / 60
        )  # Rough estimate

        return SystemHealth(
            total_agents=len(agents),
            active_agents=status_counts[AgentStatus.ACTIVE],
            idle_agents=status_counts[AgentStatus.IDLE],
            busy_agents=status_counts[AgentStatus.BUSY],
            error_agents=status_counts[AgentStatus.ERROR],
            avg_load_factor=total_load / max(1, len(agents)),
            queue_size=sum(queue_stats.queue_size_by_priority.values()),
            tasks_per_minute=tasks_per_minute,
        )

    async def _task_assignment_loop(self) -> None:
        """Main task assignment loop."""
        self.task_assignment_running = True

        while self.running and self.task_assignment_running:
            try:
                # Get available agents
                available_agents = await self.registry.list_agents(AgentStatus.IDLE)

                if not available_agents:
                    await asyncio.sleep(1)
                    continue

                # Try to assign tasks
                for agent_info in available_agents:
                    # Get next task for this agent
                    task = await self._find_suitable_task(agent_info)

                    if task:
                        # Assign task to agent
                        await self._assign_task_to_agent(task, agent_info)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error("Task assignment loop error", error=str(e))
                await asyncio.sleep(5)

    async def _find_suitable_task(self, agent_info: AgentInfo) -> Task | None:
        """Find a suitable task for the given agent based on capabilities."""
        # Simple implementation - get any task the agent can handle
        # In a more sophisticated system, this would match task requirements to agent capabilities

        task = await task_queue.get_task(agent_info.name)
        return task

    async def _assign_task_to_agent(self, task: Task, agent_info: AgentInfo) -> bool:
        """Assign a task to an agent."""
        try:
            # Update agent status
            await self.registry.update_agent_status(
                agent_info.name, AgentStatus.BUSY, task.id
            )

            # Send task to agent via message broker
            await message_broker.send_message(
                from_agent="orchestrator",
                to_agent=agent_info.name,
                topic="task_assignment",
                payload={"task_id": task.id, "task": task.model_dump()},
            )

            logger.info(
                "Task assigned to agent", task_id=task.id, agent_name=agent_info.name
            )
            return True

        except Exception as e:
            logger.error(
                "Failed to assign task",
                task_id=task.id,
                agent_name=agent_info.name,
                error=str(e),
            )
            return False

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup tasks."""
        while self.running:
            try:
                # Clean up expired tasks
                await task_queue.cleanup_expired_tasks()

                # Clean up orphaned tmux sessions
                await self._cleanup_orphaned_sessions()

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(60)

    async def _cleanup_orphaned_sessions(self) -> None:
        """Clean up tmux sessions that don't have corresponding agents using resilient tmux manager."""
        try:
            logger.debug("Starting orphaned session cleanup")

            # Get all tracked agent sessions
            agent_sessions = {
                agent.tmux_session
                for agent in self.registry.agents.values()
                if agent.tmux_session
            }

            # Use resilient tmux manager to cleanup orphaned sessions
            orphaned_sessions = (
                await self.spawner.tmux_manager.cleanup_orphaned_sessions(
                    prefix="hive-"
                )
            )

            # Filter out sessions that belong to active agents
            truly_orphaned = [
                session
                for session in orphaned_sessions
                if session not in agent_sessions
            ]

            if truly_orphaned:
                logger.info(
                    "Cleaned up orphaned tmux sessions",
                    orphaned_count=len(truly_orphaned),
                    orphaned_sessions=truly_orphaned,
                )
            else:
                logger.debug("No orphaned sessions found")

        except Exception as e:
            logger.error("Failed to cleanup orphaned sessions", error=str(e))

    def _get_default_capabilities(self, agent_type: str) -> list[str]:
        """Get default capabilities for agent type."""
        capability_map = {
            "meta": ["system_analysis", "code_generation", "self_improvement"],
            "developer": ["code_generation", "testing", "debugging"],
            "qa": ["testing", "quality_assurance", "bug_reporting"],
            "architect": ["system_design", "architecture_planning"],
            "devops": ["deployment", "infrastructure", "monitoring"],
        }

        return capability_map.get(agent_type, ["general"])


# Global orchestrator instance
orchestrator = None


async def get_orchestrator(db_url: str, project_root: Path) -> AgentOrchestrator:
    """Get the global orchestrator instance."""
    global orchestrator
    if orchestrator is None:
        orchestrator = AgentOrchestrator(db_url, project_root)
    return orchestrator
