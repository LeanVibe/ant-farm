# Refactored Agent Orchestrator - Clean architecture implementation

import asyncio
import time
import uuid
from pathlib import Path
from typing import List, Optional

import structlog

from .interfaces import (
    IAgentOrchestrator, IMessageBroker, ITaskQueue, IDatabaseManager,
    IHealthMonitor, IAgentFactory, AgentInfo, SystemHealth, AgentStatus
)
from .agent_registry import AgentRegistry
from .agent_spawner import AgentSpawner
from .health_monitor import HealthMonitor

logger = structlog.get_logger()


class AgentOrchestratorRefactored(IAgentOrchestrator):
    """Refactored Agent Orchestrator using clean architecture and dependency injection."""

    def __init__(
        self,
        message_broker: IMessageBroker,
        task_queue: ITaskQueue,
        database_manager: IDatabaseManager,
        agent_factory: IAgentFactory,
        project_root: Path,
        max_agents: int = 50
    ):
        self.message_broker = message_broker
        self.task_queue = task_queue
        self.database_manager = database_manager
        self.agent_factory = agent_factory
        self.project_root = project_root
        self.max_agents = max_agents

        # Initialize components
        self.registry = AgentRegistry(database_manager)
        self.spawner = AgentSpawner(project_root, agent_factory)
        self.health_monitor = HealthMonitor(message_broker)

        self.running = False
        self.task_assignment_running = False
        self._background_tasks: List[asyncio.Task] = []

    async def spawn_agent(self, agent_type: str, name: Optional[str] = None, capabilities: Optional[List[str]] = None) -> Optional[str]:
        """Spawn a new agent."""
        # Check agent limit
        agents = await self.list_agents()
        if len(agents) >= self.max_agents:
            logger.warning("Maximum agent limit reached", max_agents=self.max_agents)
            return None

        if not name:
            name = f"{agent_type}-{uuid.uuid4().hex[:8]}"

        if capabilities is None:
            capabilities = self._get_default_capabilities(agent_type)

        # Check if agent already exists
        existing_agent = await self.get_agent(name)
        if existing_agent:
            logger.warning("Agent name already exists", agent_name=name)
            return None

        try:
            # Create agent info
            agent_info = AgentInfo(
                id=str(uuid.uuid4()),
                name=name,
                type=agent_type,
                role=agent_type,
                status=AgentStatus.STARTING,
                capabilities=capabilities,
                tmux_session=None,  # Will be set by spawner
                last_heartbeat=time.time(),
                created_at=time.time(),
                tasks_completed=0,
                tasks_failed=0,
                current_task_id=None,
                load_factor=0.0
            )

            # Spawn the agent process
            session_name = await self.spawner.spawn_agent(agent_type, name, capabilities)
            if not session_name:
                logger.error("Failed to spawn agent process", agent_name=name)
                return None

            # Update session name and register
            agent_info.tmux_session = session_name
            success = await self.registry.register_agent(agent_info)

            if success:
                logger.info("Agent spawned and registered successfully", agent_name=name)
                return name
            else:
                # Clean up on registration failure
                await self.spawner.terminate_agent(name, session_name)
                return None

        except Exception as e:
            logger.error("Failed to spawn agent", agent_type=agent_type, agent_name=name, error=str(e))
            return None

    async def terminate_agent(self, agent_name: str) -> bool:
        """Terminate an agent."""
        try:
            agent_info = await self.get_agent(agent_name)
            if not agent_info or not agent_info.tmux_session:
                return False

            # Terminate the process
            process_terminated = await self.spawner.terminate_agent(agent_name, agent_info.tmux_session)

            # Remove from registry
            registry_removed = await self.registry.remove_agent(agent_name)

            success = process_terminated and registry_removed

            if success:
                logger.info("Agent terminated successfully", agent_name=agent_name)
            else:
                logger.error("Agent termination partially failed", agent_name=agent_name)

            return success

        except Exception as e:
            logger.error("Failed to terminate agent", agent_name=agent_name, error=str(e))
            return False

    async def get_agent(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information."""
        return await self.registry.get_agent(agent_name)

    async def list_agents(self, status_filter: Optional[str] = None) -> List[AgentInfo]:
        """List agents with optional status filter."""
        status = AgentStatus(status_filter) if status_filter else None
        return await self.registry.list_agents(status)

    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health metrics."""
        try:
            agents = await self.list_agents()

            # Count agents by status
            status_counts = {
                AgentStatus.STARTING: 0,
                AgentStatus.ACTIVE: 0,
                AgentStatus.IDLE: 0,
                AgentStatus.BUSY: 0,
                AgentStatus.STOPPED: 0,
                AgentStatus.ERROR: 0
            }

            total_load = 0.0
            for agent in agents:
                status_counts[agent.status] += 1
                total_load += agent.load_factor

            # Get queue stats
            queue_stats = await self.task_queue.get_queue_stats()

            # Calculate tasks per minute (simplified)
            tasks_per_minute = 0.0  # TODO: Implement proper calculation

            return SystemHealth(
                total_agents=len(agents),
                active_agents=status_counts[AgentStatus.ACTIVE],
                idle_agents=status_counts[AgentStatus.IDLE],
                busy_agents=status_counts[AgentStatus.BUSY],
                error_agents=status_counts[AgentStatus.ERROR],
                avg_load_factor=total_load / max(1, len(agents)),
                queue_size=queue_stats.get("total_queued", 0),
                tasks_per_minute=tasks_per_minute
            )
        except Exception as e:
            logger.error("Failed to get system health", error=str(e))
            return SystemHealth(
                total_agents=0,
                active_agents=0,
                idle_agents=0,
                busy_agents=0,
                error_agents=0,
                avg_load_factor=0.0,
                queue_size=0,
                tasks_per_minute=0.0
            )

    async def start(self) -> None:
        """Start the orchestrator."""
        if self.running:
            return

        self.running = True
        logger.info("Agent orchestrator starting")

        # Initialize registry
        await self.registry.initialize()

        # Start health monitoring
        await self.health_monitor.start_monitoring()

        # Start background tasks
        self._background_tasks.append(asyncio.create_task(self._task_assignment_loop()))
        self._background_tasks.append(asyncio.create_task(self._cleanup_loop()))

        logger.info("Agent orchestrator background tasks started")

    async def stop(self) -> None:
        """Stop the orchestrator."""
        if not self.running:
            return

        self.running = False
        self.task_assignment_running = False

        # Stop health monitoring
        await self.health_monitor.stop_monitoring()

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            try:
                await asyncio.wait(self._background_tasks, timeout=2.0)
            except Exception:
                pass

        self._background_tasks.clear()
        logger.info("Agent orchestrator stopped")

    async def _task_assignment_loop(self) -> None:
        """Main task assignment loop."""
        self.task_assignment_running = True

        while self.running and self.task_assignment_running:
            try:
                # Get idle agents
                idle_agents = await self.registry.get_idle_agents()

                if not idle_agents:
                    await asyncio.sleep(1)
                    continue

                # Try to assign tasks to idle agents
                for agent_info in idle_agents:
                    task = await self.task_queue.dequeue_task(agent_info.name)
                    if task:
                        await self._assign_task_to_agent(task, agent_info)

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                logger.error("Task assignment loop error", error=str(e))
                await asyncio.sleep(5)

    async def _assign_task_to_agent(self, task, agent_info: AgentInfo) -> bool:
        """Assign a task to an agent."""
        try:
            # Update agent status to busy
            await self.registry.update_agent_status(agent_info.name, AgentStatus.BUSY, task.id)

            # Send task to agent via message broker
            success = await self.message_broker.send_message(
                from_agent="orchestrator",
                to_agent=agent_info.name,
                topic="task_assignment",
                payload={
                    "task_id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "type": task.type,
                    "priority": task.priority
                }
            )

            if success:
                logger.info("Task assigned to agent", task_id=task.id, agent_name=agent_info.name)
                return True
            else:
                logger.error("Failed to send task assignment", task_id=task.id, agent_name=agent_info.name)
                return False

        except Exception as e:
            logger.error("Failed to assign task", task_id=task.id, agent_name=agent_info.name, error=str(e))
            return False

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup tasks."""
        while self.running:
            try:
                # Clean up expired tasks
                cleaned_count = await self.task_queue.cleanup_expired_tasks()
                if cleaned_count > 0:
                    logger.info("Cleaned up expired tasks", count=cleaned_count)

                # Clean up stale agents
                cleanup_stats = await self.registry.cleanup_stale_agents()
                if cleanup_stats:
                    logger.info("Cleaned up stale agents", **cleanup_stats)

                # Clean up orphaned tmux sessions
                orphaned_sessions = await self.spawner.cleanup_orphaned_sessions()
                if orphaned_sessions:
                    logger.info("Cleaned up orphaned sessions", count=len(orphaned_sessions))

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(60)

    def _get_default_capabilities(self, agent_type: str) -> List[str]:
        """Get default capabilities for an agent type."""
        defaults = {
            "meta": ["planning", "coordination", "analysis"],
            "dev": ["coding", "testing", "debugging"],
            "qa": ["testing", "validation", "reporting"],
            "ops": ["deployment", "monitoring", "maintenance"]
        }
        return defaults.get(agent_type, ["general"])