# Concrete implementation of IAgentOrchestrator interface

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
from .short_id import ShortIDGenerator

logger = structlog.get_logger()


class AgentOrchestratorImpl(IAgentOrchestrator):
    """Concrete implementation of the Agent Orchestrator interface."""

    def __init__(
        self,
        message_broker: IMessageBroker,
        task_queue: ITaskQueue,
        database_manager: IDatabaseManager,
        health_monitor: IHealthMonitor,
        agent_factory: IAgentFactory,
        project_root: Path,
        max_agents: int = 50
    ):
        self.message_broker = message_broker
        self.task_queue = task_queue
        self.database_manager = database_manager
        self.health_monitor = health_monitor
        self.agent_factory = agent_factory
        self.project_root = project_root
        self.max_agents = max_agents

        self.running = False
        self.task_assignment_running = False
        self._background_tasks: List[asyncio.Task] = []

    async def spawn_agent(self, agent_type: str, name: Optional[str] = None, capabilities: Optional[List[str]] = None) -> Optional[str]:
        """Spawn a new agent."""
        # Get current agent count
        agents = await self.list_agents()
        if len(agents) >= self.max_agents:
            logger.warning("Maximum agent limit reached", max_agents=self.max_agents)
            return None

        if not name:
            name = f"{agent_type}-{uuid.uuid4().hex[:8]}"

        if capabilities is None:
            capabilities = self._get_default_capabilities(agent_type)

        # Check if agent name already exists
        existing_agent = await self.get_agent(name)
        if existing_agent:
            logger.warning("Agent name already exists", agent_name=name)
            return None

        try:
            # Create agent using factory
            agent_instance = self.agent_factory.create_agent(agent_type, name, {"capabilities": capabilities})
            if not agent_instance:
                logger.error("Failed to create agent instance", agent_type=agent_type, agent_name=name)
                return None

            # Create agent info
            agent_short_id = ShortIDGenerator.generate_agent_short_id(name, name)
            agent_info = AgentInfo(
                id=str(uuid.uuid4()),
                name=name,
                type=agent_type,
                role=agent_type,  # Default role same as type
                status=AgentStatus.STARTING,
                capabilities=capabilities,
                tmux_session=f"hive-{name}",  # Simplified session name
                last_heartbeat=time.time(),
                created_at=time.time(),
                tasks_completed=0,
                tasks_failed=0,
                current_task_id=None,
                load_factor=0.0
            )

            # Save to database
            success = await self.database_manager.save_agent(agent_info)
            if not success:
                logger.error("Failed to save agent to database", agent_name=name)
                return None

            logger.info("Agent successfully spawned and registered", agent_name=name)
            return name

        except Exception as e:
            logger.error("Failed to spawn agent", agent_type=agent_type, agent_name=name, error=str(e))
            return None

    async def terminate_agent(self, agent_name: str) -> bool:
        """Terminate an agent."""
        try:
            agent_info = await self.get_agent(agent_name)
            if not agent_info:
                return False

            # Update status to stopping
            agent_info.status = AgentStatus.STOPPED
            success = await self.database_manager.update_agent(agent_name, {"status": AgentStatus.STOPPED.value})

            if success:
                # Remove from database
                await self.database_manager.delete_agent(agent_name)
                logger.info("Agent terminated successfully", agent_name=agent_name)
                return True
            else:
                logger.error("Failed to update agent status during termination", agent_name=agent_name)
                return False

        except Exception as e:
            logger.error("Failed to terminate agent", agent_name=agent_name, error=str(e))
            return False

    async def get_agent(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information."""
        try:
            db_agent = await self.database_manager.get_agent(agent_name)
            if not db_agent:
                return None

            # Convert database format to AgentInfo
            return AgentInfo(
                id=db_agent.get("id", ""),
                name=db_agent.get("name", ""),
                type=db_agent.get("type", ""),
                role=db_agent.get("role", ""),
                status=AgentStatus(db_agent.get("status", "stopped")),
                capabilities=db_agent.get("capabilities", []),
                tmux_session=db_agent.get("tmux_session"),
                last_heartbeat=db_agent.get("last_heartbeat", time.time()),
                created_at=db_agent.get("created_at", time.time()),
                tasks_completed=db_agent.get("tasks_completed", 0),
                tasks_failed=db_agent.get("tasks_failed", 0),
                current_task_id=db_agent.get("current_task_id"),
                load_factor=db_agent.get("load_factor", 0.0)
            )
        except Exception as e:
            logger.error("Failed to get agent", agent_name=agent_name, error=str(e))
            return None

    async def list_agents(self, status_filter: Optional[str] = None) -> List[AgentInfo]:
        """List agents with optional status filter."""
        try:
            db_agents = await self.database_manager.get_active_agents()
            agents = []

            for db_agent in db_agents:
                try:
                    agent_status = AgentStatus(db_agent.get("status", "stopped"))

                    # Apply status filter if provided
                    if status_filter and agent_status.value != status_filter:
                        continue

                    agent_info = AgentInfo(
                        id=db_agent.get("id", ""),
                        name=db_agent.get("name", ""),
                        type=db_agent.get("type", ""),
                        role=db_agent.get("role", ""),
                        status=agent_status,
                        capabilities=db_agent.get("capabilities", []),
                        tmux_session=db_agent.get("tmux_session"),
                        last_heartbeat=db_agent.get("last_heartbeat", time.time()),
                        created_at=db_agent.get("created_at", time.time()),
                        tasks_completed=db_agent.get("tasks_completed", 0),
                        tasks_failed=db_agent.get("tasks_failed", 0),
                        current_task_id=db_agent.get("current_task_id"),
                        load_factor=db_agent.get("load_factor", 0.0)
                    )
                    agents.append(agent_info)
                except Exception as e:
                    logger.error("Failed to convert database agent", agent_name=db_agent.get("name"), error=str(e))
                    continue

            return agents
        except Exception as e:
            logger.error("Failed to list agents", error=str(e))
            return []

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
                await asyncio.wait(self._background_tasks, timeout=1.0)
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
                idle_agents = await self.list_agents("idle")

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
            await self.database_manager.update_agent(agent_info.name, {
                "status": AgentStatus.BUSY.value,
                "current_task_id": task.id
            })

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
                cleanup_stats = await self.database_manager.cleanup_stale_agents()
                if cleanup_stats:
                    logger.info("Cleaned up stale agents", **cleanup_stats)

                await asyncio.sleep(300)  # Run every 5 minutes

            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))
                await asyncio.sleep(60)

    def _get_default_capabilities(self, agent_type: str) -> List[str]:
        """Get default capabilities for an agent type."""
        # This should be configurable, but for now use simple defaults
        defaults = {
            "meta": ["planning", "coordination", "analysis"],
            "dev": ["coding", "testing", "debugging"],
            "qa": ["testing", "validation", "reporting"],
            "ops": ["deployment", "monitoring", "maintenance"]
        }
        return defaults.get(agent_type, ["general"])