# Agent Registry - Manages agent lifecycle and state

import asyncio
import time
from typing import Dict, List, Optional

import structlog

from .interfaces import AgentInfo, AgentStatus, IDatabaseManager

logger = structlog.get_logger()


class AgentRegistry:
    """Manages agent registration, state tracking, and persistence."""

    def __init__(self, database_manager: IDatabaseManager):
        self.database_manager = database_manager
        self._agents: Dict[str, AgentInfo] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the registry and load existing agents from database."""
        async with self._lock:
            try:
                # Get all active agents from database
                db_agents = await self.database_manager.get_active_agents()

                # Convert database results to AgentInfo objects
                loaded_count = 0
                for db_agent in db_agents:
                    try:
                        agent_info = AgentInfo(
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

                        self._agents[agent_info.name] = agent_info
                        loaded_count += 1

                    except Exception as e:
                        logger.error("Failed to convert database agent", agent_name=db_agent.get("name"), error=str(e))
                        continue

                logger.info(f"Loaded {loaded_count} agents from database")

            except Exception as e:
                logger.error("Failed to load agents from database", error=str(e))

    async def register_agent(self, agent_info: AgentInfo) -> bool:
        """Register a new agent."""
        async with self._lock:
            try:
                # Check if agent already exists
                if agent_info.name in self._agents:
                    logger.warning("Agent already exists", agent_name=agent_info.name)
                    return False

                # Save to database
                success = await self.database_manager.save_agent(agent_info)
                if not success:
                    logger.error("Failed to save agent to database", agent_name=agent_info.name)
                    return False

                # Add to memory
                self._agents[agent_info.name] = agent_info
                logger.info("Agent registered successfully", agent_name=agent_info.name)
                return True

            except Exception as e:
                logger.error("Failed to register agent", agent_name=agent_info.name, error=str(e))
                return False

    async def update_agent_status(
        self,
        agent_name: str,
        status: AgentStatus,
        current_task_id: Optional[str] = None
    ) -> bool:
        """Update agent status."""
        async with self._lock:
            if agent_name not in self._agents:
                logger.warning("Agent not found for status update", agent_name=agent_name)
                return False

            try:
                # Update in memory
                agent_info = self._agents[agent_name]
                agent_info.status = status
                agent_info.last_heartbeat = time.time()
                if current_task_id is not None:
                    agent_info.current_task_id = current_task_id

                # Update in database
                updates = {"status": status.value, "last_heartbeat": time.time()}
                if current_task_id is not None:
                    updates["current_task_id"] = current_task_id

                success = await self.database_manager.update_agent(agent_name, updates)
                if success:
                    logger.info("Agent status updated", agent_name=agent_name, status=status.value)
                    return True
                else:
                    logger.error("Failed to update agent status in database", agent_name=agent_name)
                    return False

            except Exception as e:
                logger.error("Failed to update agent status", agent_name=agent_name, error=str(e))
                return False

    async def get_agent(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information."""
        async with self._lock:
            return self._agents.get(agent_name)

    async def list_agents(self, status_filter: Optional[AgentStatus] = None) -> List[AgentInfo]:
        """List all agents, optionally filtered by status."""
        async with self._lock:
            agents = list(self._agents.values())
            if status_filter:
                agents = [a for a in agents if a.status == status_filter]
            return agents

    async def remove_agent(self, agent_name: str) -> bool:
        """Remove agent from registry."""
        async with self._lock:
            if agent_name not in self._agents:
                logger.warning("Agent not found for removal", agent_name=agent_name)
                return False

            try:
                # Remove from database
                success = await self.database_manager.delete_agent(agent_name)
                if not success:
                    logger.error("Failed to delete agent from database", agent_name=agent_name)
                    return False

                # Remove from memory
                del self._agents[agent_name]
                logger.info("Agent removed successfully", agent_name=agent_name)
                return True

            except Exception as e:
                logger.error("Failed to remove agent", agent_name=agent_name, error=str(e))
                return False

    async def update_agent_load(self, agent_name: str, load_factor: float) -> bool:
        """Update agent load factor."""
        async with self._lock:
            if agent_name not in self._agents:
                return False

            try:
                # Update in memory
                self._agents[agent_name].load_factor = load_factor

                # Update in database
                success = await self.database_manager.update_agent(agent_name, {"load_factor": load_factor})
                if success:
                    logger.debug("Agent load updated", agent_name=agent_name, load_factor=load_factor)
                    return True
                else:
                    logger.error("Failed to update agent load in database", agent_name=agent_name)
                    return False

            except Exception as e:
                logger.error("Failed to update agent load", agent_name=agent_name, error=str(e))
                return False

    async def get_agents_by_capability(self, capability: str) -> List[AgentInfo]:
        """Get agents that have a specific capability."""
        async with self._lock:
            return [agent for agent in self._agents.values() if capability in agent.capabilities]

    async def get_idle_agents(self) -> List[AgentInfo]:
        """Get all idle agents."""
        return await self.list_agents(AgentStatus.IDLE)

    async def get_active_agents(self) -> List[AgentInfo]:
        """Get all active agents."""
        return await self.list_agents(AgentStatus.ACTIVE)

    def get_agent_count(self) -> int:
        """Get total number of registered agents."""
        return len(self._agents)

    async def cleanup_stale_agents(self, threshold_minutes: int = 10) -> Dict[str, str]:
        """Clean up stale agents and return cleanup statistics."""
        try:
            # Use database manager to cleanup stale agents
            cleanup_stats = await self.database_manager.cleanup_stale_agents(threshold_minutes)

            # Update in-memory registry to reflect database changes
            async with self._lock:
                # Reload agents from database to ensure consistency
                await self.initialize()

            logger.info("Stale agent cleanup completed", **cleanup_stats)
            return cleanup_stats

        except Exception as e:
            logger.error("Failed to cleanup stale agents", error=str(e))
            return {"error": str(e)}