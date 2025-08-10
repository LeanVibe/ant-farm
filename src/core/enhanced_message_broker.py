"""Enhanced agent communication system with advanced context sharing and synchronization."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog

from .message_broker import MessageBroker, MessageType, Message
from .context_engine import ContextEngine

logger = structlog.get_logger()


class ContextShareType(Enum):
    """Types of context sharing between agents."""

    WORK_SESSION = "work_session"  # Shared work session context
    KNOWLEDGE_BASE = "knowledge_base"  # Shared knowledge and learnings
    TASK_STATE = "task_state"  # Current task execution state
    PERFORMANCE_METRICS = "performance_metrics"  # Performance insights
    ERROR_PATTERNS = "error_patterns"  # Shared error patterns and solutions
    DECISION_HISTORY = "decision_history"  # Decision making history and rationale


class SyncMode(Enum):
    """Synchronization modes for real-time collaboration."""

    REAL_TIME = "real_time"  # Immediate synchronization
    BATCHED = "batched"  # Batch updates at intervals
    ON_DEMAND = "on_demand"  # Sync only when requested
    CONFLICT_RESOLUTION = "conflict_resolution"  # Handle conflicts in shared state


@dataclass
class SharedContext:
    """Shared context structure for inter-agent communication."""

    id: str
    type: ContextShareType
    owner_agent: str
    participants: Set[str] = field(default_factory=set)
    data: Dict[str, Any] = field(default_factory=dict)
    version: int = 0
    last_updated: float = field(default_factory=time.time)
    last_updated_by: str = ""
    sync_mode: SyncMode = SyncMode.REAL_TIME
    ttl: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContextUpdate:
    """Context update notification."""

    context_id: str
    update_type: str  # "create", "update", "delete", "merge"
    changes: Dict[str, Any]
    previous_version: int
    new_version: int
    timestamp: float
    updated_by: str
    conflict_resolution: Optional[Dict[str, Any]] = None


@dataclass
class AgentState:
    """Current state of an agent for synchronization."""

    agent_name: str
    current_task: Optional[str] = None
    status: str = "idle"
    capabilities: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    last_activity: float = field(default_factory=time.time)
    shared_contexts: Set[str] = field(default_factory=set)
    preferences: Dict[str, Any] = field(default_factory=dict)


class EnhancedMessageBroker(MessageBroker):
    """Enhanced message broker with context sharing and real-time synchronization."""

    def __init__(self, redis_url: str = None, context_engine: ContextEngine = None):
        super().__init__(redis_url)
        self.context_engine = context_engine
        self.shared_contexts: Dict[str, SharedContext] = {}
        self.agent_states: Dict[str, AgentState] = {}
        self.context_subscribers: Dict[
            str, Set[str]
        ] = {}  # context_id -> set of agent names
        self.sync_tasks: Dict[str, asyncio.Task] = {}

        # Enhanced key prefixes
        self.context_prefix = "hive:shared_context"
        self.sync_prefix = "hive:sync"
        self.state_prefix = "hive:agent_state"

    async def initialize(self) -> None:
        """Initialize the enhanced message broker."""
        await super().initialize()

        # Start background synchronization tasks
        self.sync_tasks["context_sync"] = asyncio.create_task(self._context_sync_loop())
        self.sync_tasks["state_sync"] = asyncio.create_task(
            self._agent_state_sync_loop()
        )

        logger.info("Enhanced message broker initialized with context sharing")

    async def create_shared_context(
        self,
        context_type: ContextShareType,
        owner_agent: str,
        initial_data: Dict[str, Any] = None,
        participants: Set[str] = None,
        sync_mode: SyncMode = SyncMode.REAL_TIME,
        ttl: Optional[int] = None,
    ) -> str:
        """Create a new shared context."""

        context_id = str(uuid.uuid4())

        shared_context = SharedContext(
            id=context_id,
            type=context_type,
            owner_agent=owner_agent,
            participants=participants or set(),
            data=initial_data or {},
            sync_mode=sync_mode,
            ttl=time.time() + ttl if ttl else None,
            last_updated_by=owner_agent,
        )

        # Store in memory and Redis
        self.shared_contexts[context_id] = shared_context
        await self._persist_shared_context(shared_context)

        # Add owner to participants
        shared_context.participants.add(owner_agent)

        # Notify participants about new context
        await self._notify_context_created(shared_context)

        logger.info(
            "Shared context created",
            context_id=context_id,
            type=context_type.value,
            owner=owner_agent,
            participants=len(shared_context.participants),
        )

        return context_id

    async def join_shared_context(self, context_id: str, agent_name: str) -> bool:
        """Join an existing shared context."""

        if context_id not in self.shared_contexts:
            # Try to load from Redis
            shared_context = await self._load_shared_context(context_id)
            if not shared_context:
                return False
            self.shared_contexts[context_id] = shared_context

        shared_context = self.shared_contexts[context_id]

        # Add agent to participants
        shared_context.participants.add(agent_name)
        await self._persist_shared_context(shared_context)

        # Add to subscribers
        if context_id not in self.context_subscribers:
            self.context_subscribers[context_id] = set()
        self.context_subscribers[context_id].add(agent_name)

        # Send current context state to joining agent
        await self.send_message(
            from_agent="context_system",
            to_agent=agent_name,
            topic="context_joined",
            payload={
                "context_id": context_id,
                "context_type": shared_context.type.value,
                "data": shared_context.data,
                "version": shared_context.version,
                "participants": list(shared_context.participants),
            },
            message_type=MessageType.NOTIFICATION,
        )

        logger.info(
            "Agent joined shared context",
            context_id=context_id,
            agent=agent_name,
            participants=len(shared_context.participants),
        )

        return True

    async def update_shared_context(
        self,
        context_id: str,
        agent_name: str,
        updates: Dict[str, Any],
        merge_strategy: str = "deep_merge",
    ) -> bool:
        """Update a shared context with conflict resolution."""

        if context_id not in self.shared_contexts:
            logger.error("Context not found", context_id=context_id)
            return False

        shared_context = self.shared_contexts[context_id]

        if agent_name not in shared_context.participants:
            logger.error(
                "Agent not participant in context",
                agent=agent_name,
                context_id=context_id,
            )
            return False

        # Create update record
        previous_version = shared_context.version
        shared_context.version += 1

        # Apply updates based on merge strategy
        if merge_strategy == "deep_merge":
            shared_context.data = self._deep_merge(shared_context.data, updates)
        elif merge_strategy == "replace":
            shared_context.data = updates
        elif merge_strategy == "patch":
            for key, value in updates.items():
                shared_context.data[key] = value

        shared_context.last_updated = time.time()
        shared_context.last_updated_by = agent_name

        # Persist to Redis
        await self._persist_shared_context(shared_context)

        # Create update notification
        context_update = ContextUpdate(
            context_id=context_id,
            update_type="update",
            changes=updates,
            previous_version=previous_version,
            new_version=shared_context.version,
            timestamp=shared_context.last_updated,
            updated_by=agent_name,
        )

        # Notify other participants based on sync mode
        if shared_context.sync_mode == SyncMode.REAL_TIME:
            await self._notify_context_updated(
                context_update, shared_context.participants - {agent_name}
            )
        elif shared_context.sync_mode == SyncMode.BATCHED:
            # Will be handled by sync loop
            pass

        logger.info(
            "Shared context updated",
            context_id=context_id,
            agent=agent_name,
            version=shared_context.version,
            sync_mode=shared_context.sync_mode.value,
        )

        return True

    async def get_shared_context(
        self, context_id: str, agent_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get shared context data for an agent."""

        if context_id not in self.shared_contexts:
            shared_context = await self._load_shared_context(context_id)
            if shared_context:
                self.shared_contexts[context_id] = shared_context

        if context_id not in self.shared_contexts:
            return None

        shared_context = self.shared_contexts[context_id]

        if agent_name not in shared_context.participants:
            return None

        return {
            "id": shared_context.id,
            "type": shared_context.type.value,
            "data": shared_context.data,
            "version": shared_context.version,
            "last_updated": shared_context.last_updated,
            "last_updated_by": shared_context.last_updated_by,
            "participants": list(shared_context.participants),
        }

    async def update_agent_state(
        self, agent_name: str, state_updates: Dict[str, Any]
    ) -> None:
        """Update agent state for real-time coordination."""

        if agent_name not in self.agent_states:
            self.agent_states[agent_name] = AgentState(agent_name=agent_name)

        agent_state = self.agent_states[agent_name]

        # Update state fields
        for key, value in state_updates.items():
            if hasattr(agent_state, key):
                setattr(agent_state, key, value)

        agent_state.last_activity = time.time()

        # Persist to Redis
        await self._persist_agent_state(agent_state)

        # Notify interested agents about state change
        await self._notify_agent_state_updated(agent_state)

        logger.debug("Agent state updated", agent=agent_name, updates=state_updates)

    async def get_agent_states(
        self, requesting_agent: str
    ) -> Dict[str, Dict[str, Any]]:
        """Get states of all agents for coordination."""

        states = {}

        for agent_name, agent_state in self.agent_states.items():
            states[agent_name] = {
                "current_task": agent_state.current_task,
                "status": agent_state.status,
                "capabilities": agent_state.capabilities,
                "performance_metrics": agent_state.performance_metrics,
                "last_activity": agent_state.last_activity,
                "shared_contexts": list(agent_state.shared_contexts),
            }

        return states

    async def send_context_aware_message(
        self,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: Dict[str, Any],
        context_ids: List[str] = None,
        include_relevant_context: bool = True,
    ) -> bool:
        """Send a message with relevant context automatically included."""

        enhanced_payload = payload.copy()

        if include_relevant_context and context_ids:
            enhanced_payload["shared_contexts"] = {}

            for context_id in context_ids:
                context_data = await self.get_shared_context(context_id, from_agent)
                if context_data:
                    enhanced_payload["shared_contexts"][context_id] = context_data

        # Add agent state if relevant
        if from_agent in self.agent_states:
            enhanced_payload["sender_state"] = {
                "current_task": self.agent_states[from_agent].current_task,
                "status": self.agent_states[from_agent].status,
                "capabilities": self.agent_states[from_agent].capabilities,
            }

        return await self.send_message(
            from_agent=from_agent,
            to_agent=to_agent,
            topic=topic,
            payload=enhanced_payload,
            message_type=MessageType.DIRECT,
        )

    async def broadcast_context_update(
        self,
        from_agent: str,
        context_id: str,
        update_data: Dict[str, Any],
        target_participants: Set[str] = None,
    ) -> None:
        """Broadcast context update to relevant agents."""

        if context_id not in self.shared_contexts:
            return

        shared_context = self.shared_contexts[context_id]
        participants = target_participants or shared_context.participants

        for participant in participants:
            if participant != from_agent:
                await self.send_message(
                    from_agent=from_agent,
                    to_agent=participant,
                    topic="context_broadcast_update",
                    payload={
                        "context_id": context_id,
                        "update_data": update_data,
                        "context_type": shared_context.type.value,
                        "timestamp": time.time(),
                    },
                    message_type=MessageType.NOTIFICATION,
                )

    async def _context_sync_loop(self) -> None:
        """Background loop for batched context synchronization."""

        while True:
            try:
                await asyncio.sleep(5)  # Sync every 5 seconds

                for context_id, shared_context in self.shared_contexts.items():
                    if shared_context.sync_mode == SyncMode.BATCHED:
                        # Check if context has pending updates
                        # This is a simplified implementation
                        pass

            except Exception as e:
                logger.error("Context sync loop error", error=str(e))

    async def _agent_state_sync_loop(self) -> None:
        """Background loop for agent state synchronization."""

        while True:
            try:
                await asyncio.sleep(10)  # Sync every 10 seconds

                # Clean up inactive agents
                current_time = time.time()
                inactive_agents = []

                for agent_name, agent_state in self.agent_states.items():
                    if current_time - agent_state.last_activity > 300:  # 5 minutes
                        inactive_agents.append(agent_name)

                for agent_name in inactive_agents:
                    await self._handle_agent_inactive(agent_name)

            except Exception as e:
                logger.error("Agent state sync loop error", error=str(e))

    async def _persist_shared_context(self, shared_context: SharedContext) -> None:
        """Persist shared context to Redis."""

        context_key = f"{self.context_prefix}:{shared_context.id}"

        context_data = {
            "id": shared_context.id,
            "type": shared_context.type.value,
            "owner_agent": shared_context.owner_agent,
            "participants": json.dumps(list(shared_context.participants)),
            "data": json.dumps(shared_context.data),
            "version": str(shared_context.version),
            "last_updated": str(shared_context.last_updated),
            "last_updated_by": shared_context.last_updated_by,
            "sync_mode": shared_context.sync_mode.value,
            "metadata": json.dumps(shared_context.metadata),
        }

        if shared_context.ttl:
            context_data["ttl"] = str(shared_context.ttl)

        await self.redis_client.hset(context_key, mapping=context_data)

        # Set TTL if specified
        if shared_context.ttl:
            ttl_seconds = max(1, int(shared_context.ttl - time.time()))
            await self.redis_client.expire(context_key, ttl_seconds)
        else:
            await self.redis_client.expire(context_key, 86400)  # 24 hours default

    async def _load_shared_context(self, context_id: str) -> Optional[SharedContext]:
        """Load shared context from Redis."""

        context_key = f"{self.context_prefix}:{context_id}"
        context_data = await self.redis_client.hgetall(context_key)

        if not context_data:
            return None

        return SharedContext(
            id=context_data["id"],
            type=ContextShareType(context_data["type"]),
            owner_agent=context_data["owner_agent"],
            participants=set(json.loads(context_data["participants"])),
            data=json.loads(context_data["data"]),
            version=int(context_data["version"]),
            last_updated=float(context_data["last_updated"]),
            last_updated_by=context_data["last_updated_by"],
            sync_mode=SyncMode(context_data["sync_mode"]),
            ttl=float(context_data["ttl"]) if context_data.get("ttl") else None,
            metadata=json.loads(context_data.get("metadata", "{}")),
        )

    async def _persist_agent_state(self, agent_state: AgentState) -> None:
        """Persist agent state to Redis."""

        state_key = f"{self.state_prefix}:{agent_state.agent_name}"

        state_data = {
            "agent_name": agent_state.agent_name,
            "current_task": agent_state.current_task or "",
            "status": agent_state.status,
            "capabilities": json.dumps(agent_state.capabilities),
            "performance_metrics": json.dumps(agent_state.performance_metrics),
            "last_activity": str(agent_state.last_activity),
            "shared_contexts": json.dumps(list(agent_state.shared_contexts)),
            "preferences": json.dumps(agent_state.preferences),
        }

        await self.redis_client.hset(state_key, mapping=state_data)
        await self.redis_client.expire(state_key, 3600)  # 1 hour TTL

    async def _notify_context_created(self, shared_context: SharedContext) -> None:
        """Notify participants about new context creation."""

        for participant in shared_context.participants:
            if participant != shared_context.owner_agent:
                await self.send_message(
                    from_agent="context_system",
                    to_agent=participant,
                    topic="context_created",
                    payload={
                        "context_id": shared_context.id,
                        "context_type": shared_context.type.value,
                        "owner": shared_context.owner_agent,
                        "data": shared_context.data,
                    },
                    message_type=MessageType.NOTIFICATION,
                )

    async def _notify_context_updated(
        self, update: ContextUpdate, participants: Set[str]
    ) -> None:
        """Notify participants about context updates."""

        for participant in participants:
            await self.send_message(
                from_agent="context_system",
                to_agent=participant,
                topic="context_updated",
                payload={
                    "context_id": update.context_id,
                    "update_type": update.update_type,
                    "changes": update.changes,
                    "previous_version": update.previous_version,
                    "new_version": update.new_version,
                    "updated_by": update.updated_by,
                    "timestamp": update.timestamp,
                },
                message_type=MessageType.NOTIFICATION,
            )

    async def _notify_agent_state_updated(self, agent_state: AgentState) -> None:
        """Notify interested agents about state changes."""

        # Find agents that share contexts with this agent
        interested_agents = set()

        for context_id in agent_state.shared_contexts:
            if context_id in self.shared_contexts:
                interested_agents.update(self.shared_contexts[context_id].participants)

        # Remove the agent itself
        interested_agents.discard(agent_state.agent_name)

        state_update = {
            "agent_name": agent_state.agent_name,
            "status": agent_state.status,
            "current_task": agent_state.current_task,
            "last_activity": agent_state.last_activity,
        }

        for interested_agent in interested_agents:
            await self.send_message(
                from_agent="coordination_system",
                to_agent=interested_agent,
                topic="agent_state_updated",
                payload=state_update,
                message_type=MessageType.NOTIFICATION,
            )

    async def _handle_agent_inactive(self, agent_name: str) -> None:
        """Handle agent becoming inactive."""

        if agent_name in self.agent_states:
            agent_state = self.agent_states[agent_name]
            agent_state.status = "inactive"

            # Update shared contexts
            for context_id in agent_state.shared_contexts:
                if context_id in self.shared_contexts:
                    shared_context = self.shared_contexts[context_id]
                    shared_context.participants.discard(agent_name)
                    await self._persist_shared_context(shared_context)

            # Notify other agents
            await self._notify_agent_state_updated(agent_state)

            logger.info("Agent marked as inactive", agent=agent_name)

    def _deep_merge(
        self, base: Dict[str, Any], updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries."""

        result = base.copy()

        for key, value in updates.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value

        return result

    async def shutdown(self) -> None:
        """Shutdown enhanced message broker."""

        # Cancel sync tasks
        for task_name, task in self.sync_tasks.items():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await super().shutdown()
        logger.info("Enhanced message broker shutdown complete")


# Global enhanced message broker instance
enhanced_message_broker = None


def get_enhanced_message_broker(
    context_engine: ContextEngine = None,
) -> EnhancedMessageBroker:
    """Get enhanced message broker instance."""
    global enhanced_message_broker

    if enhanced_message_broker is None:
        enhanced_message_broker = EnhancedMessageBroker(context_engine=context_engine)

    return enhanced_message_broker
