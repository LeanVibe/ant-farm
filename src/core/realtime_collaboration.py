"""Real-time collaboration synchronization system for multi-agent coordination."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable

import structlog

from .enhanced_message_broker import (
    EnhancedMessageBroker,
    ContextShareType,
    SyncMode,
    AgentState,
)
from .communication_monitor import get_communication_monitor

logger = structlog.get_logger()


class CollaborationState(Enum):
    """States of collaborative work sessions."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    SYNCHRONIZING = "synchronizing"
    COMPLETED = "completed"
    FAILED = "failed"


class ConflictResolutionStrategy(Enum):
    """Strategies for resolving conflicts in collaborative editing."""

    LAST_WRITER_WINS = "last_writer_wins"
    MERGE_AUTOMATIC = "merge_automatic"
    MERGE_MANUAL = "merge_manual"
    VERSION_BRANCHING = "version_branching"
    CONSENSUS_REQUIRED = "consensus_required"


@dataclass
class CollaborativeSession:
    """A real-time collaborative session between agents."""

    id: str
    title: str
    participants: Set[str] = field(default_factory=set)
    coordinator: str = ""
    state: CollaborationState = CollaborationState.INITIALIZING
    shared_resources: Dict[str, Any] = field(default_factory=dict)
    version_history: List[Dict[str, Any]] = field(default_factory=list)
    current_version: int = 0
    conflict_resolution: ConflictResolutionStrategy = (
        ConflictResolutionStrategy.MERGE_AUTOMATIC
    )
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncOperation:
    """An operation to be synchronized across agents."""

    id: str
    operation_type: str  # "create", "update", "delete", "move", "copy"
    resource_path: str
    data: Any
    author: str
    timestamp: float
    depends_on: List[str] = field(default_factory=list)
    conflicts_with: List[str] = field(default_factory=list)


@dataclass
class ConflictResult:
    """Result of conflict resolution."""

    conflict_id: str
    resolution_strategy: ConflictResolutionStrategy
    resolved_data: Any
    resolution_author: str
    resolution_timestamp: float
    involved_operations: List[str]
    manual_review_required: bool = False


class RealTimeCollaborationSync:
    """Real-time synchronization system for collaborative agent work."""

    def __init__(self, enhanced_broker: EnhancedMessageBroker):
        self.broker = enhanced_broker
        self.active_sessions: Dict[str, CollaborativeSession] = {}
        self.pending_operations: Dict[
            str, List[SyncOperation]
        ] = {}  # session_id -> operations
        self.operation_handlers: Dict[str, Callable] = {}
        self.conflict_resolvers: Dict[ConflictResolutionStrategy, Callable] = {
            ConflictResolutionStrategy.LAST_WRITER_WINS: self._resolve_last_writer_wins,
            ConflictResolutionStrategy.MERGE_AUTOMATIC: self._resolve_merge_automatic,
            ConflictResolutionStrategy.VERSION_BRANCHING: self._resolve_version_branching,
        }

        # Sync metrics
        self.sync_metrics = {
            "operations_processed": 0,
            "conflicts_resolved": 0,
            "sessions_active": 0,
            "average_sync_latency": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize the collaboration sync system."""

        # Register message handlers
        self.operation_handlers = {
            "start_collaboration": self._handle_start_collaboration,
            "join_collaboration": self._handle_join_collaboration,
            "sync_operation": self._handle_sync_operation,
            "resolve_conflict": self._handle_resolve_conflict,
            "collaboration_state_change": self._handle_state_change,
        }

        # Start background sync processor
        asyncio.create_task(self._sync_processor_loop())
        asyncio.create_task(self._session_monitor_loop())

        logger.info("Real-time collaboration sync initialized")

    async def start_collaboration_session(
        self,
        title: str,
        coordinator: str,
        initial_participants: Set[str] = None,
        shared_resources: Dict[str, Any] = None,
        conflict_resolution: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE_AUTOMATIC,
    ) -> str:
        """Start a new collaborative session."""

        start_time = time.time()
        communication_monitor = get_communication_monitor()

        session_id = str(uuid.uuid4())

        session = CollaborativeSession(
            id=session_id,
            title=title,
            coordinator=coordinator,
            participants=initial_participants or {coordinator},
            shared_resources=shared_resources or {},
            conflict_resolution=conflict_resolution,
        )

        # Create shared context for the session
        context_id = await self.broker.create_shared_context(
            context_type=ContextShareType.WORK_SESSION,
            owner_agent=coordinator,
            initial_data={
                "session_id": session_id,
                "title": title,
                "shared_resources": session.shared_resources,
                "state": session.state.value,
            },
            participants=session.participants,
            sync_mode=SyncMode.REAL_TIME,
        )

        session.metadata["context_id"] = context_id
        self.active_sessions[session_id] = session
        self.pending_operations[session_id] = []

        # Notify participants
        await self._notify_session_started(session)

        # Transition to active state
        session.state = CollaborationState.ACTIVE
        await self._update_session_state(session)

        self.sync_metrics["sessions_active"] += 1

        # Record collaboration metrics
        session_start_time = time.time() - start_time
        await self._record_collaboration_metrics(
            "session_started",
            coordinator,
            {
                "session_id": session_id,
                "participant_count": len(session.participants),
                "setup_time": session_start_time,
                "has_shared_resources": bool(shared_resources),
                "conflict_resolution": conflict_resolution.value,
            },
        )

        logger.info(
            "Collaboration session started",
            session_id=session_id,
            coordinator=coordinator,
            participants=len(session.participants),
            setup_time=session_start_time,
        )

        return session_id

    async def join_collaboration_session(
        self, session_id: str, agent_name: str
    ) -> bool:
        """Join an existing collaboration session."""

        if session_id not in self.active_sessions:
            logger.error("Session not found", session_id=session_id)
            return False

        session = self.active_sessions[session_id]

        if session.state != CollaborationState.ACTIVE:
            logger.error(
                "Session not active", session_id=session_id, state=session.state.value
            )
            return False

        # Add to participants
        session.participants.add(agent_name)
        session.last_activity = time.time()

        # Join shared context
        context_id = session.metadata.get("context_id")
        if context_id:
            await self.broker.join_shared_context(context_id, agent_name)

        # Send current session state to new participant
        await self.broker.send_message(
            from_agent="collaboration_system",
            to_agent=agent_name,
            topic="session_joined",
            payload={
                "session_id": session_id,
                "title": session.title,
                "shared_resources": session.shared_resources,
                "current_version": session.current_version,
                "participants": list(session.participants),
                "coordinator": session.coordinator,
            },
        )

        # Notify other participants
        await self._notify_participant_joined(session, agent_name)

        logger.info(
            "Agent joined collaboration session",
            session_id=session_id,
            agent=agent_name,
        )

        return True

    async def submit_sync_operation(
        self,
        session_id: str,
        operation_type: str,
        resource_path: str,
        data: Any,
        author: str,
        depends_on: List[str] = None,
    ) -> str:
        """Submit an operation for synchronization."""

        if session_id not in self.active_sessions:
            raise ValueError(f"Session {session_id} not found")

        session = self.active_sessions[session_id]

        if author not in session.participants:
            raise ValueError(f"Agent {author} not participant in session")

        operation = SyncOperation(
            id=str(uuid.uuid4()),
            operation_type=operation_type,
            resource_path=resource_path,
            data=data,
            author=author,
            timestamp=time.time(),
            depends_on=depends_on or [],
        )

        # Add to pending operations
        self.pending_operations[session_id].append(operation)

        # Notify other participants immediately for real-time sync
        await self._broadcast_operation(session, operation)

        logger.debug(
            "Sync operation submitted",
            session_id=session_id,
            operation_id=operation.id,
            type=operation_type,
            author=author,
        )

        return operation.id

    async def get_session_state(
        self, session_id: str, agent_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get current state of a collaboration session."""

        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        if agent_name not in session.participants:
            return None

        return {
            "id": session.id,
            "title": session.title,
            "state": session.state.value,
            "participants": list(session.participants),
            "coordinator": session.coordinator,
            "shared_resources": session.shared_resources,
            "current_version": session.current_version,
            "last_activity": session.last_activity,
            "pending_operations": len(self.pending_operations.get(session_id, [])),
        }

    async def pause_collaboration(self, session_id: str, agent_name: str) -> bool:
        """Pause a collaboration session."""

        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        if agent_name != session.coordinator:
            logger.error(
                "Only coordinator can pause session",
                session_id=session_id,
                agent=agent_name,
            )
            return False

        session.state = CollaborationState.PAUSED
        await self._update_session_state(session)

        # Notify all participants
        await self._notify_session_paused(session)

        logger.info(
            "Collaboration session paused", session_id=session_id, by=agent_name
        )

        return True

    async def resume_collaboration(self, session_id: str, agent_name: str) -> bool:
        """Resume a paused collaboration session."""

        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        if agent_name != session.coordinator:
            logger.error(
                "Only coordinator can resume session",
                session_id=session_id,
                agent=agent_name,
            )
            return False

        if session.state != CollaborationState.PAUSED:
            return False

        session.state = CollaborationState.ACTIVE
        await self._update_session_state(session)

        # Process any pending operations
        await self._process_pending_operations(session_id)

        # Notify all participants
        await self._notify_session_resumed(session)

        logger.info(
            "Collaboration session resumed", session_id=session_id, by=agent_name
        )

        return True

    async def _sync_processor_loop(self) -> None:
        """Background loop to process synchronization operations."""

        while True:
            try:
                await asyncio.sleep(0.1)  # Process every 100ms for real-time feel

                for session_id in list(self.active_sessions.keys()):
                    await self._process_pending_operations(session_id)

            except Exception as e:
                logger.error("Sync processor error", error=str(e))

    async def _session_monitor_loop(self) -> None:
        """Background loop to monitor session health and cleanup."""

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                current_time = time.time()
                inactive_sessions = []

                for session_id, session in self.active_sessions.items():
                    # Mark sessions inactive after 1 hour of no activity
                    if current_time - session.last_activity > 3600:
                        inactive_sessions.append(session_id)

                for session_id in inactive_sessions:
                    await self._cleanup_inactive_session(session_id)

            except Exception as e:
                logger.error("Session monitor error", error=str(e))

    async def _process_pending_operations(self, session_id: str) -> None:
        """Process pending operations for a session."""

        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]

        if session.state != CollaborationState.ACTIVE:
            return

        operations = self.pending_operations.get(session_id, [])
        if not operations:
            return

        # Set synchronizing state
        if len(operations) > 5:  # Only for significant batches
            session.state = CollaborationState.SYNCHRONIZING
            await self._update_session_state(session)

        processed_operations = []
        conflicts_detected = []

        for operation in operations:
            try:
                # Check for conflicts with previous operations
                conflicts = await self._detect_conflicts(
                    session, operation, processed_operations
                )

                if conflicts:
                    conflicts_detected.extend(conflicts)
                    continue

                # Apply operation
                await self._apply_operation(session, operation)
                processed_operations.append(operation)

                self.sync_metrics["operations_processed"] += 1

            except Exception as e:
                logger.error(
                    "Failed to process operation",
                    session_id=session_id,
                    operation_id=operation.id,
                    error=str(e),
                )

        # Remove processed operations
        self.pending_operations[session_id] = [
            op for op in operations if op not in processed_operations
        ]

        # Handle conflicts
        if conflicts_detected:
            await self._handle_conflicts(session, conflicts_detected)

        # Update session version if operations were processed
        if processed_operations:
            session.current_version += len(processed_operations)
            session.last_activity = time.time()

            # Create version history entry
            session.version_history.append(
                {
                    "version": session.current_version,
                    "operations": [op.id for op in processed_operations],
                    "timestamp": time.time(),
                    "conflicts_resolved": len(conflicts_detected),
                }
            )

        # Restore active state
        if session.state == CollaborationState.SYNCHRONIZING:
            session.state = CollaborationState.ACTIVE
            await self._update_session_state(session)

    async def _detect_conflicts(
        self,
        session: CollaborativeSession,
        operation: SyncOperation,
        processed_ops: List[SyncOperation],
    ) -> List[SyncOperation]:
        """Detect conflicts between operations."""

        conflicts = []

        for processed_op in processed_ops:
            # Check for resource path conflicts
            if (
                operation.resource_path == processed_op.resource_path
                and operation.operation_type in ["update", "delete"]
                and processed_op.operation_type in ["update", "delete"]
            ):
                # Time-based conflict detection (operations within 1 second)
                if abs(operation.timestamp - processed_op.timestamp) < 1.0:
                    conflicts.append(processed_op)
                    operation.conflicts_with.append(processed_op.id)

        return conflicts

    async def _apply_operation(
        self, session: CollaborativeSession, operation: SyncOperation
    ) -> None:
        """Apply a synchronization operation to the session."""

        if operation.operation_type == "create":
            if operation.resource_path not in session.shared_resources:
                session.shared_resources[operation.resource_path] = operation.data

        elif operation.operation_type == "update":
            if operation.resource_path in session.shared_resources:
                if isinstance(operation.data, dict) and isinstance(
                    session.shared_resources[operation.resource_path], dict
                ):
                    # Merge dictionaries
                    session.shared_resources[operation.resource_path].update(
                        operation.data
                    )
                else:
                    session.shared_resources[operation.resource_path] = operation.data

        elif operation.operation_type == "delete":
            session.shared_resources.pop(operation.resource_path, None)

        elif operation.operation_type == "move":
            if (
                "old_path" in operation.data
                and operation.data["old_path"] in session.shared_resources
            ):
                value = session.shared_resources.pop(operation.data["old_path"])
                session.shared_resources[operation.resource_path] = value

        # Update shared context
        context_id = session.metadata.get("context_id")
        if context_id:
            await self.broker.update_shared_context(
                context_id=context_id,
                agent_name="collaboration_system",
                updates={
                    "shared_resources": session.shared_resources,
                    "current_version": session.current_version,
                    "last_operation": {
                        "id": operation.id,
                        "type": operation.operation_type,
                        "author": operation.author,
                        "timestamp": operation.timestamp,
                    },
                },
            )

    async def _handle_conflicts(
        self, session: CollaborativeSession, conflicts: List[SyncOperation]
    ) -> None:
        """Handle detected conflicts using the session's resolution strategy."""

        resolver = self.conflict_resolvers.get(session.conflict_resolution)
        if not resolver:
            logger.error(
                "No conflict resolver", strategy=session.conflict_resolution.value
            )
            return

        for conflict in conflicts:
            try:
                resolution = await resolver(session, conflict)

                if resolution.manual_review_required:
                    await self._request_manual_conflict_resolution(
                        session, conflict, resolution
                    )
                else:
                    await self._apply_conflict_resolution(session, resolution)

                self.sync_metrics["conflicts_resolved"] += 1

            except Exception as e:
                logger.error(
                    "Conflict resolution failed", conflict_id=conflict.id, error=str(e)
                )

    async def _resolve_last_writer_wins(
        self, session: CollaborativeSession, conflict: SyncOperation
    ) -> ConflictResult:
        """Resolve conflict using last writer wins strategy."""

        return ConflictResult(
            conflict_id=conflict.id,
            resolution_strategy=ConflictResolutionStrategy.LAST_WRITER_WINS,
            resolved_data=conflict.data,
            resolution_author="system",
            resolution_timestamp=time.time(),
            involved_operations=[conflict.id],
        )

    async def _resolve_merge_automatic(
        self, session: CollaborativeSession, conflict: SyncOperation
    ) -> ConflictResult:
        """Resolve conflict using automatic merge strategy."""

        # Simple merge strategy - prefer new data but preserve existing structure
        current_data = session.shared_resources.get(conflict.resource_path, {})

        if isinstance(current_data, dict) and isinstance(conflict.data, dict):
            merged_data = {**current_data, **conflict.data}
        else:
            merged_data = conflict.data

        return ConflictResult(
            conflict_id=conflict.id,
            resolution_strategy=ConflictResolutionStrategy.MERGE_AUTOMATIC,
            resolved_data=merged_data,
            resolution_author="system",
            resolution_timestamp=time.time(),
            involved_operations=[conflict.id],
        )

    async def _resolve_version_branching(
        self, session: CollaborativeSession, conflict: SyncOperation
    ) -> ConflictResult:
        """Resolve conflict using version branching strategy."""

        # Create a branch for the conflicting change
        branch_path = (
            f"{conflict.resource_path}_branch_{conflict.author}_{int(time.time())}"
        )

        return ConflictResult(
            conflict_id=conflict.id,
            resolution_strategy=ConflictResolutionStrategy.VERSION_BRANCHING,
            resolved_data={
                "original_path": conflict.resource_path,
                "branch_path": branch_path,
            },
            resolution_author="system",
            resolution_timestamp=time.time(),
            involved_operations=[conflict.id],
            manual_review_required=True,
        )

    async def _broadcast_operation(
        self, session: CollaborativeSession, operation: SyncOperation
    ) -> None:
        """Broadcast operation to all session participants."""

        for participant in session.participants:
            if participant != operation.author:
                await self.broker.send_message(
                    from_agent="collaboration_system",
                    to_agent=participant,
                    topic="sync_operation_broadcast",
                    payload={
                        "session_id": session.id,
                        "operation": {
                            "id": operation.id,
                            "type": operation.operation_type,
                            "resource_path": operation.resource_path,
                            "data": operation.data,
                            "author": operation.author,
                            "timestamp": operation.timestamp,
                        },
                    },
                )

    async def _update_session_state(self, session: CollaborativeSession) -> None:
        """Update session state in shared context."""

        context_id = session.metadata.get("context_id")
        if context_id:
            await self.broker.update_shared_context(
                context_id=context_id,
                agent_name="collaboration_system",
                updates={
                    "state": session.state.value,
                    "last_activity": session.last_activity,
                    "current_version": session.current_version,
                },
            )

    async def _notify_session_started(self, session: CollaborativeSession) -> None:
        """Notify participants that session has started."""

        for participant in session.participants:
            await self.broker.send_message(
                from_agent="collaboration_system",
                to_agent=participant,
                topic="collaboration_started",
                payload={
                    "session_id": session.id,
                    "title": session.title,
                    "coordinator": session.coordinator,
                    "participants": list(session.participants),
                },
            )

    async def _notify_participant_joined(
        self, session: CollaborativeSession, new_participant: str
    ) -> None:
        """Notify participants that someone joined."""

        for participant in session.participants:
            if participant != new_participant:
                await self.broker.send_message(
                    from_agent="collaboration_system",
                    to_agent=participant,
                    topic="participant_joined",
                    payload={
                        "session_id": session.id,
                        "new_participant": new_participant,
                        "total_participants": len(session.participants),
                    },
                )

    async def _notify_session_paused(self, session: CollaborativeSession) -> None:
        """Notify participants that session is paused."""

        for participant in session.participants:
            await self.broker.send_message(
                from_agent="collaboration_system",
                to_agent=participant,
                topic="collaboration_paused",
                payload={"session_id": session.id},
            )

    async def _notify_session_resumed(self, session: CollaborativeSession) -> None:
        """Notify participants that session is resumed."""

        for participant in session.participants:
            await self.broker.send_message(
                from_agent="collaboration_system",
                to_agent=participant,
                topic="collaboration_resumed",
                payload={"session_id": session.id},
            )

    async def _cleanup_inactive_session(self, session_id: str) -> None:
        """Clean up inactive collaboration session."""

        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.state = CollaborationState.COMPLETED

            # Notify participants
            for participant in session.participants:
                await self.broker.send_message(
                    from_agent="collaboration_system",
                    to_agent=participant,
                    topic="collaboration_completed",
                    payload={
                        "session_id": session_id,
                        "reason": "inactive",
                        "final_version": session.current_version,
                    },
                )

            # Clean up
            del self.active_sessions[session_id]
            self.pending_operations.pop(session_id, None)
            self.sync_metrics["sessions_active"] -= 1

            logger.info(
                "Inactive collaboration session cleaned up", session_id=session_id
            )

    async def get_sync_metrics(self) -> Dict[str, Any]:
        """Get synchronization system metrics."""

        self.sync_metrics["sessions_active"] = len(self.active_sessions)

        return self.sync_metrics.copy()

    # Message handlers for enhanced broker integration
    async def _handle_start_collaboration(self, message) -> Dict[str, Any]:
        """Handle start collaboration request."""

        payload = message.payload

        session_id = await self.start_collaboration_session(
            title=payload["title"],
            coordinator=message.from_agent,
            initial_participants=set(payload.get("participants", [message.from_agent])),
            shared_resources=payload.get("shared_resources", {}),
            conflict_resolution=ConflictResolutionStrategy(
                payload.get("conflict_resolution", "merge_automatic")
            ),
        )

        return {"session_id": session_id, "status": "started"}

    async def _handle_join_collaboration(self, message) -> Dict[str, Any]:
        """Handle join collaboration request."""

        session_id = message.payload["session_id"]
        success = await self.join_collaboration_session(session_id, message.from_agent)

        return {"status": "joined" if success else "failed"}

    async def _handle_sync_operation(self, message) -> Dict[str, Any]:
        """Handle sync operation submission."""

        payload = message.payload

        operation_id = await self.submit_sync_operation(
            session_id=payload["session_id"],
            operation_type=payload["operation_type"],
            resource_path=payload["resource_path"],
            data=payload["data"],
            author=message.from_agent,
            depends_on=payload.get("depends_on", []),
        )

        return {"operation_id": operation_id, "status": "submitted"}

    async def _handle_resolve_conflict(self, message) -> Dict[str, Any]:
        """Handle manual conflict resolution."""

        # Implementation for manual conflict resolution
        return {"status": "resolved"}

    async def _handle_state_change(self, message) -> Dict[str, Any]:
        """Handle collaboration state change requests."""

        payload = message.payload
        session_id = payload["session_id"]
        new_state = payload["state"]

        if new_state == "paused":
            success = await self.pause_collaboration(session_id, message.from_agent)
        elif new_state == "active":
            success = await self.resume_collaboration(session_id, message.from_agent)
        else:
            success = False

        return {"status": "changed" if success else "failed"}

    async def _record_collaboration_metrics(
        self,
        metric_type: str,
        agent_name: str,
        metric_data: Dict[str, Any],
    ) -> None:
        """Record collaboration-specific performance metrics."""

        communication_monitor = get_communication_monitor()

        if hasattr(communication_monitor, "metrics_buffer"):
            from .communication_monitor import CommunicationMetric, MetricType

            # Map collaboration metrics to communication metric types
            metric_type_mapping = {
                "session_started": MetricType.THROUGHPUT,
                "session_joined": MetricType.THROUGHPUT,
                "operation_submitted": MetricType.LATENCY,
                "sync_completed": MetricType.RELIABILITY,
                "conflict_resolved": MetricType.ERROR_RATE,
            }

            comm_metric_type = metric_type_mapping.get(
                metric_type, MetricType.THROUGHPUT
            )

            # Extract meaningful value based on metric type
            value = 1.0  # Default value for count-based metrics
            if metric_type == "operation_submitted" and "latency" in metric_data:
                value = metric_data["latency"]
            elif metric_type == "session_started" and "setup_time" in metric_data:
                value = metric_data["setup_time"] * 1000  # Convert to milliseconds

            metric = CommunicationMetric(
                metric_type=comm_metric_type,
                value=value,
                timestamp=time.time(),
                agent_name=agent_name,
                metadata={
                    "collaboration_metric": metric_type,
                    **metric_data,
                },
            )

            communication_monitor.metrics_buffer.append(metric)

    async def get_collaboration_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for collaboration features."""

        current_time = time.time()

        # Calculate session statistics
        active_sessions = len(self.active_sessions)
        total_participants = sum(
            len(session.participants) for session in self.active_sessions.values()
        )

        # Calculate average session duration for completed sessions
        session_durations = []
        for session in self.active_sessions.values():
            if session.state == CollaborationState.COMPLETED:
                duration = current_time - session.created_at
                session_durations.append(duration)

        avg_session_duration = (
            sum(session_durations) / len(session_durations) if session_durations else 0
        )

        # Calculate sync efficiency
        total_operations = sum(len(ops) for ops in self.pending_operations.values())

        return {
            "active_sessions": active_sessions,
            "total_participants": total_participants,
            "average_participants_per_session": (
                total_participants / active_sessions if active_sessions > 0 else 0
            ),
            "pending_operations": total_operations,
            "average_session_duration": avg_session_duration,
            "sync_metrics": self.sync_metrics.copy(),
            "timestamp": current_time,
        }


# Global collaboration sync instance
collaboration_sync = None


def get_collaboration_sync(
    enhanced_broker: EnhancedMessageBroker,
) -> RealTimeCollaborationSync:
    """Get collaboration sync instance."""
    global collaboration_sync

    if collaboration_sync is None:
        collaboration_sync = RealTimeCollaborationSync(enhanced_broker)

    return collaboration_sync
