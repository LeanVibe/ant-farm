"""Agent Lifecycle State Machine.

This module provides comprehensive lifecycle management for agents including:
- State machine with well-defined transitions
- Lifecycle event handling
- State persistence and recovery
- Lifecycle monitoring and metrics
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import traceback

try:
    import structlog

    logger = structlog.get_logger()
    use_structlog = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    use_structlog = False


def log_info(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.info(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{message} {extra_info}" if extra_info else message)


def log_warning(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.warning(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.warning(f"{message} {extra_info}" if extra_info else message)


def log_error(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.error(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.error(f"{message} {extra_info}" if extra_info else message)


def log_debug(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.debug(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"{message} {extra_info}" if extra_info else message)


class AgentState(Enum):
    """Agent lifecycle states."""

    CREATED = "created"  # Agent object created but not initialized
    INITIALIZING = "initializing"  # Agent is starting up and initializing components
    READY = "ready"  # Agent is ready but not actively working
    ACTIVE = "active"  # Agent is actively processing tasks
    BUSY = "busy"  # Agent is at capacity
    DEGRADED = "degraded"  # Agent is working but with reduced performance
    RECOVERING = "recovering"  # Agent is recovering from an error or failure
    PAUSED = "paused"  # Agent is temporarily paused
    STOPPING = "stopping"  # Agent is gracefully shutting down
    STOPPED = "stopped"  # Agent has stopped and resources cleaned up
    ERROR = "error"  # Agent is in an error state
    TERMINATED = "terminated"  # Agent has been forcefully terminated


class LifecycleEvent(Enum):
    """Lifecycle events that trigger state transitions."""

    START = "start"
    INITIALIZE = "initialize"
    ACTIVATE = "activate"
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    HEALTH_DEGRADED = "health_degraded"
    HEALTH_RECOVERED = "health_recovered"
    ERROR_OCCURRED = "error_occurred"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    RECOVERY_FAILED = "recovery_failed"
    PAUSE = "pause"
    RESUME = "resume"
    STOP = "stop"
    TERMINATE = "terminate"
    SHUTDOWN = "shutdown"


@dataclass
class StateTransition:
    """Represents a state transition."""

    from_state: AgentState
    to_state: AgentState
    event: LifecycleEvent
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    duration_ms: Optional[float] = None


@dataclass
class LifecycleConfig:
    """Configuration for agent lifecycle behavior."""

    startup_timeout: float = 60.0  # seconds
    shutdown_timeout: float = 30.0  # seconds
    recovery_timeout: float = 300.0  # seconds
    health_check_interval: float = 30.0  # seconds
    state_persistence: bool = True
    enable_auto_recovery: bool = True
    max_recovery_attempts: int = 3
    degraded_performance_threshold: float = 0.5  # 50% performance
    busy_threshold: int = 10  # max concurrent tasks


class AgentLifecycle:
    """Manages the lifecycle of a single agent."""

    def __init__(self, agent_name: str, config: Optional[LifecycleConfig] = None):
        self.agent_name = agent_name
        self.config = config or LifecycleConfig()
        self.current_state = AgentState.CREATED
        self.previous_state: Optional[AgentState] = None
        self.state_entered_at = time.time()
        self.state_history: List[StateTransition] = []
        self.event_handlers: Dict[LifecycleEvent, List[Callable]] = {}
        self.state_handlers: Dict[AgentState, List[Callable]] = {}
        self.transition_guards: Dict[
            tuple, List[Callable]
        ] = {}  # (from_state, event) -> guards
        self.active_tasks: Set[str] = set()
        self.performance_metrics: Dict[str, float] = {}
        self.recovery_attempts = 0
        self.lifecycle_task: Optional[asyncio.Task] = None

        # Define valid state transitions
        self.valid_transitions = self._define_valid_transitions()

        # Initialize event handlers
        for event in LifecycleEvent:
            self.event_handlers[event] = []
        for state in AgentState:
            self.state_handlers[state] = []

    def _define_valid_transitions(self) -> Dict[tuple, AgentState]:
        """Define valid state transitions."""
        return {
            # Initial startup flow
            (AgentState.CREATED, LifecycleEvent.START): AgentState.INITIALIZING,
            (AgentState.INITIALIZING, LifecycleEvent.INITIALIZE): AgentState.READY,
            (AgentState.READY, LifecycleEvent.ACTIVATE): AgentState.ACTIVE,
            # Task handling
            (AgentState.ACTIVE, LifecycleEvent.TASK_ASSIGNED): AgentState.BUSY,
            (AgentState.BUSY, LifecycleEvent.TASK_COMPLETED): AgentState.ACTIVE,
            (AgentState.BUSY, LifecycleEvent.TASK_FAILED): AgentState.ACTIVE,
            (AgentState.ACTIVE, LifecycleEvent.TASK_COMPLETED): AgentState.ACTIVE,
            (AgentState.ACTIVE, LifecycleEvent.TASK_FAILED): AgentState.ACTIVE,
            # Health degradation
            (AgentState.ACTIVE, LifecycleEvent.HEALTH_DEGRADED): AgentState.DEGRADED,
            (AgentState.BUSY, LifecycleEvent.HEALTH_DEGRADED): AgentState.DEGRADED,
            (AgentState.DEGRADED, LifecycleEvent.HEALTH_RECOVERED): AgentState.ACTIVE,
            # Error handling
            (AgentState.ACTIVE, LifecycleEvent.ERROR_OCCURRED): AgentState.ERROR,
            (AgentState.BUSY, LifecycleEvent.ERROR_OCCURRED): AgentState.ERROR,
            (AgentState.DEGRADED, LifecycleEvent.ERROR_OCCURRED): AgentState.ERROR,
            (AgentState.ERROR, LifecycleEvent.RECOVERY_STARTED): AgentState.RECOVERING,
            (
                AgentState.RECOVERING,
                LifecycleEvent.RECOVERY_COMPLETED,
            ): AgentState.ACTIVE,
            (AgentState.RECOVERING, LifecycleEvent.RECOVERY_FAILED): AgentState.ERROR,
            # Pause/Resume
            (AgentState.ACTIVE, LifecycleEvent.PAUSE): AgentState.PAUSED,
            (AgentState.BUSY, LifecycleEvent.PAUSE): AgentState.PAUSED,
            (AgentState.DEGRADED, LifecycleEvent.PAUSE): AgentState.PAUSED,
            (AgentState.PAUSED, LifecycleEvent.RESUME): AgentState.ACTIVE,
            # Shutdown
            (AgentState.READY, LifecycleEvent.STOP): AgentState.STOPPING,
            (AgentState.ACTIVE, LifecycleEvent.STOP): AgentState.STOPPING,
            (AgentState.BUSY, LifecycleEvent.STOP): AgentState.STOPPING,
            (AgentState.DEGRADED, LifecycleEvent.STOP): AgentState.STOPPING,
            (AgentState.PAUSED, LifecycleEvent.STOP): AgentState.STOPPING,
            (AgentState.ERROR, LifecycleEvent.STOP): AgentState.STOPPING,
            (AgentState.RECOVERING, LifecycleEvent.STOP): AgentState.STOPPING,
            (AgentState.STOPPING, LifecycleEvent.SHUTDOWN): AgentState.STOPPED,
            # Termination (can happen from any state)
            (AgentState.CREATED, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.INITIALIZING, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.READY, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.ACTIVE, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.BUSY, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.DEGRADED, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.PAUSED, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.ERROR, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.RECOVERING, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
            (AgentState.STOPPING, LifecycleEvent.TERMINATE): AgentState.TERMINATED,
        }

    def add_event_handler(self, event: LifecycleEvent, handler: Callable):
        """Add an event handler."""
        self.event_handlers[event].append(handler)

    def add_state_handler(self, state: AgentState, handler: Callable):
        """Add a state entry handler."""
        self.state_handlers[state].append(handler)

    def add_transition_guard(
        self, from_state: AgentState, event: LifecycleEvent, guard: Callable
    ):
        """Add a guard function for a transition."""
        key = (from_state, event)
        if key not in self.transition_guards:
            self.transition_guards[key] = []
        self.transition_guards[key].append(guard)

    async def trigger_event(
        self, event: LifecycleEvent, metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Trigger a lifecycle event."""
        metadata = metadata or {}

        log_debug(
            "Triggering lifecycle event",
            agent=self.agent_name,
            event=event.value,
            current_state=self.current_state.value,
        )

        # Check if transition is valid
        transition_key = (self.current_state, event)
        if transition_key not in self.valid_transitions:
            log_warning(
                "Invalid state transition",
                agent=self.agent_name,
                from_state=self.current_state.value,
                event=event.value,
            )
            return False

        new_state = self.valid_transitions[transition_key]

        # Check transition guards
        if not await self._check_transition_guards(self.current_state, event, metadata):
            log_warning(
                "Transition blocked by guard",
                agent=self.agent_name,
                from_state=self.current_state.value,
                event=event.value,
            )
            return False

        # Perform the transition
        return await self._perform_transition(new_state, event, metadata)

    async def _check_transition_guards(
        self, from_state: AgentState, event: LifecycleEvent, metadata: Dict[str, Any]
    ) -> bool:
        """Check if transition guards allow the transition."""
        key = (from_state, event)
        guards = self.transition_guards.get(key, [])

        for guard in guards:
            try:
                if asyncio.iscoroutinefunction(guard):
                    allowed = await guard(self, from_state, event, metadata)
                else:
                    allowed = guard(self, from_state, event, metadata)

                if not allowed:
                    return False
            except Exception as e:
                log_error(
                    "Transition guard failed",
                    agent=self.agent_name,
                    guard=str(guard),
                    error=str(e),
                )
                return False

        return True

    async def _perform_transition(
        self, new_state: AgentState, event: LifecycleEvent, metadata: Dict[str, Any]
    ) -> bool:
        """Perform the state transition."""
        start_time = time.time()
        old_state = self.current_state

        try:
            # Run event handlers
            await self._run_event_handlers(event, metadata)

            # Update state
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_entered_at = time.time()

            # Record transition
            transition = StateTransition(
                from_state=old_state,
                to_state=new_state,
                event=event,
                timestamp=start_time,
                metadata=metadata,
                duration_ms=(time.time() - start_time) * 1000,
            )
            self.state_history.append(transition)

            # Run state entry handlers
            await self._run_state_handlers(new_state)

            log_info(
                "State transition completed",
                agent=self.agent_name,
                from_state=old_state.value,
                to_state=new_state.value,
                event=event.value,
                duration_ms=transition.duration_ms,
            )

            return True

        except Exception as e:
            log_error(
                "State transition failed",
                agent=self.agent_name,
                from_state=old_state.value,
                to_state=new_state.value,
                event=event.value,
                error=str(e),
            )
            return False

    async def _run_event_handlers(
        self, event: LifecycleEvent, metadata: Dict[str, Any]
    ):
        """Run event handlers."""
        handlers = self.event_handlers[event]

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self, event, metadata)
                else:
                    handler(self, event, metadata)
            except Exception as e:
                log_error(
                    "Event handler failed",
                    agent=self.agent_name,
                    event=event.value,
                    handler=str(handler),
                    error=str(e),
                )

    async def _run_state_handlers(self, state: AgentState):
        """Run state entry handlers."""
        handlers = self.state_handlers[state]

        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self, state)
                else:
                    handler(self, state)
            except Exception as e:
                log_error(
                    "State handler failed",
                    agent=self.agent_name,
                    state=state.value,
                    handler=str(handler),
                    error=str(e),
                )

    def add_task(self, task_id: str):
        """Add an active task."""
        self.active_tasks.add(task_id)

        # Check if we should transition to busy state
        if (
            self.current_state == AgentState.ACTIVE
            and len(self.active_tasks) >= self.config.busy_threshold
        ):
            asyncio.create_task(self.trigger_event(LifecycleEvent.TASK_ASSIGNED))

    def remove_task(self, task_id: str):
        """Remove an active task."""
        self.active_tasks.discard(task_id)

        # Check if we should transition back to active state
        if (
            self.current_state == AgentState.BUSY
            and len(self.active_tasks) < self.config.busy_threshold
        ):
            asyncio.create_task(self.trigger_event(LifecycleEvent.TASK_COMPLETED))

    def update_performance_metric(self, metric_name: str, value: float):
        """Update a performance metric."""
        self.performance_metrics[metric_name] = value

        # Check for performance degradation
        if (
            metric_name == "success_rate"
            and value < self.config.degraded_performance_threshold
            and self.current_state == AgentState.ACTIVE
        ):
            asyncio.create_task(self.trigger_event(LifecycleEvent.HEALTH_DEGRADED))
        elif (
            metric_name == "success_rate"
            and value >= self.config.degraded_performance_threshold
            and self.current_state == AgentState.DEGRADED
        ):
            asyncio.create_task(self.trigger_event(LifecycleEvent.HEALTH_RECOVERED))

    async def start_lifecycle_monitoring(self):
        """Start lifecycle monitoring."""
        if self.lifecycle_task and not self.lifecycle_task.done():
            log_warning("Lifecycle monitoring already running", agent=self.agent_name)
            return

        self.lifecycle_task = asyncio.create_task(self._lifecycle_monitoring_loop())
        log_info("Lifecycle monitoring started", agent=self.agent_name)

    async def stop_lifecycle_monitoring(self):
        """Stop lifecycle monitoring."""
        if self.lifecycle_task:
            self.lifecycle_task.cancel()
            try:
                await self.lifecycle_task
            except asyncio.CancelledError:
                pass
        log_info("Lifecycle monitoring stopped", agent=self.agent_name)

    async def _lifecycle_monitoring_loop(self):
        """Main lifecycle monitoring loop."""
        while True:
            try:
                await self._check_state_timeouts()
                await self._check_auto_recovery()
                await asyncio.sleep(self.config.health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(
                    "Lifecycle monitoring error", agent=self.agent_name, error=str(e)
                )
                await asyncio.sleep(self.config.health_check_interval)

    async def _check_state_timeouts(self):
        """Check for state timeouts."""
        time_in_state = time.time() - self.state_entered_at

        # Check startup timeout
        if (
            self.current_state == AgentState.INITIALIZING
            and time_in_state > self.config.startup_timeout
        ):
            log_error(
                "Agent startup timeout",
                agent=self.agent_name,
                time_in_state=time_in_state,
            )
            await self.trigger_event(
                LifecycleEvent.ERROR_OCCURRED, {"reason": "startup_timeout"}
            )

        # Check recovery timeout
        elif (
            self.current_state == AgentState.RECOVERING
            and time_in_state > self.config.recovery_timeout
        ):
            log_error(
                "Agent recovery timeout",
                agent=self.agent_name,
                time_in_state=time_in_state,
            )
            await self.trigger_event(
                LifecycleEvent.RECOVERY_FAILED, {"reason": "recovery_timeout"}
            )

    async def _check_auto_recovery(self):
        """Check if auto recovery should be triggered."""
        if (
            self.config.enable_auto_recovery
            and self.current_state == AgentState.ERROR
            and self.recovery_attempts < self.config.max_recovery_attempts
        ):
            log_info(
                "Starting auto recovery",
                agent=self.agent_name,
                attempt=self.recovery_attempts + 1,
            )
            self.recovery_attempts += 1
            await self.trigger_event(
                LifecycleEvent.RECOVERY_STARTED, {"attempt": self.recovery_attempts}
            )

    def get_lifecycle_status(self) -> Dict[str, Any]:
        """Get current lifecycle status."""
        time_in_state = time.time() - self.state_entered_at

        return {
            "agent_name": self.agent_name,
            "current_state": self.current_state.value,
            "previous_state": self.previous_state.value
            if self.previous_state
            else None,
            "time_in_state": time_in_state,
            "state_entered_at": self.state_entered_at,
            "active_tasks": list(self.active_tasks),
            "task_count": len(self.active_tasks),
            "performance_metrics": self.performance_metrics.copy(),
            "recovery_attempts": self.recovery_attempts,
            "state_history": [
                {
                    "from_state": t.from_state.value,
                    "to_state": t.to_state.value,
                    "event": t.event.value,
                    "timestamp": t.timestamp,
                    "duration_ms": t.duration_ms,
                }
                for t in self.state_history[-10:]  # Last 10 transitions
            ],
        }

    def is_operational(self) -> bool:
        """Check if agent is in an operational state."""
        return self.current_state in [
            AgentState.READY,
            AgentState.ACTIVE,
            AgentState.BUSY,
            AgentState.DEGRADED,
        ]

    def can_accept_tasks(self) -> bool:
        """Check if agent can accept new tasks."""
        return (
            self.current_state in [AgentState.READY, AgentState.ACTIVE]
            and len(self.active_tasks) < self.config.busy_threshold
        )


class LifecycleSystemManager:
    """Manages lifecycles for all agents in the system."""

    def __init__(self):
        self.lifecycles: Dict[str, AgentLifecycle] = {}
        self.system_config = LifecycleConfig()

    def register_agent(self, agent_name: str, config: Optional[LifecycleConfig] = None):
        """Register an agent for lifecycle management."""
        if agent_name in self.lifecycles:
            log_warning(
                "Agent already registered for lifecycle management", agent=agent_name
            )
            return

        lifecycle = AgentLifecycle(agent_name, config or self.system_config)
        self.lifecycles[agent_name] = lifecycle

        log_info("Agent registered for lifecycle management", agent=agent_name)

    def unregister_agent(self, agent_name: str):
        """Unregister an agent from lifecycle management."""
        if agent_name in self.lifecycles:
            del self.lifecycles[agent_name]
            log_info("Agent unregistered from lifecycle management", agent=agent_name)

    def get_lifecycle(self, agent_name: str) -> Optional[AgentLifecycle]:
        """Get lifecycle manager for an agent."""
        return self.lifecycles.get(agent_name)

    async def start_all_monitoring(self):
        """Start lifecycle monitoring for all agents."""
        for lifecycle in self.lifecycles.values():
            await lifecycle.start_lifecycle_monitoring()
        log_info("Lifecycle monitoring started for all agents")

    async def stop_all_monitoring(self):
        """Stop lifecycle monitoring for all agents."""
        for lifecycle in self.lifecycles.values():
            await lifecycle.stop_lifecycle_monitoring()
        log_info("Lifecycle monitoring stopped for all agents")

    async def shutdown_all_agents(self):
        """Gracefully shutdown all agents."""
        shutdown_tasks = []

        for lifecycle in self.lifecycles.values():
            if lifecycle.is_operational():
                shutdown_tasks.append(lifecycle.trigger_event(LifecycleEvent.STOP))

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        # Wait a bit for graceful shutdown
        await asyncio.sleep(2.0)

        # Force shutdown any remaining agents
        shutdown_tasks = []
        for lifecycle in self.lifecycles.values():
            if lifecycle.current_state == AgentState.STOPPING:
                shutdown_tasks.append(lifecycle.trigger_event(LifecycleEvent.SHUTDOWN))

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        log_info("All agents shutdown completed")

    def get_system_status(self) -> Dict[str, Any]:
        """Get system-wide lifecycle status."""
        states_count = {}
        operational_agents = 0
        task_accepting_agents = 0

        agent_statuses = {}
        for agent_name, lifecycle in self.lifecycles.items():
            status = lifecycle.get_lifecycle_status()
            agent_statuses[agent_name] = status

            state = status["current_state"]
            states_count[state] = states_count.get(state, 0) + 1

            if lifecycle.is_operational():
                operational_agents += 1
            if lifecycle.can_accept_tasks():
                task_accepting_agents += 1

        return {
            "total_agents": len(self.lifecycles),
            "operational_agents": operational_agents,
            "task_accepting_agents": task_accepting_agents,
            "states_distribution": states_count,
            "system_health_score": operational_agents / len(self.lifecycles)
            if self.lifecycles
            else 1.0,
            "agents": agent_statuses,
        }


# Global lifecycle system instance
lifecycle_system = LifecycleSystemManager()


def get_lifecycle_system() -> LifecycleSystemManager:
    """Get the global lifecycle system manager."""
    return lifecycle_system
