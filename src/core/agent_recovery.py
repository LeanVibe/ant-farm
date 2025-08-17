"""Agent Recovery and Failover System.

This module provides automatic recovery mechanisms for agents including:
- Automatic restart of failed agents
- Failover to backup agents
- Resource migration and cleanup
- Recovery state management
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


class RecoveryStrategy(Enum):
    """Recovery strategy types."""

    RESTART = "restart"  # Restart the same agent
    FAILOVER = "failover"  # Failover to backup agent
    MIGRATE = "migrate"  # Migrate to different node
    TERMINATE = "terminate"  # Terminate and cleanup


class RecoveryState(Enum):
    """Recovery operation states."""

    NONE = "none"
    DETECTING = "detecting"
    INITIATING = "initiating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class FailureCategory(Enum):
    """Failure categories for recovery decisions."""

    TRANSIENT = "transient"  # Temporary failure, retry
    RESOURCE = "resource"  # Resource exhaustion
    CONFIGURATION = "configuration"  # Configuration issue
    DEPENDENCY = "dependency"  # External dependency failure
    CRITICAL = "critical"  # Critical system failure


@dataclass
class RecoveryAttempt:
    """Record of a recovery attempt."""

    timestamp: float
    strategy: RecoveryStrategy
    failure_reason: str
    recovery_duration: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class AgentFailure:
    """Record of an agent failure."""

    agent_name: str
    timestamp: float
    failure_category: FailureCategory
    error_message: str
    stack_trace: Optional[str] = None
    previous_failures: List["AgentFailure"] = field(default_factory=list)

    @property
    def failure_count(self) -> int:
        """Get total failure count including previous failures."""
        return 1 + len(self.previous_failures)


@dataclass
class RecoveryPolicy:
    """Policy for agent recovery behavior."""

    max_restart_attempts: int = 3
    restart_backoff_base: float = 1.0  # Base delay in seconds
    restart_backoff_multiplier: float = 2.0
    max_restart_delay: float = 300.0  # 5 minutes max
    failure_window: float = 3600.0  # 1 hour window
    max_failures_in_window: int = 5
    enable_failover: bool = True
    enable_migration: bool = False
    health_check_timeout: float = 30.0
    recovery_timeout: float = 600.0  # 10 minutes


class AgentRecoveryManager:
    """Manages automatic recovery for a single agent."""

    def __init__(self, agent_name: str, policy: Optional[RecoveryPolicy] = None):
        self.agent_name = agent_name
        self.policy = policy or RecoveryPolicy()
        self.recovery_state = RecoveryState.NONE
        self.current_recovery_task: Optional[asyncio.Task] = None
        self.failure_history: List[AgentFailure] = []
        self.recovery_attempts: List[RecoveryAttempt] = []
        self.agent_factory: Optional[Callable] = None
        self.backup_agents: List[str] = []
        self.resource_cleanup_handlers: List[Callable] = []
        self.health_monitor = None  # Will be set by recovery system

    def set_agent_factory(self, factory: Callable):
        """Set the factory function for creating new agent instances."""
        self.agent_factory = factory

    def add_backup_agent(self, backup_name: str):
        """Add a backup agent for failover."""
        if backup_name not in self.backup_agents:
            self.backup_agents.append(backup_name)
            log_debug("Backup agent added", agent=self.agent_name, backup=backup_name)

    def add_cleanup_handler(self, handler: Callable):
        """Add a resource cleanup handler."""
        self.resource_cleanup_handlers.append(handler)

    async def handle_agent_failure(self, failure: AgentFailure):
        """Handle an agent failure and initiate recovery."""
        if self.recovery_state in [RecoveryState.IN_PROGRESS, RecoveryState.INITIATING]:
            log_warning(
                "Recovery already in progress, queuing failure",
                agent=self.agent_name,
                current_state=self.recovery_state.value,
            )
            self.failure_history.append(failure)
            return

        # Add to failure history
        self.failure_history.append(failure)

        # Clean up old failures outside the window
        cutoff_time = time.time() - self.policy.failure_window
        self.failure_history = [
            f for f in self.failure_history if f.timestamp > cutoff_time
        ]

        log_error(
            "Agent failure detected, initiating recovery",
            agent=self.agent_name,
            failure_category=failure.failure_category.value,
            error=failure.error_message,
            failure_count=len(self.failure_history),
        )

        # Start recovery process
        self.recovery_state = RecoveryState.INITIATING
        self.current_recovery_task = asyncio.create_task(
            self._perform_recovery(failure)
        )

    async def _perform_recovery(self, failure: AgentFailure):
        """Perform the recovery process."""
        try:
            self.recovery_state = RecoveryState.IN_PROGRESS
            start_time = time.time()

            # Determine recovery strategy
            strategy = self._select_recovery_strategy(failure)

            log_info(
                "Starting recovery",
                agent=self.agent_name,
                strategy=strategy.value,
                failure_category=failure.failure_category.value,
            )

            # Execute recovery strategy
            success = await self._execute_recovery_strategy(strategy, failure)

            # Record recovery attempt
            duration = time.time() - start_time
            attempt = RecoveryAttempt(
                timestamp=start_time,
                strategy=strategy,
                failure_reason=failure.error_message,
                recovery_duration=duration,
                success=success,
            )
            self.recovery_attempts.append(attempt)

            if success:
                self.recovery_state = RecoveryState.COMPLETED
                log_info(
                    "Recovery completed successfully",
                    agent=self.agent_name,
                    strategy=strategy.value,
                    duration=duration,
                )
            else:
                self.recovery_state = RecoveryState.FAILED
                log_error(
                    "Recovery failed",
                    agent=self.agent_name,
                    strategy=strategy.value,
                    duration=duration,
                )

        except Exception as e:
            self.recovery_state = RecoveryState.FAILED
            log_error(
                "Recovery process failed with exception",
                agent=self.agent_name,
                error=str(e),
                traceback=traceback.format_exc(),
            )
        finally:
            # Reset recovery state after a delay
            await asyncio.sleep(5.0)
            if self.recovery_state in [RecoveryState.COMPLETED, RecoveryState.FAILED]:
                self.recovery_state = RecoveryState.NONE

    def _select_recovery_strategy(self, failure: AgentFailure) -> RecoveryStrategy:
        """Select the appropriate recovery strategy based on failure analysis."""
        recent_failures = len(self.failure_history)
        restart_attempts = len(
            [
                a
                for a in self.recovery_attempts
                if a.strategy == RecoveryStrategy.RESTART
            ]
        )

        # If too many failures in window, use failover or terminate
        if recent_failures >= self.policy.max_failures_in_window:
            if self.policy.enable_failover and self.backup_agents:
                return RecoveryStrategy.FAILOVER
            else:
                return RecoveryStrategy.TERMINATE

        # If too many restart attempts, try failover
        if restart_attempts >= self.policy.max_restart_attempts:
            if self.policy.enable_failover and self.backup_agents:
                return RecoveryStrategy.FAILOVER
            elif self.policy.enable_migration:
                return RecoveryStrategy.MIGRATE
            else:
                return RecoveryStrategy.TERMINATE

        # For transient failures, try restart
        if failure.failure_category == FailureCategory.TRANSIENT:
            return RecoveryStrategy.RESTART

        # For resource failures, try migration if available
        if failure.failure_category == FailureCategory.RESOURCE:
            if self.policy.enable_migration:
                return RecoveryStrategy.MIGRATE
            elif self.policy.enable_failover and self.backup_agents:
                return RecoveryStrategy.FAILOVER
            else:
                return RecoveryStrategy.RESTART

        # For critical failures, terminate
        if failure.failure_category == FailureCategory.CRITICAL:
            return RecoveryStrategy.TERMINATE

        # Default to restart
        return RecoveryStrategy.RESTART

    async def _execute_recovery_strategy(
        self, strategy: RecoveryStrategy, failure: AgentFailure
    ) -> bool:
        """Execute the selected recovery strategy."""
        try:
            if strategy == RecoveryStrategy.RESTART:
                return await self._perform_restart()
            elif strategy == RecoveryStrategy.FAILOVER:
                return await self._perform_failover()
            elif strategy == RecoveryStrategy.MIGRATE:
                return await self._perform_migration()
            elif strategy == RecoveryStrategy.TERMINATE:
                return await self._perform_termination()
            else:
                log_error("Unknown recovery strategy", strategy=strategy.value)
                return False
        except Exception as e:
            log_error(
                "Recovery strategy execution failed",
                agent=self.agent_name,
                strategy=strategy.value,
                error=str(e),
            )
            return False

    async def _perform_restart(self) -> bool:
        """Restart the failed agent."""
        if not self.agent_factory:
            log_error("No agent factory available for restart", agent=self.agent_name)
            return False

        # Calculate backoff delay
        restart_attempts = len(
            [
                a
                for a in self.recovery_attempts
                if a.strategy == RecoveryStrategy.RESTART
            ]
        )
        delay = min(
            self.policy.restart_backoff_base
            * (self.policy.restart_backoff_multiplier**restart_attempts),
            self.policy.max_restart_delay,
        )

        if delay > 0:
            log_info(
                "Waiting before restart",
                agent=self.agent_name,
                delay=delay,
                attempt=restart_attempts + 1,
            )
            await asyncio.sleep(delay)

        try:
            # Cleanup existing resources
            await self._cleanup_resources()

            # Create new agent instance
            log_info("Creating new agent instance", agent=self.agent_name)
            new_agent = await self.agent_factory()

            # Verify agent health
            if await self._verify_agent_health(new_agent):
                log_info("Agent restart successful", agent=self.agent_name)
                return True
            else:
                log_error(
                    "Agent health check failed after restart", agent=self.agent_name
                )
                return False

        except Exception as e:
            log_error("Agent restart failed", agent=self.agent_name, error=str(e))
            return False

    async def _perform_failover(self) -> bool:
        """Failover to a backup agent."""
        if not self.backup_agents:
            log_error("No backup agents available for failover", agent=self.agent_name)
            return False

        # Try each backup agent
        for backup_name in self.backup_agents:
            try:
                log_info(
                    "Attempting failover to backup agent",
                    primary=self.agent_name,
                    backup=backup_name,
                )

                # Here we would implement the actual failover logic
                # This is a placeholder for the backup agent activation
                success = await self._activate_backup_agent(backup_name)

                if success:
                    log_info(
                        "Failover successful",
                        primary=self.agent_name,
                        backup=backup_name,
                    )
                    return True

            except Exception as e:
                log_error(
                    "Failover attempt failed",
                    primary=self.agent_name,
                    backup=backup_name,
                    error=str(e),
                )
                continue

        log_error("All failover attempts failed", agent=self.agent_name)
        return False

    async def _perform_migration(self) -> bool:
        """Migrate agent to a different node."""
        # This is a placeholder for migration logic
        # In a real implementation, this would move the agent to a different node/container
        log_info("Migration not implemented yet", agent=self.agent_name)
        return False

    async def _perform_termination(self) -> bool:
        """Terminate the agent and cleanup resources."""
        log_warning("Terminating agent after recovery failure", agent=self.agent_name)

        try:
            await self._cleanup_resources()
            log_info("Agent termination completed", agent=self.agent_name)
            return True
        except Exception as e:
            log_error("Agent termination failed", agent=self.agent_name, error=str(e))
            return False

    async def _cleanup_resources(self):
        """Cleanup agent resources."""
        for handler in self.resource_cleanup_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self.agent_name)
                else:
                    handler(self.agent_name)
            except Exception as e:
                log_error(
                    "Resource cleanup handler failed",
                    agent=self.agent_name,
                    error=str(e),
                )

    async def _activate_backup_agent(self, backup_name: str) -> bool:
        """Activate a backup agent."""
        # This is a placeholder for backup agent activation logic
        # In a real implementation, this would start the backup agent
        # and transfer any necessary state
        log_debug("Activating backup agent", backup=backup_name)
        await asyncio.sleep(1.0)  # Simulate activation time
        return True

    async def _verify_agent_health(self, agent) -> bool:
        """Verify that an agent is healthy."""
        try:
            # Use health monitor if available
            if self.health_monitor:
                health_status = await asyncio.wait_for(
                    self.health_monitor.perform_health_check(),
                    timeout=self.policy.health_check_timeout,
                )
                from src.core.agent_health import HealthStatus

                return health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
            else:
                # Basic health check - verify agent is responsive
                if hasattr(agent, "health_check"):
                    health = await asyncio.wait_for(
                        agent.health_check(), timeout=self.policy.health_check_timeout
                    )
                    return health.value in ["healthy", "degraded"]
                else:
                    # If no health check method, assume healthy if agent exists
                    return agent is not None

        except asyncio.TimeoutError:
            log_error("Health check timeout", agent=self.agent_name)
            return False
        except Exception as e:
            log_error("Health check failed", agent=self.agent_name, error=str(e))
            return False

    def get_recovery_status(self) -> Dict[str, Any]:
        """Get current recovery status."""
        return {
            "agent_name": self.agent_name,
            "recovery_state": self.recovery_state.value,
            "failure_count": len(self.failure_history),
            "recovery_attempts": len(self.recovery_attempts),
            "last_failure": self.failure_history[-1].timestamp
            if self.failure_history
            else None,
            "last_recovery": self.recovery_attempts[-1].timestamp
            if self.recovery_attempts
            else None,
            "backup_agents": self.backup_agents,
            "recent_failures": [
                {
                    "timestamp": f.timestamp,
                    "category": f.failure_category.value,
                    "error": f.error_message,
                }
                for f in self.failure_history[-5:]  # Last 5 failures
            ],
            "recent_recovery_attempts": [
                {
                    "timestamp": a.timestamp,
                    "strategy": a.strategy.value,
                    "success": a.success,
                    "duration": a.recovery_duration,
                }
                for a in self.recovery_attempts[-3:]  # Last 3 attempts
            ],
        }


class RecoverySystemManager:
    """Manages recovery for all agents in the system."""

    def __init__(self):
        self.recovery_managers: Dict[str, AgentRecoveryManager] = {}
        self.system_policy = RecoveryPolicy()
        self.monitoring_task: Optional[asyncio.Task] = None
        self.failure_detectors: List[Callable] = []

    def register_agent(self, agent_name: str, policy: Optional[RecoveryPolicy] = None):
        """Register an agent for recovery management."""
        if agent_name in self.recovery_managers:
            log_warning("Agent already registered for recovery", agent=agent_name)
            return

        recovery_manager = AgentRecoveryManager(
            agent_name, policy or self.system_policy
        )
        self.recovery_managers[agent_name] = recovery_manager

        log_info("Agent registered for recovery management", agent=agent_name)

    def unregister_agent(self, agent_name: str):
        """Unregister an agent from recovery management."""
        if agent_name in self.recovery_managers:
            del self.recovery_managers[agent_name]
            log_info("Agent unregistered from recovery management", agent=agent_name)

    def set_agent_factory(self, agent_name: str, factory: Callable):
        """Set the factory function for an agent."""
        if agent_name in self.recovery_managers:
            self.recovery_managers[agent_name].set_agent_factory(factory)

    def add_backup_agent(self, primary_name: str, backup_name: str):
        """Add a backup agent relationship."""
        if primary_name in self.recovery_managers:
            self.recovery_managers[primary_name].add_backup_agent(backup_name)

    def add_cleanup_handler(self, agent_name: str, handler: Callable):
        """Add a resource cleanup handler for an agent."""
        if agent_name in self.recovery_managers:
            self.recovery_managers[agent_name].add_cleanup_handler(handler)

    def add_failure_detector(self, detector: Callable):
        """Add a failure detection function."""
        self.failure_detectors.append(detector)

    async def start_monitoring(self):
        """Start system-wide recovery monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            log_warning("Recovery monitoring already running")
            return

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        log_info("Recovery system monitoring started")

    async def stop_monitoring(self):
        """Stop recovery monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
            log_info("Recovery system monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for failure detection."""
        while True:
            try:
                # Run failure detectors
                for detector in self.failure_detectors:
                    try:
                        failures = await detector()
                        if failures:
                            for failure in failures:
                                await self.handle_failure(failure)
                    except Exception as e:
                        log_error("Failure detector error", error=str(e))

                await asyncio.sleep(10.0)  # Check every 10 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Monitoring loop error", error=str(e))
                await asyncio.sleep(10.0)

    async def handle_failure(self, failure: AgentFailure):
        """Handle a detected agent failure."""
        agent_name = failure.agent_name

        if agent_name not in self.recovery_managers:
            log_warning(
                "Failure detected for unregistered agent",
                agent=agent_name,
                error=failure.error_message,
            )
            return

        recovery_manager = self.recovery_managers[agent_name]
        await recovery_manager.handle_agent_failure(failure)

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall recovery system status."""
        total_agents = len(self.recovery_managers)
        recovering_agents = sum(
            1
            for rm in self.recovery_managers.values()
            if rm.recovery_state != RecoveryState.NONE
        )

        failed_agents = sum(
            1 for rm in self.recovery_managers.values() if rm.failure_history
        )

        return {
            "total_agents": total_agents,
            "recovering_agents": recovering_agents,
            "failed_agents": failed_agents,
            "system_health_score": (total_agents - recovering_agents) / total_agents
            if total_agents > 0
            else 1.0,
            "agent_status": {
                name: rm.get_recovery_status()
                for name, rm in self.recovery_managers.items()
            },
        }


# Global recovery system instance
recovery_system = RecoverySystemManager()


def get_recovery_system() -> RecoverySystemManager:
    """Get the global recovery system manager."""
    return recovery_system
