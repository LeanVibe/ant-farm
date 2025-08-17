"""Enhanced BaseAgent with integrated health monitoring, recovery, and lifecycle management.

This module extends the existing BaseAgent with:
- Health monitoring with circuit breakers
- Automatic recovery and failover
- Resource management and cleanup
- Lifecycle state machine
"""

import asyncio
import time
from typing import Any, Dict, Optional

try:
    import structlog

    logger = structlog.get_logger()
    use_structlog = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    use_structlog = False

# Import the new health systems
try:
    from ..core.agent_health import (
        AgentHealthMonitor,
        HealthStatus,
        CircuitBreakerConfig,
        get_health_monitor_registry,
    )
    from ..core.agent_recovery import (
        AgentRecoveryManager,
        RecoveryPolicy,
        AgentFailure,
        FailureCategory,
        get_recovery_system,
    )
    from ..core.resource_management import (
        ResourceManager,
        ResourceQuota,
        ResourceType,
        get_resource_system,
        managed_resource,
    )
    from ..core.agent_lifecycle import (
        AgentLifecycle,
        AgentState,
        LifecycleEvent,
        LifecycleConfig,
        get_lifecycle_system,
    )
except ImportError:
    # Fallback imports for direct execution
    from src.core.agent_health import (
        AgentHealthMonitor,
        HealthStatus,
        CircuitBreakerConfig,
        get_health_monitor_registry,
    )
    from src.core.agent_recovery import (
        AgentRecoveryManager,
        RecoveryPolicy,
        AgentFailure,
        FailureCategory,
        get_recovery_system,
    )
    from src.core.resource_management import (
        ResourceManager,
        ResourceQuota,
        ResourceType,
        get_resource_system,
        managed_resource,
    )
    from src.core.agent_lifecycle import (
        AgentLifecycle,
        AgentState,
        LifecycleEvent,
        LifecycleConfig,
        get_lifecycle_system,
    )

# Import existing base agent
try:
    from .base_agent import BaseAgent, TaskResult, ToolResult
except ImportError:
    from src.agents.base_agent import BaseAgent, TaskResult, ToolResult


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


def log_debug(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.debug(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"{message} {extra_info}" if extra_info else message)


def log_error(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.error(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.error(f"{message} {extra_info}" if extra_info else message)


class EnhancedBaseAgent(BaseAgent):
    """Enhanced base agent with integrated health monitoring and lifecycle management."""

    def __init__(
        self,
        name: str,
        agent_type: str,
        role: str,
        enhanced_communication: bool = True,
        health_config: Optional[Dict[str, Any]] = None,
        recovery_config: Optional[RecoveryPolicy] = None,
        resource_config: Optional[ResourceQuota] = None,
        lifecycle_config: Optional[LifecycleConfig] = None,
    ):
        # Initialize parent class
        super().__init__(name, agent_type, role, enhanced_communication)

        # Initialize health monitoring
        self.health_monitor = AgentHealthMonitor(name)
        self.recovery_manager = AgentRecoveryManager(name, recovery_config)
        self.resource_manager = ResourceManager(name, resource_config)
        self.lifecycle = AgentLifecycle(name, lifecycle_config)

        # Register with global systems
        get_health_monitor_registry().register_monitor(name, self.health_monitor)
        get_recovery_system().register_agent(name, recovery_config)
        get_resource_system().register_agent(name, resource_config)
        get_lifecycle_system().register_agent(name, lifecycle_config)

        # Set up health checks
        self._setup_health_checks()

        # Set up circuit breakers
        self._setup_circuit_breakers()

        # Set up lifecycle handlers
        self._setup_lifecycle_handlers()

        # Set up recovery handlers
        self._setup_recovery_handlers()

        log_info(
            "Enhanced agent initialized",
            agent=self.name,
            type=self.agent_type,
            health_monitoring=True,
            recovery_enabled=True,
            resource_management=True,
            lifecycle_management=True,
        )

    def _setup_health_checks(self):
        """Set up health check functions."""
        self.health_monitor.add_health_check("cli_tools", self._check_cli_tools_health)
        self.health_monitor.add_health_check("database", self._check_database_health)
        self.health_monitor.add_health_check(
            "message_broker", self._check_message_broker_health
        )
        self.health_monitor.add_health_check(
            "task_queue", self._check_task_queue_health
        )
        self.health_monitor.add_health_check("memory_usage", self._check_memory_usage)

    def _setup_circuit_breakers(self):
        """Set up circuit breakers for critical operations."""
        # CLI tool execution circuit breaker
        self.health_monitor.add_circuit_breaker(
            "cli_execution",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=30.0,
                max_response_time=60.0,
            ),
        )

        # Task processing circuit breaker
        self.health_monitor.add_circuit_breaker(
            "task_processing",
            CircuitBreakerConfig(
                failure_threshold=5,
                success_threshold=3,
                timeout=60.0,
                max_response_time=300.0,
            ),
        )

        # Database operations circuit breaker
        self.health_monitor.add_circuit_breaker(
            "database_operations",
            CircuitBreakerConfig(
                failure_threshold=3,
                success_threshold=2,
                timeout=20.0,
                max_response_time=30.0,
            ),
        )

    def _setup_lifecycle_handlers(self):
        """Set up lifecycle event and state handlers."""
        # Event handlers
        self.lifecycle.add_event_handler(LifecycleEvent.START, self._on_start_event)
        self.lifecycle.add_event_handler(LifecycleEvent.STOP, self._on_stop_event)
        self.lifecycle.add_event_handler(
            LifecycleEvent.ERROR_OCCURRED, self._on_error_event
        )
        self.lifecycle.add_event_handler(
            LifecycleEvent.TASK_ASSIGNED, self._on_task_assigned_event
        )
        self.lifecycle.add_event_handler(
            LifecycleEvent.TASK_COMPLETED, self._on_task_completed_event
        )

        # State handlers
        self.lifecycle.add_state_handler(AgentState.ACTIVE, self._on_active_state)
        self.lifecycle.add_state_handler(AgentState.ERROR, self._on_error_state)
        self.lifecycle.add_state_handler(AgentState.STOPPING, self._on_stopping_state)

    def _setup_recovery_handlers(self):
        """Set up recovery and cleanup handlers."""
        self.recovery_manager.set_agent_factory(self._create_agent_instance)
        self.recovery_manager.add_cleanup_handler(self._cleanup_agent_resources)
        self.resource_manager.add_cleanup_handler(self._cleanup_custom_resources)

    async def start(self) -> None:
        """Enhanced start method with lifecycle management."""
        try:
            # Trigger lifecycle start
            await self.lifecycle.trigger_event(LifecycleEvent.START)

            # Start monitoring systems
            await self.health_monitor.start_monitoring()
            await self.resource_manager.start_monitoring()
            await self.lifecycle.start_lifecycle_monitoring()

            # Initialize agent (triggers INITIALIZE event)
            await self.lifecycle.trigger_event(LifecycleEvent.INITIALIZE)

            # Call parent start method
            await super().start()

            # Activate agent (triggers ACTIVATE event)
            await self.lifecycle.trigger_event(LifecycleEvent.ACTIVATE)

        except Exception as e:
            # Report failure to recovery system
            failure = AgentFailure(
                agent_name=self.name,
                timestamp=time.time(),
                failure_category=FailureCategory.CRITICAL,
                error_message=str(e),
                stack_trace=str(e),
            )
            await self.recovery_manager.handle_agent_failure(failure)
            await self.lifecycle.trigger_event(
                LifecycleEvent.ERROR_OCCURRED, {"error": str(e)}
            )
            raise

    async def cleanup(self) -> None:
        """Enhanced cleanup method."""
        try:
            # Trigger stopping lifecycle event
            await self.lifecycle.trigger_event(LifecycleEvent.STOP)

            # Stop monitoring
            await self.health_monitor.stop_monitoring()
            await self.resource_manager.stop_monitoring()
            await self.lifecycle.stop_lifecycle_monitoring()

            # Cleanup resources
            await self.resource_manager.cleanup_all()

            # Call parent cleanup
            await super().cleanup()

            # Trigger shutdown complete
            await self.lifecycle.trigger_event(LifecycleEvent.SHUTDOWN)

        except Exception as e:
            log_error("Cleanup failed", agent=self.name, error=str(e))
            await self.lifecycle.trigger_event(LifecycleEvent.TERMINATE)

    async def execute_with_cli_tool(
        self,
        prompt: str,
        tool_override=None,
        prompt_file: str | None = None,
    ) -> ToolResult:
        """Enhanced CLI tool execution with circuit breaker protection."""
        try:
            # Execute with circuit breaker
            result = await self.health_monitor.execute_with_circuit_breaker(
                "cli_execution",
                super().execute_with_cli_tool,
                prompt,
                tool_override,
                prompt_file,
            )

            # Record metrics
            self.health_monitor.record_operation_success(
                "cli_execution", result.execution_time
            )

            return result

        except Exception as e:
            # Record failure
            self.health_monitor.record_operation_failure("cli_execution", str(e))

            # Check if this is a critical failure requiring recovery
            if self._is_critical_failure(e):
                failure = AgentFailure(
                    agent_name=self.name,
                    timestamp=time.time(),
                    failure_category=self._categorize_failure(e),
                    error_message=str(e),
                )
                await self.recovery_manager.handle_agent_failure(failure)
                await self.lifecycle.trigger_event(
                    LifecycleEvent.ERROR_OCCURRED, {"error": str(e)}
                )

            raise

    async def process_task(self, task) -> TaskResult:
        """Enhanced task processing with lifecycle integration."""
        task_id = str(getattr(task, "id", id(task)))

        try:
            # Add task to lifecycle tracking
            self.lifecycle.add_task(task_id)
            await self.lifecycle.trigger_event(
                LifecycleEvent.TASK_ASSIGNED, {"task_id": task_id}
            )

            # Register task resource
            self.resource_manager.register_resource(
                f"task_{task_id}",
                ResourceType.CUSTOM,
                task,
                metadata={"task_type": getattr(task, "task_type", "unknown")},
            )

            # Process with circuit breaker protection
            result = await self.health_monitor.execute_with_circuit_breaker(
                "task_processing", super().process_task, task
            )

            # Update metrics
            if result.success:
                self.health_monitor.record_operation_success("task_processing", 0)
                self.lifecycle.update_performance_metric("success_rate", 1.0)
                await self.lifecycle.trigger_event(
                    LifecycleEvent.TASK_COMPLETED, {"task_id": task_id}
                )
            else:
                self.health_monitor.record_operation_failure(
                    "task_processing", result.error or "Unknown error"
                )
                self.lifecycle.update_performance_metric("success_rate", 0.0)
                await self.lifecycle.trigger_event(
                    LifecycleEvent.TASK_FAILED, {"task_id": task_id}
                )

            return result

        except Exception as e:
            # Record failure
            self.health_monitor.record_operation_failure("task_processing", str(e))
            await self.lifecycle.trigger_event(
                LifecycleEvent.TASK_FAILED, {"task_id": task_id, "error": str(e)}
            )

            # Check for critical failure
            if self._is_critical_failure(e):
                failure = AgentFailure(
                    agent_name=self.name,
                    timestamp=time.time(),
                    failure_category=self._categorize_failure(e),
                    error_message=str(e),
                )
                await self.recovery_manager.handle_agent_failure(failure)
                await self.lifecycle.trigger_event(
                    LifecycleEvent.ERROR_OCCURRED, {"error": str(e)}
                )

            raise
        finally:
            # Remove task from tracking
            self.lifecycle.remove_task(task_id)
            self.resource_manager.unregister_resource(f"task_{task_id}")

    async def health_check(self) -> HealthStatus:
        """Enhanced health check using the health monitor."""
        return await self.health_monitor.perform_health_check()

    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status including all systems."""
        return {
            "agent_name": self.name,
            "agent_type": self.agent_type,
            "timestamp": time.time(),
            "health": self.health_monitor.get_health_report(),
            "lifecycle": self.lifecycle.get_lifecycle_status(),
            "recovery": self.recovery_manager.get_recovery_status(),
            "resources": self.resource_manager.get_resource_status(),
            "base_status": {
                "status": self.status,
                "uptime": time.time() - self.start_time,
                "tasks_completed": self.tasks_completed,
                "tasks_failed": self.tasks_failed,
                "current_task_id": self.current_task_id,
            },
        }

    # Health check implementations
    async def _check_cli_tools_health(self) -> bool:
        """Check CLI tools health."""
        try:
            return (
                len(self.cli_tools.available_tools) > 0
                and self.cli_tools.preferred_tool is not None
            )
        except Exception:
            return False

    async def _check_database_health(self) -> bool:
        """Check database health."""
        try:
            if self.async_db_manager:
                return await self.async_db_manager.health_check()
            return True  # No database manager means no dependency
        except Exception:
            return False

    async def _check_message_broker_health(self) -> bool:
        """Check message broker health."""
        try:
            # Simple ping test
            return True  # Simplified for now
        except Exception:
            return False

    async def _check_task_queue_health(self) -> bool:
        """Check task queue health."""
        try:
            # Check if task queue is responsive
            return True  # Simplified for now
        except Exception:
            return False

    async def _check_memory_usage(self) -> bool:
        """Check memory usage."""
        try:
            usage = self.resource_manager.current_usage
            quota = self.resource_manager.quota
            if quota.max_memory_mb:
                return usage.memory_mb < quota.max_memory_mb * 0.9  # 90% threshold
            return True
        except Exception:
            return False

    # Lifecycle event handlers
    async def _on_start_event(self, lifecycle, event, metadata):
        """Handle start event."""
        log_info("Agent starting", agent=self.name)

    async def _on_stop_event(self, lifecycle, event, metadata):
        """Handle stop event."""
        log_info("Agent stopping", agent=self.name)

    async def _on_error_event(self, lifecycle, event, metadata):
        """Handle error event."""
        log_error("Agent error event", agent=self.name, metadata=metadata)

    async def _on_task_assigned_event(self, lifecycle, event, metadata):
        """Handle task assigned event."""
        log_debug("Task assigned", agent=self.name, task_id=metadata.get("task_id"))

    async def _on_task_completed_event(self, lifecycle, event, metadata):
        """Handle task completed event."""
        log_debug("Task completed", agent=self.name, task_id=metadata.get("task_id"))

    # Lifecycle state handlers
    async def _on_active_state(self, lifecycle, state):
        """Handle entering active state."""
        log_info("Agent entered active state", agent=self.name)

    async def _on_error_state(self, lifecycle, state):
        """Handle entering error state."""
        log_warning("Agent entered error state", agent=self.name)

    async def _on_stopping_state(self, lifecycle, state):
        """Handle entering stopping state."""
        log_info("Agent entering stopping state", agent=self.name)

    # Recovery and cleanup handlers
    async def _create_agent_instance(self):
        """Create a new agent instance for recovery."""
        # This would create a new instance of the same agent type
        # Implementation depends on the specific agent factory
        return None  # Placeholder

    async def _cleanup_agent_resources(self, agent_name: str):
        """Cleanup agent-specific resources."""
        log_info("Cleaning up agent resources", agent=agent_name)

    async def _cleanup_custom_resources(self, agent_name: str):
        """Cleanup custom resources."""
        log_info("Cleaning up custom resources", agent=agent_name)

    # Utility methods
    def _is_critical_failure(self, exception: Exception) -> bool:
        """Determine if an exception represents a critical failure."""
        critical_exceptions = (MemoryError, SystemError, KeyboardInterrupt, SystemExit)
        return isinstance(exception, critical_exceptions)

    def _categorize_failure(self, exception: Exception) -> FailureCategory:
        """Categorize the type of failure."""
        if isinstance(exception, TimeoutError):
            return FailureCategory.TRANSIENT
        elif isinstance(exception, MemoryError):
            return FailureCategory.RESOURCE
        elif isinstance(exception, (ConnectionError, OSError)):
            return FailureCategory.DEPENDENCY
        elif isinstance(exception, (SystemError, SystemExit)):
            return FailureCategory.CRITICAL
        else:
            return FailureCategory.TRANSIENT
