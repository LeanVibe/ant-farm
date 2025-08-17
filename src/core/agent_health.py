"""Agent Health Monitoring System with Circuit Breakers.

This module provides comprehensive health monitoring for agents including:
- Health status tracking
- Circuit breaker patterns
- Automatic recovery mechanisms
- Health metrics collection
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import structlog

    logger = structlog.get_logger()
    use_structlog = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    use_structlog = False


def log_info(message: str, **kwargs):
    """Helper function for logging that handles both structlog and standard logging."""
    if use_structlog:
        logger.info(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{message} {extra_info}" if extra_info else message)


def log_warning(message: str, **kwargs):
    """Helper function for logging that handles both structlog and standard logging."""
    if use_structlog:
        logger.warning(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.warning(f"{message} {extra_info}" if extra_info else message)


def log_error(message: str, **kwargs):
    """Helper function for logging that handles both structlog and standard logging."""
    if use_structlog:
        logger.error(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.error(f"{message} {extra_info}" if extra_info else message)


def log_debug(message: str, **kwargs):
    """Helper function for logging that handles both structlog and standard logging."""
    if use_structlog:
        logger.debug(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.debug(f"{message} {extra_info}" if extra_info else message)


class HealthStatus(Enum):
    """Agent health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class HealthMetrics:
    """Health metrics for an agent."""

    success_count: int = 0
    failure_count: int = 0
    last_success_time: Optional[float] = None
    last_failure_time: Optional[float] = None
    response_times: List[float] = field(default_factory=list)
    error_rate: float = 0.0
    avg_response_time: float = 0.0
    uptime: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Number of failures to open circuit
    success_threshold: int = 3  # Number of successes to close circuit
    timeout: float = 60.0  # Timeout before attempting half-open
    recovery_timeout: float = 300.0  # Maximum recovery time
    max_response_time: float = 30.0  # Maximum acceptable response time


class CircuitBreaker:
    """Circuit breaker implementation for agent operations."""

    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self.next_attempt_time: Optional[float] = None

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                log_info("Circuit breaker entering half-open state", name=self.name)
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is open")

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time

            if execution_time > self.config.max_response_time:
                raise TimeoutError(
                    f"Operation exceeded max response time: {execution_time}s"
                )

            await self._on_success()
            return result

        except Exception as e:
            await self._on_failure(e)
            raise

    async def _on_success(self):
        """Handle successful operation."""
        self.failure_count = 0

        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                log_info(
                    "Circuit breaker closed after successful recovery", name=self.name
                )

    async def _on_failure(self, exception: Exception):
        """Handle failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            self._schedule_next_attempt()
            log_warning(
                "Circuit breaker reopened after failure during recovery",
                name=self.name,
                error=str(exception),
            )
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self._schedule_next_attempt()
            log_warning(
                "Circuit breaker opened due to failure threshold",
                name=self.name,
                failure_count=self.failure_count,
                error=str(exception),
            )

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset."""
        if self.next_attempt_time is None:
            return False
        return time.time() >= self.next_attempt_time

    def _schedule_next_attempt(self):
        """Schedule next attempt time."""
        self.next_attempt_time = time.time() + self.config.timeout


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class AgentHealthMonitor:
    """Comprehensive health monitoring for agents."""

    def __init__(self, agent_name: str, check_interval: float = 30.0):
        self.agent_name = agent_name
        self.check_interval = check_interval
        self.metrics = HealthMetrics()
        self.health_status = HealthStatus.UNKNOWN
        self.start_time = time.time()
        self.last_check_time: Optional[float] = None
        self.health_checks: Dict[str, Callable] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_history: List[Dict[str, Any]] = []
        self.max_history_size = 100

    def add_health_check(self, name: str, check_func: Callable[[], bool]):
        """Add a health check function."""
        self.health_checks[name] = check_func
        log_debug("Health check added", agent=self.agent_name, check_name=name)

    def add_circuit_breaker(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ):
        """Add a circuit breaker for a specific operation."""
        if config is None:
            config = CircuitBreakerConfig()

        self.circuit_breakers[name] = CircuitBreaker(
            f"{self.agent_name}:{name}", config
        )
        log_debug("Circuit breaker added", agent=self.agent_name, breaker_name=name)

    async def execute_with_circuit_breaker(
        self, breaker_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Execute function with named circuit breaker protection."""
        if breaker_name not in self.circuit_breakers:
            raise ValueError(f"Circuit breaker '{breaker_name}' not found")

        circuit_breaker = self.circuit_breakers[breaker_name]
        return await circuit_breaker.call(func, *args, **kwargs)

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            log_warning("Health monitoring already running", agent=self.agent_name)
            return

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        log_info(
            "Health monitoring started",
            agent=self.agent_name,
            check_interval=self.check_interval,
        )

    async def stop_monitoring(self):
        """Stop health monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        log_info("Health monitoring stopped", agent=self.agent_name)

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self.perform_health_check()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(
                    "Health monitoring error", agent=self.agent_name, error=str(e)
                )
                await asyncio.sleep(self.check_interval)

    async def perform_health_check(self) -> HealthStatus:
        """Perform comprehensive health check."""
        start_time = time.time()
        check_results = {}

        # Run all registered health checks
        for check_name, check_func in self.health_checks.items():
            try:
                result = await self._run_health_check(check_func)
                check_results[check_name] = result
            except Exception as e:
                check_results[check_name] = False
                log_warning(
                    "Health check failed",
                    agent=self.agent_name,
                    check_name=check_name,
                    error=str(e),
                )

        # Update metrics
        self.last_check_time = time.time()
        check_duration = self.last_check_time - start_time
        self.metrics.response_times.append(check_duration)

        # Keep only recent response times
        if len(self.metrics.response_times) > 50:
            self.metrics.response_times = self.metrics.response_times[-50:]

        self.metrics.avg_response_time = sum(self.metrics.response_times) / len(
            self.metrics.response_times
        )
        self.metrics.uptime = self.last_check_time - self.start_time

        # Determine overall health status
        self.health_status = self._calculate_health_status(check_results)

        # Record health history
        self._record_health_history(check_results, check_duration)

        log_debug(
            "Health check completed",
            agent=self.agent_name,
            status=self.health_status.value,
            duration=check_duration,
            checks_passed=sum(check_results.values()),
            total_checks=len(check_results),
        )

        return self.health_status

    async def _run_health_check(self, check_func: Callable) -> bool:
        """Run a single health check with timeout."""
        try:
            if asyncio.iscoroutinefunction(check_func):
                result = await asyncio.wait_for(check_func(), timeout=10.0)
            else:
                result = check_func()
            return bool(result)
        except asyncio.TimeoutError:
            return False
        except Exception:
            return False

    def _calculate_health_status(self, check_results: Dict[str, bool]) -> HealthStatus:
        """Calculate overall health status based on check results."""
        if not check_results:
            return HealthStatus.UNKNOWN

        total_checks = len(check_results)
        passed_checks = sum(check_results.values())
        pass_rate = passed_checks / total_checks

        # Check circuit breaker states
        critical_circuits_open = sum(
            1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN
        )

        if critical_circuits_open > 0:
            return HealthStatus.CRITICAL

        if pass_rate >= 0.9:
            return HealthStatus.HEALTHY
        elif pass_rate >= 0.7:
            return HealthStatus.DEGRADED
        elif pass_rate >= 0.5:
            return HealthStatus.UNHEALTHY
        else:
            return HealthStatus.CRITICAL

    def _record_health_history(self, check_results: Dict[str, bool], duration: float):
        """Record health check in history."""
        history_entry = {
            "timestamp": self.last_check_time,
            "status": self.health_status.value,
            "check_results": check_results.copy(),
            "duration": duration,
            "metrics": {
                "success_count": self.metrics.success_count,
                "failure_count": self.metrics.failure_count,
                "error_rate": self.metrics.error_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "uptime": self.metrics.uptime,
            },
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count,
                }
                for name, cb in self.circuit_breakers.items()
            },
        }

        self.health_history.append(history_entry)

        # Maintain history size limit
        if len(self.health_history) > self.max_history_size:
            self.health_history = self.health_history[-self.max_history_size :]

    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        return {
            "agent_name": self.agent_name,
            "current_status": self.health_status.value,
            "last_check": self.last_check_time,
            "uptime": self.metrics.uptime,
            "metrics": {
                "success_count": self.metrics.success_count,
                "failure_count": self.metrics.failure_count,
                "error_rate": self.metrics.error_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "memory_usage": self.metrics.memory_usage,
                "cpu_usage": self.metrics.cpu_usage,
            },
            "health_checks": list(self.health_checks.keys()),
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count,
                    "last_failure": cb.last_failure_time,
                }
                for name, cb in self.circuit_breakers.items()
            },
            "recent_history": self.health_history[-10:] if self.health_history else [],
        }

    def record_operation_success(self, operation_name: str, duration: float):
        """Record successful operation."""
        self.metrics.success_count += 1
        self.metrics.last_success_time = time.time()
        self.metrics.response_times.append(duration)
        self._update_error_rate()

    def record_operation_failure(self, operation_name: str, error: str):
        """Record failed operation."""
        self.metrics.failure_count += 1
        self.metrics.last_failure_time = time.time()
        self._update_error_rate()

    def _update_error_rate(self):
        """Update error rate calculation."""
        total_operations = self.metrics.success_count + self.metrics.failure_count
        if total_operations > 0:
            self.metrics.error_rate = self.metrics.failure_count / total_operations
        else:
            self.metrics.error_rate = 0.0


class HealthMonitorRegistry:
    """Registry for managing health monitors across agents."""

    def __init__(self):
        self.monitors: Dict[str, AgentHealthMonitor] = {}
        self._lock = asyncio.Lock()

    async def register_monitor(self, agent_name: str, monitor: AgentHealthMonitor):
        """Register a health monitor for an agent."""
        async with self._lock:
            self.monitors[agent_name] = monitor
            await monitor.start_monitoring()
            log_info("Health monitor registered and started", agent=agent_name)

    async def unregister_monitor(self, agent_name: str):
        """Unregister and stop a health monitor."""
        async with self._lock:
            if agent_name in self.monitors:
                monitor = self.monitors[agent_name]
                await monitor.stop_monitoring()
                del self.monitors[agent_name]
                log_info("Health monitor unregistered and stopped", agent=agent_name)

    def get_monitor(self, agent_name: str) -> Optional[AgentHealthMonitor]:
        """Get health monitor for an agent."""
        return self.monitors.get(agent_name)

    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get health report for all monitored agents."""
        reports = {}

        for agent_name, monitor in self.monitors.items():
            reports[agent_name] = monitor.get_health_report()

        # Calculate system-wide metrics
        total_agents = len(self.monitors)
        healthy_agents = sum(
            1
            for report in reports.values()
            if report["current_status"] == HealthStatus.HEALTHY.value
        )

        degraded_agents = sum(
            1
            for report in reports.values()
            if report["current_status"] == HealthStatus.DEGRADED.value
        )

        unhealthy_agents = sum(
            1
            for report in reports.values()
            if report["current_status"]
            in [HealthStatus.UNHEALTHY.value, HealthStatus.CRITICAL.value]
        )

        return {
            "system_health": {
                "total_agents": total_agents,
                "healthy_agents": healthy_agents,
                "degraded_agents": degraded_agents,
                "unhealthy_agents": unhealthy_agents,
                "system_health_score": healthy_agents / total_agents
                if total_agents > 0
                else 0.0,
            },
            "agent_reports": reports,
            "timestamp": time.time(),
        }

    async def perform_system_health_check(self) -> Dict[str, HealthStatus]:
        """Perform health check on all registered agents."""
        results = {}

        tasks = []
        for agent_name, monitor in self.monitors.items():
            task = asyncio.create_task(monitor.perform_health_check())
            tasks.append((agent_name, task))

        for agent_name, task in tasks:
            try:
                status = await task
                results[agent_name] = status
            except Exception as e:
                log_error(
                    "Health check failed for agent", agent=agent_name, error=str(e)
                )
                results[agent_name] = HealthStatus.UNKNOWN

        return results


# Global registry instance
health_monitor_registry = HealthMonitorRegistry()


def get_health_monitor_registry() -> HealthMonitorRegistry:
    """Get the global health monitor registry."""
    return health_monitor_registry
