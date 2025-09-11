"""Circuit Breaker Pattern Implementation for System Resilience.

This module provides circuit breaker functionality to prevent cascade failures
by temporarily stopping calls to failing services and allowing them to recover.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
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


class CircuitBreakerState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, calls pass through
    OPEN = "open"  # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""

    def __init__(self, service_name: str, state: CircuitBreakerState):
        self.service_name = service_name
        self.state = state
        super().__init__(f"Circuit breaker for {service_name} is {state.value}")


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""

    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: float = 60.0  # Seconds to wait before trying half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds
    name: str = "default"
    monitor_failures: bool = True  # Whether to track failure metrics


@dataclass
class CircuitBreakerMetrics:
    """Metrics for circuit breaker performance."""

    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_calls == 0:
            return 0.0
        return self.failed_calls / self.total_calls

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_calls == 0:
            return 0.0
        return self.successful_calls / self.total_calls


class CircuitBreaker:
    """Circuit breaker implementation with configurable behavior."""

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CircuitBreakerMetrics()
        self.last_state_change = time.time()
        self.half_open_success_count = 0

        log_info(
            "Circuit breaker initialized",
            name=config.name,
            failure_threshold=config.failure_threshold,
            recovery_timeout=config.recovery_timeout,
        )

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from open to half-open."""
        if self.state != CircuitBreakerState.OPEN:
            return False

        time_since_last_failure = time.time() - (self.metrics.last_failure_time or 0)
        return time_since_last_failure >= self.config.recovery_timeout

    def _record_success(self):
        """Record a successful call."""
        self.metrics.total_calls += 1
        self.metrics.successful_calls += 1
        self.metrics.consecutive_failures = 0
        self.metrics.consecutive_successes += 1
        self.metrics.last_success_time = time.time()

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_success_count += 1
            if self.half_open_success_count >= self.config.success_threshold:
                self._change_state(CircuitBreakerState.CLOSED)
                self.half_open_success_count = 0

    def _record_failure(self, error: Exception):
        """Record a failed call."""
        self.metrics.total_calls += 1
        self.metrics.failed_calls += 1
        self.metrics.consecutive_failures += 1
        self.metrics.consecutive_successes = 0
        self.metrics.last_failure_time = time.time()

        if self.state == CircuitBreakerState.CLOSED:
            if self.metrics.consecutive_failures >= self.config.failure_threshold:
                self._change_state(CircuitBreakerState.OPEN)
        elif self.state == CircuitBreakerState.HALF_OPEN:
            self._change_state(CircuitBreakerState.OPEN)
            self.half_open_success_count = 0

    def _change_state(self, new_state: CircuitBreakerState):
        """Change circuit breaker state."""
        old_state = self.state
        self.state = new_state
        self.last_state_change = time.time()

        state_change = {
            "timestamp": self.last_state_change,
            "from_state": old_state.value,
            "to_state": new_state.value,
            "consecutive_failures": self.metrics.consecutive_failures,
            "total_calls": self.metrics.total_calls,
        }
        self.metrics.state_changes.append(state_change)

        # Keep only last 100 state changes
        if len(self.metrics.state_changes) > 100:
            self.metrics.state_changes = self.metrics.state_changes[-100:]

        log_info(
            "Circuit breaker state changed",
            name=self.config.name,
            from_state=old_state.value,
            to_state=new_state.value,
            consecutive_failures=self.metrics.consecutive_failures,
        )

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function through the circuit breaker."""
        # Check if we should attempt reset
        if self.state == CircuitBreakerState.OPEN and self._should_attempt_reset():
            self._change_state(CircuitBreakerState.HALF_OPEN)

        # Fail fast if circuit is open
        if self.state == CircuitBreakerState.OPEN:
            raise CircuitBreakerError(self.config.name, self.state)

        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(
                    func(*args, **kwargs), timeout=self.config.timeout
                )
            else:
                # Run sync function in thread pool
                result = await asyncio.get_event_loop().run_in_executor(
                    None, func, *args, **kwargs
                )

            self._record_success()
            return result

        except asyncio.TimeoutError:
            error = asyncio.TimeoutError(f"Call to {self.config.name} timed out")
            self._record_failure(error)
            raise error
        except Exception as e:
            self._record_failure(e)
            raise e

    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout": self.config.recovery_timeout,
            "metrics": {
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "failure_rate": self.metrics.failure_rate,
                "success_rate": self.metrics.success_rate,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
            },
            "last_state_change": self.last_state_change,
            "time_since_last_state_change": time.time() - self.last_state_change,
            "can_attempt_reset": self._should_attempt_reset(),
        }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = CircuitBreakerConfig()

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        if name not in self.breakers:
            if config is None:
                config = CircuitBreakerConfig(name=name)
            self.breakers[name] = CircuitBreaker(config)

        return self.breakers[name]

    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.breakers.get(name)

    def get_all_breakers(self) -> Dict[str, CircuitBreaker]:
        """Get all registered circuit breakers."""
        return self.breakers.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        return {
            name: breaker.get_status() for name, breaker in self.breakers.items()
        }

    def reset_all(self):
        """Reset all circuit breakers to closed state."""
        for breaker in self.breakers.values():
            if breaker.state != CircuitBreakerState.CLOSED:
                breaker._change_state(CircuitBreakerState.CLOSED)
                breaker.metrics.consecutive_failures = 0
                breaker.half_open_success_count = 0

        log_info("All circuit breakers reset to closed state")


# Global registry instance
circuit_breaker_registry = CircuitBreakerRegistry()


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return circuit_breaker_registry


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 3,
    timeout: float = 30.0,
    monitor_failures: bool = True,
):
    """Decorator to apply circuit breaker to a function."""

    def decorator(func: Callable):
        config = CircuitBreakerConfig(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            success_threshold=success_threshold,
            timeout=timeout,
            monitor_failures=monitor_failures,
        )
        breaker = circuit_breaker_registry.get_or_create(name, config)

        async def async_wrapper(*args, **kwargs):
            return await breaker.call(func, *args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            # For sync functions, we need to get the event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is already running, we need to handle differently
                    # This is a simplified approach - in production you might want
                    # to use asyncio.run_in_executor or similar
                    raise RuntimeError(
                        "Cannot use circuit breaker decorator on sync function "
                        "when event loop is already running. Use async version."
                    )
                else:
                    return loop.run_until_complete(breaker.call(func, *args, **kwargs))
            except RuntimeError:
                # No event loop, create one
                return asyncio.run(breaker.call(func, *args, **kwargs))

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Convenience functions for common use cases
def get_message_broker_breaker() -> CircuitBreaker:
    """Get circuit breaker for message broker operations."""
    return circuit_breaker_registry.get_or_create(
        "message_broker",
        CircuitBreakerConfig(
            name="message_broker",
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=10.0,
        ),
    )


def get_database_breaker() -> CircuitBreaker:
    """Get circuit breaker for database operations."""
    return circuit_breaker_registry.get_or_create(
        "database",
        CircuitBreakerConfig(
            name="database",
            failure_threshold=5,
            recovery_timeout=60.0,
            timeout=15.0,
        ),
    )


def get_external_api_breaker() -> CircuitBreaker:
    """Get circuit breaker for external API calls."""
    return circuit_breaker_registry.get_or_create(
        "external_api",
        CircuitBreakerConfig(
            name="external_api",
            failure_threshold=3,
            recovery_timeout=120.0,
            timeout=30.0,
        ),
    )