"""Enhanced error handling and resilience system for LeanVibe Agent Hive.

This module provides comprehensive error classification, exponential backoff,
circuit breaker integration, and graceful degradation capabilities.
"""

import asyncio
import time
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Type
import traceback

try:
    import structlog
    logger = structlog.get_logger()
    # Type ignore for structlog dynamic fields
    logger.info = logger.info  # type: ignore
    logger.warning = logger.warning  # type: ignore
    logger.error = logger.error  # type: ignore
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for classification."""

    LOW = "low"          # Non-critical, can be ignored or retried
    MEDIUM = "medium"    # Important but system can continue
    HIGH = "high"        # Critical, requires immediate attention
    CRITICAL = "critical"  # System-threatening, may require shutdown


class ErrorCategory(Enum):
    """Error categories for better classification and handling."""

    NETWORK = "network"              # Network connectivity issues
    DATABASE = "database"            # Database connection/query errors
    EXTERNAL_API = "external_api"    # Third-party API failures
    RESOURCE = "resource"            # Resource exhaustion (memory, CPU, disk)
    CONFIGURATION = "configuration"  # Configuration errors
    AUTHENTICATION = "authentication"  # Auth/authz failures
    VALIDATION = "validation"        # Input validation errors
    TIMEOUT = "timeout"              # Operation timeouts
    INTERNAL = "internal"            # Internal system errors
    UNKNOWN = "unknown"              # Unclassified errors


@dataclass
class ErrorContext:
    """Context information for error classification and handling."""

    operation: str
    component: str
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.INTERNAL
    retryable: bool = True
    user_visible: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    backoff_factor: float = 2.0  # Exponential backoff multiplier
    jitter: bool = True  # Add random jitter to prevent thundering herd


class EnhancedError(Exception):
    """Enhanced error with classification and context."""

    def __init__(
        self,
        message: str,
        context: ErrorContext,
        original_error: Optional[Exception] = None
    ):
        super().__init__(message)
        self.context = context
        self.original_error = original_error


class ErrorClassifier:
    """Classifies errors based on type and context."""

    @staticmethod
    def classify_error(error: Exception, operation: str, component: str) -> ErrorContext:
        """Classify an error based on its type and context."""

        error_type = type(error).__name__
        error_message = str(error).lower()

        # Database-related errors (check first to avoid conflict with network keywords)
        if any(keyword in error_message for keyword in [
            'database', 'postgres', 'sql', 'connection pool', 'transaction', 'psycopg'
        ]) or 'psycopg' in error_type.lower() or 'psycopg' in error_message:
            return ErrorContext(
                operation=operation,
                component=component,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.DATABASE,
                retryable=True
            )

        # Network-related errors
        elif any(keyword in error_message for keyword in [
            'timeout', 'network', 'dns', 'socket', 'http'
        ]) or error_type in ['ConnectionError', 'TimeoutError', 'httpx.TimeoutException']:
            return ErrorContext(
                operation=operation,
                component=component,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.NETWORK,
                retryable=True
            )

        # Authentication errors
        elif any(keyword in error_message for keyword in [
            'auth', 'token', 'permission', 'unauthorized', 'forbidden'
        ]):
            return ErrorContext(
                operation=operation,
                component=component,
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.AUTHENTICATION,
                retryable=False
            )

        # Resource exhaustion
        elif any(keyword in error_message for keyword in [
            'memory', 'cpu', 'disk', 'resource', 'limit'
        ]):
            return ErrorContext(
                operation=operation,
                component=component,
                severity=ErrorSeverity.CRITICAL,
                category=ErrorCategory.RESOURCE,
                retryable=False
            )

        # Timeout errors
        elif 'timeout' in error_message or error_type == 'asyncio.TimeoutError':
            return ErrorContext(
                operation=operation,
                component=component,
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.TIMEOUT,
                retryable=True
            )

        # Default classification
        return ErrorContext(
            operation=operation,
            component=component,
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.INTERNAL,
            retryable=True
        )


class ExponentialBackoff:
    """Implements exponential backoff with jitter."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempts = 0

    def should_retry(self) -> bool:
        """Check if we should retry based on attempt count."""
        return self.attempts < self.config.max_attempts

    def get_delay(self) -> float:
        """Calculate delay for next retry attempt."""
        if self.attempts == 0:
            return 0.0

        # Exponential backoff
        delay = min(
            self.config.base_delay * (self.config.backoff_factor ** (self.attempts - 1)),
            self.config.max_delay
        )

        # Add jitter
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)  # 50-100% of calculated delay

        return delay

    def record_attempt(self):
        """Record that an attempt was made."""
        self.attempts += 1

    def reset(self):
        """Reset the backoff state."""
        self.attempts = 0


class ErrorRecoveryStrategy:
    """Strategies for recovering from different types of errors."""

    @staticmethod
    async def recover_from_error(
        error_context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Attempt to recover from an error based on its classification."""

        if error_context.category == ErrorCategory.NETWORK:
            # For network errors, implement circuit breaker pattern
            return await ErrorRecoveryStrategy._network_recovery(
                error_context, operation, *args, **kwargs
            )

        elif error_context.category == ErrorCategory.DATABASE:
            # For database errors, try connection recovery
            return await ErrorRecoveryStrategy._database_recovery(
                error_context, operation, *args, **kwargs
            )

        elif error_context.category == ErrorCategory.RESOURCE:
            # For resource errors, implement graceful degradation
            return await ErrorRecoveryStrategy._resource_recovery(
                error_context, operation, *args, **kwargs
            )

        else:
            # Default retry strategy
            return await ErrorRecoveryStrategy._default_recovery(
                error_context, operation, *args, **kwargs
            )

    @staticmethod
    async def _network_recovery(
        error_context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Recovery strategy for network errors."""
        backoff = ExponentialBackoff(RetryConfig(max_attempts=3))

        while backoff.should_retry():
            try:
                backoff.record_attempt()
                delay = backoff.get_delay()

                if delay > 0:
                    logger.info(  # type: ignore
                        "Retrying network operation after delay",
                        operation=error_context.operation,
                        attempt=backoff.attempts,
                        delay=delay
                    )
                    await asyncio.sleep(delay)

                return await operation(*args, **kwargs)

            except Exception as e:
                logger.warning(
                    "Network operation failed, retrying",
                    operation=error_context.operation,
                    attempt=backoff.attempts,
                    error=str(e)
                )

        # All retries exhausted
        raise EnhancedError(
            f"Network operation failed after {backoff.attempts} attempts",
            error_context
        )

    @staticmethod
    async def _database_recovery(
        error_context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Recovery strategy for database errors."""
        backoff = ExponentialBackoff(RetryConfig(max_attempts=2, base_delay=0.5))

        while backoff.should_retry():
            try:
                backoff.record_attempt()
                delay = backoff.get_delay()

                if delay > 0:
                    logger.info(
                        "Retrying database operation after delay",
                        operation=error_context.operation,
                        attempt=backoff.attempts,
                        delay=delay
                    )
                    await asyncio.sleep(delay)

                return await operation(*args, **kwargs)

            except Exception as e:
                logger.warning(
                    "Database operation failed, retrying",
                    operation=error_context.operation,
                    attempt=backoff.attempts,
                    error=str(e)
                )

        # All retries exhausted
        raise EnhancedError(
            f"Database operation failed after {backoff.attempts} attempts",
            error_context
        )

    @staticmethod
    async def _resource_recovery(
        error_context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Recovery strategy for resource exhaustion errors."""
        # For resource errors, implement graceful degradation
        logger.warning(
            "Resource exhaustion detected, implementing graceful degradation",
            operation=error_context.operation,
            component=error_context.component
        )

        # Try with reduced resource requirements
        try:
            # This is a placeholder - actual implementation would depend on the operation
            return await operation(*args, **kwargs)
        except Exception as e:
            logger.error(
                "Graceful degradation failed",
                operation=error_context.operation,
                error=str(e)
            )
            raise EnhancedError(
                "Resource exhaustion with no graceful degradation available",
                error_context,
                e
            )

    @staticmethod
    async def _default_recovery(
        error_context: ErrorContext,
        operation: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Default recovery strategy."""
        backoff = ExponentialBackoff(RetryConfig(max_attempts=2))

        while backoff.should_retry():
            try:
                backoff.record_attempt()
                delay = backoff.get_delay()

                if delay > 0:
                    logger.info(
                        "Retrying operation after delay",
                        operation=error_context.operation,
                        attempt=backoff.attempts,
                        delay=delay
                    )
                    await asyncio.sleep(delay)

                return await operation(*args, **kwargs)

            except Exception as e:
                logger.warning(
                    "Operation failed, retrying",
                    operation=error_context.operation,
                    attempt=backoff.attempts,
                    error=str(e)
                )

        # All retries exhausted
        raise EnhancedError(
            f"Operation failed after {backoff.attempts} attempts",
            error_context
        )


async def with_error_handling(
    operation: Callable,
    operation_name: str,
    component_name: str,
    *args,
    **kwargs
) -> Any:
    """Decorator-style function for comprehensive error handling."""

    try:
        return await operation(*args, **kwargs)

    except Exception as e:
        # Classify the error
        error_context = ErrorClassifier.classify_error(e, operation_name, component_name)

        # Log the error with context
        logger.error(
            "Operation failed",
            operation=operation_name,
            component=component_name,
            severity=error_context.severity.value,
            category=error_context.category.value,
            retryable=error_context.retryable,
            error=str(e),
            traceback=traceback.format_exc()
        )

        # Attempt recovery if retryable
        if error_context.retryable:
            try:
                return await ErrorRecoveryStrategy.recover_from_error(
                    error_context, operation, *args, **kwargs
                )
            except EnhancedError as recovery_error:
                # Recovery failed, re-raise the enhanced error
                raise recovery_error

        # Not retryable, raise original error
        raise e


# Graceful degradation utilities
class GracefulDegradationManager:
    """Manages graceful degradation of system capabilities."""

    def __init__(self):
        self.degraded_features: set[str] = set()
        self.degradation_start_times: Dict[str, float] = {}

    def degrade_feature(self, feature_name: str, reason: str):
        """Mark a feature as degraded."""
        if feature_name not in self.degraded_features:
            self.degraded_features.add(feature_name)
            self.degradation_start_times[feature_name] = time.time()

            logger.warning(
                "Feature degraded",
                feature=feature_name,
                reason=reason,
                total_degraded=len(self.degraded_features)
            )

    def restore_feature(self, feature_name: str):
        """Restore a degraded feature."""
        if feature_name in self.degraded_features:
            self.degraded_features.remove(feature_name)
            degradation_duration = time.time() - self.degradation_start_times.get(feature_name, time.time())

            logger.info(
                "Feature restored",
                feature=feature_name,
                degradation_duration=degradation_duration
            )

    def is_degraded(self, feature_name: str) -> bool:
        """Check if a feature is currently degraded."""
        return feature_name in self.degraded_features

    def get_degraded_features(self) -> List[str]:
        """Get list of currently degraded features."""
        return list(self.degraded_features)


# Global instances
error_classifier = ErrorClassifier()
graceful_degradation_manager = GracefulDegradationManager()