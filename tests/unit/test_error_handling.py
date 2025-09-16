"""Tests for enhanced error handling and resilience system."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.core.error_handling import (
    ErrorClassifier,
    ErrorContext,
    ErrorCategory,
    ErrorSeverity,
    ExponentialBackoff,
    RetryConfig,
    GracefulDegradationManager,
    EnhancedError,
    with_error_handling,
    error_classifier,
    graceful_degradation_manager,
)


class TestErrorClassifier:
    """Test error classification functionality."""

    def test_classify_network_error(self):
        """Test classification of network-related errors."""
        error = ConnectionError("Connection refused")
        context = ErrorClassifier.classify_error(error, "test_operation", "test_component")

        assert context.category == ErrorCategory.NETWORK
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.retryable is True
        assert context.operation == "test_operation"
        assert context.component == "test_component"

    def test_classify_timeout_error(self):
        """Test classification of timeout errors."""
        error = asyncio.TimeoutError("Operation timed out")
        context = ErrorClassifier.classify_error(error, "test_operation", "test_component")

        # Note: asyncio.TimeoutError gets classified as NETWORK due to "timeout" keyword
        # This is expected behavior based on the current classification logic
        assert context.category == ErrorCategory.NETWORK
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.retryable is True

    def test_classify_database_error(self):
        """Test classification of database errors."""
        error = Exception("psycopg2.OperationalError: connection failed")
        context = ErrorClassifier.classify_error(error, "test_operation", "test_component")

        assert context.category == ErrorCategory.DATABASE
        assert context.severity == ErrorSeverity.HIGH
        assert context.retryable is True

    def test_classify_authentication_error(self):
        """Test classification of authentication errors."""
        error = Exception("401 Unauthorized")
        context = ErrorClassifier.classify_error(error, "test_operation", "test_component")

        assert context.category == ErrorCategory.AUTHENTICATION
        assert context.severity == ErrorSeverity.HIGH
        assert context.retryable is False

    def test_classify_resource_error(self):
        """Test classification of resource exhaustion errors."""
        error = Exception("MemoryError: out of memory")
        context = ErrorClassifier.classify_error(error, "test_operation", "test_component")

        assert context.category == ErrorCategory.RESOURCE
        assert context.severity == ErrorSeverity.CRITICAL
        assert context.retryable is False

    def test_classify_unknown_error(self):
        """Test classification of unknown errors."""
        error = Exception("Some random error")
        context = ErrorClassifier.classify_error(error, "test_operation", "test_component")

        assert context.category == ErrorCategory.INTERNAL
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.retryable is True


class TestExponentialBackoff:
    """Test exponential backoff functionality."""

    def test_initial_state(self):
        """Test initial backoff state."""
        config = RetryConfig(max_attempts=3, base_delay=1.0, backoff_factor=2.0)
        backoff = ExponentialBackoff(config)

        assert backoff.should_retry() is True
        assert backoff.get_delay() == 0.0
        assert backoff.attempts == 0

    def test_backoff_delays(self):
        """Test backoff delay calculation."""
        config = RetryConfig(max_attempts=4, base_delay=1.0, backoff_factor=2.0, jitter=False)
        backoff = ExponentialBackoff(config)

        # First attempt (no delay)
        assert backoff.get_delay() == 0.0
        backoff.record_attempt()

        # Second attempt
        assert backoff.get_delay() == 1.0
        backoff.record_attempt()

        # Third attempt
        assert backoff.get_delay() == 2.0
        backoff.record_attempt()

        # Fourth attempt
        assert backoff.get_delay() == 4.0
        backoff.record_attempt()

        # Should not retry anymore
        assert backoff.should_retry() is False

    def test_max_delay_limit(self):
        """Test that delays are capped at max_delay."""
        config = RetryConfig(max_attempts=10, base_delay=1.0, backoff_factor=2.0, max_delay=5.0, jitter=False)
        backoff = ExponentialBackoff(config)

        # Record many attempts
        for i in range(10):
            backoff.record_attempt()
            delay = backoff.get_delay()
            assert delay <= 5.0

    def test_reset_functionality(self):
        """Test backoff reset functionality."""
        config = RetryConfig(max_attempts=3)
        backoff = ExponentialBackoff(config)

        backoff.record_attempt()
        backoff.record_attempt()
        assert backoff.attempts == 2

        backoff.reset()
        assert backoff.attempts == 0
        assert backoff.should_retry() is True


class TestGracefulDegradationManager:
    """Test graceful degradation functionality."""

    def test_initial_state(self):
        """Test initial degradation manager state."""
        manager = GracefulDegradationManager()

        assert len(manager.degraded_features) == 0
        assert len(manager.degradation_start_times) == 0
        assert manager.is_degraded("test_feature") is False

    def test_degrade_feature(self):
        """Test feature degradation."""
        manager = GracefulDegradationManager()

        manager.degrade_feature("test_feature", "Test degradation")

        assert manager.is_degraded("test_feature") is True
        assert len(manager.degraded_features) == 1
        assert "test_feature" in manager.degraded_features
        assert "test_feature" in manager.degradation_start_times

    def test_restore_feature(self):
        """Test feature restoration."""
        manager = GracefulDegradationManager()

        manager.degrade_feature("test_feature", "Test degradation")
        assert manager.is_degraded("test_feature") is True

        manager.restore_feature("test_feature")
        assert manager.is_degraded("test_feature") is False
        assert len(manager.degraded_features) == 0

    def test_get_degraded_features(self):
        """Test getting list of degraded features."""
        manager = GracefulDegradationManager()

        manager.degrade_feature("feature1", "reason1")
        manager.degrade_feature("feature2", "reason2")

        degraded = manager.get_degraded_features()
        assert len(degraded) == 2
        assert "feature1" in degraded
        assert "feature2" in degraded

    def test_restore_nonexistent_feature(self):
        """Test restoring a feature that was never degraded."""
        manager = GracefulDegradationManager()

        # Should not raise an error
        manager.restore_feature("nonexistent_feature")
        assert len(manager.degraded_features) == 0


class TestEnhancedError:
    """Test enhanced error functionality."""

    def test_enhanced_error_creation(self):
        """Test creating an enhanced error."""
        context = ErrorContext(
            operation="test_op",
            component="test_comp",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.NETWORK
        )
        original_error = ValueError("Original error")

        error = EnhancedError("Enhanced message", context, original_error)

        assert str(error) == "Enhanced message"
        assert error.context == context
        assert error.original_error == original_error


@pytest.mark.asyncio
class TestErrorHandlingDecorator:
    """Test the with_error_handling decorator functionality."""

    async def test_successful_operation(self):
        """Test successful operation with error handling."""
        async def successful_operation():
            return "success"

        result = await with_error_handling(
            successful_operation,
            "test_operation",
            "test_component"
        )

        assert result == "success"

    async def test_retryable_error_recovery(self):
        """Test recovery from retryable errors."""
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Temporary connection error")
            return "recovered"

        result = await with_error_handling(
            failing_operation,
            "test_operation",
            "test_component"
        )

        assert result == "recovered"
        assert call_count == 3  # Should have been called 3 times (initial + 2 retries)

    async def test_non_retryable_error(self):
        """Test non-retryable errors are not retried."""
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise Exception("Authentication failed")  # This should be classified as non-retryable

        with pytest.raises(Exception, match="Authentication failed"):
            await with_error_handling(
                failing_operation,
                "test_operation",
                "test_component"
            )

        assert call_count == 1  # Should only be called once (no retries)

    async def test_max_retries_exhausted(self):
        """Test that max retries are respected."""
        call_count = 0

        async def always_failing_operation():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent connection error")

        with pytest.raises(EnhancedError):
            await with_error_handling(
                always_failing_operation,
                "test_operation",
                "test_component"
            )

        # Should be called max_attempts + 1 times (initial + retries)
        assert call_count == 4  # 1 initial + 3 retries (default max_attempts)


class TestGlobalInstances:
    """Test global instances are properly initialized."""

    def test_error_classifier_instance(self):
        """Test global error classifier instance."""
        assert isinstance(error_classifier, ErrorClassifier)

    def test_graceful_degradation_manager_instance(self):
        """Test global graceful degradation manager instance."""
        assert isinstance(graceful_degradation_manager, GracefulDegradationManager)


class TestErrorContext:
    """Test ErrorContext dataclass."""

    def test_default_values(self):
        """Test ErrorContext default values."""
        context = ErrorContext(operation="test", component="test")

        assert context.operation == "test"
        assert context.component == "test"
        assert context.severity == ErrorSeverity.MEDIUM
        assert context.category == ErrorCategory.INTERNAL
        assert context.retryable is True
        assert context.user_visible is False
        assert isinstance(context.metadata, dict)
        assert isinstance(context.timestamp, float)

    def test_custom_values(self):
        """Test ErrorContext with custom values."""
        metadata = {"key": "value"}
        context = ErrorContext(
            operation="test_op",
            component="test_comp",
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.DATABASE,
            retryable=False,
            user_visible=True,
            metadata=metadata
        )

        assert context.operation == "test_op"
        assert context.component == "test_comp"
        assert context.severity == ErrorSeverity.CRITICAL
        assert context.category == ErrorCategory.DATABASE
        assert context.retryable is False
        assert context.user_visible is True
        assert context.metadata == metadata


class TestRetryConfig:
    """Test RetryConfig dataclass."""

    def test_default_values(self):
        """Test RetryConfig default values."""
        config = RetryConfig()

        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.backoff_factor == 2.0
        assert config.jitter is True

    def test_custom_values(self):
        """Test RetryConfig with custom values."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=30.0,
            backoff_factor=1.5,
            jitter=False
        )

        assert config.max_attempts == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.backoff_factor == 1.5
        assert config.jitter is False