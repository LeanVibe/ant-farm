"""Configuration constants for LeanVibe Agent Hive 2.0.

This module centralizes all hardcoded values to improve configurability
and maintainability. All timing intervals, thresholds, and other constants
should be defined here rather than scattered throughout the codebase.
"""

from enum import Enum


class Intervals:
    """Time intervals in seconds."""

    # Agent lifecycle intervals
    AGENT_HEARTBEAT = 30
    AGENT_STATUS_CHECK = 5
    AGENT_HEALTH_CHECK = 60
    AGENT_REGISTRATION_RETRY = 10

    # Task coordination intervals
    TASK_COORDINATION_CYCLE = 5
    TASK_QUEUE_POLL = 1
    TASK_CLEANUP = 300
    TASK_TIMEOUT_CHECK = 30

    # Emergency and safety intervals
    EMERGENCY_CHECK = 10
    SAFETY_VIOLATION_CHECK = 30
    INTERVENTION_COOLDOWN = 60

    # System health and monitoring
    SYSTEM_HEALTH_CHECK = 60
    METRICS_COLLECTION = 30
    LOG_ROTATION_CHECK = 3600

    # Database and connection intervals
    DB_CONNECTION_RETRY = 5
    DB_POOL_CLEANUP = 300
    CONNECTION_TIMEOUT = 30

    # Message broker intervals
    MESSAGE_PROCESSING_CYCLE = 1
    MESSAGE_CLEANUP = 300
    SUBSCRIPTION_HEALTH_CHECK = 60

    # CLI and user interaction
    CLI_OPERATION_TIMEOUT = 300
    USER_INPUT_TIMEOUT = 30
    COMMAND_RETRY_DELAY = 2


class Limits:
    """Various system limits and thresholds."""

    # Agent limits
    MAX_CONCURRENT_AGENTS = 50
    MAX_AGENT_RETRIES = 3
    MAX_AGENT_MEMORY_MB = 500

    # Task limits
    MAX_TASK_RETRIES = 3
    MAX_TASK_DURATION = 1800  # 30 minutes
    MAX_QUEUE_SIZE = 1000

    # Message limits
    MAX_MESSAGE_SIZE = 1048576  # 1MB
    MAX_MESSAGE_HISTORY = 1000
    MESSAGE_RETENTION_HOURS = 168  # 1 week

    # Database limits
    DB_POOL_SIZE = 10
    DB_MAX_OVERFLOW = 20
    MAX_QUERY_EXECUTION_TIME = 30

    # File system limits
    MAX_LOG_FILE_SIZE_MB = 100
    MAX_TEMP_FILES = 1000
    MAX_WORKSPACE_SIZE_GB = 10


class Timeouts:
    """Timeout values in seconds."""

    # Network timeouts
    HTTP_REQUEST_TIMEOUT = 30
    WEBSOCKET_PING_TIMEOUT = 10
    REDIS_OPERATION_TIMEOUT = 5

    # Process timeouts
    SUBPROCESS_TIMEOUT = 300
    SHELL_COMMAND_TIMEOUT = 120
    CLI_TOOL_TIMEOUT = 300

    # Lock timeouts
    DISTRIBUTED_LOCK_TIMEOUT = 30
    FILE_LOCK_TIMEOUT = 10

    # Shutdown timeouts
    GRACEFUL_SHUTDOWN_TIMEOUT = 30
    FORCE_SHUTDOWN_TIMEOUT = 10


class Priorities:
    """Priority levels for various operations."""

    # Task priorities (1-9, where 9 is highest)
    CRITICAL_TASK = 9
    HIGH_TASK = 7
    NORMAL_TASK = 5
    LOW_TASK = 3
    BACKGROUND_TASK = 1

    # Agent priorities
    META_AGENT_PRIORITY = 9
    ARCHITECT_AGENT_PRIORITY = 8
    DEVELOPER_AGENT_PRIORITY = 7
    QA_AGENT_PRIORITY = 6
    DEVOPS_AGENT_PRIORITY = 5


class Thresholds:
    """Various threshold values."""

    # Performance thresholds
    CPU_USAGE_WARNING = 80.0  # percentage
    MEMORY_USAGE_WARNING = 85.0  # percentage
    DISK_USAGE_WARNING = 90.0  # percentage

    # Quality thresholds
    CODE_COVERAGE_MINIMUM = 90.0  # percentage
    PERFORMANCE_REGRESSION_THRESHOLD = 0.1  # 10% slower

    # Safety thresholds
    ERROR_RATE_THRESHOLD = 0.05  # 5% error rate
    RESPONSE_TIME_THRESHOLD = 1.0  # 1 second

    # Resource thresholds
    MAX_OPEN_FILES = 1000
    MAX_CONCURRENT_CONNECTIONS = 100


class RetryConfig:
    """Configuration for retry logic."""

    # Exponential backoff
    BASE_DELAY = 1
    MAX_DELAY = 60
    BACKOFF_MULTIPLIER = 2
    JITTER = True

    # Retry counts by operation type
    DATABASE_RETRIES = 3
    NETWORK_RETRIES = 3
    FILE_OPERATION_RETRIES = 3
    CLI_TOOL_RETRIES = 2


class LogLevels:
    """Logging configuration."""

    # Log levels by component
    DEFAULT_LEVEL = "INFO"
    DATABASE_LEVEL = "WARNING"
    REDIS_LEVEL = "WARNING"
    HTTP_LEVEL = "INFO"

    # Sensitive operations
    SECURITY_LEVEL = "WARNING"
    AUTH_LEVEL = "INFO"


class CacheConfig:
    """Cache configuration constants."""

    # TTL values in seconds
    AGENT_STATUS_TTL = 60
    TASK_RESULT_TTL = 3600
    CONTEXT_SEARCH_TTL = 300
    USER_SESSION_TTL = 1800

    # Cache sizes
    MAX_CACHE_ENTRIES = 10000
    CACHE_CLEANUP_THRESHOLD = 0.8  # Clean when 80% full


class FeatureFlags:
    """Feature flag constants."""

    # Self-improvement features
    SELF_MODIFICATION_ENABLED = True
    AUTONOMOUS_REFACTORING = True
    EMERGENCY_INTERVENTION = True

    # Advanced features
    PREDICTIVE_SCALING = False
    ADVANCED_ANALYTICS = True
    DISTRIBUTED_EXECUTION = False

    # Development features
    DEBUG_MODE = False
    VERBOSE_LOGGING = False
    PROFILING_ENABLED = False


# Backward compatibility - these can be imported directly
AGENT_HEARTBEAT_INTERVAL = Intervals.AGENT_HEARTBEAT
TASK_COORDINATION_CYCLE = Intervals.TASK_COORDINATION_CYCLE
EMERGENCY_CHECK_INTERVAL = Intervals.EMERGENCY_CHECK
SYSTEM_HEALTH_CHECK_INTERVAL = Intervals.SYSTEM_HEALTH_CHECK
