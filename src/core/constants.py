"""
Configuration constants for the LeanVibe Agent Hive system.

This module centralizes all timing, interval, and threshold constants
to improve maintainability and testing.
"""

from typing import Final


class Intervals:
    """Timing intervals for various system components."""

    # Agent monitoring and heartbeat intervals
    AGENT_HEARTBEAT: Final[int] = 30
    AGENT_BRIEF_DELAY: Final[int] = 1
    AGENT_STARTUP_DELAY: Final[int] = 1
    AGENT_SHUTDOWN_GRACE: Final[int] = 1

    # Task coordination intervals
    TASK_COORDINATION_CYCLE: Final[int] = 5
    TASK_COORDINATOR_ERROR_DELAY: Final[int] = 30

    # Agent-specific intervals
    DEVELOPER_AGENT_CYCLE: Final[int] = 1
    DEVELOPER_AGENT_ERROR_DELAY: Final[int] = 5
    ARCHITECT_AGENT_PLANNING_DELAY: Final[int] = 10
    ARCHITECT_AGENT_CYCLE: Final[int] = 15
    QA_AGENT_ERROR_DELAY: Final[int] = 5
    QA_AGENT_CYCLE: Final[int] = 10
    META_AGENT_CYCLE: Final[int] = 10
    META_AGENT_ERROR_DELAY: Final[int] = 30
    DEVOPS_AGENT_MONITORING_CYCLE: Final[int] = 30
    DEVOPS_AGENT_CYCLE: Final[int] = 20
    DEVOPS_AGENT_COMMAND_DELAY: Final[int] = 1
    DEVOPS_AGENT_SCALING_DELAY: Final[int] = 2

    # System monitoring intervals
    SYSTEM_HEALTH_CHECK: Final[int] = 60
    SYSTEM_STARTUP_DELAY: Final[int] = 3
    SYSTEM_RESTART_DELAY: Final[int] = 2

    # Emergency and safety intervals
    EMERGENCY_CHECK_INTERVAL: Final[int] = 10
    EMERGENCY_ERROR_BACKOFF: Final[int] = 30

    # Analytics and monitoring intervals
    ANALYTICS_DEFAULT_INTERVAL: Final[int] = 60
    AUTONOMOUS_DASHBOARD_DEFAULT: Final[int] = 5

    # Agent runner intervals
    RUNNER_TASK_CHECK: Final[int] = 5
    RUNNER_ERROR_DELAY: Final[int] = 30
    RUNNER_SYSTEM_HEALTH_CHECK: Final[int] = 10
    RUNNER_IMPROVEMENT_CHECK: Final[int] = 30
    RUNNER_PROACTIVE_CYCLE: Final[int] = 15
    RUNNER_MONITORING_CYCLE: Final[int] = 30
    RUNNER_MAINTENANCE_CYCLE: Final[int] = 20
    RUNNER_CLEANUP_CYCLE: Final[int] = 30
    RUNNER_BRIEF_DELAY: Final[int] = 1

    # Sleep/wake manager intervals
    SLEEP_WAKE_MONITOR_CYCLE: Final[int] = 60
    SLEEP_WAKE_ERROR_DELAY: Final[int] = 300
    SLEEP_WAKE_DURATION: Final[int] = 300
    SLEEP_WAKE_CHECK_DURING_SLEEP: Final[int] = 30

    # Orchestrator intervals
    ORCHESTRATOR_STARTUP_DELAY: Final[int] = 3
    ORCHESTRATOR_AGENT_DELAY: Final[int] = 2
    ORCHESTRATOR_HEARTBEAT_CHECK: Final[int] = 1
    ORCHESTRATOR_MONITOR_CYCLE: Final[int] = 1
    ORCHESTRATOR_ERROR_DELAY: Final[int] = 5
    ORCHESTRATOR_HEALTH_CHECK: Final[int] = 300
    ORCHESTRATOR_CLEANUP_CYCLE: Final[int] = 60

    # Persistence and caching intervals
    PERSISTENT_CLI_HEARTBEAT: Final[int] = 30
    PERSISTENT_CLI_ERROR_DELAY: Final[int] = 5
    CACHE_RETRY_BACKOFF_BASE: Final[float] = 1.0

    # Resource monitoring intervals
    RESOURCE_GUARDIAN_CHECK: Final[float] = 0.1
    RESOURCE_GUARDIAN_MONITOR_CYCLE: Final[int] = 60

    # ADW system intervals
    ADW_SESSION_TEST_DELAY: Final[int] = 1
    ADW_SESSION_CODE_DELAY: Final[int] = 1
    ADW_MICRO_DEVELOPMENT_DELAYS: Final[float] = 0.1

    # Testing intervals
    EXTENDED_SESSION_MONITOR: Final[int] = 5
    EXTENDED_SESSION_HEALTH_CHECK: Final[int] = 30


class Thresholds:
    """Threshold values for system monitoring and decision making."""

    # Performance thresholds
    MAX_MEMORY_PER_AGENT: Final[int] = 500_000_000  # 500MB in bytes
    MAX_CPU_PERCENTAGE: Final[float] = 80.0
    MAX_RESPONSE_TIME_MS: Final[int] = 100

    # Queue and coordination thresholds
    MAX_QUEUE_DEPTH: Final[int] = 1000
    MAX_CONCURRENT_AGENTS: Final[int] = 50

    # Health check thresholds
    MIN_HEALTH_SCORE: Final[float] = 0.7
    MAX_ERROR_RATE: Final[float] = 0.1

    # Cognitive load thresholds
    COGNITIVE_LOAD_WARNING: Final[float] = 0.7
    COGNITIVE_LOAD_CRITICAL: Final[float] = 0.9


class Timeouts:
    """Timeout values for various operations."""

    # API timeouts
    API_REQUEST_TIMEOUT: Final[int] = 30
    WEBSOCKET_TIMEOUT: Final[int] = 60

    # Database timeouts
    DATABASE_QUERY_TIMEOUT: Final[int] = 30
    DATABASE_CONNECTION_TIMEOUT: Final[int] = 10

    # Agent operation timeouts
    AGENT_SPAWN_TIMEOUT: Final[int] = 60
    AGENT_STOP_TIMEOUT: Final[int] = 30
    CLI_TOOL_TIMEOUT: Final[int] = 300

    # Task execution timeouts
    TASK_EXECUTION_TIMEOUT: Final[int] = 3600  # 1 hour
    EMERGENCY_RESPONSE_TIMEOUT: Final[int] = 10


class RetryLimits:
    """Retry limits for various operations."""

    # Network retries
    API_RETRY_ATTEMPTS: Final[int] = 3
    DATABASE_RETRY_ATTEMPTS: Final[int] = 3

    # Agent operation retries
    AGENT_SPAWN_RETRIES: Final[int] = 3
    CLI_TOOL_RETRIES: Final[int] = 2

    # Cache operation retries
    CACHE_RETRY_ATTEMPTS: Final[int] = 3


class Paths:
    """Path constants for the system."""

    # Log paths
    LOG_DIR: Final[str] = "logs"
    AGENT_LOG_DIR: Final[str] = "logs/agents"
    SYSTEM_LOG_DIR: Final[str] = "logs/system"

    # Data paths
    DATA_DIR: Final[str] = "data"
    BACKUP_DIR: Final[str] = "data/backups"
    CONTEXT_DATA_DIR: Final[str] = "data/context"


# Legacy constants for backwards compatibility
# TODO: These should be gradually phased out in favor of the class-based approach above

# Agent intervals (deprecated - use Intervals class)
AGENT_HEARTBEAT_INTERVAL = Intervals.AGENT_HEARTBEAT
TASK_COORDINATION_CYCLE = Intervals.TASK_COORDINATION_CYCLE

# System intervals (deprecated - use Intervals class)
SYSTEM_HEALTH_CHECK_INTERVAL = Intervals.SYSTEM_HEALTH_CHECK
EMERGENCY_CHECK_INTERVAL = Intervals.EMERGENCY_CHECK_INTERVAL
