"""Shared enumerations for LeanVibe Agent Hive 2.0."""

from enum import Enum


class AgentStatus(str, Enum):
    """Agent status enumeration - shared across all modules."""

    STARTING = "starting"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"
    INACTIVE = "inactive"  # For compatibility with existing API


class TaskStatus(str, Enum):
    """Task status enumeration."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    """Task priority enumeration."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
