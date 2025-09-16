# Core interfaces for dependency injection and clean architecture

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol
from dataclasses import dataclass
from enum import Enum


# Domain Models
@dataclass
class AgentInfo:
    """Agent information structure."""
    id: str
    name: str
    type: str
    role: str
    status: str
    capabilities: List[str]
    tmux_session: Optional[str]
    last_heartbeat: float
    created_at: float
    tasks_completed: int
    tasks_failed: int
    current_task_id: Optional[str] = None
    load_factor: float = 0.0


@dataclass
class Task:
    """Task structure."""
    id: str
    title: str
    description: str
    type: str
    priority: int
    status: str
    assigned_to: Optional[str]
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    result: Optional[Dict[str, Any]]


@dataclass
class Message:
    """Message structure."""
    id: str
    from_agent: str
    to_agent: str
    topic: str
    message_type: str
    payload: Dict[str, Any]
    timestamp: float
    expires_at: Optional[float] = None
    reply_to: Optional[str] = None
    correlation_id: Optional[str] = None
    priority: int = 5


@dataclass
class SystemHealth:
    """System health metrics."""
    total_agents: int
    active_agents: int
    idle_agents: int
    busy_agents: int
    error_agents: int
    avg_load_factor: float
    queue_size: int
    tasks_per_minute: float


# Core Service Interfaces
class IAgentOrchestrator(Protocol):
    """Interface for agent orchestration operations."""

    @abstractmethod
    async def spawn_agent(self, agent_type: str, name: Optional[str] = None, capabilities: Optional[List[str]] = None) -> Optional[str]:
        """Spawn a new agent."""
        pass

    @abstractmethod
    async def terminate_agent(self, agent_name: str) -> bool:
        """Terminate an agent."""
        pass

    @abstractmethod
    async def get_agent(self, agent_name: str) -> Optional[AgentInfo]:
        """Get agent information."""
        pass

    @abstractmethod
    async def list_agents(self, status_filter: Optional[str] = None) -> List[AgentInfo]:
        """List agents with optional status filter."""
        pass

    @abstractmethod
    async def get_system_health(self) -> SystemHealth:
        """Get comprehensive system health metrics."""
        pass

    @abstractmethod
    async def start(self) -> None:
        """Start the orchestrator."""
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop the orchestrator."""
        pass


class IMessageBroker(Protocol):
    """Interface for message broker operations."""

    @abstractmethod
    async def send_message(self, from_agent: str, to_agent: str, topic: str, payload: Dict[str, Any],
                          message_type: str = "direct", priority: int = 5,
                          expires_in: Optional[int] = None, correlation_id: Optional[str] = None) -> bool:
        """Send a message to an agent."""
        pass

    @abstractmethod
    async def send_message_with_result(self, from_agent: str, to_agent: str, topic: str,
                                     payload: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Send a message and wait for result."""
        pass

    @abstractmethod
    async def broadcast_message(self, from_agent: str, topic: str, payload: Dict[str, Any],
                               priority: int = 5) -> bool:
        """Broadcast message to all agents."""
        pass

    @abstractmethod
    async def start_listening(self, agent_name: str, handler: Any) -> None:
        """Start listening for messages for an agent."""
        pass

    @abstractmethod
    async def stop_listening(self, agent_name: str) -> None:
        """Stop listening for messages for an agent."""
        pass

    @abstractmethod
    async def get_message_history(self, agent_name: str, limit: int = 100,
                                since: Optional[float] = None) -> List[Message]:
        """Get message history for an agent."""
        pass


class ITaskQueue(Protocol):
    """Interface for task queue operations."""

    @abstractmethod
    async def enqueue_task(self, task: Task) -> str:
        """Add a task to the queue."""
        pass

    @abstractmethod
    async def dequeue_task(self, agent_name: str) -> Optional[Task]:
        """Get next task for an agent."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get specific task by ID."""
        pass

    @abstractmethod
    async def update_task_status(self, task_id: str, status: str,
                               result: Optional[Dict[str, Any]] = None) -> bool:
        """Update task status."""
        pass

    @abstractmethod
    async def fail_task(self, task_id: str, error_message: str) -> bool:
        """Mark task as failed."""
        pass

    @abstractmethod
    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        pass

    @abstractmethod
    async def cleanup_expired_tasks(self) -> int:
        """Clean up expired tasks, return count cleaned."""
        pass


class IDatabaseManager(Protocol):
    """Interface for database operations."""

    @abstractmethod
    async def get_active_agents(self) -> List[Dict[str, Any]]:
        """Get all active agents from database."""
        pass

    @abstractmethod
    async def save_agent(self, agent_info: AgentInfo) -> bool:
        """Save agent to database."""
        pass

    @abstractmethod
    async def update_agent(self, agent_name: str, updates: Dict[str, Any]) -> bool:
        """Update agent in database."""
        pass

    @abstractmethod
    async def delete_agent(self, agent_name: str) -> bool:
        """Delete agent from database."""
        pass

    @abstractmethod
    async def get_agent(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent from database."""
        pass

    @abstractmethod
    async def save_task(self, task: Task) -> bool:
        """Save task to database."""
        pass

    @abstractmethod
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task from database."""
        pass

    @abstractmethod
    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update task in database."""
        pass

    @abstractmethod
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check database health."""
        pass

    @abstractmethod
    async def cleanup_stale_agents(self, threshold_minutes: int = 10) -> Dict[str, int]:
        """Clean up stale agents, return cleanup stats."""
        pass


class IHealthMonitor(Protocol):
    """Interface for health monitoring operations."""

    @abstractmethod
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        pass

    @abstractmethod
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        pass

    @abstractmethod
    async def check_agent_health(self, agent_name: str) -> Dict[str, Any]:
        """Check specific agent health."""
        pass

    @abstractmethod
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        pass

    @abstractmethod
    async def get_health_score(self) -> float:
        """Get overall system health score (0-100)."""
        pass


class IAgentFactory(Protocol):
    """Interface for agent factory operations."""

    @abstractmethod
    def create_agent(self, agent_type: str, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create an agent instance."""
        pass

    @abstractmethod
    def get_available_types(self) -> List[str]:
        """Get list of available agent types."""
        pass

    @abstractmethod
    def get_agent_capabilities(self, agent_type: str) -> List[str]:
        """Get capabilities for agent type."""
        pass

    @abstractmethod
    def validate_config(self, agent_type: str, config: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate agent configuration."""
        pass


class IMetricsCollector(Protocol):
    """Interface for metrics collection operations."""

    @abstractmethod
    async def start_collection(self) -> None:
        """Start metrics collection."""
        pass

    @abstractmethod
    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        pass

    @abstractmethod
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value."""
        pass

    @abstractmethod
    def get_metric_values(self, name: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent metric values."""
        pass

    @abstractmethod
    def get_system_health_score(self) -> float:
        """Calculate system health score."""
        pass

    @abstractmethod
    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics."""
        pass


class ICacheManager(Protocol):
    """Interface for caching operations."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass

    @abstractmethod
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        pass


class IConfigurationManager(Protocol):
    """Interface for configuration management."""

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        pass

    @abstractmethod
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values."""
        pass

    @abstractmethod
    def load_from_file(self, file_path: str) -> bool:
        """Load configuration from file."""
        pass

    @abstractmethod
    def save_to_file(self, file_path: str) -> bool:
        """Save configuration to file."""
        pass


# Service Result Types
@dataclass
class ServiceResult:
    """Standardized service operation result."""
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @classmethod
    def ok(cls, data: Any = None, metadata: Optional[Dict[str, Any]] = None) -> 'ServiceResult':
        """Create successful result."""
        return cls(success=True, data=data, metadata=metadata or {})

    @classmethod
    def error(cls, error_msg: str, metadata: Optional[Dict[str, Any]] = None) -> 'ServiceResult':
        """Create error result."""
        return cls(success=False, error_message=error_msg, metadata=metadata or {})


# Enums
class AgentStatus(Enum):
    """Agent status enumeration."""
    STARTING = "starting"
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    STOPPED = "stopped"
    ERROR = "error"


class TaskStatus(Enum):
    """Task status enumeration."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(Enum):
    """Message type enumeration."""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    MULTICAST = "multicast"
    REQUEST = "request"
    REPLY = "reply"
    NOTIFICATION = "notification"