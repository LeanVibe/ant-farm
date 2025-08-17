"""Agent Resource Management and Cleanup System.

This module provides comprehensive resource management for agents including:
- Resource tracking and allocation
- Automatic cleanup on agent shutdown
- Resource quotas and limits
- Memory and CPU monitoring
- Cleanup coordination across agents
"""

import asyncio
import gc
import os
import psutil
import time
import weakref
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
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


class ResourceType(Enum):
    """Types of resources that can be managed."""

    FILE_HANDLE = "file_handle"
    MEMORY = "memory"
    NETWORK_CONNECTION = "network_connection"
    DATABASE_CONNECTION = "database_connection"
    THREAD = "thread"
    PROCESS = "process"
    TEMPORARY_FILE = "temporary_file"
    CACHE_ENTRY = "cache_entry"
    LOCK = "lock"
    CUSTOM = "custom"


class ResourceState(Enum):
    """States of a managed resource."""

    ALLOCATED = "allocated"
    ACTIVE = "active"
    IDLE = "idle"
    RELEASING = "releasing"
    RELEASED = "released"
    ERROR = "error"


@dataclass
class ResourceQuota:
    """Resource quota configuration."""

    max_memory_mb: Optional[int] = None
    max_file_handles: Optional[int] = None
    max_network_connections: Optional[int] = None
    max_database_connections: Optional[int] = None
    max_threads: Optional[int] = None
    max_processes: Optional[int] = None
    max_temporary_files: Optional[int] = None
    max_cache_size_mb: Optional[int] = None


@dataclass
class ResourceUsage:
    """Current resource usage."""

    memory_mb: float = 0.0
    cpu_percent: float = 0.0
    file_handles: int = 0
    network_connections: int = 0
    database_connections: int = 0
    threads: int = 0
    processes: int = 0
    temporary_files: int = 0
    cache_size_mb: float = 0.0

    def exceeds_quota(self, quota: ResourceQuota) -> List[str]:
        """Check which quotas are exceeded."""
        violations = []

        if quota.max_memory_mb and self.memory_mb > quota.max_memory_mb:
            violations.append(
                f"memory: {self.memory_mb:.1f}MB > {quota.max_memory_mb}MB"
            )
        if quota.max_file_handles and self.file_handles > quota.max_file_handles:
            violations.append(
                f"file_handles: {self.file_handles} > {quota.max_file_handles}"
            )
        if (
            quota.max_network_connections
            and self.network_connections > quota.max_network_connections
        ):
            violations.append(
                f"network_connections: {self.network_connections} > {quota.max_network_connections}"
            )
        if (
            quota.max_database_connections
            and self.database_connections > quota.max_database_connections
        ):
            violations.append(
                f"database_connections: {self.database_connections} > {quota.max_database_connections}"
            )
        if quota.max_threads and self.threads > quota.max_threads:
            violations.append(f"threads: {self.threads} > {quota.max_threads}")
        if quota.max_processes and self.processes > quota.max_processes:
            violations.append(f"processes: {self.processes} > {quota.max_processes}")
        if (
            quota.max_temporary_files
            and self.temporary_files > quota.max_temporary_files
        ):
            violations.append(
                f"temporary_files: {self.temporary_files} > {quota.max_temporary_files}"
            )
        if quota.max_cache_size_mb and self.cache_size_mb > quota.max_cache_size_mb:
            violations.append(
                f"cache_size: {self.cache_size_mb:.1f}MB > {quota.max_cache_size_mb}MB"
            )

        return violations


class ManagedResource:
    """Represents a managed resource with automatic cleanup."""

    def __init__(
        self,
        resource_id: str,
        resource_type: ResourceType,
        resource_obj: Any,
        cleanup_func: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.resource_obj = resource_obj
        self.cleanup_func = cleanup_func
        self.metadata = metadata or {}
        self.state = ResourceState.ALLOCATED
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.size_bytes = 0

        # Try to estimate resource size
        self._estimate_size()

    def _estimate_size(self):
        """Estimate the size of the resource."""
        try:
            if hasattr(self.resource_obj, "__sizeof__"):
                self.size_bytes = self.resource_obj.__sizeof__()
            elif isinstance(self.resource_obj, (str, bytes)):
                self.size_bytes = len(self.resource_obj)
            elif isinstance(self.resource_obj, (list, dict, set, tuple)):
                import sys

                self.size_bytes = sys.getsizeof(self.resource_obj)
        except Exception:
            self.size_bytes = 0

    def access(self):
        """Mark resource as accessed."""
        self.last_accessed = time.time()
        self.access_count += 1
        if self.state == ResourceState.IDLE:
            self.state = ResourceState.ACTIVE

    def mark_idle(self):
        """Mark resource as idle."""
        if self.state == ResourceState.ACTIVE:
            self.state = ResourceState.IDLE

    async def cleanup(self) -> bool:
        """Cleanup the resource."""
        if self.state == ResourceState.RELEASED:
            return True

        self.state = ResourceState.RELEASING

        try:
            if self.cleanup_func:
                if asyncio.iscoroutinefunction(self.cleanup_func):
                    await self.cleanup_func(self.resource_obj)
                else:
                    self.cleanup_func(self.resource_obj)
            else:
                # Default cleanup strategies
                await self._default_cleanup()

            self.state = ResourceState.RELEASED
            return True

        except Exception as e:
            self.state = ResourceState.ERROR
            log_error(
                "Resource cleanup failed",
                resource_id=self.resource_id,
                resource_type=self.resource_type.value,
                error=str(e),
            )
            return False

    async def _default_cleanup(self):
        """Default cleanup strategies for common resource types."""
        try:
            # File handles
            if hasattr(self.resource_obj, "close"):
                self.resource_obj.close()

            # Network connections
            elif hasattr(self.resource_obj, "disconnect"):
                await self.resource_obj.disconnect()
            elif hasattr(self.resource_obj, "shutdown"):
                self.resource_obj.shutdown()

            # Database connections
            elif hasattr(self.resource_obj, "close"):
                self.resource_obj.close()

            # Processes
            elif hasattr(self.resource_obj, "terminate"):
                self.resource_obj.terminate()
                try:
                    self.resource_obj.wait(timeout=5)
                except:
                    self.resource_obj.kill()

            # Temporary files
            elif self.resource_type == ResourceType.TEMPORARY_FILE:
                if isinstance(self.resource_obj, str) and os.path.exists(
                    self.resource_obj
                ):
                    os.unlink(self.resource_obj)

        except Exception as e:
            log_warning(
                "Default cleanup failed", resource_id=self.resource_id, error=str(e)
            )


class ResourceManager:
    """Manages resources for a single agent."""

    def __init__(self, agent_name: str, quota: Optional[ResourceQuota] = None):
        self.agent_name = agent_name
        self.quota = quota or ResourceQuota()
        self.resources: Dict[str, ManagedResource] = {}
        self.resource_types: Dict[ResourceType, Set[str]] = {}
        self.cleanup_handlers: List[Callable] = []
        self.monitoring_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.current_usage = ResourceUsage()
        self.usage_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        self.monitor_interval = 30.0  # seconds
        self.cleanup_interval = 300.0  # 5 minutes
        self.idle_timeout = 600.0  # 10 minutes

        # Initialize resource type tracking
        for resource_type in ResourceType:
            self.resource_types[resource_type] = set()

    def register_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        resource_obj: Any,
        cleanup_func: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ManagedResource:
        """Register a resource for management."""
        if resource_id in self.resources:
            log_warning(
                "Resource already registered",
                agent=self.agent_name,
                resource_id=resource_id,
            )
            return self.resources[resource_id]

        managed_resource = ManagedResource(
            resource_id, resource_type, resource_obj, cleanup_func, metadata
        )

        self.resources[resource_id] = managed_resource
        self.resource_types[resource_type].add(resource_id)

        log_debug(
            "Resource registered",
            agent=self.agent_name,
            resource_id=resource_id,
            resource_type=resource_type.value,
        )

        return managed_resource

    def unregister_resource(self, resource_id: str) -> bool:
        """Unregister a resource."""
        if resource_id not in self.resources:
            return False

        resource = self.resources[resource_id]
        self.resource_types[resource.resource_type].discard(resource_id)
        del self.resources[resource_id]

        log_debug(
            "Resource unregistered", agent=self.agent_name, resource_id=resource_id
        )

        return True

    def get_resource(self, resource_id: str) -> Optional[ManagedResource]:
        """Get a managed resource."""
        resource = self.resources.get(resource_id)
        if resource:
            resource.access()
        return resource

    def add_cleanup_handler(self, handler: Callable):
        """Add a cleanup handler function."""
        self.cleanup_handlers.append(handler)

    async def start_monitoring(self):
        """Start resource monitoring."""
        if self.monitoring_task and not self.monitoring_task.done():
            log_warning("Resource monitoring already running", agent=self.agent_name)
            return

        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())

        log_info("Resource monitoring started", agent=self.agent_name)

    async def stop_monitoring(self):
        """Stop resource monitoring."""
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        log_info("Resource monitoring stopped", agent=self.agent_name)

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                await self.update_usage()
                await self.check_quotas()
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error(
                    "Resource monitoring error", agent=self.agent_name, error=str(e)
                )
                await asyncio.sleep(self.monitor_interval)

    async def _cleanup_loop(self):
        """Automatic cleanup loop."""
        while True:
            try:
                await self.cleanup_idle_resources()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Resource cleanup error", agent=self.agent_name, error=str(e))
                await asyncio.sleep(self.cleanup_interval)

    async def update_usage(self):
        """Update current resource usage."""
        try:
            # Get process info
            process = psutil.Process()

            # Memory usage
            memory_info = process.memory_info()
            self.current_usage.memory_mb = memory_info.rss / 1024 / 1024

            # CPU usage
            self.current_usage.cpu_percent = process.cpu_percent()

            # File handles
            try:
                self.current_usage.file_handles = len(process.open_files())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.current_usage.file_handles = 0

            # Network connections
            try:
                self.current_usage.network_connections = len(process.connections())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                self.current_usage.network_connections = 0

            # Count managed resources
            self.current_usage.database_connections = len(
                self.resource_types[ResourceType.DATABASE_CONNECTION]
            )
            self.current_usage.threads = len(self.resource_types[ResourceType.THREAD])
            self.current_usage.processes = len(
                self.resource_types[ResourceType.PROCESS]
            )
            self.current_usage.temporary_files = len(
                self.resource_types[ResourceType.TEMPORARY_FILE]
            )

            # Cache size
            cache_size = sum(
                self.resources[res_id].size_bytes
                for res_id in self.resource_types[ResourceType.CACHE_ENTRY]
            )
            self.current_usage.cache_size_mb = cache_size / 1024 / 1024

            # Record usage history
            self._record_usage_history()

        except Exception as e:
            log_error(
                "Failed to update resource usage", agent=self.agent_name, error=str(e)
            )

    async def check_quotas(self):
        """Check for quota violations."""
        violations = self.current_usage.exceeds_quota(self.quota)

        if violations:
            log_warning(
                "Resource quota violations detected",
                agent=self.agent_name,
                violations=violations,
            )

            # Trigger cleanup to try to resolve violations
            await self.cleanup_idle_resources()
            await self.force_cleanup_if_needed()

    async def cleanup_idle_resources(self):
        """Cleanup idle resources."""
        current_time = time.time()
        idle_resources = []

        for resource in self.resources.values():
            if (
                resource.state == ResourceState.IDLE
                and current_time - resource.last_accessed > self.idle_timeout
            ):
                idle_resources.append(resource)

        if idle_resources:
            log_info(
                "Cleaning up idle resources",
                agent=self.agent_name,
                count=len(idle_resources),
            )

            for resource in idle_resources:
                await self._cleanup_resource(resource)

    async def force_cleanup_if_needed(self):
        """Force cleanup if quotas are severely exceeded."""
        violations = self.current_usage.exceeds_quota(self.quota)

        if not violations:
            return

        # Sort resources by last access time (oldest first)
        resources_by_age = sorted(
            self.resources.values(), key=lambda r: r.last_accessed
        )

        # Cleanup oldest resources until quotas are met
        for resource in resources_by_age:
            if resource.state not in [ResourceState.RELEASING, ResourceState.RELEASED]:
                await self._cleanup_resource(resource)

                # Re-check quotas after each cleanup
                await self.update_usage()
                violations = self.current_usage.exceeds_quota(self.quota)
                if not violations:
                    break

    async def _cleanup_resource(self, resource: ManagedResource):
        """Cleanup a single resource."""
        try:
            success = await resource.cleanup()
            if success:
                self.unregister_resource(resource.resource_id)
                log_debug(
                    "Resource cleaned up",
                    agent=self.agent_name,
                    resource_id=resource.resource_id,
                )
        except Exception as e:
            log_error(
                "Resource cleanup failed",
                agent=self.agent_name,
                resource_id=resource.resource_id,
                error=str(e),
            )

    def _record_usage_history(self):
        """Record current usage in history."""
        history_entry = {
            "timestamp": time.time(),
            "memory_mb": self.current_usage.memory_mb,
            "cpu_percent": self.current_usage.cpu_percent,
            "file_handles": self.current_usage.file_handles,
            "network_connections": self.current_usage.network_connections,
            "database_connections": self.current_usage.database_connections,
            "threads": self.current_usage.threads,
            "processes": self.current_usage.processes,
            "temporary_files": self.current_usage.temporary_files,
            "cache_size_mb": self.current_usage.cache_size_mb,
            "total_resources": len(self.resources),
        }

        self.usage_history.append(history_entry)

        # Maintain history size limit
        if len(self.usage_history) > self.max_history_size:
            self.usage_history = self.usage_history[-self.max_history_size :]

    async def cleanup_all(self):
        """Cleanup all resources."""
        log_info(
            "Starting cleanup of all resources",
            agent=self.agent_name,
            resource_count=len(self.resources),
        )

        # Stop monitoring
        await self.stop_monitoring()

        # Run custom cleanup handlers first
        for handler in self.cleanup_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(self.agent_name)
                else:
                    handler(self.agent_name)
            except Exception as e:
                log_error("Cleanup handler failed", agent=self.agent_name, error=str(e))

        # Cleanup all managed resources
        cleanup_tasks = []
        for resource in list(self.resources.values()):
            cleanup_tasks.append(self._cleanup_resource(resource))

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        # Force garbage collection
        gc.collect()

        log_info(
            "Resource cleanup completed",
            agent=self.agent_name,
            remaining_resources=len(self.resources),
        )

    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        return {
            "agent_name": self.agent_name,
            "current_usage": {
                "memory_mb": self.current_usage.memory_mb,
                "cpu_percent": self.current_usage.cpu_percent,
                "file_handles": self.current_usage.file_handles,
                "network_connections": self.current_usage.network_connections,
                "database_connections": self.current_usage.database_connections,
                "threads": self.current_usage.threads,
                "processes": self.current_usage.processes,
                "temporary_files": self.current_usage.temporary_files,
                "cache_size_mb": self.current_usage.cache_size_mb,
            },
            "quota": {
                "max_memory_mb": self.quota.max_memory_mb,
                "max_file_handles": self.quota.max_file_handles,
                "max_network_connections": self.quota.max_network_connections,
                "max_database_connections": self.quota.max_database_connections,
                "max_threads": self.quota.max_threads,
                "max_processes": self.quota.max_processes,
                "max_temporary_files": self.quota.max_temporary_files,
                "max_cache_size_mb": self.quota.max_cache_size_mb,
            },
            "quota_violations": self.current_usage.exceeds_quota(self.quota),
            "resource_count": len(self.resources),
            "resources_by_type": {
                resource_type.value: len(resource_ids)
                for resource_type, resource_ids in self.resource_types.items()
            },
            "recent_usage": self.usage_history[-10:] if self.usage_history else [],
        }


class ResourceSystemManager:
    """Manages resources across all agents in the system."""

    def __init__(self):
        self.resource_managers: Dict[str, ResourceManager] = {}
        self.system_quota = ResourceQuota(
            max_memory_mb=1024,  # 1GB default
            max_file_handles=1000,
            max_network_connections=100,
            max_database_connections=20,
            max_threads=50,
            max_processes=10,
            max_temporary_files=100,
            max_cache_size_mb=256,
        )

    def register_agent(self, agent_name: str, quota: Optional[ResourceQuota] = None):
        """Register an agent for resource management."""
        if agent_name in self.resource_managers:
            log_warning(
                "Agent already registered for resource management", agent=agent_name
            )
            return

        resource_manager = ResourceManager(agent_name, quota or self.system_quota)
        self.resource_managers[agent_name] = resource_manager

        log_info("Agent registered for resource management", agent=agent_name)

    def unregister_agent(self, agent_name: str):
        """Unregister an agent from resource management."""
        if agent_name in self.resource_managers:
            del self.resource_managers[agent_name]
            log_info("Agent unregistered from resource management", agent=agent_name)

    def get_resource_manager(self, agent_name: str) -> Optional[ResourceManager]:
        """Get resource manager for an agent."""
        return self.resource_managers.get(agent_name)

    async def start_all_monitoring(self):
        """Start monitoring for all agents."""
        for resource_manager in self.resource_managers.values():
            await resource_manager.start_monitoring()
        log_info("Resource monitoring started for all agents")

    async def stop_all_monitoring(self):
        """Stop monitoring for all agents."""
        for resource_manager in self.resource_managers.values():
            await resource_manager.stop_monitoring()
        log_info("Resource monitoring stopped for all agents")

    async def cleanup_all_agents(self):
        """Cleanup resources for all agents."""
        cleanup_tasks = []
        for resource_manager in self.resource_managers.values():
            cleanup_tasks.append(resource_manager.cleanup_all())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        log_info("System-wide resource cleanup completed")

    def get_system_status(self) -> Dict[str, Any]:
        """Get system-wide resource status."""
        total_memory = 0.0
        total_cpu = 0.0
        total_resources = 0

        agent_statuses = {}
        for agent_name, resource_manager in self.resource_managers.items():
            status = resource_manager.get_resource_status()
            agent_statuses[agent_name] = status

            total_memory += status["current_usage"]["memory_mb"]
            total_cpu += status["current_usage"]["cpu_percent"]
            total_resources += status["resource_count"]

        return {
            "total_agents": len(self.resource_managers),
            "system_usage": {
                "total_memory_mb": total_memory,
                "avg_cpu_percent": total_cpu / len(self.resource_managers)
                if self.resource_managers
                else 0.0,
                "total_resources": total_resources,
            },
            "agents": agent_statuses,
        }


# Global resource system instance
resource_system = ResourceSystemManager()


def get_resource_system() -> ResourceSystemManager:
    """Get the global resource system manager."""
    return resource_system


# Context manager for automatic resource management
class managed_resource:
    """Context manager for automatic resource cleanup."""

    def __init__(
        self,
        agent_name: str,
        resource_id: str,
        resource_type: ResourceType,
        cleanup_func: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.agent_name = agent_name
        self.resource_id = resource_id
        self.resource_type = resource_type
        self.cleanup_func = cleanup_func
        self.metadata = metadata
        self.resource_obj = None
        self.managed_resource = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.managed_resource:
            asyncio.create_task(self.managed_resource.cleanup())

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.managed_resource:
            await self.managed_resource.cleanup()

    def register(self, resource_obj: Any) -> Any:
        """Register a resource object."""
        self.resource_obj = resource_obj
        resource_manager = resource_system.get_resource_manager(self.agent_name)

        if resource_manager:
            self.managed_resource = resource_manager.register_resource(
                self.resource_id,
                self.resource_type,
                resource_obj,
                self.cleanup_func,
                self.metadata,
            )

        return resource_obj
