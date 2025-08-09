"""Redis-based task queue system with priorities, retries, and dependencies."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import redis.asyncio as redis
import structlog
from pydantic import BaseModel, Field

from .caching import TASK_QUEUE_CACHE_CONFIG, CacheKey, get_cache_manager

logger = structlog.get_logger()


class TaskPriority(IntEnum):
    """Task priority levels (lower number = higher priority)."""

    CRITICAL = 1
    HIGH = 3
    NORMAL = 5
    LOW = 7
    BACKGROUND = 9


class TaskStatus:
    """Task status constants."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(BaseModel):
    """Task data structure with validation."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    description: str
    task_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    status: str = TaskStatus.PENDING
    agent_id: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    created_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error_message: str | None = None


@dataclass
class QueueStats:
    """Statistics for queue monitoring."""

    pending_tasks: int
    assigned_tasks: int
    in_progress_tasks: int
    completed_tasks: int
    failed_tasks: int
    total_tasks: int
    average_completion_time: float
    queue_size_by_priority: dict[int, int]


class TaskQueue:
    """
    Redis-based priority task queue with advanced features.

    Required interface:
    - async def get_failed_tasks(self) -> list[Task]: Returns a list of failed tasks.
    - async def get_failed_tasks_count(self) -> int: Returns the count of failed tasks.
    - async def get_total_tasks(self) -> int: Returns the total number of tasks.
    - async def get_completed_tasks(self) -> int: Returns the count of completed tasks.
    - async def get_queue_depth(self) -> int: Returns the current queue depth.

    Any mock or subclass used in tests or other components must implement these methods.
    """

    async def get_failed_tasks(self) -> list[Task]:
        """Return a list of failed tasks."""
        return await self.list_tasks(status=TaskStatus.FAILED)

    """Redis-based priority task queue with advanced features."""

    def __init__(self, redis_url: str = None):
        if redis_url is None:
            from .config import get_settings

            redis_url = get_settings().redis_url
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.queue_prefix = "hive:queue"
        self.task_prefix = "hive:task"
        self.dependency_prefix = "hive:dep"
        self.stats_key = "hive:stats"
        self.cache_manager = None

    async def initialize(self) -> None:
        """Initialize the task queue system."""
        try:
            # Test Redis connection with timeout
            await asyncio.wait_for(self.redis_client.ping(), timeout=5.0)

            # Configure Redis client for better performance
            # Enable connection pooling and set appropriate timeouts
            if hasattr(self.redis_client, "connection_pool"):
                pool = self.redis_client.connection_pool
                pool.connection_kwargs.update(
                    {
                        "socket_timeout": 10.0,
                        "socket_connect_timeout": 5.0,
                        "socket_keepalive": True,
                        "socket_keepalive_options": {},
                    }
                )

            # Initialize cache manager
            try:
                self.cache_manager = await get_cache_manager()
            except Exception as e:
                logger.warning(
                    "Cache manager initialization failed, continuing without caching",
                    error=str(e),
                )
                self.cache_manager = None

            logger.info(
                "Task queue initialized with optimized connection settings and caching"
            )
        except TimeoutError:
            logger.error("Redis connection timeout during task queue initialization")
            raise
        except Exception as e:
            logger.error("Task queue initialization failed", error=str(e))
            raise

    async def submit_task(self, task: Task) -> str:
        """Submit a task to the queue with dependency checking."""
        # Store task data
        task_key = f"{self.task_prefix}:{task.id}"
        task_data = task.model_dump()
        task_data["priority"] = task.priority.value

        # Convert timestamp fields to strings for Redis
        for field in ["created_at", "started_at", "completed_at"]:
            if task_data.get(field) is not None:
                task_data[field] = str(task_data[field])
            else:
                task_data[field] = ""

        # Convert complex fields to JSON strings for Redis
        for field in ["payload", "dependencies", "result"]:
            if task_data.get(field) is not None:
                task_data[field] = json.dumps(task_data[field])
            else:
                task_data[field] = ""

        # Convert None values to empty strings
        for key, value in task_data.items():
            if value is None:
                task_data[key] = ""

        await self.redis_client.hset(task_key, mapping=task_data)

        # Check if dependencies are satisfied
        if await self._dependencies_satisfied(task):
            # Add to priority queue using sorted set (lower score = higher priority)
            await self.redis_client.zadd(
                f"{self.queue_prefix}:priority", {task.id: task.priority.value}
            )
            logger.info(
                "Task submitted to queue", task_id=task.id, priority=task.priority
            )
        else:
            # Store in dependency wait list
            dep_key = f"{self.dependency_prefix}:{task.id}"
            await self.redis_client.sadd(dep_key, *task.dependencies)
            logger.info(
                "Task waiting for dependencies",
                task_id=task.id,
                dependencies=task.dependencies,
            )

        # Update stats and invalidate cache
        await self._update_stats("submitted")
        if self.cache_manager:
            await self.cache_manager.invalidate_dependency("task_queue_stats")
            if task.agent_id:
                await self.cache_manager.invalidate_dependency(f"agent:{task.agent_id}")

        return task.id

    async def add_task(self, task: Task) -> str:
        """Alias for submit_task to maintain API compatibility."""
        return await self.submit_task(task)

    async def _dependencies_satisfied(self, task: Task) -> bool:
        """Check if all task dependencies are completed."""
        for dep_id in task.dependencies:
            dep_key = f"{self.task_prefix}:{dep_id}"
            dep_data = await self.redis_client.hgetall(dep_key)
            if not dep_data or dep_data.get("status") != TaskStatus.COMPLETED:
                return False
        return True

    async def _process_dependent_tasks(self, completed_task_id: str) -> None:
        """Process tasks that were waiting for this task to complete."""
        # Find tasks waiting for this dependency
        dep_keys = await self.redis_client.keys(f"{self.dependency_prefix}:*")

        for dep_key in dep_keys:
            if await self.redis_client.sismember(dep_key, completed_task_id):
                # Remove this dependency
                await self.redis_client.srem(dep_key, completed_task_id)

                # Check if all dependencies are now satisfied
                remaining_deps = await self.redis_client.smembers(dep_key)
                if not remaining_deps:
                    # All dependencies satisfied, queue the task
                    task_id = dep_key.split(":")[-1]
                    task_data = await self.redis_client.hgetall(
                        f"{self.task_prefix}:{task_id}"
                    )

                    if task_data:
                        priority = int(task_data.get("priority", TaskPriority.NORMAL))
                        queue_key = f"{self.queue_prefix}:p{priority}"
                        await self.redis_client.lpush(queue_key, task_id)

                        # Clean up dependency tracking
                        await self.redis_client.delete(dep_key)

                        logger.info(
                            "Task dependencies satisfied, queued", task_id=task_id
                        )

    async def _dict_to_task(self, task_data: dict[str, str]) -> Task:
        """Convert Redis hash data back to Task object."""
        # Convert string fields back to appropriate types
        data = dict(task_data)

        # Handle priority separately
        if "priority" in data and data["priority"]:
            try:
                data["priority"] = TaskPriority(int(data["priority"]))
            except (ValueError, TypeError):
                data["priority"] = TaskPriority.NORMAL

        # Handle float fields
        for field in ["created_at", "started_at", "completed_at"]:
            if field in data and data[field] and data[field] != "None":
                data[field] = float(data[field])
            else:
                data[field] = None

        # Handle JSON fields
        for field in ["payload", "dependencies", "result"]:
            if field in data and data[field]:
                try:
                    data[field] = json.loads(data[field])
                except json.JSONDecodeError:
                    data[field] = {} if field in ["payload", "result"] else []
            else:
                data[field] = {} if field in ["payload", "result"] else []

        return Task(**data)

    async def _update_task(self, task: Task) -> None:
        """Update task data in Redis."""
        task_key = f"{self.task_prefix}:{task.id}"
        task_data = task.model_dump()
        task_data["priority"] = task.priority.value

        # Convert fields for Redis storage
        for field in ["created_at", "started_at", "completed_at"]:
            if task_data.get(field) is not None:
                task_data[field] = str(task_data[field])
            else:
                task_data[field] = ""

        for field in ["payload", "dependencies", "result"]:
            if task_data.get(field) is not None:
                task_data[field] = json.dumps(task_data[field])
            else:
                task_data[field] = ""

        # Convert None values to empty strings
        for key, value in task_data.items():
            if value is None:
                task_data[key] = ""

        await self.redis_client.hset(task_key, mapping=task_data)

    async def _update_stats(self, operation: str) -> None:
        """Update queue statistics."""
        stats_key = f"{self.stats_key}:{operation}"
        await self.redis_client.incr(stats_key)
        await self.redis_client.expire(stats_key, 86400)  # 24 hours

    async def get_unassigned_tasks(self) -> list[Task]:
        """Get all unassigned tasks for coordination."""
        unassigned_tasks = []
        all_task_keys = await self.redis_client.keys(f"{self.task_prefix}:*")
        for task_key in all_task_keys:
            task_data = await self.redis_client.hgetall(task_key)
            if (
                task_data
                and task_data.get("status") == TaskStatus.PENDING
                and not task_data.get("agent_id")
            ):
                unassigned_tasks.append(await self._dict_to_task(task_data))
        return unassigned_tasks

    async def get_agent_active_task_count(self, agent_id: str) -> int:
        """Get count of active tasks for a specific agent with caching."""

        # Generate cache key
        cache_key = CacheKey.generate("agent_task_count", agent_id=agent_id)

        # Try cache first
        if self.cache_manager:
            cache = self.cache_manager.get_cache(
                "task_queue_stats", TASK_QUEUE_CACHE_CONFIG
            )
            cached_count = await cache.get(cache_key)
            if cached_count is not None:
                logger.debug("Agent task count cache hit", agent_id=agent_id)
                return cached_count

        count = 0

        # Search through all tasks to find active ones for this agent using SCAN
        pattern = f"{self.task_prefix}:*"
        async for task_key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(task_key)

            if task_data and task_data.get("agent_id") == agent_id:
                status = task_data.get("status")
                if status in [TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS]:
                    count += 1

        # Cache the result with short TTL since this can change frequently
        if self.cache_manager:
            await cache.set(
                cache_key,
                count,
                ttl=30,  # 30 seconds
                dependencies=[f"agent:{agent_id}"],
            )

        return count

    async def get_total_tasks(self) -> int:
        """Get total number of tasks in the system with caching."""

        # Generate cache key
        cache_key = CacheKey.generate("total_tasks")

        # Try cache first
        if self.cache_manager:
            cache = self.cache_manager.get_cache(
                "task_queue_stats", TASK_QUEUE_CACHE_CONFIG
            )
            cached_count = await cache.get(cache_key)
            if cached_count is not None:
                logger.debug("Total tasks cache hit")
                return cached_count

        count = 0
        pattern = f"{self.task_prefix}:*"
        async for _task_key in self.redis_client.scan_iter(match=pattern):
            count += 1

        # Cache the result
        if self.cache_manager:
            await cache.set(cache_key, count, dependencies=["task_queue_stats"])

        return count

    async def get_completed_tasks_count(self) -> int:
        """Get count of completed tasks with caching."""

        # Generate cache key
        cache_key = CacheKey.generate("completed_tasks")

        # Try cache first
        if self.cache_manager:
            cache = self.cache_manager.get_cache(
                "task_queue_stats", TASK_QUEUE_CACHE_CONFIG
            )
            cached_count = await cache.get(cache_key)
            if cached_count is not None:
                logger.debug("Completed tasks cache hit")
                return cached_count

        count = 0
        pattern = f"{self.task_prefix}:*"
        async for task_key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(task_key)
            if task_data and task_data.get("status") == TaskStatus.COMPLETED:
                count += 1

        # Cache the result
        if self.cache_manager:
            await cache.set(cache_key, count, dependencies=["task_queue_stats"])

        return count

    async def get_failed_tasks_count(self) -> int:
        """Get count of failed tasks with caching."""

        # Generate cache key
        cache_key = CacheKey.generate("failed_tasks")

        # Try cache first
        if self.cache_manager:
            cache = self.cache_manager.get_cache(
                "task_queue_stats", TASK_QUEUE_CACHE_CONFIG
            )
            cached_count = await cache.get(cache_key)
            if cached_count is not None:
                logger.debug("Failed tasks cache hit")
                return cached_count

        count = 0
        pattern = f"{self.task_prefix}:*"
        async for task_key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(task_key)
            if task_data and task_data.get("status") == TaskStatus.FAILED:
                count += 1

        # Cache the result
        if self.cache_manager:
            await cache.set(cache_key, count, dependencies=["task_queue_stats"])

        return count

    async def get_task(self, task_id_or_agent: str = None) -> Task | None:
        """Get task by ID if provided, otherwise get next available task."""
        # If parameter looks like a task ID (UUID format), get specific task
        if task_id_or_agent and (
            "-" in task_id_or_agent and len(task_id_or_agent) > 20
        ):
            return await self._get_task_by_id(task_id_or_agent)
        else:
            # Get next available task for agent
            return await self._get_next_available_task(task_id_or_agent)

    async def _get_task_by_id(self, task_id: str) -> Task | None:
        """Get a specific task by ID."""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                return None

            return await self._dict_to_task(task_data)

        except Exception as e:
            logger.error("Failed to get task by ID", task_id=task_id, error=str(e))
            return None

    async def _get_next_available_task(self, agent_id: str = None) -> Task | None:
        """Get next available task from priority queues using BRPOP."""
        try:
            # Check queues in priority order (lower number = higher priority)
            queue_keys = []
            for priority in sorted(TaskPriority):
                queue_key = f"{self.queue_prefix}:p{priority.value}"
                queue_keys.append(queue_key)

            # Use BRPOP to atomically get task from highest priority queue
            # Timeout of 1 second to avoid blocking forever
            result = await self.redis_client.brpop(queue_keys, timeout=1)

            if not result:
                return None

            queue_key, task_id = result

            # Get task data
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                logger.warning("Task data not found", task_id=task_id)
                return None

            # Convert to Task object
            task = await self._dict_to_task(task_data)

            # Assign to agent if provided
            if agent_id:
                task.agent_id = agent_id
                task.status = TaskStatus.ASSIGNED
                task.started_at = time.time()
                await self._update_task(task)

            logger.info("Task retrieved from queue", task_id=task_id, agent_id=agent_id)
            return task

        except Exception as e:
            logger.error("Failed to get task from queue", error=str(e))
            return None

    async def list_tasks(
        self, status: str = None, assigned_to: str = None
    ) -> list[Task]:
        """List tasks with optional filtering."""
        tasks = []
        try:
            pattern = f"{self.task_prefix}:*"
            async for task_key in self.redis_client.scan_iter(match=pattern):
                task_data = await self.redis_client.hgetall(task_key)
                if task_data:
                    task = await self._dict_to_task(task_data)

                    # Apply filters
                    if status and task.status != status:
                        continue
                    if assigned_to and task.agent_id != assigned_to:
                        continue

                    tasks.append(task)

            # Sort by created_at (newest first)
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            return tasks

        except Exception as e:
            logger.error("Failed to list tasks", error=str(e))
            return []

    async def get_queue_depth(self) -> int:
        """Get total number of pending tasks across all priority queues."""
        try:
            total_depth = 0
            for priority in TaskPriority:
                queue_key = f"{self.queue_prefix}:p{priority.value}"
                depth = await self.redis_client.llen(queue_key)
                total_depth += depth
            return total_depth

        except Exception as e:
            logger.error("Failed to get queue depth", error=str(e))
            return 0

    async def get_queue_stats(self) -> QueueStats:
        """Get comprehensive queue statistics."""
        try:
            # Count tasks by status
            status_counts = {
                TaskStatus.PENDING: 0,
                TaskStatus.ASSIGNED: 0,
                TaskStatus.IN_PROGRESS: 0,
                TaskStatus.COMPLETED: 0,
                TaskStatus.FAILED: 0,
            }

            # Count tasks by priority
            priority_counts = {}

            completion_times = []
            total_tasks = 0

            pattern = f"{self.task_prefix}:*"
            async for task_key in self.redis_client.scan_iter(match=pattern):
                task_data = await self.redis_client.hgetall(task_key)
                if task_data:
                    total_tasks += 1
                    status = task_data.get("status", TaskStatus.PENDING)
                    if status in status_counts:
                        status_counts[status] += 1

                    priority = int(task_data.get("priority", TaskPriority.NORMAL.value))
                    priority_counts[priority] = priority_counts.get(priority, 0) + 1

                    # Calculate completion time if task is completed
                    if status == TaskStatus.COMPLETED:
                        created_at = float(task_data.get("created_at", 0))
                        completed_at = float(task_data.get("completed_at", 0))
                        if created_at > 0 and completed_at > 0:
                            completion_times.append(completed_at - created_at)

            avg_completion_time = (
                sum(completion_times) / len(completion_times)
                if completion_times
                else 0.0
            )

            return QueueStats(
                pending_tasks=status_counts[TaskStatus.PENDING],
                assigned_tasks=status_counts[TaskStatus.ASSIGNED],
                in_progress_tasks=status_counts[TaskStatus.IN_PROGRESS],
                completed_tasks=status_counts[TaskStatus.COMPLETED],
                failed_tasks=status_counts[TaskStatus.FAILED],
                total_tasks=total_tasks,
                average_completion_time=avg_completion_time,
                queue_size_by_priority=priority_counts,
            )

        except Exception as e:
            logger.error("Failed to get queue stats", error=str(e))
            return QueueStats(0, 0, 0, 0, 0, 0, 0.0, {})

    async def fail_task(
        self, task_id: str, error_message: str, retry: bool = True
    ) -> bool:
        """Mark a task as failed with optional retry."""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                logger.warning("Task not found for failure", task_id=task_id)
                return False

            task = await self._dict_to_task(task_data)
            task.status = TaskStatus.FAILED
            task.error_message = error_message
            task.completed_at = time.time()

            # Handle retry logic
            if retry and task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                task.error_message = None
                task.completed_at = None

                # Re-queue the task
                queue_key = f"{self.queue_prefix}:p{task.priority.value}"
                await self.redis_client.lpush(queue_key, task_id)

                logger.info(
                    "Task queued for retry",
                    task_id=task_id,
                    retry_count=task.retry_count,
                )
            else:
                logger.info(
                    "Task marked as failed", task_id=task_id, error=error_message
                )

            await self._update_task(task)
            await self._update_stats("failed")

            return True

        except Exception as e:
            logger.error("Failed to mark task as failed", task_id=task_id, error=str(e))
            return False

    async def start_task(self, task_id: str) -> bool:
        """Mark a task as started/in progress."""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                logger.warning("Task not found for starting", task_id=task_id)
                return False

            task = await self._dict_to_task(task_data)
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = time.time()

            await self._update_task(task)
            await self._update_stats("started")

            logger.info("Task marked as started", task_id=task_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to mark task as started", task_id=task_id, error=str(e)
            )
            return False

    async def complete_task(self, task_id: str, result: dict = None) -> bool:
        """Mark a task as completed."""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                logger.warning("Task not found for completion", task_id=task_id)
                return False

            task = await self._dict_to_task(task_data)
            task.status = TaskStatus.COMPLETED
            task.result = result or {}
            task.completed_at = time.time()

            await self._update_task(task)
            await self._update_stats("completed")

            # Remove from priority queue
            await self.redis_client.zrem(f"{self.queue_prefix}:priority", task_id)

            # Process dependent tasks
            await self._process_dependent_tasks(task_id)

            logger.info("Task marked as completed", task_id=task_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to mark task as completed", task_id=task_id, error=str(e)
            )
            return False

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                return False

            task = await self._dict_to_task(task_data)
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()

            await self._update_task(task)
            await self._update_stats("cancelled")

            logger.info("Task cancelled", task_id=task_id)
            return True

        except Exception as e:
            logger.error("Failed to cancel task", task_id=task_id, error=str(e))
            return False

    async def update_task_status(
        self, task_id: str, status: str, agent_id: str = None
    ) -> bool:
        """Update a task's status and optionally assign to agent."""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                logger.warning("Task not found for status update", task_id=task_id)
                return False

            task = await self._dict_to_task(task_data)
            old_status = task.status
            task.status = status

            if agent_id:
                task.agent_id = agent_id

            if status == TaskStatus.IN_PROGRESS:
                task.started_at = time.time()
            elif status in [
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            ]:
                task.completed_at = time.time()

            await self._update_task(task)

            # Update stats if status changed
            if old_status != status:
                await self._update_stats(f"status_change_{status}")

            logger.info(
                "Task status updated",
                task_id=task_id,
                old_status=old_status,
                new_status=status,
            )
            return True

        except Exception as e:
            logger.error("Failed to update task status", task_id=task_id, error=str(e))
            return False

    async def fail_task(
        self, task_id: str, error_message: str = None, retry: bool = True
    ) -> bool:
        """Mark a task as failed with optional error message and retry control."""
        try:
            task_key = f"{self.task_prefix}:{task_id}"
            task_data = await self.redis_client.hgetall(task_key)

            if not task_data:
                return False

            task = await self._dict_to_task(task_data)
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            task.error_message = error_message
            task.retry_count += 1

            await self._update_task(task)
            await self._update_stats("failed")

            # Check if task should be retried
            if retry and task.retry_count < task.max_retries:
                logger.info(
                    "Task failed, will retry",
                    task_id=task_id,
                    retry_count=task.retry_count,
                )
                # Reset task for retry
                task.status = TaskStatus.PENDING
                task.started_at = None
                task.completed_at = None
                task.agent_id = None
                await self._update_task(task)

                # Re-add to appropriate priority queue
                queue_key = f"{self.queue_prefix}:p{task.priority.value}"
                await self.redis_client.lpush(queue_key, task_id)
            else:
                logger.error(
                    "Task failed permanently",
                    task_id=task_id,
                    max_retries=task.max_retries,
                )

            return True

        except Exception as e:
            logger.error("Failed to mark task as failed", task_id=task_id, error=str(e))
            return False

    async def cleanup_expired_tasks(self) -> int:
        """Clean up expired tasks and return count of cleaned tasks."""
        cleaned_count = 0
        current_time = time.time()

        try:
            pattern = f"{self.task_prefix}:*"
            async for task_key in self.redis_client.scan_iter(match=pattern):
                task_data = await self.redis_client.hgetall(task_key)
                if not task_data:
                    continue

                status = task_data.get("status")
                started_at = task_data.get("started_at")
                timeout = int(task_data.get("timeout_seconds", 300))

                if status == TaskStatus.IN_PROGRESS and started_at:
                    if current_time - float(started_at) > timeout:
                        task_id = task_key.split(":")[-1]
                        task = await self._dict_to_task(task_data)
                        task.error_message = "Task timeout"
                        await self._update_task(task)
                        await self.fail_task(task_id, "Task timeout", retry=True)
                        cleaned_count += 1

            if cleaned_count > 0:
                logger.info("Cleaned up expired tasks", count=cleaned_count)

            return cleaned_count

        except Exception as e:
            logger.error("Failed to cleanup expired tasks", error=str(e))
            return 0

    async def get_next_task(self, agent_id: str = None) -> Task | None:
        """Get the next highest priority task from the queue."""
        try:
            # Use sorted set to get highest priority task (lowest score)
            task_ids = await self.redis_client.zrange(
                f"{self.queue_prefix}:priority", 0, 0
            )

            if not task_ids:
                return None

            task_id = task_ids[0]
            task = await self.get_task(task_id)

            if task and task.status == TaskStatus.PENDING:
                # Remove from priority queue
                await self.redis_client.zrem(f"{self.queue_prefix}:priority", task_id)
                return task

            return None

        except Exception as e:
            logger.error("Failed to get next task", error=str(e))
            return None

    async def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to a specific agent."""
        try:
            task = await self.get_task(task_id)
            if not task:
                logger.warning("Task not found for assignment", task_id=task_id)
                return False

            # Update task with agent assignment
            task.agent_id = agent_id
            task.status = TaskStatus.ASSIGNED
            await self._update_task(task)

            logger.info("Task assigned to agent", task_id=task_id, agent_id=agent_id)
            return True

        except Exception as e:
            logger.error(
                "Failed to assign task",
                task_id=task_id,
                agent_id=agent_id,
                error=str(e),
            )
            return False

    async def get_task_status(self, task_id: str) -> Task | None:
        """Get current task status (alias for get_task)."""
        return await self.get_task(task_id)

    async def get_timed_out_tasks(self) -> list[Task]:
        """Get list of tasks that have timed out."""
        try:
            timed_out = []
            current_time = time.time()

            # Get all in-progress tasks
            in_progress_tasks = await self.list_tasks(status=TaskStatus.IN_PROGRESS)

            for task in in_progress_tasks:
                if (
                    task.started_at
                    and current_time - task.started_at > task.timeout_seconds
                ):
                    timed_out.append(task)

            return timed_out

        except Exception as e:
            logger.error("Failed to get timed out tasks", error=str(e))
            return []


# Global task queue instance
task_queue = TaskQueue()
