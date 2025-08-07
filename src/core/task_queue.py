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
    priority: int = TaskPriority.NORMAL
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
    """Redis-based priority task queue with advanced features."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.queue_prefix = "hive:queue"
        self.task_prefix = "hive:task"
        self.dependency_prefix = "hive:dep"
        self.stats_key = "hive:stats"

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

            logger.info("Task queue initialized with optimized connection settings")
        except asyncio.TimeoutError:
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
            # Add to appropriate priority queue
            queue_key = f"{self.queue_prefix}:p{task.priority}"
            await self.redis_client.lpush(queue_key, task.id)
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

        # Update stats
        await self._update_stats("submitted")
        return task.id

    async def get_task(
        self, agent_id: str, priorities: list[int] | None = None, timeout: int = 60
    ) -> Task | None:
        """Get next available task for agent with priority ordering."""
        if priorities is None:
            priorities = [1, 3, 5, 7, 9]  # All priorities

        # Check queues in priority order
        for priority in sorted(priorities):
            queue_key = f"{self.queue_prefix}:p{priority}"

            # Try to pop a task (non-blocking first check)
            task_id = await self.redis_client.rpop(queue_key)
            if not task_id:
                continue

            # Get task data
            task_data = await self.redis_client.hgetall(f"{self.task_prefix}:{task_id}")
            if not task_data:
                logger.warning("Task data not found", task_id=task_id)
                continue

            # Convert back to Task object
            task = await self._dict_to_task(task_data)

            # Assign to agent
            task.status = TaskStatus.ASSIGNED
            task.agent_id = agent_id
            task.started_at = time.time()

            # Update in Redis
            await self._update_task(task)

            logger.info(
                "Task assigned", task_id=task_id, agent_id=agent_id, priority=priority
            )
            await self._update_stats("assigned")
            return task

        # No tasks available in any priority queue
        return None

    async def start_task(self, task_id: str) -> bool:
        """Mark task as in progress."""
        task_key = f"{self.task_prefix}:{task_id}"
        task_data = await self.redis_client.hgetall(task_key)

        if not task_data:
            return False

        # Update status
        await self.redis_client.hset(task_key, "status", TaskStatus.IN_PROGRESS)
        await self._update_stats("started")

        logger.info("Task started", task_id=task_id)
        return True

    async def complete_task(
        self, task_id: str, result: dict[str, Any] | None = None
    ) -> bool:
        """Mark task as completed and process dependent tasks."""
        task_key = f"{self.task_prefix}:{task_id}"
        task_data = await self.redis_client.hgetall(task_key)

        if not task_data:
            return False

        # Update task
        updates = {"status": TaskStatus.COMPLETED, "completed_at": str(time.time())}
        if result:
            updates["result"] = json.dumps(result)

        await self.redis_client.hset(task_key, mapping=updates)

        # Process dependent tasks
        await self._process_dependent_tasks(task_id)

        logger.info("Task completed", task_id=task_id)
        await self._update_stats("completed")
        return True

    async def fail_task(self, task_id: str, error: str, retry: bool = True) -> bool:
        """Mark task as failed and handle retry logic."""
        task_key = f"{self.task_prefix}:{task_id}"
        task_data = await self.redis_client.hgetall(task_key)

        if not task_data:
            return False

        task = await self._dict_to_task(task_data)

        # Check if we should retry
        if retry and task.retry_count < task.max_retries:
            # Increment retry count and requeue with exponential backoff
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.agent_id = None

            # Calculate delay (exponential backoff)
            delay = min(300, 2**task.retry_count)  # Max 5 minutes

            await self._update_task(task)

            # Schedule retry
            await asyncio.sleep(delay)
            queue_key = f"{self.queue_prefix}:p{task.priority}"
            await self.redis_client.lpush(queue_key, task_id)

            logger.info(
                "Task scheduled for retry",
                task_id=task_id,
                retry_count=task.retry_count,
                delay=delay,
            )
            return True
        else:
            # Mark as permanently failed
            updates = {
                "status": TaskStatus.FAILED,
                "error_message": error,
                "completed_at": str(time.time()),
            }
            await self.redis_client.hset(task_key, mapping=updates)

            logger.error("Task failed permanently", task_id=task_id, error=error)
            await self._update_stats("failed")
            return True

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or assigned task."""
        task_key = f"{self.task_prefix}:{task_id}"
        task_data = await self.redis_client.hgetall(task_key)

        if not task_data:
            return False

        # Remove from queue if still pending
        task = await self._dict_to_task(task_data)
        if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
            queue_key = f"{self.queue_prefix}:p{task.priority}"
            await self.redis_client.lrem(queue_key, 1, task_id)

        # Update status
        await self.redis_client.hset(task_key, "status", TaskStatus.CANCELLED)

        logger.info("Task cancelled", task_id=task_id)
        await self._update_stats("cancelled")
        return True

    async def get_task_status(self, task_id: str) -> Task | None:
        """Get current status of a task."""
        task_key = f"{self.task_prefix}:{task_id}"
        task_data = await self.redis_client.hgetall(task_key)

        if not task_data:
            return None

        return await self._dict_to_task(task_data)

    async def get_queue_stats(self) -> QueueStats:
        """Get comprehensive queue statistics."""
        # Count tasks by status
        all_task_keys = await self.redis_client.keys(f"{self.task_prefix}:*")

        status_counts = {
            TaskStatus.PENDING: 0,
            TaskStatus.ASSIGNED: 0,
            TaskStatus.IN_PROGRESS: 0,
            TaskStatus.COMPLETED: 0,
            TaskStatus.FAILED: 0,
        }

        completion_times = []

        for task_key in all_task_keys:
            task_data = await self.redis_client.hgetall(task_key)
            if not task_data:
                continue

            status = task_data.get("status", TaskStatus.PENDING)
            status_counts[status] = status_counts.get(status, 0) + 1

            # Calculate completion time for completed tasks
            if status == TaskStatus.COMPLETED:
                started = task_data.get("started_at")
                completed = task_data.get("completed_at")
                if started and completed:
                    completion_times.append(float(completed) - float(started))

        # Count queue sizes by priority
        queue_sizes = {}
        for priority in [1, 3, 5, 7, 9]:
            queue_key = f"{self.queue_prefix}:p{priority}"
            size = await self.redis_client.llen(queue_key)
            queue_sizes[priority] = size

        avg_completion_time = (
            sum(completion_times) / len(completion_times) if completion_times else 0.0
        )

        return QueueStats(
            pending_tasks=status_counts[TaskStatus.PENDING],
            assigned_tasks=status_counts[TaskStatus.ASSIGNED],
            in_progress_tasks=status_counts[TaskStatus.IN_PROGRESS],
            completed_tasks=status_counts[TaskStatus.COMPLETED],
            failed_tasks=status_counts[TaskStatus.FAILED],
            total_tasks=sum(status_counts.values()),
            average_completion_time=avg_completion_time,
            queue_size_by_priority=queue_sizes,
        )

    async def cleanup_expired_tasks(self) -> int:
        """Clean up expired tasks and reassign them."""
        current_time = time.time()
        cleaned_count = 0

        # Find tasks that have timed out
        all_task_keys = await self.redis_client.keys(f"{self.task_prefix}:*")

        for task_key in all_task_keys:
            task_data = await self.redis_client.hgetall(task_key)
            if not task_data:
                continue

            status = task_data.get("status")
            started_at = task_data.get("started_at")
            timeout = int(task_data.get("timeout_seconds", 300))

            if status == TaskStatus.IN_PROGRESS and started_at:
                if current_time - float(started_at) > timeout:
                    task_id = task_key.split(":")[-1]
                    await self.fail_task(task_id, "Task timeout", retry=True)
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info("Cleaned up expired tasks", count=cleaned_count)

        return cleaned_count

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

        # Handle numeric fields
        for field in ["priority", "retry_count", "max_retries", "timeout_seconds"]:
            if field in data and data[field]:
                data[field] = int(data[field])

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

        # Check all priority queues
        for priority in range(1, 10):
            queue_key = f"{self.queue_prefix}:p{priority}"
            task_ids = await self.redis_client.lrange(queue_key, 0, -1)

            for task_id in task_ids:
                task_key = f"{self.task_prefix}:{task_id}"
                task_data = await self.redis_client.hgetall(task_key)

                if task_data:
                    task = await self._dict_to_task(task_data)
                    if task.status == TaskStatus.PENDING and not task.agent_id:
                        unassigned_tasks.append(task)

        return unassigned_tasks

    async def get_agent_active_task_count(self, agent_id: str) -> int:
        """Get count of active tasks for a specific agent."""
        count = 0

        # Search through all tasks to find active ones for this agent
        pattern = f"{self.task_prefix}:*"
        async for task_key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(task_key)

            if task_data and task_data.get("agent_id") == agent_id:
                status = task_data.get("status")
                if status in [TaskStatus.ASSIGNED, TaskStatus.RUNNING]:
                    count += 1

        return count

    async def get_total_tasks(self) -> int:
        """Get total number of tasks in the system."""
        count = 0
        pattern = f"{self.task_prefix}:*"
        async for _task_key in self.redis_client.scan_iter(match=pattern):
            count += 1
        return count

    async def get_completed_tasks_count(self) -> int:
        """Get count of completed tasks."""
        count = 0
        pattern = f"{self.task_prefix}:*"
        async for task_key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(task_key)
            if task_data and task_data.get("status") == TaskStatus.COMPLETED:
                count += 1
        return count

    async def get_failed_tasks_count(self) -> int:
        """Get count of failed tasks."""
        count = 0
        pattern = f"{self.task_prefix}:*"
        async for task_key in self.redis_client.scan_iter(match=pattern):
            task_data = await self.redis_client.hgetall(task_key)
            if task_data and task_data.get("status") == TaskStatus.FAILED:
                count += 1
        return count


# Global task queue instance
task_queue = TaskQueue()
