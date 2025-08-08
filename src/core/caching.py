"""Advanced Redis-based caching system with intelligent invalidation.

This module provides a comprehensive caching layer for the LeanVibe Agent Hive 2.0
system, focusing on optimizing database queries and improving response times.

Performance Target: <50ms p95 response time for cached operations.
"""

import asyncio
import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import redis.asyncio as redis
import structlog

logger = structlog.get_logger()


class CacheStrategy(Enum):
    """Cache invalidation strategies."""

    TTL = "ttl"  # Time-based expiration
    LRU = "lru"  # Least recently used
    WRITE_THROUGH = "write_through"  # Update cache on write
    WRITE_BEHIND = "write_behind"  # Async cache updates
    DEPENDENCY = "dependency"  # Invalidate based on dependencies


class CacheLevel(Enum):
    """Cache storage levels for different data types."""

    L1_MEMORY = "l1_memory"  # In-process memory cache
    L2_REDIS = "l2_redis"  # Redis cache layer
    L3_PERSISTENT = "l3_persistent"  # Persistent cache with Redis backup


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    ttl_seconds: int = 300  # 5 minutes default
    max_size: int = 1000  # Max items in cache
    strategy: CacheStrategy = CacheStrategy.TTL
    level: CacheLevel = CacheLevel.L2_REDIS
    compress: bool = False  # Compress large values
    serialize_json: bool = True  # JSON serialization
    key_prefix: str = "hive:cache"


@dataclass
class CacheStats:
    """Cache performance statistics."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    average_response_time_ms: float = 0.0
    cache_size_bytes: int = 0
    hit_rate: float = 0.0


class CacheKey:
    """Utility for generating consistent cache keys."""

    @staticmethod
    def generate(namespace: str, *args, **kwargs) -> str:
        """Generate a consistent cache key."""
        # Create hash from arguments
        key_data = {"args": args, "kwargs": sorted(kwargs.items()) if kwargs else {}}
        key_string = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]

        return f"hive:cache:{namespace}:{key_hash}"

    @staticmethod
    def pattern(namespace: str) -> str:
        """Generate a pattern for bulk operations."""
        return f"hive:cache:{namespace}:*"


class CacheInvalidator:
    """Handles cache invalidation based on dependencies."""

    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.dependencies_key = "hive:cache:dependencies"

    async def add_dependency(self, cache_key: str, dependency: str):
        """Add a dependency relationship."""
        await self.redis_client.sadd(f"{self.dependencies_key}:{dependency}", cache_key)

    async def invalidate_by_dependency(self, dependency: str) -> int:
        """Invalidate all cache entries dependent on a resource."""
        dependency_key = f"{self.dependencies_key}:{dependency}"
        dependent_keys = await self.redis_client.smembers(dependency_key)

        if dependent_keys:
            # Delete all dependent cache entries
            await self.redis_client.delete(*dependent_keys)
            # Clean up dependency tracking
            await self.redis_client.delete(dependency_key)

            logger.info(
                "Cache invalidated by dependency",
                dependency=dependency,
                invalidated_keys=len(dependent_keys),
            )

            return len(dependent_keys)

        return 0


class MultiLevelCache:
    """Multi-level cache with L1 (memory) and L2 (Redis) support."""

    def __init__(self, redis_client: redis.Redis, config: CacheConfig):
        self.redis_client = redis_client
        self.config = config
        self.l1_cache = {}  # In-memory L1 cache
        self.l1_access_times = {}  # Track access for LRU
        self.stats = CacheStats()
        self.invalidator = CacheInvalidator(redis_client)

    async def get(self, key: str) -> Any | None:
        """Get value from cache with L1/L2 fallback."""
        start_time = time.time()
        self.stats.total_requests += 1

        try:
            # Try L1 cache first
            if self.config.level in [CacheLevel.L1_MEMORY, CacheLevel.L3_PERSISTENT]:
                if key in self.l1_cache:
                    self.l1_access_times[key] = time.time()
                    self.stats.hits += 1
                    self._update_response_time(start_time)
                    logger.debug("Cache hit (L1)", key=key)
                    return self.l1_cache[key]

            # Try L2 Redis cache
            redis_value = await self.redis_client.get(key)
            if redis_value is not None:
                value = self._deserialize(redis_value)

                # Promote to L1 if using multi-level
                if self.config.level in [
                    CacheLevel.L1_MEMORY,
                    CacheLevel.L3_PERSISTENT,
                ]:
                    await self._set_l1(key, value)

                self.stats.hits += 1
                self._update_response_time(start_time)
                logger.debug("Cache hit (L2)", key=key)
                return value

            # Cache miss
            self.stats.misses += 1
            self._update_response_time(start_time)
            logger.debug("Cache miss", key=key)
            return None

        except Exception as e:
            logger.error("Cache get error", key=key, error=str(e))
            self.stats.misses += 1
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
        dependencies: list[str] | None = None,
    ):
        """Set value in cache with optional TTL and dependencies."""
        try:
            ttl = ttl or self.config.ttl_seconds
            serialized_value = self._serialize(value)

            # Set in Redis with TTL
            await self.redis_client.setex(key, ttl, serialized_value)

            # Set in L1 if using multi-level
            if self.config.level in [CacheLevel.L1_MEMORY, CacheLevel.L3_PERSISTENT]:
                await self._set_l1(key, value)

            # Add dependency tracking
            if dependencies:
                for dep in dependencies:
                    await self.invalidator.add_dependency(key, dep)

            logger.debug("Cache set", key=key, ttl=ttl, dependencies=dependencies)

        except Exception as e:
            logger.error("Cache set error", key=key, error=str(e))

    async def delete(self, key: str):
        """Delete from all cache levels."""
        try:
            # Delete from Redis
            await self.redis_client.delete(key)

            # Delete from L1
            if key in self.l1_cache:
                del self.l1_cache[key]
                del self.l1_access_times[key]

            logger.debug("Cache deleted", key=key)

        except Exception as e:
            logger.error("Cache delete error", key=key, error=str(e))

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching a pattern."""
        try:
            keys = []
            # Handle scan_iter properly for async redis
            if hasattr(self.redis_client, "scan_iter"):
                scan_iter = self.redis_client.scan_iter(match=pattern)
                if asyncio.iscoroutine(scan_iter):
                    scan_iter = await scan_iter

                if hasattr(scan_iter, "__aiter__"):
                    async for key in scan_iter:
                        keys.append(key)
                elif hasattr(scan_iter, "__iter__"):
                    for key in scan_iter:
                        keys.append(key)
                else:
                    # Fallback for mock objects
                    if callable(scan_iter):
                        try:
                            result = scan_iter()
                            if isinstance(result, list):
                                keys = result
                        except Exception:
                            pass

            if keys:
                await self.redis_client.delete(*keys)

                # Clear from L1 as well
                for key in keys:
                    if key in self.l1_cache:
                        del self.l1_cache[key]
                        del self.l1_access_times[key]

                logger.info(
                    "Cache pattern cleared", pattern=pattern, keys_cleared=len(keys)
                )
                return len(keys)

        except Exception as e:
            logger.error("Cache pattern clear error", pattern=pattern, error=str(e))

        return 0

    async def invalidate_dependencies(self, dependency: str) -> int:
        """Invalidate all entries dependent on a resource."""
        return await self.invalidator.invalidate_by_dependency(dependency)

    async def get_stats(self) -> CacheStats:
        """Get cache performance statistics."""
        # Update hit rate
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.hits / self.stats.total_requests

        # Estimate cache size
        try:
            memory_info = await self.redis_client.memory_usage("hive:cache:*")
            self.stats.cache_size_bytes = memory_info if memory_info else 0
        except Exception:
            # Fallback estimation
            self.stats.cache_size_bytes = len(self.l1_cache) * 1024  # Rough estimate

        return self.stats

    async def _set_l1(self, key: str, value: Any):
        """Set value in L1 cache with LRU eviction."""
        # Check if we need to evict
        if len(self.l1_cache) >= self.config.max_size and key not in self.l1_cache:
            await self._evict_l1_lru()

        self.l1_cache[key] = value
        self.l1_access_times[key] = time.time()

    async def _evict_l1_lru(self):
        """Evict least recently used item from L1."""
        if not self.l1_access_times:
            return

        lru_key = min(self.l1_access_times, key=self.l1_access_times.get)
        if lru_key in self.l1_cache:
            del self.l1_cache[lru_key]
        if lru_key in self.l1_access_times:
            del self.l1_access_times[lru_key]
        self.stats.evictions += 1

    def _serialize(self, value: Any) -> str:
        """Serialize value for storage."""
        if self.config.serialize_json:
            return json.dumps(value, default=str)
        return str(value)

    def _deserialize(self, value: str) -> Any:
        """Deserialize value from storage."""
        if self.config.serialize_json:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    def _update_response_time(self, start_time: float):
        """Update average response time statistics."""
        response_time = (time.time() - start_time) * 1000  # Convert to ms

        # Running average
        if self.stats.total_requests == 1:
            self.stats.average_response_time_ms = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.stats.average_response_time_ms = (
                alpha * response_time
                + (1 - alpha) * self.stats.average_response_time_ms
            )


class CacheManager:
    """Main cache manager for the system."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.caches = {}  # Namespace -> MultiLevelCache mapping
        self.default_config = CacheConfig()

    async def initialize(self):
        """Initialize the cache manager with timeout and retry logic."""
        max_retries = 3
        base_timeout = 5.0

        for attempt in range(max_retries):
            try:
                # Add timeout to Redis ping
                await asyncio.wait_for(self.redis_client.ping(), timeout=base_timeout)
                logger.info("Cache manager initialized")
                return
            except asyncio.TimeoutError:
                logger.warning(
                    f"Redis ping timeout on attempt {attempt + 1}/{max_retries}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Progressive backoff
                    continue
                else:
                    logger.error("Redis connection failed after all retries")
                    raise ConnectionError("Redis connection timeout after retries")
            except Exception as e:
                logger.error(
                    "Cache manager initialization failed",
                    error=str(e),
                    attempt=attempt + 1,
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                else:
                    raise ConnectionError(f"Failed to connect to Redis: {e}") from e

    def get_cache(
        self, namespace: str, config: CacheConfig | None = None
    ) -> MultiLevelCache:
        """Get or create a cache for a namespace."""
        if namespace not in self.caches:
            cache_config = config or self.default_config
            self.caches[namespace] = MultiLevelCache(self.redis_client, cache_config)

        return self.caches[namespace]

    def cached(
        self,
        namespace: str,
        ttl: int | None = None,
        dependencies: list[str] | None = None,
        key_generator: Callable | None = None,
    ):
        """Decorator for automatic caching of function results."""

        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_generator:
                    cache_key = key_generator(*args, **kwargs)
                else:
                    cache_key = CacheKey.generate(namespace, *args, **kwargs)

                # Try to get from cache
                cache = self.get_cache(namespace)
                cached_result = await cache.get(cache_key)

                if cached_result is not None:
                    return cached_result

                # Execute function and cache result
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result, ttl=ttl, dependencies=dependencies)

                return result

            return wrapper

        return decorator

    async def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all cache entries in a namespace."""
        if namespace in self.caches:
            pattern = CacheKey.pattern(namespace)
            return await self.caches[namespace].clear_pattern(pattern)
        return 0

    async def invalidate_dependency(self, dependency: str) -> int:
        """Invalidate all entries dependent on a resource across all namespaces."""
        total_invalidated = 0

        for cache in self.caches.values():
            count = await cache.invalidate_dependencies(dependency)
            total_invalidated += count

        return total_invalidated

    async def get_global_stats(self) -> dict[str, CacheStats]:
        """Get performance statistics for all caches."""
        stats = {}

        for namespace, cache in self.caches.items():
            stats[namespace] = await cache.get_stats()

        return stats

    async def health_check(self) -> dict[str, Any]:
        """Perform cache system health check."""
        try:
            # Test Redis connectivity
            ping_start = time.time()
            await self.redis_client.ping()
            ping_time = (time.time() - ping_start) * 1000

            # Get memory usage
            try:
                memory_info = await self.redis_client.info("memory")
                memory_usage = memory_info.get("used_memory", 0)
                memory_peak = memory_info.get("used_memory_peak", 0)
            except Exception:
                memory_usage = memory_peak = 0

            # Get cache statistics
            global_stats = await self.get_global_stats()

            return {
                "status": "healthy",
                "redis_ping_ms": ping_time,
                "memory_usage_bytes": memory_usage,
                "memory_peak_bytes": memory_peak,
                "cache_namespaces": len(self.caches),
                "cache_stats": global_stats,
                "performance_target_met": all(
                    stats.average_response_time_ms < 50.0
                    for stats in global_stats.values()
                ),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "performance_target_met": False,
            }


# Pre-configured cache instances for common use cases
CONTEXT_CACHE_CONFIG = CacheConfig(
    ttl_seconds=600,  # 10 minutes for context queries
    max_size=5000,
    strategy=CacheStrategy.TTL,
    level=CacheLevel.L3_PERSISTENT,
)

TASK_QUEUE_CACHE_CONFIG = CacheConfig(
    ttl_seconds=60,  # 1 minute for task queue stats
    max_size=1000,
    strategy=CacheStrategy.TTL,
    level=CacheLevel.L2_REDIS,
)

SESSION_CACHE_CONFIG = CacheConfig(
    ttl_seconds=1800,  # 30 minutes for session data
    max_size=2000,
    strategy=CacheStrategy.WRITE_THROUGH,
    level=CacheLevel.L3_PERSISTENT,
)


# Global cache manager instance
cache_manager = None


async def get_cache_manager(redis_url: str = None) -> CacheManager:
    """Get the global cache manager instance."""
    global cache_manager
    if cache_manager is None:
        if redis_url is None:
            from .config import get_settings

            redis_url = get_settings().redis_url
        cache_manager = CacheManager(redis_url)
        await cache_manager.initialize()
    return cache_manager


# Utility functions for common caching patterns
async def cache_database_query(
    namespace: str,
    query_func: Callable,
    *args,
    ttl: int = 300,
    dependencies: list[str] | None = None,
    **kwargs,
):
    """Cache a database query result."""
    manager = await get_cache_manager()
    cache = manager.get_cache(namespace)

    cache_key = CacheKey.generate(namespace, query_func.__name__, *args, **kwargs)

    # Try cache first
    result = await cache.get(cache_key)
    if result is not None:
        return result

    # Execute query and cache result
    result = await query_func(*args, **kwargs)
    await cache.set(cache_key, result, ttl=ttl, dependencies=dependencies)

    return result


async def invalidate_model_cache(model_name: str, model_id: str | None = None):
    """Invalidate cache entries for a specific model."""
    manager = await get_cache_manager()

    if model_id:
        dependency = f"{model_name}:{model_id}"
    else:
        dependency = model_name

    return await manager.invalidate_dependency(dependency)
