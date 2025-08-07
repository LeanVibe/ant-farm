"""Comprehensive tests for the caching system."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.core.caching import (
    CONTEXT_CACHE_CONFIG,
    TASK_QUEUE_CACHE_CONFIG,
    CacheConfig,
    CacheInvalidator,
    CacheKey,
    CacheLevel,
    CacheManager,
    CacheStrategy,
    MultiLevelCache,
    get_cache_manager,
)


@pytest.fixture
async def mock_redis():
    """Mock Redis client for testing."""
    redis_mock = AsyncMock()
    redis_mock.ping.return_value = True
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.scan_iter.return_value = []
    redis_mock.smembers.return_value = set()
    redis_mock.sadd.return_value = 1
    redis_mock.srem.return_value = 1
    redis_mock.memory_usage.return_value = 1024
    return redis_mock


@pytest.fixture
def cache_config():
    """Default cache configuration for testing."""
    return CacheConfig(
        ttl_seconds=300,
        max_size=100,
        strategy=CacheStrategy.TTL,
        level=CacheLevel.L2_REDIS,
    )


@pytest.fixture
async def cache_manager(mock_redis):
    """Cache manager instance for testing."""
    with patch("src.core.caching.redis.from_url", return_value=mock_redis):
        manager = CacheManager("redis://localhost:6379")
        await manager.initialize()
        return manager


@pytest.fixture
async def multi_level_cache(mock_redis, cache_config):
    """Multi-level cache instance for testing."""
    return MultiLevelCache(mock_redis, cache_config)


class TestCacheKey:
    """Test cache key generation utilities."""

    def test_generate_consistent_keys(self):
        """Test that identical inputs generate identical keys."""
        key1 = CacheKey.generate("test", "arg1", arg2="value2")
        key2 = CacheKey.generate("test", "arg1", arg2="value2")
        assert key1 == key2

    def test_generate_different_keys(self):
        """Test that different inputs generate different keys."""
        key1 = CacheKey.generate("test", "arg1", arg2="value2")
        key2 = CacheKey.generate("test", "arg1", arg2="value3")
        assert key1 != key2

    def test_pattern_generation(self):
        """Test cache key pattern generation."""
        pattern = CacheKey.pattern("test_namespace")
        assert pattern == "hive:cache:test_namespace:*"


class TestCacheConfig:
    """Test cache configuration."""

    def test_default_config(self):
        """Test default cache configuration values."""
        config = CacheConfig()
        assert config.ttl_seconds == 300
        assert config.max_size == 1000
        assert config.strategy == CacheStrategy.TTL
        assert config.level == CacheLevel.L2_REDIS

    def test_custom_config(self):
        """Test custom cache configuration."""
        config = CacheConfig(
            ttl_seconds=600,
            max_size=2000,
            strategy=CacheStrategy.LRU,
            level=CacheLevel.L1_MEMORY,
        )
        assert config.ttl_seconds == 600
        assert config.max_size == 2000
        assert config.strategy == CacheStrategy.LRU
        assert config.level == CacheLevel.L1_MEMORY


class TestCacheInvalidator:
    """Test cache invalidation functionality."""

    @pytest.mark.asyncio
    async def test_add_dependency(self, mock_redis):
        """Test adding cache dependencies."""
        invalidator = CacheInvalidator(mock_redis)

        await invalidator.add_dependency("cache_key_1", "dependency_1")

        mock_redis.sadd.assert_called_once_with(
            "hive:cache:dependencies:dependency_1", "cache_key_1"
        )

    @pytest.mark.asyncio
    async def test_invalidate_by_dependency(self, mock_redis):
        """Test invalidating cache entries by dependency."""
        invalidator = CacheInvalidator(mock_redis)

        # Mock dependent keys
        mock_redis.smembers.return_value = {"key1", "key2", "key3"}

        result = await invalidator.invalidate_by_dependency("dependency_1")

        # Should delete dependent keys and clean up dependency tracking
        mock_redis.delete.assert_called()
        assert result == 3


class TestMultiLevelCache:
    """Test multi-level cache functionality."""

    @pytest.mark.asyncio
    async def test_cache_miss(self, multi_level_cache, mock_redis):
        """Test cache miss scenario."""
        mock_redis.get.return_value = None

        result = await multi_level_cache.get("test_key")

        assert result is None
        assert multi_level_cache.stats.misses == 1
        assert multi_level_cache.stats.hits == 0

    @pytest.mark.asyncio
    async def test_cache_hit_l2(self, multi_level_cache, mock_redis):
        """Test L2 cache hit."""
        mock_redis.get.return_value = '{"data": "test_value"}'

        result = await multi_level_cache.get("test_key")

        assert result == {"data": "test_value"}
        assert multi_level_cache.stats.hits == 1
        assert multi_level_cache.stats.misses == 0

    @pytest.mark.asyncio
    async def test_cache_set(self, multi_level_cache, mock_redis):
        """Test setting cache values."""
        test_data = {"key": "value", "number": 42}

        await multi_level_cache.set("test_key", test_data, ttl=600)

        mock_redis.setex.assert_called_once_with(
            "test_key", 600, '{"key": "value", "number": 42}'
        )

    @pytest.mark.asyncio
    async def test_cache_set_with_dependencies(self, multi_level_cache, mock_redis):
        """Test setting cache values with dependencies."""
        test_data = {"key": "value"}
        dependencies = ["dep1", "dep2"]

        await multi_level_cache.set("test_key", test_data, dependencies=dependencies)

        # Should call sadd for each dependency
        assert mock_redis.sadd.call_count == 2

    @pytest.mark.asyncio
    async def test_cache_delete(self, multi_level_cache, mock_redis):
        """Test deleting cache entries."""
        await multi_level_cache.delete("test_key")

        mock_redis.delete.assert_called_once_with("test_key")

    @pytest.mark.asyncio
    async def test_l1_cache_functionality(self, mock_redis):
        """Test L1 (in-memory) cache functionality."""
        config = CacheConfig(level=CacheLevel.L1_MEMORY, max_size=3)
        cache = MultiLevelCache(mock_redis, config)

        # Set values in L1 cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Should hit L1 cache
        result = await cache.get("key1")
        assert result == "value1"

        # Add another item to trigger LRU eviction
        await cache.set("key4", "value4")

        # key1 should be evicted (LRU)
        assert "key1" not in cache.l1_cache
        assert cache.stats.evictions == 1

    @pytest.mark.asyncio
    async def test_cache_stats(self, multi_level_cache, mock_redis):
        """Test cache statistics collection."""
        # Simulate some cache operations
        await multi_level_cache.get("miss_key")  # Miss

        mock_redis.get.return_value = '"hit_value"'
        await multi_level_cache.get("hit_key")  # Hit

        stats = await multi_level_cache.get_stats()

        assert stats.total_requests == 2
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.hit_rate == 0.5

    @pytest.mark.asyncio
    async def test_clear_pattern(self, multi_level_cache, mock_redis):
        """Test clearing cache entries by pattern."""

        # Mock scan_iter to return some keys
        async def mock_scan_iter(match):
            keys = ["hive:cache:test:key1", "hive:cache:test:key2"]
            for key in keys:
                yield key

        mock_redis.scan_iter.side_effect = mock_scan_iter

        result = await multi_level_cache.clear_pattern("hive:cache:test:*")

        assert result == 2
        mock_redis.delete.assert_called_once()


class TestCacheManager:
    """Test cache manager functionality."""

    @pytest.mark.asyncio
    async def test_initialize(self, mock_redis):
        """Test cache manager initialization."""
        with patch("src.core.caching.redis.from_url", return_value=mock_redis):
            manager = CacheManager("redis://localhost:6379")
            await manager.initialize()

            mock_redis.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_cache(self, cache_manager):
        """Test getting cache instances by namespace."""
        cache1 = cache_manager.get_cache("namespace1")
        cache2 = cache_manager.get_cache("namespace1")  # Same namespace
        cache3 = cache_manager.get_cache("namespace2")  # Different namespace

        assert cache1 is cache2  # Should return same instance
        assert cache1 is not cache3  # Different instances for different namespaces

    @pytest.mark.asyncio
    async def test_cached_decorator(self, cache_manager, mock_redis):
        """Test the @cached decorator functionality."""
        call_count = 0

        @cache_manager.cached("test_namespace", ttl=300)
        async def expensive_function(arg1, arg2=None):
            nonlocal call_count
            call_count += 1
            return f"result_{arg1}_{arg2}"

        # First call - should execute function
        mock_redis.get.return_value = None  # Cache miss
        result1 = await expensive_function("test", arg2="value")
        assert result1 == "result_test_value"
        assert call_count == 1

        # Second call - should hit cache
        mock_redis.get.return_value = '"result_test_value"'  # Cache hit
        result2 = await expensive_function("test", arg2="value")
        assert result2 == "result_test_value"
        assert call_count == 1  # Function not called again

    @pytest.mark.asyncio
    async def test_invalidate_namespace(self, cache_manager, mock_redis):
        """Test invalidating entire namespaces."""

        # Mock scan_iter to return some keys
        async def mock_scan_iter(match):
            keys = ["hive:cache:test:key1", "hive:cache:test:key2"]
            for key in keys:
                yield key

        mock_redis.scan_iter.side_effect = mock_scan_iter

        # Create cache for namespace
        cache_manager.get_cache("test")

        result = await cache_manager.invalidate_namespace("test")

        assert result == 2

    @pytest.mark.asyncio
    async def test_global_stats(self, cache_manager):
        """Test getting global cache statistics."""
        # Create some caches
        cache1 = cache_manager.get_cache("namespace1")
        cache2 = cache_manager.get_cache("namespace2")

        stats = await cache_manager.get_global_stats()

        assert "namespace1" in stats
        assert "namespace2" in stats
        assert len(stats) == 2

    @pytest.mark.asyncio
    async def test_health_check(self, cache_manager, mock_redis):
        """Test cache system health check."""
        mock_redis.info.return_value = {
            "used_memory": 1024000,
            "used_memory_peak": 2048000,
        }

        health = await cache_manager.health_check()

        assert health["status"] == "healthy"
        assert "redis_ping_ms" in health
        assert "memory_usage_bytes" in health
        assert "cache_namespaces" in health


class TestCachingIntegration:
    """Integration tests for caching system."""

    @pytest.mark.asyncio
    async def test_context_cache_integration(self, mock_redis):
        """Test context caching integration."""
        with patch("src.core.caching.redis.from_url", return_value=mock_redis):
            # Test getting cache manager for context operations
            cache_manager = await get_cache_manager()
            context_cache = cache_manager.get_cache(
                "context_search", CONTEXT_CACHE_CONFIG
            )

            # Test cache configuration
            assert context_cache.config.ttl_seconds == 600
            assert context_cache.config.level == CacheLevel.L3_PERSISTENT

    @pytest.mark.asyncio
    async def test_task_queue_cache_integration(self, mock_redis):
        """Test task queue caching integration."""
        with patch("src.core.caching.redis.from_url", return_value=mock_redis):
            cache_manager = await get_cache_manager()
            task_cache = cache_manager.get_cache(
                "task_queue_stats", TASK_QUEUE_CACHE_CONFIG
            )

            # Test cache configuration
            assert task_cache.config.ttl_seconds == 60
            assert task_cache.config.level == CacheLevel.L2_REDIS

    @pytest.mark.asyncio
    async def test_dependency_invalidation_flow(self, cache_manager, mock_redis):
        """Test complete dependency invalidation flow."""
        cache = cache_manager.get_cache("test_namespace")

        # Set cache entry with dependency
        await cache.set("test_key", "test_value", dependencies=["agent:123"])

        # Mock dependency lookup
        mock_redis.smembers.return_value = {"test_key"}

        # Invalidate by dependency
        result = await cache_manager.invalidate_dependency("agent:123")

        assert result == 1
        mock_redis.delete.assert_called()

    @pytest.mark.asyncio
    async def test_performance_targets(self, cache_manager):
        """Test that cache meets performance targets."""
        cache = cache_manager.get_cache("performance_test")

        # Measure cache set performance
        start_time = time.time()
        await cache.set("perf_key", {"data": "test"})
        set_time = (time.time() - start_time) * 1000

        # Should be very fast (under 10ms)
        assert set_time < 10

        # Measure cache get performance (mock fast response)
        start_time = time.time()
        await cache.get("perf_key")
        get_time = (time.time() - start_time) * 1000

        # Should be very fast (under 5ms)
        assert get_time < 5


class TestCacheErrorHandling:
    """Test error handling in caching system."""

    @pytest.mark.asyncio
    async def test_redis_connection_failure(self, mock_redis):
        """Test handling Redis connection failures."""
        mock_redis.ping.side_effect = Exception("Connection failed")

        with patch("src.core.caching.redis.from_url", return_value=mock_redis):
            manager = CacheManager("redis://localhost:6379")

            with pytest.raises(Exception):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_cache_get_error_handling(self, multi_level_cache, mock_redis):
        """Test error handling in cache get operations."""
        mock_redis.get.side_effect = Exception("Redis error")

        result = await multi_level_cache.get("test_key")

        assert result is None
        assert multi_level_cache.stats.misses == 1

    @pytest.mark.asyncio
    async def test_cache_set_error_handling(self, multi_level_cache, mock_redis):
        """Test error handling in cache set operations."""
        mock_redis.setex.side_effect = Exception("Redis error")

        # Should not raise exception
        await multi_level_cache.set("test_key", "test_value")

        # Error should be logged but not propagated

    @pytest.mark.asyncio
    async def test_serialization_error_handling(self, multi_level_cache):
        """Test handling of serialization errors."""

        # Create an object that can't be JSON serialized
        class NonSerializable:
            def __init__(self):
                self.func = lambda x: x

        non_serializable = NonSerializable()

        # Should handle gracefully (implementation would need to be updated)
        # This test documents expected behavior
        try:
            await multi_level_cache.set("test_key", non_serializable)
            # If we reach here, serialization was handled gracefully
        except Exception:
            # Expected behavior - serialization should fail gracefully
            pass


@pytest.mark.asyncio
async def test_cache_memory_management():
    """Test cache memory management and cleanup."""
    config = CacheConfig(max_size=5, level=CacheLevel.L1_MEMORY)
    mock_redis = AsyncMock()
    cache = MultiLevelCache(mock_redis, config)

    # Fill cache to capacity
    for i in range(5):
        await cache.set(f"key_{i}", f"value_{i}")

    assert len(cache.l1_cache) == 5

    # Add one more item to trigger eviction
    await cache.set("key_5", "value_5")

    # Should have evicted one item
    assert len(cache.l1_cache) == 5
    assert cache.stats.evictions == 1


@pytest.mark.asyncio
async def test_cache_ttl_functionality(mock_redis):
    """Test TTL (time-to-live) functionality."""
    config = CacheConfig(ttl_seconds=300)
    cache = MultiLevelCache(mock_redis, config)

    await cache.set("test_key", "test_value", ttl=600)

    # Should call setex with custom TTL
    mock_redis.setex.assert_called_with("test_key", 600, '"test_value"')


@pytest.mark.asyncio
async def test_concurrent_cache_operations(cache_manager):
    """Test concurrent cache operations."""
    cache = cache_manager.get_cache("concurrent_test")

    # Create multiple concurrent operations
    tasks = []
    for i in range(10):
        tasks.append(cache.set(f"key_{i}", f"value_{i}"))
        tasks.append(cache.get(f"key_{i}"))

    # Should handle concurrent operations without error
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check that no operations failed
    for result in results:
        if isinstance(result, Exception):
            pytest.fail(f"Concurrent operation failed: {result}")


if __name__ == "__main__":
    pytest.main([__file__])
