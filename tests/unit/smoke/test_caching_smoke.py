from unittest.mock import AsyncMock, patch

import pytest

from src.core.caching import CacheManager, CacheKey


@pytest.mark.asyncio
async def test_cache_set_get_smoke():
    with patch("src.core.caching.redis.from_url") as mock_from_url:
        client = AsyncMock()
        client.ping = AsyncMock(return_value=True)
        client.setex = AsyncMock(return_value=True)
        client.get = AsyncMock(return_value='{"answer": 42}')
        client.memory_usage = AsyncMock(return_value=1024)
        client.info = AsyncMock(return_value={"used_memory": 1024, "used_memory_peak": 2048})
        mock_from_url.return_value = client

        manager = await CacheManager("redis://localhost:6379").initialize() or None
        # Recreate via helper to align with typical code paths
        with patch("src.core.caching.get_cache_manager") as get_mgr:
            get_mgr.return_value = CacheManager("redis://localhost:6379")
            get_mgr.return_value.redis_client = client

            cache = get_mgr.return_value.get_cache("smoke")
            key = CacheKey.generate("smoke", 1, name="x")

            await cache.set(key, {"answer": 42}, ttl=10)
            value = await cache.get(key)

            assert value == {"answer": 42}
            stats = await cache.get_stats()
            assert stats.cache_size_bytes >= 0
