import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from src.core.message_broker import MessageBroker, MessageType


@pytest.mark.asyncio
async def test_message_broker_send_message_smoke():
    async def run():
        with patch("src.core.message_broker.redis.from_url") as mock_from_url:
            # Minimal async redis client mock
            redis_client = AsyncMock()
            redis_client.ping = AsyncMock(return_value=True)
            redis_client.publish = AsyncMock(return_value=1)
            redis_client.hset = AsyncMock(return_value=True)
            redis_client.expire = AsyncMock(return_value=True)
            redis_client.zadd = AsyncMock(return_value=1)
            mock_from_url.return_value = redis_client

            broker = MessageBroker("redis://localhost:6379")
            await broker.initialize()

            ok = await broker.send_message(
                from_agent="smoke-a",
                to_agent="smoke-b",
                topic="smoke",
                payload={"ping": 1},
                message_type=MessageType.DIRECT,
            )

            assert ok is True
            redis_client.publish.assert_awaited()

    await run()
