"""Test structured API adoption for message broker."""

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest

from src.core.message_broker import MessageBroker, MessageType, BrokerSendResult, BrokerSendReason


class TestMessageBrokerStructuredAPI:
    """Test structured API functionality."""

    @pytest.fixture
    async def message_broker(self):
        """Create MessageBroker instance with mocked Redis."""
        with patch("src.core.message_broker.redis.from_url") as mock_redis:
            # Create a mock Redis client
            rc = AsyncMock()
            rc.ping = AsyncMock(return_value=True)
            rc.set = AsyncMock(return_value=True)
            rc.publish = AsyncMock(return_value=1)
            rc.hset = AsyncMock(return_value=True)
            rc.expire = AsyncMock(return_value=True)
            rc.zadd = AsyncMock(return_value=True)
            rc.close = AsyncMock(return_value=None)
            mock_redis.return_value = rc

            broker = MessageBroker("redis://localhost:6379")
            await broker.initialize()
            return broker

    @pytest.mark.asyncio
    async def test_send_message_with_result_success(self, message_broker):
        """Test successful send with structured result."""
        result = await message_broker.send_message_with_result(
            from_agent="test_agent",
            to_agent="target_agent",
            topic="test_topic",
            payload={"data": "test"},
            message_type=MessageType.DIRECT,
        )

        assert isinstance(result, BrokerSendResult)
        assert result.success is True
        assert result.reason == BrokerSendReason.OK
        assert result.message_id is not None
        assert isinstance(result.message_id, str)

    @pytest.mark.asyncio
    async def test_send_message_with_result_idempotency(self, message_broker):
        """Test idempotency handling with structured result."""
        idempotency_key = "test_idempotent_key"

        # First send should succeed
        result1 = await message_broker.send_message_with_result(
            from_agent="test_agent",
            to_agent="target_agent",
            topic="test_topic",
            payload={"data": "test1"},
            idempotency_key=idempotency_key,
        )

        assert result1.success is True
        assert result1.reason == BrokerSendReason.OK

        # Second send with same key should return idempotent duplicate
        result2 = await message_broker.send_message_with_result(
            from_agent="test_agent",
            to_agent="target_agent",
            topic="test_topic",
            payload={"data": "test2"},
            idempotency_key=idempotency_key,
        )

        assert result2.success is True
        assert result2.reason == BrokerSendReason.IDEMPOTENT_DUPLICATE

    @pytest.mark.asyncio
    async def test_send_message_with_result_error(self, message_broker):
        """Test error handling with structured result."""
        # Mock Redis to raise an error
        message_broker.redis_client.publish.side_effect = Exception("Redis connection error")

        result = await message_broker.send_message_with_result(
            from_agent="test_agent",
            to_agent="target_agent",
            topic="test_topic",
            payload={"data": "test"},
        )

        assert isinstance(result, BrokerSendResult)
        assert result.success is False
        assert result.reason == BrokerSendReason.PUBLISH_ERROR
        assert result.message_id is None
        assert "Redis connection error" in str(result.details)

    @pytest.mark.asyncio
    async def test_backward_compatibility(self, message_broker):
        """Test that legacy boolean API still works."""
        # Test legacy send_message method
        result = await message_broker.send_message(
            from_agent="test_agent",
            to_agent="target_agent",
            topic="test_topic",
            payload={"data": "test"},
        )

        assert isinstance(result, bool)
        assert result is True

    @pytest.mark.asyncio
    async def test_orchestrator_ping_with_structured_api(self):
        """Test orchestrator ping functionality with structured API."""
        from src.core.orchestrator import HealthMonitor
        from src.core.enums import AgentStatus
        from src.agents.base_agent import AgentInfo

        # Create a mock registry
        class MockRegistry:
            def __init__(self):
                self.agents = {
                    "test_agent": AgentInfo(
                        id="test_id",
                        name="test_agent",
                        type="test_type",
                        role="test_role",
                        status=AgentStatus.ACTIVE,
                        capabilities=[],
                        tmux_session=None,
                        last_heartbeat=asyncio.get_event_loop().time(),
                        created_at=asyncio.get_event_loop().time(),
                        tasks_completed=0,
                        tasks_failed=0,
                    )
                }

        registry = MockRegistry()
        monitor = HealthMonitor(registry)

        # Mock message broker to simulate ping response
        with patch("src.core.orchestrator.message_broker") as mock_broker:
            # Create a mock that returns a structured result
            mock_result = BrokerSendResult(
                success=True,
                reason=BrokerSendReason.OK,
                message_id="test_message_id"
            )
            mock_broker.send_message_with_result.return_value = mock_result

            # Test ping functionality
            is_alive = await monitor._ping_agent("test_agent")

            assert isinstance(is_alive, bool)
            # The ping should succeed since we mocked a successful structured result
            assert is_alive is True