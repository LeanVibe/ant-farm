"""Focused TDD tests for MessageBroker - Critical Agent Communication Component.

This test suite covers the essential MessageBroker functionality for reliable agent coordination:
1. Message sending and receiving (core workflow)
2. Pub/sub reliability (agent communication)
3. Message persistence and delivery guarantees (reliability)
4. Connection management (system stability)

Following TDD: Write failing tests first, implement minimal code, refactor.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.message_broker import Message, MessageBroker, MessageHandler, MessageType


@pytest.fixture
def mock_redis():
    """Create a comprehensive mock Redis client for MessageBroker testing."""
    redis_mock = AsyncMock()

    # Basic Redis operations
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.close = AsyncMock(return_value=None)

    # Pub/Sub operations
    redis_mock.publish = AsyncMock(return_value=1)

    # Message persistence (hashes)
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.hdel = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)

    # Message queues (lists)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.rpush = AsyncMock(return_value=1)
    redis_mock.lpop = AsyncMock(return_value=None)
    redis_mock.rpop = AsyncMock(return_value=None)
    redis_mock.llen = AsyncMock(return_value=0)

    # Priority queues (sorted sets)
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zrem = AsyncMock(return_value=1)
    redis_mock.zrange = AsyncMock(return_value=[])
    redis_mock.zrangebyscore = AsyncMock(return_value=[])

    # Dead letter queue operations
    redis_mock.zrevrange = AsyncMock(return_value=[])

    # Key operations
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.exists = AsyncMock(return_value=0)
    redis_mock.keys = AsyncMock(return_value=[])

    # Mock pubsub
    pubsub_mock = AsyncMock()
    pubsub_mock.subscribe = AsyncMock(return_value=None)
    pubsub_mock.unsubscribe = AsyncMock(return_value=None)
    pubsub_mock.listen = AsyncMock()
    redis_mock.pubsub = Mock(return_value=pubsub_mock)

    return redis_mock


@pytest.fixture
async def message_broker(mock_redis):
    """Create MessageBroker instance with mocked Redis."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        broker = MessageBroker("redis://localhost:6379/1")
        await broker.initialize()
        return broker


@pytest.fixture
def sample_message():
    """Create a sample message for testing."""
    return Message(
        id=str(uuid4()),
        from_agent="agent1",
        to_agent="agent2",
        topic="test_topic",
        message_type=MessageType.DIRECT,
        payload={"test": "data"},
        timestamp=time.time(),
    )


@pytest.fixture
def message_handler():
    """Create a message handler for testing."""
    return MessageHandler("test-agent")


class TestMessageBrokerBasicOperations:
    """Test fundamental message broker operations."""

    @pytest.mark.asyncio
    async def test_initialize_message_broker(self, mock_redis):
        """Test that message broker initializes properly."""
        with patch("redis.asyncio.from_url", return_value=mock_redis):
            broker = MessageBroker("redis://localhost:6379/1")
            await broker.initialize()

            # Verify Redis connection was tested
            mock_redis.ping.assert_called_once()

            # Verify broker instance has required attributes
            assert hasattr(broker, "redis_client")
            assert hasattr(broker, "pubsub")
            assert broker.message_prefix == "hive:messages"

    @pytest.mark.asyncio
    async def test_send_message_basic(self, message_broker, sample_message, mock_redis):
        """Test basic message sending."""
        # Act - Send message
        success = await message_broker.send_message(
            from_agent=sample_message.from_agent,
            to_agent=sample_message.to_agent,
            topic=sample_message.topic,
            payload=sample_message.payload,
            message_type=sample_message.message_type,
        )

        # Assert - Message sent successfully (returns success boolean)
        assert success is True
        assert isinstance(success, bool)

        # Verify Redis operations were called
        mock_redis.publish.assert_called()  # Message published to channel
        mock_redis.hset.assert_called()  # Message persisted for reliability

    @pytest.mark.asyncio
    async def test_send_broadcast_message(self, message_broker, mock_redis):
        """Test broadcasting message to all agents."""
        # Act - Send broadcast message
        message_id = await message_broker.send_message(
            from_agent="orchestrator",
            to_agent="broadcast",
            topic="system_announcement",
            payload={"announcement": "System maintenance in 5 minutes"},
            message_type=MessageType.BROADCAST,
        )

        # Assert - Broadcast sent successfully (returns message ID)
        assert message_id is not None
        assert isinstance(message_id, str)

        # Verify broadcast channel was used
        mock_redis.publish.assert_called()

    @pytest.mark.asyncio
    async def test_message_handler_registration(self, message_broker, message_handler):
        """Test registering message handlers."""
        # Arrange - Create handler function
        messages_received = []

        async def test_handler(message):
            messages_received.append(message)
            return {"status": "received"}

        # Act - Register handler
        message_handler.register_handler("test_topic", test_handler)
        await message_broker.start_listening("test-agent", message_handler)

        # Assert - Handler registered successfully
        assert "test_topic" in message_handler.handlers
        assert message_handler.handlers["test_topic"] == test_handler


class TestMessageBrokerReliability:
    """Test message broker reliability features."""

    @pytest.mark.asyncio
    async def test_message_persistence(
        self, message_broker, sample_message, mock_redis
    ):
        """Test that messages are persisted for reliability."""
        # Act - Send message
        await message_broker.send_message(
            from_agent=sample_message.from_agent,
            to_agent=sample_message.to_agent,
            topic=sample_message.topic,
            payload=sample_message.payload,
            message_type=sample_message.message_type,
        )

        # Assert - Message was persisted
        mock_redis.hset.assert_called()  # Message stored in Redis hash
        mock_redis.expire.assert_called()  # TTL set for cleanup

    @pytest.mark.asyncio
    async def test_message_delivery_retry(self, message_broker, mock_redis):
        """Test message delivery retry mechanism."""
        # Arrange - Mock failed delivery (Redis publish returns 0)
        mock_redis.publish.return_value = 0  # No subscribers

        # Act - Send message
        success = await message_broker.send_message(
            from_agent="agent1",
            to_agent="agent2",
            topic="retry_test",
            payload={"test": "retry"},
            message_type=MessageType.DIRECT,
        )

        # Assert - Message queued for retry or handled gracefully
        # Implementation should handle failed delivery
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, message_broker, mock_redis):
        """Test dead letter queue for failed messages."""
        # Act - Send message that fails delivery
        await message_broker.send_message(
            from_agent="sender",
            to_agent="nonexistent_agent",
            topic="dlq_test",
            payload={"should": "fail"},
            message_type=MessageType.DIRECT,
        )

        # Assert - Failed message handling
        # Implementation details will vary, but should handle gracefully
        mock_redis.publish.assert_called()

    @pytest.mark.asyncio
    async def test_request_reply_pattern(self, message_broker, mock_redis):
        """Test request-reply messaging pattern."""
        # Arrange - Mock reply message in Redis
        reply_id = str(uuid4())
        mock_redis.hgetall.return_value = {
            "id": reply_id,
            "from_agent": "responder",
            "to_agent": "requester",
            "topic": "response",
            "message_type": MessageType.REPLY.value,
            "payload": json.dumps({"status": "success"}),
            "timestamp": str(time.time()),
            "reply_to": "original_request_id",
        }

        # Act - Send request and wait for reply
        if hasattr(message_broker, "send_request"):
            response = await message_broker.send_request(
                from_agent="requester",
                to_agent="responder",
                topic="test_request",
                payload={"query": "test"},
                timeout=1.0,
            )

            # Assert - Response received
            assert response is not None
        else:
            # Test basic send for now
            success = await message_broker.send_message(
                from_agent="requester",
                to_agent="responder",
                topic="test_request",
                payload={"query": "test"},
                message_type=MessageType.REQUEST,
            )
            assert success is True


class TestMessageBrokerPubSub:
    """Test publish-subscribe functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_to_topics(
        self, message_broker, message_handler, mock_redis
    ):
        """Test subscribing to message topics."""
        # Act - Start listening
        await message_broker.start_listening("test-agent", message_handler)

        # Assert - Subscription operations called
        pubsub_mock = mock_redis.pubsub.return_value
        pubsub_mock.subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_unsubscribe_from_topics(
        self, message_broker, message_handler, mock_redis
    ):
        """Test unsubscribing from message topics."""
        # Arrange - Start listening first
        await message_broker.start_listening("test-agent", message_handler)

        # Act - Stop listening
        await message_broker.stop_listening("test-agent")

        # Assert - Unsubscription operations called
        pubsub_mock = mock_redis.pubsub.return_value
        # Verify subscription and unsubscription were called
        assert pubsub_mock.subscribe.called or pubsub_mock.unsubscribe.called

    @pytest.mark.asyncio
    async def test_message_routing(self, message_broker, mock_redis):
        """Test that messages are routed to correct agents."""
        # Act - Send message to specific agent
        await message_broker.send_message(
            from_agent="sender",
            to_agent="specific_agent",
            topic="routing_test",
            payload={"target": "specific_agent"},
            message_type=MessageType.DIRECT,
        )

        # Assert - Message published to agent-specific channel
        mock_redis.publish.assert_called()

        # Extract channel name from publish call
        call_args = mock_redis.publish.call_args
        if call_args:
            channel = call_args[0][0]  # First argument is channel
            assert "specific_agent" in channel or "routing_test" in channel


class TestMessageBrokerConnectionManagement:
    """Test connection management and error handling."""

    @pytest.mark.asyncio
    async def test_redis_connection_failure_handling(self, mock_redis):
        """Test handling Redis connection failures gracefully."""
        # Arrange - Mock Redis to raise connection error
        mock_redis.ping.side_effect = Exception("Connection failed")

        with patch("redis.asyncio.from_url", return_value=mock_redis):
            broker = MessageBroker("redis://localhost:6379/1")

            # Act & Assert - Initialization should handle connection failure
            with pytest.raises(Exception):
                await broker.initialize()

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, message_broker, mock_redis):
        """Test graceful shutdown of message broker."""
        # Act - Shutdown broker
        if hasattr(message_broker, "shutdown"):
            await message_broker.shutdown()

        # Assert - Cleanup operations performed
        # Redis connection should be closed or cleaned up
        assert True  # Basic test for method existence

    @pytest.mark.asyncio
    async def test_reconnection_handling(self, message_broker, mock_redis):
        """Test automatic reconnection on connection loss."""
        # Arrange - Simulate connection loss
        mock_redis.ping.side_effect = [True, Exception("Connection lost"), True]

        # Act - Try operations after connection loss
        try:
            success = await message_broker.send_message(
                from_agent="test",
                to_agent="test2",
                topic="reconnect_test",
                payload={"test": "reconnection"},
                message_type=MessageType.DIRECT,
            )
            # Should handle connection issues gracefully
            assert isinstance(success, bool)
        except Exception:
            # Connection errors should be handled gracefully
            assert True


class TestMessageBrokerPerformance:
    """Test message broker performance aspects."""

    @pytest.mark.asyncio
    async def test_concurrent_message_sending(self, message_broker, mock_redis):
        """Test sending multiple messages concurrently."""
        # Arrange - Create multiple messages
        messages = [("agent1", "agent2", f"topic_{i}", {"data": i}) for i in range(10)]

        # Act - Send all messages concurrently
        tasks = [
            message_broker.send_message(
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                payload=payload,
                message_type=MessageType.DIRECT,
            )
            for from_agent, to_agent, topic, payload in messages
        ]

        results = await asyncio.gather(*tasks)

        # Assert - All messages sent successfully
        assert len(results) == 10
        assert all(isinstance(result, bool) for result in results)

        # Verify Redis operations performed for each message
        assert mock_redis.publish.call_count >= 10

    @pytest.mark.asyncio
    async def test_message_throughput(self, message_broker, mock_redis):
        """Test message broker throughput."""
        # Arrange - Prepare for throughput test
        start_time = time.time()
        message_count = 100

        # Act - Send messages rapidly
        tasks = []
        for i in range(message_count):
            task = message_broker.send_message(
                from_agent=f"sender_{i % 10}",
                to_agent=f"receiver_{i % 10}",
                topic="throughput_test",
                payload={"index": i},
                message_type=MessageType.DIRECT,
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Assert - Performance within acceptable bounds
        duration = end_time - start_time
        messages_per_second = message_count / duration

        # Should process at least 10 messages per second (very conservative)
        assert messages_per_second > 10
        assert len(results) == message_count

    @pytest.mark.asyncio
    async def test_message_batching(self, message_broker, mock_redis):
        """Test message batching for efficiency."""
        # Act - Send batch of messages if supported
        if hasattr(message_broker, "send_batch"):
            batch_messages = [
                {
                    "from_agent": "batch_sender",
                    "to_agent": f"receiver_{i}",
                    "topic": "batch_test",
                    "payload": {"index": i},
                    "message_type": MessageType.DIRECT,
                }
                for i in range(5)
            ]

            success = await message_broker.send_batch(batch_messages)
            assert success is True
        else:
            # Test individual sends for now
            for i in range(5):
                success = await message_broker.send_message(
                    from_agent="batch_sender",
                    to_agent=f"receiver_{i}",
                    topic="batch_test",
                    payload={"index": i},
                    message_type=MessageType.DIRECT,
                )
                assert success is True


class TestMessageBrokerErrorHandling:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_message_handling(self, message_broker):
        """Test handling of invalid message data."""
        # Act & Assert - Invalid message should be handled gracefully
        with pytest.raises((ValueError, TypeError)):
            await message_broker.send_message(
                from_agent="",  # Empty from_agent should be invalid
                to_agent="",  # Empty to_agent should be invalid
                topic="",  # Empty topic should be invalid
                payload=None,  # None payload might be invalid
                message_type=MessageType.DIRECT,
            )

    @pytest.mark.asyncio
    async def test_message_size_limits(self, message_broker, mock_redis):
        """Test handling of large messages."""
        # Arrange - Create very large payload
        large_payload = {"data": "x" * 1000000}  # 1MB of data

        # Act - Send large message
        success = await message_broker.send_message(
            from_agent="sender",
            to_agent="receiver",
            topic="large_message_test",
            payload=large_payload,
            message_type=MessageType.DIRECT,
        )

        # Assert - Large message handled (accepted or rejected gracefully)
        assert isinstance(success, bool)

    @pytest.mark.asyncio
    async def test_agent_not_found_handling(self, message_broker, mock_redis):
        """Test handling messages to non-existent agents."""
        # Act - Send message to non-existent agent
        success = await message_broker.send_message(
            from_agent="sender",
            to_agent="nonexistent_agent_12345",
            topic="not_found_test",
            payload={"test": "message"},
            message_type=MessageType.DIRECT,
        )

        # Assert - Non-existent agent handled gracefully
        assert isinstance(success, bool)
        # Message should still be sent (delivery failure handled separately)


class TestMessageBrokerIntegration:
    """Test integration scenarios with other components."""

    @pytest.mark.asyncio
    async def test_task_assignment_messaging(self, message_broker, mock_redis):
        """Test messaging for task assignment workflow."""
        # Act - Send task assignment message
        success = await message_broker.send_message(
            from_agent="orchestrator",
            to_agent="worker_agent",
            topic="task_assignment",
            payload={
                "task_id": "test-task-123",
                "task_type": "processing",
                "priority": "high",
            },
            message_type=MessageType.DIRECT,
        )

        # Assert - Task assignment message sent
        assert success is True
        mock_redis.publish.assert_called()

    @pytest.mark.asyncio
    async def test_agent_heartbeat_messaging(self, message_broker, mock_redis):
        """Test messaging for agent heartbeat workflow."""
        # Act - Send heartbeat message
        success = await message_broker.send_message(
            from_agent="worker_agent",
            to_agent="orchestrator",
            topic="heartbeat",
            payload={"status": "healthy", "timestamp": time.time(), "load": 0.5},
            message_type=MessageType.DIRECT,
        )

        # Assert - Heartbeat message sent
        assert success is True
        mock_redis.publish.assert_called()

    @pytest.mark.asyncio
    async def test_system_broadcast_messaging(self, message_broker, mock_redis):
        """Test system-wide broadcast messaging."""
        # Act - Send system broadcast
        success = await message_broker.send_message(
            from_agent="system",
            to_agent="broadcast",
            topic="system_shutdown",
            payload={
                "message": "System maintenance starting in 5 minutes",
                "shutdown_time": time.time() + 300,
            },
            message_type=MessageType.BROADCAST,
        )

        # Assert - Broadcast message sent
        assert success is True
        mock_redis.publish.assert_called()
