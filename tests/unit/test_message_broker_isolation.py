"""Comprehensive isolation tests for MessageBroker component.

These tests ensure the MessageBroker operates correctly in complete isolation
from Redis and other external dependencies. All Redis operations are mocked
to verify component behavior without external system dependencies.
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List

import pytest
import structlog

from src.core.message_broker import Message, MessageBroker, MessageHandler, MessageType
from tests.unit.test_component_isolation_framework import (
    ComponentIsolationTestFramework,
    ComponentTestConfig,
    create_isolated_message_broker,
)

logger = structlog.get_logger()


class TestMessageBrokerIsolation:
    """Comprehensive isolation tests for MessageBroker."""

    @pytest.fixture
    async def isolated_broker(self):
        """Fixture providing an isolated MessageBroker."""
        broker, framework = await create_isolated_message_broker()
        yield broker, framework

    @pytest.mark.asyncio
    async def test_broker_initialization_isolation(self, isolated_broker):
        """Test broker initialization without external Redis dependency."""
        broker, framework = isolated_broker

        # Initialize broker - should only interact with mocked Redis
        await broker.initialize()

        # Verify initialization succeeded
        assert broker.pubsub is not None
        assert broker.running is False  # Not yet listening

        # Verify only expected Redis interactions
        framework.assert_redis_interactions([{"method": "ping"}])

        # Ensure no external calls leaked
        framework.assert_no_external_calls()

    @pytest.mark.asyncio
    async def test_send_message_isolation(self, isolated_broker):
        """Test message sending in complete isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send a direct message
        success = await broker.send_message(
            from_agent="test_sender",
            to_agent="test_receiver",
            topic="test_topic",
            payload={"data": "test_value", "number": 42},
            message_type=MessageType.DIRECT,
            priority=3,
        )

        # Verify send succeeded
        assert success is True

        # Verify expected Redis operations
        redis_interactions = [
            i
            for i in framework.mock_redis_client.interactions
            if i.component == "redis"
        ]

        # Should have: hset (persist), expire (TTL), publish (send)
        assert len(redis_interactions) >= 3

        methods_called = [i.method for i in redis_interactions]
        assert "hset" in methods_called  # Message persistence
        assert "expire" in methods_called  # TTL setting
        assert "publish" in methods_called  # Message publishing

        # Verify message was published to correct channel
        publish_calls = [i for i in redis_interactions if i.method == "publish"]
        assert len(publish_calls) == 1

        channel, message_data = publish_calls[0].args
        assert channel == "agent:test_receiver"

        # Verify message structure
        published_message = json.loads(message_data)
        assert published_message["from_agent"] == "test_sender"
        assert published_message["to_agent"] == "test_receiver"
        assert published_message["topic"] == "test_topic"
        assert published_message["message_type"] == "direct"
        assert published_message["payload"]["data"] == "test_value"
        assert published_message["payload"]["number"] == 42
        assert published_message["priority"] == 3

    @pytest.mark.asyncio
    async def test_broadcast_message_isolation(self, isolated_broker):
        """Test broadcast message functionality in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send broadcast message
        success = await broker.broadcast_message(
            from_agent="broadcaster",
            topic="system_announcement",
            payload={"announcement": "System maintenance at 2 AM"},
            priority=1,
        )

        # Verify broadcast succeeded
        assert success is True

        # Verify published to broadcast channel
        redis_interactions = framework.mock_redis_client.interactions
        publish_calls = [i for i in redis_interactions if i.method == "publish"]
        assert len(publish_calls) == 1

        channel, message_data = publish_calls[0].args
        assert channel == "broadcast"

        published_message = json.loads(message_data)
        assert published_message["to_agent"] == "broadcast"
        assert published_message["message_type"] == "broadcast"

    @pytest.mark.asyncio
    async def test_request_reply_pattern_isolation(self, isolated_broker):
        """Test request-reply messaging pattern in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Setup a mock handler for the target agent
        handler = MessageHandler("target_agent")

        async def request_handler(message: Message) -> dict[str, Any]:
            return {
                "status": "processed",
                "data": f"Handled: {message.payload.get('request')}",
            }

        handler.register_handler("process_request", request_handler)

        # Start listening (this should not make external calls in isolation)
        await broker.start_listening("target_agent", handler)

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send request (this will timeout in isolation, but we can test the send part)
        try:
            # Use a very short timeout for testing
            response = await asyncio.wait_for(
                broker.send_request(
                    from_agent="requester",
                    to_agent="target_agent",
                    topic="process_request",
                    payload={"request": "test_data"},
                    timeout=0.1,  # Very short timeout
                ),
                timeout=0.2,
            )
        except TimeoutError:
            # Expected in isolation since no real message delivery
            pass

        # Verify request was sent correctly
        redis_interactions = framework.mock_redis_client.interactions
        publish_calls = [i for i in redis_interactions if i.method == "publish"]
        assert len(publish_calls) >= 1

        # Find the request message
        request_message = None
        for call in publish_calls:
            channel, message_data = call.args
            message = json.loads(message_data)
            if message.get("message_type") == "request":
                request_message = message
                break

        assert request_message is not None
        assert request_message["topic"] == "process_request"
        assert request_message["correlation_id"] is not None

    @pytest.mark.asyncio
    async def test_message_persistence_isolation(self, isolated_broker):
        """Test message persistence functionality in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send message with TTL
        await broker.send_message(
            from_agent="sender",
            to_agent="receiver",
            topic="test_persistence",
            payload={"data": "persistent_data"},
            expires_in=3600,  # 1 hour
        )

        # Verify message was persisted with correct TTL
        redis_interactions = framework.mock_redis_client.interactions

        hset_calls = [i for i in redis_interactions if i.method == "hset"]
        expire_calls = [i for i in redis_interactions if i.method == "expire"]

        assert len(hset_calls) == 1
        assert len(expire_calls) == 1

        # Verify TTL was set correctly
        expire_call = expire_calls[0]
        key, ttl = expire_call.args
        assert key.startswith("hive:messages:")
        assert ttl <= 3600  # Should be <= because of processing time
        assert ttl > 3590  # Should be close to 3600

    @pytest.mark.asyncio
    async def test_multicast_message_isolation(self, isolated_broker):
        """Test multicast messaging in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send multicast message
        target_agents = ["agent1", "agent2", "agent3"]
        message_ids = await broker.multicast_message(
            from_agent="sender",
            to_agents=target_agents,
            topic="team_update",
            payload={"update": "New feature deployed"},
            priority=2,
        )

        # Verify correct number of messages sent
        assert len(message_ids) == len(target_agents)

        # Verify each agent received a message
        redis_interactions = framework.mock_redis_client.interactions
        publish_calls = [i for i in redis_interactions if i.method == "publish"]

        # Should have one publish call per target agent
        assert len(publish_calls) == len(target_agents)

        published_channels = [call.args[0] for call in publish_calls]
        expected_channels = [f"agent:{agent}" for agent in target_agents]

        for expected_channel in expected_channels:
            assert expected_channel in published_channels

    @pytest.mark.asyncio
    async def test_message_history_isolation(self, isolated_broker):
        """Test message history functionality in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Pre-populate mock Redis with some history data
        history_key = "hive:history:test_agent"
        message_id1 = str(uuid.uuid4())
        message_id2 = str(uuid.uuid4())

        # Add messages to sorted set (simulating history)
        await framework.mock_redis_client.zadd(
            history_key, {message_id1: time.time() - 100, message_id2: time.time() - 50}
        )

        # Add message data
        message1_key = f"hive:messages:{message_id1}"
        message2_key = f"hive:messages:{message_id2}"

        await framework.mock_redis_client.hset(
            message1_key,
            {
                "id": message_id1,
                "from_agent": "sender1",
                "to_agent": "test_agent",
                "topic": "old_topic",
                "message_type": "direct",
                "payload": json.dumps({"old": "data"}),
                "timestamp": str(time.time() - 100),
                "priority": "5",
            },
        )

        await framework.mock_redis_client.hset(
            message2_key,
            {
                "id": message_id2,
                "from_agent": "sender2",
                "to_agent": "test_agent",
                "topic": "recent_topic",
                "message_type": "direct",
                "payload": json.dumps({"recent": "data"}),
                "timestamp": str(time.time() - 50),
                "priority": "3",
            },
        )

        # Clear interaction history to focus on get_message_history call
        framework.mock_redis_client.interactions.clear()

        # Get message history
        history = await broker.get_message_history("test_agent", limit=10)

        # Verify history retrieval
        assert len(history) == 2

        # Verify messages are returned in correct order (newest first typically)
        assert history[0].topic == "recent_topic"
        assert history[1].topic == "old_topic"

        # Verify expected Redis operations
        redis_interactions = framework.mock_redis_client.interactions
        zrevrange_calls = [i for i in redis_interactions if i.method == "zrevrange"]
        hgetall_calls = [i for i in redis_interactions if i.method == "hgetall"]

        assert len(zrevrange_calls) == 1
        assert len(hgetall_calls) == 2  # One for each message

    @pytest.mark.asyncio
    async def test_offline_message_queue_isolation(self, isolated_broker):
        """Test offline message queuing in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Pre-populate offline message queue
        queue_key = "hive:msgqueue:offline_agent"
        message_id1 = str(uuid.uuid4())
        message_id2 = str(uuid.uuid4())

        # Add messages to queue
        await framework.mock_redis_client.lpush(queue_key, message_id1, message_id2)

        # Add message data
        for msg_id in [message_id1, message_id2]:
            message_key = f"hive:messages:{msg_id}"
            await framework.mock_redis_client.hset(
                message_key,
                {
                    "id": msg_id,
                    "from_agent": "sender",
                    "to_agent": "offline_agent",
                    "topic": "queued_message",
                    "message_type": "direct",
                    "payload": json.dumps({"queued": "data"}),
                    "timestamp": str(time.time()),
                    "priority": "5",
                },
            )

        # Clear interaction history
        framework.mock_redis_client.interactions.clear()

        # Get offline messages
        offline_messages = await broker.get_offline_messages("offline_agent")

        # Verify messages retrieved
        assert len(offline_messages) == 2

        for message in offline_messages:
            assert message.to_agent == "offline_agent"
            assert message.topic == "queued_message"

        # Verify expected Redis operations
        redis_interactions = framework.mock_redis_client.interactions
        rpop_calls = [i for i in redis_interactions if i.method == "rpop"]
        hgetall_calls = [i for i in redis_interactions if i.method == "hgetall"]

        # Should have rpop calls until queue is empty, plus hgetall for each message
        assert len(rpop_calls) >= 2
        assert len(hgetall_calls) == 2

    @pytest.mark.asyncio
    async def test_error_handling_isolation(self, isolated_broker):
        """Test error handling in isolation using simulated Redis errors."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Simulate Redis connection error
        framework.simulate_redis_error("publish", ConnectionError)

        # Attempt to send message - should handle error gracefully
        success = await broker.send_message(
            from_agent="sender",
            to_agent="receiver",
            topic="test_error",
            payload={"data": "test"},
        )

        # Should return False on error
        assert success is False

        # Verify error was handled (no exception raised)
        # The framework should have recorded the failed interaction
        redis_interactions = framework.mock_redis_client.interactions

        # Should still have tried to persist and set TTL before the publish error
        hset_calls = [i for i in redis_interactions if i.method == "hset"]
        assert len(hset_calls) >= 1  # Message persistence should have succeeded

    @pytest.mark.asyncio
    async def test_performance_constraints_isolation(self, isolated_broker):
        """Test that message broker meets performance constraints in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Send multiple messages to test performance
        start_time = time.time()

        for i in range(10):
            await broker.send_message(
                from_agent="perf_test",
                to_agent=f"target_{i}",
                topic="performance_test",
                payload={"index": i, "data": f"message_{i}"},
            )

        end_time = time.time()
        total_duration = end_time - start_time

        # Verify performance constraints
        framework.assert_performance_constraints(
            max_calls=50,  # Should be reasonable for 10 messages
            max_duration=1.0,  # Should complete within 1 second
        )

        # Get detailed performance metrics
        metrics = framework.get_performance_metrics()

        assert metrics["total_interactions"] <= 50
        assert "redis" in metrics["components_used"]

        # Verify average Redis operation time is reasonable
        redis_metrics = metrics["by_component"]["redis"]
        if redis_metrics["call_count"] > 0:
            avg_redis_time = (
                redis_metrics["total_duration"] / redis_metrics["call_count"]
            )
            assert avg_redis_time < 0.1  # Each Redis operation should be < 100ms

    @pytest.mark.asyncio
    async def test_message_expiration_isolation(self, isolated_broker):
        """Test message expiration handling in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Send message with short expiration
        current_time = time.time()
        await broker.send_message(
            from_agent="sender",
            to_agent="receiver",
            topic="expiring_message",
            payload={"data": "will_expire"},
            expires_in=1,  # 1 second
        )

        # Simulate time passing by modifying mock Redis state
        message_id = None
        for interaction in framework.mock_redis_client.interactions:
            if interaction.method == "hset":
                # Extract message ID from the key
                key = interaction.args[0]
                if key.startswith("hive:messages:"):
                    message_id = key.split(":")[-1]
                    break

        assert message_id is not None

        # Manually expire the message in mock Redis
        message_key = f"hive:messages:{message_id}"
        if message_key in framework.mock_redis_client.expirations:
            framework.mock_redis_client.expirations[message_key] = (
                current_time - 1
            )  # Expired

        # Clear interaction history
        framework.mock_redis_client.interactions.clear()

        # Try to retrieve expired message
        message_data = await framework.mock_redis_client.hgetall(message_key)

        # Should return empty dict for expired message
        assert message_data == {}

        # Verify expiration was checked
        redis_interactions = framework.mock_redis_client.interactions
        hgetall_calls = [i for i in redis_interactions if i.method == "hgetall"]
        assert len(hgetall_calls) == 1

    @pytest.mark.asyncio
    async def test_shutdown_isolation(self, isolated_broker):
        """Test broker shutdown in isolation."""
        broker, framework = isolated_broker
        await broker.initialize()

        # Clear previous interactions
        framework.mock_redis_client.interactions.clear()

        # Shutdown broker
        await broker.shutdown()

        # Verify shutdown operations
        redis_interactions = framework.mock_redis_client.interactions
        close_calls = [i for i in redis_interactions if i.method in ["close", "aclose"]]

        # Should have closed Redis connection
        assert len(close_calls) >= 1

        # Verify no external resources were leaked
        framework.assert_no_external_calls()


class TestMessageHandlerIsolation:
    """Test MessageHandler component in isolation."""

    @pytest.mark.asyncio
    async def test_message_handler_isolation(self):
        """Test MessageHandler operates correctly in isolation."""

        # Create handler with no external dependencies
        handler = MessageHandler("test_agent")

        # Register a test handler
        responses = []

        async def test_handler(message: Message) -> dict[str, Any]:
            responses.append(message.payload)
            return {"status": "processed", "data": message.payload.get("data")}

        handler.register_handler("test_topic", test_handler)

        # Create test message
        test_message = Message(
            id=str(uuid.uuid4()),
            from_agent="sender",
            to_agent="test_agent",
            topic="test_topic",
            message_type=MessageType.REQUEST,
            payload={"data": "test_payload"},
            timestamp=time.time(),
        )

        # Handle message
        reply = await handler.handle_message(test_message)

        # Verify handler was called
        assert len(responses) == 1
        assert responses[0]["data"] == "test_payload"

        # Verify reply was generated for request
        assert reply is not None
        assert reply.message_type == MessageType.REPLY
        assert reply.to_agent == "sender"
        assert reply.reply_to == test_message.id
        assert reply.payload["status"] == "processed"

    @pytest.mark.asyncio
    async def test_message_handler_error_isolation(self):
        """Test MessageHandler error handling in isolation."""

        handler = MessageHandler("error_agent")

        # Register handler that raises exception
        async def failing_handler(message: Message) -> dict[str, Any]:
            raise ValueError("Simulated handler error")

        handler.register_handler("error_topic", failing_handler)

        # Create test message
        test_message = Message(
            id=str(uuid.uuid4()),
            from_agent="sender",
            to_agent="error_agent",
            topic="error_topic",
            message_type=MessageType.REQUEST,
            payload={"data": "test"},
            timestamp=time.time(),
        )

        # Handle message - should not raise exception
        reply = await handler.handle_message(test_message)

        # Should get error reply
        assert reply is not None
        assert reply.message_type == MessageType.REPLY
        assert reply.topic == "error_topic_error"
        assert "error" in reply.payload
        assert "Simulated handler error" in reply.payload["error"]


# Performance and stress tests
class TestMessageBrokerPerformanceIsolation:
    """Performance tests for MessageBroker in isolation."""

    @pytest.mark.asyncio
    async def test_high_volume_messaging_isolation(self):
        """Test high-volume messaging performance in isolation."""

        config = ComponentTestConfig(
            record_interactions=False
        )  # Disable for performance
        framework = ComponentIsolationTestFramework(config)

        async with framework.isolate_component(
            lambda: None, redis_url="redis://mock:6379"
        ) as _:
            from src.core.message_broker import MessageBroker

            broker = MessageBroker("redis://mock:6379")
            broker.redis_client = framework.mock_redis_client

            await broker.initialize()

            # Send many messages
            start_time = time.time()
            message_count = 100

            for i in range(message_count):
                await broker.send_message(
                    from_agent="sender",
                    to_agent=f"receiver_{i % 10}",  # 10 different receivers
                    topic="bulk_test",
                    payload={"index": i, "batch": "performance_test"},
                )

            end_time = time.time()
            duration = end_time - start_time

            # Performance assertions
            messages_per_second = message_count / duration
            assert messages_per_second > 50  # Should handle at least 50 messages/second
            assert duration < 10.0  # Should complete within 10 seconds

            logger.info(
                f"Performance test completed: {message_count} messages in {duration:.2f}s "
                f"({messages_per_second:.1f} msg/s)"
            )


if __name__ == "__main__":
    # Run individual test for debugging
    async def run_single_test():
        test_instance = TestMessageBrokerIsolation()
        broker, framework = await create_isolated_message_broker()
        await test_instance.test_send_message_isolation((broker, framework))
        print("Single test completed successfully")

    asyncio.run(run_single_test())
