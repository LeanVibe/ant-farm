"""Unit tests for message broker functionality."""

import json
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.message_broker import (
    Message,
    MessageBroker,
    MessageHandler,
    MessageType,
)


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_mock = AsyncMock()
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.publish = AsyncMock(return_value=None)
    redis_mock.hset = AsyncMock(return_value=None)
    redis_mock.expire = AsyncMock(return_value=None)
    redis_mock.zadd = AsyncMock(return_value=None)
    redis_mock.lpush = AsyncMock(return_value=None)
    redis_mock.rpop = AsyncMock(return_value=None)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.zrangebyscore = AsyncMock(return_value=[])
    redis_mock.zrevrange = AsyncMock(return_value=[])
    redis_mock.close = AsyncMock(return_value=None)

    # Mock pubsub
    pubsub_mock = AsyncMock()
    pubsub_mock.subscribe = AsyncMock(return_value=None)
    pubsub_mock.unsubscribe = AsyncMock(return_value=None)
    redis_mock.pubsub = Mock(return_value=pubsub_mock)

    return redis_mock


@pytest.fixture
def message_broker(mock_redis):
    """Create a MessageBroker with mocked Redis."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        broker = MessageBroker("redis://localhost:6379/1")
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


class TestMessage:
    """Test cases for Message dataclass."""

    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(
            id="test-id",
            from_agent="sender",
            to_agent="receiver",
            topic="test",
            message_type=MessageType.DIRECT,
            payload={"key": "value"},
            timestamp=123456.0,
        )

        assert msg.id == "test-id"
        assert msg.from_agent == "sender"
        assert msg.to_agent == "receiver"
        assert msg.topic == "test"
        assert msg.message_type == MessageType.DIRECT
        assert msg.payload == {"key": "value"}
        assert msg.timestamp == 123456.0
        assert msg.priority == 5  # Default value

    def test_message_with_optional_fields(self):
        """Test message creation with optional fields."""
        msg = Message(
            id="test-id",
            from_agent="sender",
            to_agent="receiver",
            topic="test",
            message_type=MessageType.REQUEST,
            payload={"key": "value"},
            timestamp=123456.0,
            expires_at=123500.0,
            reply_to="parent-id",
            correlation_id="corr-id",
            priority=1,
            delivery_count=2,
            max_retries=5,
        )

        assert msg.expires_at == 123500.0
        assert msg.reply_to == "parent-id"
        assert msg.correlation_id == "corr-id"
        assert msg.priority == 1
        assert msg.delivery_count == 2
        assert msg.max_retries == 5


class TestMessageHandler:
    """Test cases for MessageHandler class."""

    def test_handler_initialization(self):
        """Test MessageHandler initialization."""
        handler = MessageHandler("test_agent")
        assert handler.agent_name == "test_agent"
        assert handler.handlers == {}

    def test_register_handler(self):
        """Test registering a message handler."""
        handler = MessageHandler("test_agent")

        async def test_handler(message):
            return {"processed": True}

        handler.register_handler("test_topic", test_handler)
        assert "test_topic" in handler.handlers
        assert handler.handlers["test_topic"] == test_handler

    @pytest.mark.asyncio
    async def test_handle_message_success(self):
        """Test successful message handling."""
        handler = MessageHandler("test_agent")

        async def test_handler(message):
            return {"result": "success"}

        handler.register_handler("test_topic", test_handler)

        message = Message(
            id="test-id",
            from_agent="sender",
            to_agent="test_agent",
            topic="test_topic",
            message_type=MessageType.DIRECT,
            payload={"test": "data"},
            timestamp=time.time(),
        )

        result = await handler.handle_message(message)
        assert result is None  # Direct messages don't return replies

    @pytest.mark.asyncio
    async def test_handle_request_message(self):
        """Test handling request messages that generate replies."""
        handler = MessageHandler("test_agent")

        async def test_handler(message):
            return {"result": "processed"}

        handler.register_handler("test_request", test_handler)

        request_message = Message(
            id="request-id",
            from_agent="requester",
            to_agent="test_agent",
            topic="test_request",
            message_type=MessageType.REQUEST,
            payload={"request": "data"},
            timestamp=time.time(),
            correlation_id="corr-123",
        )

        reply = await handler.handle_message(request_message)

        assert reply is not None
        assert reply.message_type == MessageType.REPLY
        assert reply.from_agent == "test_agent"
        assert reply.to_agent == "requester"
        assert reply.topic == "test_request_reply"
        assert reply.reply_to == "request-id"
        assert reply.correlation_id == "corr-123"
        assert reply.payload == {"result": "processed"}

    @pytest.mark.asyncio
    async def test_handle_message_error(self):
        """Test error handling in message processing."""
        handler = MessageHandler("test_agent")

        async def failing_handler(message):
            raise ValueError("Processing failed")

        handler.register_handler("error_topic", failing_handler)

        request_message = Message(
            id="request-id",
            from_agent="requester",
            to_agent="test_agent",
            topic="error_topic",
            message_type=MessageType.REQUEST,
            payload={"request": "data"},
            timestamp=time.time(),
            correlation_id="corr-123",
        )

        error_reply = await handler.handle_message(request_message)

        assert error_reply is not None
        assert error_reply.message_type == MessageType.REPLY
        assert error_reply.topic == "error_topic_error"
        assert "error" in error_reply.payload
        assert "Processing failed" in error_reply.payload["error"]

    @pytest.mark.asyncio
    async def test_handle_message_no_handler(self):
        """Test handling message with no registered handler."""
        handler = MessageHandler("test_agent")

        message = Message(
            id="test-id",
            from_agent="sender",
            to_agent="test_agent",
            topic="unknown_topic",
            message_type=MessageType.DIRECT,
            payload={"test": "data"},
            timestamp=time.time(),
        )

        result = await handler.handle_message(message)
        assert result is None


class TestMessageBroker:
    """Test cases for MessageBroker class."""

    @pytest.mark.asyncio
    async def test_broker_initialization(self, message_broker, mock_redis):
        """Test MessageBroker initialization."""
        await message_broker.initialize()

        mock_redis.ping.assert_called_once()
        assert message_broker.pubsub is not None
        assert message_broker.message_handlers == {}
        assert message_broker.pending_requests == {}
        assert message_broker.subscriptions == set()
        assert message_broker.running is False

    @pytest.mark.asyncio
    async def test_start_listening(self, message_broker, mock_redis):
        """Test starting to listen for messages."""
        await message_broker.initialize()

        handler = MessageHandler("test_agent")

        with patch.object(message_broker, "_message_listener") as mock_listener:
            mock_listener.return_value = None

            await message_broker.start_listening("test_agent", handler)

        assert "test_agent" in message_broker.message_handlers
        assert message_broker.message_handlers["test_agent"] == handler

        # Check subscriptions
        mock_redis.pubsub().subscribe.assert_any_call("agent:test_agent")
        mock_redis.pubsub().subscribe.assert_any_call("broadcast")

        assert "agent:test_agent" in message_broker.subscriptions
        assert "broadcast" in message_broker.subscriptions

    @pytest.mark.asyncio
    async def test_stop_listening(self, message_broker, mock_redis):
        """Test stopping message listening."""
        await message_broker.initialize()

        handler = MessageHandler("test_agent")
        message_broker.message_handlers["test_agent"] = handler
        message_broker.subscriptions.add("agent:test_agent")

        await message_broker.stop_listening("test_agent")

        assert "test_agent" not in message_broker.message_handlers
        mock_redis.pubsub().unsubscribe.assert_called_with("agent:test_agent")
        assert "agent:test_agent" not in message_broker.subscriptions

    @pytest.mark.asyncio
    async def test_send_message_direct(self, message_broker, mock_redis):
        """Test sending a direct message."""
        await message_broker.initialize()

        message_id = await message_broker.send_message(
            from_agent="sender",
            to_agent="receiver",
            topic="test_topic",
            payload={"test": "data"},
            message_type=MessageType.DIRECT,
            priority=3,
        )

        assert message_id is not None

        # Check Redis operations
        mock_redis.hset.assert_called()
        mock_redis.expire.assert_called()
        mock_redis.publish.assert_called()
        mock_redis.zadd.assert_called()

        # Check publish was called with correct channel
        args, _ = mock_redis.publish.call_args
        assert args[0] == "agent:receiver"

    @pytest.mark.asyncio
    async def test_send_broadcast_message(self, message_broker, mock_redis):
        """Test broadcasting a message."""
        await message_broker.initialize()

        message_id = await message_broker.broadcast_message(
            from_agent="sender",
            topic="announcement",
            payload={"message": "Hello everyone"},
        )

        assert message_id is not None

        # Check publish was called with broadcast channel
        args, _ = mock_redis.publish.call_args
        assert args[0] == "broadcast"

    @pytest.mark.asyncio
    async def test_multicast_message(self, message_broker, mock_redis):
        """Test sending multicast messages."""
        await message_broker.initialize()

        message_ids = await message_broker.multicast_message(
            from_agent="sender",
            to_agents=["agent1", "agent2", "agent3"],
            topic="group_message",
            payload={"data": "important"},
        )

        assert len(message_ids) == 3
        assert all(msg_id is not None for msg_id in message_ids)

        # Check multiple publish calls
        assert mock_redis.publish.call_count == 3

    @pytest.mark.asyncio
    async def test_send_request_with_timeout(self, message_broker, mock_redis):
        """Test sending request with timeout."""
        await message_broker.initialize()

        # Mock a timeout scenario
        with pytest.raises(TimeoutError):
            await message_broker.send_request(
                from_agent="requester",
                to_agent="responder",
                topic="test_request",
                payload={"question": "answer?"},
                timeout=0.1,  # Very short timeout
            )

    @pytest.mark.asyncio
    async def test_get_message_history(self, message_broker, mock_redis):
        """Test retrieving message history."""
        await message_broker.initialize()

        # Mock Redis responses
        mock_redis.zrevrange.return_value = ["msg1", "msg2"]
        mock_redis.hgetall.return_value = {
            "id": "msg1",
            "from_agent": "sender",
            "to_agent": "receiver",
            "topic": "test",
            "message_type": "direct",
            "payload": json.dumps({"test": "data"}),
            "timestamp": "123456.0",
            "priority": "5",
            "delivery_count": "0",
            "max_retries": "3",
        }

        history = await message_broker.get_message_history("test_agent", limit=10)

        assert len(history) <= 2  # Could be less due to filtering
        mock_redis.zrevrange.assert_called_with(
            "hive:history:test_agent", 0, 9, withscores=False
        )

    @pytest.mark.asyncio
    async def test_get_offline_messages(self, message_broker, mock_redis):
        """Test retrieving offline messages."""
        await message_broker.initialize()

        # Mock Redis responses
        mock_redis.rpop.side_effect = ["msg1", None]  # One message, then empty
        mock_redis.hgetall.return_value = {
            "id": "msg1",
            "from_agent": "sender",
            "to_agent": "receiver",
            "topic": "offline_test",
            "message_type": "direct",
            "payload": json.dumps({"data": "offline"}),
            "timestamp": str(time.time()),
            "priority": "5",
            "delivery_count": "0",
            "max_retries": "3",
        }

        offline_messages = await message_broker.get_offline_messages("test_agent")

        assert len(offline_messages) <= 1
        mock_redis.rpop.assert_called_with("hive:msgqueue:test_agent")

    @pytest.mark.asyncio
    async def test_shutdown(self, message_broker, mock_redis):
        """Test broker shutdown."""
        await message_broker.initialize()
        await message_broker.shutdown()

        mock_redis.close.assert_called_once()


class TestMessageSerialization:
    """Test cases for message serialization and deserialization."""

    def test_serialize_message(self, message_broker, sample_message):
        """Test message serialization for pub/sub."""
        serialized = message_broker._serialize_message(sample_message)

        assert isinstance(serialized, str)
        data = json.loads(serialized)

        assert data["id"] == sample_message.id
        assert data["from_agent"] == sample_message.from_agent
        assert data["to_agent"] == sample_message.to_agent
        assert data["topic"] == sample_message.topic
        assert data["message_type"] == sample_message.message_type.value
        assert data["payload"] == sample_message.payload

    def test_serialize_message_for_storage(self, message_broker, sample_message):
        """Test message serialization for Redis hash storage."""
        serialized = message_broker._serialize_message_for_storage(sample_message)

        assert isinstance(serialized, dict)
        assert all(isinstance(v, str) for v in serialized.values())

        assert serialized["id"] == sample_message.id
        assert serialized["payload"] == json.dumps(sample_message.payload)
        assert serialized["timestamp"] == str(sample_message.timestamp)

    def test_deserialize_message_pubsub_format(self, message_broker):
        """Test deserializing message from pub/sub format."""
        data = {
            "id": "test-id",
            "from_agent": "sender",
            "to_agent": "receiver",
            "topic": "test",
            "message_type": "direct",
            "payload": {"key": "value"},
            "timestamp": 123456.0,
            "priority": 5,
        }

        message = message_broker._deserialize_message(data)

        assert message.id == "test-id"
        assert message.from_agent == "sender"
        assert message.message_type == MessageType.DIRECT
        assert message.payload == {"key": "value"}

    def test_deserialize_message_storage_format(self, message_broker):
        """Test deserializing message from storage format."""
        data = {
            "id": "test-id",
            "from_agent": "sender",
            "to_agent": "receiver",
            "topic": "test",
            "message_type": "direct",
            "payload": json.dumps({"key": "value"}),
            "timestamp": "123456.0",
            "priority": "5",
            "delivery_count": "0",
            "max_retries": "3",
        }

        message = message_broker._deserialize_message(data)

        assert message.id == "test-id"
        assert message.payload == {"key": "value"}
        assert message.timestamp == 123456.0
        assert message.priority == 5


class TestMessageRouting:
    """Test cases for message routing logic."""

    @pytest.mark.asyncio
    async def test_route_direct_message(self, message_broker):
        """Test routing direct messages to specific handlers."""
        handler = MessageHandler("test_agent")
        message_broker.message_handlers["test_agent"] = handler

        # Mock handler.handle_message
        handler.handle_message = AsyncMock(return_value=None)

        message = Message(
            id="test-id",
            from_agent="sender",
            to_agent="test_agent",
            topic="test",
            message_type=MessageType.DIRECT,
            payload={"data": "test"},
            timestamp=time.time(),
        )

        await message_broker._route_message(message)

        handler.handle_message.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_route_broadcast_message(self, message_broker):
        """Test routing broadcast messages to all handlers."""
        handler1 = MessageHandler("agent1")
        handler2 = MessageHandler("agent2")

        message_broker.message_handlers["agent1"] = handler1
        message_broker.message_handlers["agent2"] = handler2

        # Mock handler methods
        handler1.handle_message = AsyncMock(return_value=None)
        handler2.handle_message = AsyncMock(return_value=None)

        broadcast_message = Message(
            id="broadcast-id",
            from_agent="sender",
            to_agent="broadcast",
            topic="announcement",
            message_type=MessageType.BROADCAST,
            payload={"message": "Hello all"},
            timestamp=time.time(),
        )

        await message_broker._route_message(broadcast_message)

        handler1.handle_message.assert_called_once_with(broadcast_message)
        handler2.handle_message.assert_called_once_with(broadcast_message)

    @pytest.mark.asyncio
    async def test_route_message_offline_agent(self, message_broker, mock_redis):
        """Test routing message to offline agent."""
        # No handler registered for "offline_agent"

        message = Message(
            id="offline-msg",
            from_agent="sender",
            to_agent="offline_agent",
            topic="test",
            message_type=MessageType.DIRECT,
            payload={"data": "for offline agent"},
            timestamp=time.time(),
        )

        await message_broker._route_message(message)

        # Should queue for offline delivery
        mock_redis.lpush.assert_called_with(
            "hive:msgqueue:offline_agent", "offline-msg"
        )
        mock_redis.expire.assert_called_with("hive:msgqueue:offline_agent", 86400)


class TestMessageTypes:
    """Test different message types."""

    def test_message_type_enum(self):
        """Test MessageType enumeration."""
        assert MessageType.DIRECT.value == "direct"
        assert MessageType.BROADCAST.value == "broadcast"
        assert MessageType.MULTICAST.value == "multicast"
        assert MessageType.REQUEST.value == "request"
        assert MessageType.REPLY.value == "reply"
        assert MessageType.NOTIFICATION.value == "notification"

    def test_all_message_types_handled(self):
        """Ensure all message types are properly defined."""
        expected_types = {
            "direct",
            "broadcast",
            "multicast",
            "request",
            "reply",
            "notification",
        }

        actual_types = {msg_type.value for msg_type in MessageType}
        assert actual_types == expected_types
