"""Redis-based message broker for inter-agent communication."""

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import redis.asyncio as redis
import structlog

logger = structlog.get_logger()


class MessageType(Enum):
    """Message type enumeration."""
    DIRECT = "direct"         # Agent-to-agent direct message
    BROADCAST = "broadcast"   # Message to all agents
    MULTICAST = "multicast"   # Message to multiple specific agents
    REQUEST = "request"       # Request-reply pattern
    REPLY = "reply"          # Reply to a request
    NOTIFICATION = "notification"  # System notifications


@dataclass
class Message:
    """Message structure."""
    id: str
    from_agent: str
    to_agent: str  # Can be "broadcast" or specific agent
    topic: str
    message_type: MessageType
    payload: dict[str, Any]
    timestamp: float
    expires_at: float | None = None
    reply_to: str | None = None
    correlation_id: str | None = None
    priority: int = 5  # 1=highest, 9=lowest
    delivery_count: int = 0
    max_retries: int = 3


class MessageHandler:
    """Base class for message handlers."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.handlers: dict[str, Callable] = {}

    def register_handler(self, topic: str, handler: Callable):
        """Register a handler for a specific topic."""
        self.handlers[topic] = handler
        logger.debug("Handler registered", agent=self.agent_name, topic=topic)

    async def handle_message(self, message: Message) -> Message | None:
        """Handle an incoming message."""
        handler = self.handlers.get(message.topic)
        if handler:
            try:
                result = await handler(message)

                # If this was a request, prepare reply
                if message.message_type == MessageType.REQUEST:
                    reply = Message(
                        id=str(uuid.uuid4()),
                        from_agent=self.agent_name,
                        to_agent=message.from_agent,
                        topic=f"{message.topic}_reply",
                        message_type=MessageType.REPLY,
                        payload=result or {},
                        timestamp=time.time(),
                        reply_to=message.id,
                        correlation_id=message.correlation_id
                    )
                    return reply

                return None

            except Exception as e:
                logger.error("Message handler error",
                           agent=self.agent_name,
                           topic=message.topic,
                           error=str(e))

                # Send error reply for requests
                if message.message_type == MessageType.REQUEST:
                    error_reply = Message(
                        id=str(uuid.uuid4()),
                        from_agent=self.agent_name,
                        to_agent=message.from_agent,
                        topic=f"{message.topic}_error",
                        message_type=MessageType.REPLY,
                        payload={"error": str(e)},
                        timestamp=time.time(),
                        reply_to=message.id,
                        correlation_id=message.correlation_id
                    )
                    return error_reply
        else:
            logger.warning("No handler for topic",
                         agent=self.agent_name,
                         topic=message.topic)

        return None


class MessageBroker:
    """Redis-based message broker with pub/sub and persistence."""

    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.pubsub = None
        self.message_handlers: dict[str, MessageHandler] = {}
        self.pending_requests: dict[str, asyncio.Future] = {}
        self.subscriptions: set[str] = set()
        self.running = False

        # Key prefixes
        self.message_prefix = "hive:messages"
        self.queue_prefix = "hive:msgqueue"
        self.dlq_prefix = "hive:dlq"  # Dead letter queue
        self.history_prefix = "hive:history"

    async def initialize(self) -> None:
        """Initialize the message broker."""
        await self.redis_client.ping()
        self.pubsub = self.redis_client.pubsub()
        logger.info("Message broker initialized")

    async def start_listening(self, agent_name: str, handler: MessageHandler) -> None:
        """Start listening for messages for a specific agent."""
        self.message_handlers[agent_name] = handler

        # Subscribe to agent-specific channel
        agent_channel = f"agent:{agent_name}"
        await self.pubsub.subscribe(agent_channel)
        self.subscriptions.add(agent_channel)

        # Subscribe to broadcast channel
        broadcast_channel = "broadcast"
        await self.pubsub.subscribe(broadcast_channel)
        self.subscriptions.add(broadcast_channel)

        logger.info("Started listening for messages", agent=agent_name)

        if not self.running:
            self.running = True
            asyncio.create_task(self._message_listener())

    async def stop_listening(self, agent_name: str) -> None:
        """Stop listening for messages for an agent."""
        if agent_name in self.message_handlers:
            del self.message_handlers[agent_name]

        agent_channel = f"agent:{agent_name}"
        if agent_channel in self.subscriptions:
            await self.pubsub.unsubscribe(agent_channel)
            self.subscriptions.remove(agent_channel)

        logger.info("Stopped listening for messages", agent=agent_name)

    async def send_message(self,
                          from_agent: str,
                          to_agent: str,
                          topic: str,
                          payload: dict[str, Any],
                          message_type: MessageType = MessageType.DIRECT,
                          priority: int = 5,
                          expires_in: int | None = None,
                          correlation_id: str | None = None) -> str:
        """Send a message to an agent or broadcast."""

        message = Message(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            topic=topic,
            message_type=message_type,
            payload=payload,
            timestamp=time.time(),
            expires_at=time.time() + expires_in if expires_in else None,
            priority=priority,
            correlation_id=correlation_id
        )

        # Persist message
        await self._persist_message(message)

        # Determine channel
        if to_agent == "broadcast":
            channel = "broadcast"
        else:
            channel = f"agent:{to_agent}"

        # Publish message
        message_data = self._serialize_message(message)
        await self.redis_client.publish(channel, message_data)

        # Store in message history
        await self._store_message_history(message)

        logger.info("Message sent",
                   from_agent=from_agent,
                   to_agent=to_agent,
                   topic=topic,
                   message_id=message.id)

        return message.id

    async def send_request(self,
                          from_agent: str,
                          to_agent: str,
                          topic: str,
                          payload: dict[str, Any],
                          timeout: int = 30) -> dict[str, Any]:
        """Send a request and wait for reply."""

        correlation_id = str(uuid.uuid4())

        # Create future for the reply
        reply_future = asyncio.Future()
        self.pending_requests[correlation_id] = reply_future

        try:
            # Send request
            await self.send_message(
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                payload=payload,
                message_type=MessageType.REQUEST,
                correlation_id=correlation_id
            )

            # Wait for reply
            reply = await asyncio.wait_for(reply_future, timeout=timeout)
            return reply.payload

        except TimeoutError:
            logger.error("Request timeout",
                        from_agent=from_agent,
                        to_agent=to_agent,
                        topic=topic)
            raise
        finally:
            # Clean up
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]

    async def broadcast_message(self,
                               from_agent: str,
                               topic: str,
                               payload: dict[str, Any],
                               priority: int = 5) -> str:
        """Broadcast a message to all agents."""

        return await self.send_message(
            from_agent=from_agent,
            to_agent="broadcast",
            topic=topic,
            payload=payload,
            message_type=MessageType.BROADCAST,
            priority=priority
        )

    async def multicast_message(self,
                               from_agent: str,
                               to_agents: list[str],
                               topic: str,
                               payload: dict[str, Any],
                               priority: int = 5) -> list[str]:
        """Send a message to multiple specific agents."""

        message_ids = []

        for to_agent in to_agents:
            message_id = await self.send_message(
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                payload=payload,
                message_type=MessageType.MULTICAST,
                priority=priority
            )
            message_ids.append(message_id)

        return message_ids

    async def get_message_history(self,
                                 agent_name: str,
                                 limit: int = 100,
                                 since: float | None = None) -> list[Message]:
        """Get message history for an agent."""

        history_key = f"{self.history_prefix}:{agent_name}"

        # Get message IDs from sorted set
        if since:
            message_ids = await self.redis_client.zrangebyscore(
                history_key, since, "+inf", withscores=False
            )
        else:
            message_ids = await self.redis_client.zrevrange(
                history_key, 0, limit - 1, withscores=False
            )

        # Retrieve message data
        messages = []
        for message_id in message_ids:
            message_key = f"{self.message_prefix}:{message_id}"
            message_data = await self.redis_client.hgetall(message_key)

            if message_data:
                message = self._deserialize_message(message_data)
                messages.append(message)

        return messages

    async def get_offline_messages(self, agent_name: str) -> list[Message]:
        """Get messages that arrived while agent was offline."""

        queue_key = f"{self.queue_prefix}:{agent_name}"
        offline_messages = []

        # Process all queued messages
        while True:
            message_id = await self.redis_client.rpop(queue_key)
            if not message_id:
                break

            message_key = f"{self.message_prefix}:{message_id}"
            message_data = await self.redis_client.hgetall(message_key)

            if message_data:
                message = self._deserialize_message(message_data)

                # Check if message has expired
                if message.expires_at and time.time() > message.expires_at:
                    await self._move_to_dlq(message, "expired")
                    continue

                offline_messages.append(message)

        return offline_messages

    async def _message_listener(self) -> None:
        """Main message listening loop."""

        async for message in self.pubsub.listen():
            if message["type"] == "message":
                try:
                    # Deserialize message
                    message_data = json.loads(message["data"])
                    msg = self._deserialize_message(message_data)

                    # Handle replies
                    if msg.message_type == MessageType.REPLY and msg.correlation_id:
                        if msg.correlation_id in self.pending_requests:
                            future = self.pending_requests[msg.correlation_id]
                            if not future.done():
                                future.set_result(msg)
                            continue

                    # Route to appropriate handler
                    await self._route_message(msg)

                except Exception as e:
                    logger.error("Message processing error", error=str(e))

    async def _route_message(self, message: Message) -> None:
        """Route message to appropriate handler."""

        # For broadcast messages, send to all handlers
        if message.message_type == MessageType.BROADCAST:
            for agent_name, handler in self.message_handlers.items():
                try:
                    reply = await handler.handle_message(message)
                    if reply:
                        await self._send_reply(reply)
                except Exception as e:
                    logger.error("Handler error", agent=agent_name, error=str(e))

        # For direct messages, send to specific handler
        elif message.to_agent in self.message_handlers:
            handler = self.message_handlers[message.to_agent]
            try:
                reply = await handler.handle_message(message)
                if reply:
                    await self._send_reply(reply)
            except Exception as e:
                logger.error("Handler error", agent=message.to_agent, error=str(e))

        # If handler not available, queue for offline delivery
        else:
            await self._queue_for_offline_delivery(message)

    async def _send_reply(self, reply: Message) -> None:
        """Send a reply message."""

        await self._persist_message(reply)

        # Determine channel
        channel = f"agent:{reply.to_agent}"

        # Publish reply
        message_data = self._serialize_message(reply)
        await self.redis_client.publish(channel, message_data)

        # Store in history
        await self._store_message_history(reply)

    async def _persist_message(self, message: Message) -> None:
        """Persist message to Redis."""

        message_key = f"{self.message_prefix}:{message.id}"
        message_data = self._serialize_message_for_storage(message)

        # Store message with TTL
        ttl = 86400  # 24 hours default
        if message.expires_at:
            ttl = max(1, int(message.expires_at - time.time()))

        await self.redis_client.hset(message_key, mapping=message_data)
        await self.redis_client.expire(message_key, ttl)

    async def _queue_for_offline_delivery(self, message: Message) -> None:
        """Queue message for offline agent."""

        queue_key = f"{self.queue_prefix}:{message.to_agent}"
        await self.redis_client.lpush(queue_key, message.id)

        # Set TTL for queue
        await self.redis_client.expire(queue_key, 86400)  # 24 hours

        logger.debug("Message queued for offline delivery",
                    to_agent=message.to_agent,
                    message_id=message.id)

    async def _store_message_history(self, message: Message) -> None:
        """Store message in history for both sender and receiver."""

        # Store for sender
        sender_key = f"{self.history_prefix}:{message.from_agent}"
        await self.redis_client.zadd(sender_key, {message.id: message.timestamp})
        await self.redis_client.expire(sender_key, 604800)  # 7 days

        # Store for receiver (unless broadcast)
        if message.to_agent != "broadcast":
            receiver_key = f"{self.history_prefix}:{message.to_agent}"
            await self.redis_client.zadd(receiver_key, {message.id: message.timestamp})
            await self.redis_client.expire(receiver_key, 604800)  # 7 days

    async def _move_to_dlq(self, message: Message, reason: str) -> None:
        """Move message to dead letter queue."""

        dlq_key = f"{self.dlq_prefix}:{message.to_agent}"
        dlq_data = {
            "message_id": message.id,
            "reason": reason,
            "timestamp": time.time()
        }

        await self.redis_client.lpush(dlq_key, json.dumps(dlq_data))
        await self.redis_client.expire(dlq_key, 604800)  # 7 days

        logger.warning("Message moved to dead letter queue",
                      message_id=message.id,
                      reason=reason)

    def _serialize_message(self, message: Message) -> str:
        """Serialize message for Redis pub/sub."""

        data = {
            "id": message.id,
            "from_agent": message.from_agent,
            "to_agent": message.to_agent,
            "topic": message.topic,
            "message_type": message.message_type.value,
            "payload": message.payload,
            "timestamp": message.timestamp,
            "expires_at": message.expires_at,
            "reply_to": message.reply_to,
            "correlation_id": message.correlation_id,
            "priority": message.priority,
            "delivery_count": message.delivery_count,
            "max_retries": message.max_retries
        }

        return json.dumps(data)

    def _serialize_message_for_storage(self, message: Message) -> dict[str, str]:
        """Serialize message for Redis hash storage."""

        data = {
            "id": message.id,
            "from_agent": message.from_agent,
            "to_agent": message.to_agent,
            "topic": message.topic,
            "message_type": message.message_type.value,
            "payload": json.dumps(message.payload),
            "timestamp": str(message.timestamp),
            "priority": str(message.priority),
            "delivery_count": str(message.delivery_count),
            "max_retries": str(message.max_retries)
        }

        # Optional fields
        if message.expires_at:
            data["expires_at"] = str(message.expires_at)
        if message.reply_to:
            data["reply_to"] = message.reply_to
        if message.correlation_id:
            data["correlation_id"] = message.correlation_id

        return data

    def _deserialize_message(self, data: dict[str, Any]) -> Message:
        """Deserialize message from Redis data."""

        # Handle both pub/sub format and storage format
        if isinstance(data.get("payload"), str):
            payload = json.loads(data["payload"])
        else:
            payload = data["payload"]

        message_type = MessageType(data["message_type"])

        return Message(
            id=data["id"],
            from_agent=data["from_agent"],
            to_agent=data["to_agent"],
            topic=data["topic"],
            message_type=message_type,
            payload=payload,
            timestamp=float(data["timestamp"]),
            expires_at=float(data["expires_at"]) if data.get("expires_at") else None,
            reply_to=data.get("reply_to"),
            correlation_id=data.get("correlation_id"),
            priority=int(data.get("priority", 5)),
            delivery_count=int(data.get("delivery_count", 0)),
            max_retries=int(data.get("max_retries", 3))
        )


# Global message broker instance
message_broker = MessageBroker()
