"""Redis-based message broker for inter-agent communication."""

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import redis.asyncio as redis
import structlog

logger = structlog.get_logger()
from .contracts import BrokerSendResult, BrokerSendReason
from .circuit_breaker import get_message_broker_breaker, CircuitBreakerError

# Circuit breaker for message broker operations
message_broker_breaker = get_message_broker_breaker()


class MessageType(Enum):
    """Message type enumeration."""

    DIRECT = "direct"  # Agent-to-agent direct message
    BROADCAST = "broadcast"  # Message to all agents
    MULTICAST = "multicast"  # Message to multiple specific agents
    REQUEST = "request"  # Request-reply pattern
    REPLY = "reply"  # Reply to a request
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
    idempotency_key: str | None = None


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
                        correlation_id=message.correlation_id,
                    )
                    return reply

                return None

            except Exception as e:
                logger.error(
                    "Message handler error",
                    agent=self.agent_name,
                    topic=message.topic,
                    error=str(e),
                )

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
                        correlation_id=message.correlation_id,
                    )
                    return error_reply
        else:
            logger.warning(
                "No handler for topic", agent=self.agent_name, topic=message.topic
            )

        return None


class MessageBroker:
    """Redis-based message broker with pub/sub and persistence."""

    def __init__(self, redis_url: str = None, pubsub=None):
        """Initialize MessageBroker.

        Args:
            redis_url: Redis connection URL
            pubsub: Optional pubsub client for dependency injection (testing)
        """
        if redis_url is None:
            from .config import get_settings

            redis_url = get_settings().redis_url
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.pubsub = pubsub
        self.message_handlers: dict[str, MessageHandler] = {}
        self.pending_requests: dict[str, asyncio.Future] = {}
        self.subscriptions: set[str] = set()
        self.running = False

        # Key prefixes
        self.message_prefix = "hive:messages"
        self.queue_prefix = "hive:msgqueue"
        self.dlq_prefix = "hive:dlq"  # Dead letter queue
        self.history_prefix = "hive:history"

        # DLQ metrics tracking
        self._dlq_reasons: dict[str, int] = {}
        self._last_dlq_flush = 0.0
        self._dlq_flush_interval = 60.0  # Flush every 60 seconds
        self._async_db_manager = None  # Will be set when needed

        # Circuit breaker for Redis operations
        self.redis_breaker = get_message_broker_breaker()

    async def initialize(self) -> None:
        """Initialize the message broker."""
        await self.redis_client.ping()
        if self.pubsub is None:
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

        # Also unsubscribe from broadcast channel if no other agents are listening
        broadcast_channel = "broadcast"
        # Check if any other agents are still subscribed to broadcast
        other_agents_subscribed = any(
            channel.startswith("agent:") and channel != agent_channel
            for channel in self.subscriptions
        )
        if broadcast_channel in self.subscriptions and not other_agents_subscribed:
            await self.pubsub.unsubscribe(broadcast_channel)
            self.subscriptions.remove(broadcast_channel)

        logger.info("Stopped listening for messages", agent=agent_name)

    async def send_message(
        self,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: dict[str, Any],
        message_type: MessageType = MessageType.DIRECT,
        priority: int = 5,
        expires_in: int | None = None,
        correlation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> bool:
        """Send a message to an agent or broadcast."""
        try:
            # Idempotency check (best-effort)
            if idempotency_key:
                idem_key = f"{self.message_prefix}:idem:{idempotency_key}"
                # set key with NX and 24h TTL; if already exists, consider success
                try:
                    ok = await self.redis_breaker.call(self.redis_client.set, idem_key, "1", ex=86400, nx=True)
                    if not ok:
                        logger.info(
                            "Idempotent duplicate suppressed",
                            idempotency_key=idempotency_key,
                        )
                        return True
                except Exception as e:
                    logger.warning("Idempotency check failed; continuing", error=str(e))

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
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )

            # Persist message
            await self._persist_message(message)

            # Determine channel
            if to_agent == "broadcast":
                channel = "broadcast"
            else:
                channel = f"agent:{to_agent}"

            # Publish message
            message_dict = {
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
                "max_retries": message.max_retries,
                "idempotency_key": message.idempotency_key,
            }

            # Protect Redis publish with circuit breaker
            await self.redis_breaker.call(
                self.redis_client.publish,
                channel, json.dumps(message_dict, default=str)
            )

            # Store in history
            await self._store_message_history(message)

            logger.info(
                "Message sent",
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                message_id=message.id,
            )

            return True

        except CircuitBreakerError as e:
            logger.warning(
                "Circuit breaker open, failing fast",
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                circuit_breaker=e.service_name,
                state=e.state.value,
            )
            # Don't move to DLQ for circuit breaker failures - they're expected
            return False
        except Exception as e:
            logger.error(
                "Failed to send message",
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                error=str(e),
            )
            # Move to DLQ on publish failure
            try:
                await self._move_to_dlq(
                    Message(
                        id=str(uuid.uuid4()),
                        from_agent=from_agent,
                        to_agent=to_agent,
                        topic=topic,
                        message_type=MessageType.DIRECT,
                        payload=payload,
                        timestamp=time.time(),
                        priority=priority,
                        correlation_id=correlation_id,
                        idempotency_key=idempotency_key,
                    ),
                    reason="publish_error",
                )
            except Exception:
                pass
            return False

    async def send_message_with_result(
        self,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: dict[str, Any],
        message_type: MessageType = MessageType.DIRECT,
        priority: int = 5,
        expires_in: int | None = None,
        correlation_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> BrokerSendResult:
        """Structured send API with explicit result semantics."""
        try:
            # Idempotency check
            if idempotency_key:
                idem_key = f"{self.message_prefix}:idem:{idempotency_key}"
                try:
                    ok = await self.redis_breaker.call(self.redis_client.set, idem_key, "1", ex=86400, nx=True)
                    if not ok:
                        return BrokerSendResult(
                            success=True,
                            reason=BrokerSendReason.IDEMPOTENT_DUPLICATE,
                            message_id=None,
                        )
                except Exception:
                    pass

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
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
            )

            await self._persist_message(message)
            channel = "broadcast" if to_agent == "broadcast" else f"agent:{to_agent}"
            # Protect Redis publish with circuit breaker
            await self.redis_breaker.call(
                self.redis_client.publish,
                channel, json.dumps(self._serialize_message_for_storage(message))
            )
            return BrokerSendResult(
                success=True, reason=BrokerSendReason.OK, message_id=message.id
            )
        except CircuitBreakerError as e:
            logger.warning(
                "Circuit breaker open, failing fast",
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                circuit_breaker=e.service_name,
                state=e.state.value,
            )
            return BrokerSendResult(
                success=False,
                reason=BrokerSendReason.PUBLISH_ERROR,
                message_id=None,
                details=f"Circuit breaker open: {e.service_name}",
            )
        except Exception as e:
            try:
                await self._move_to_dlq(
                    Message(
                        id=str(uuid.uuid4()),
                        from_agent=from_agent,
                        to_agent=to_agent,
                        topic=topic,
                        message_type=MessageType.DIRECT,
                        payload=payload,
                        timestamp=time.time(),
                        priority=priority,
                        correlation_id=correlation_id,
                        idempotency_key=idempotency_key,
                    ),
                    reason="publish_error",
                )
            except Exception:
                pass
            return BrokerSendResult(
                success=False,
                reason=BrokerSendReason.PUBLISH_ERROR,
                message_id=None,
                details=str(e),
            )

    async def send_request(
        self,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: dict[str, Any],
        timeout: int = 30,
    ) -> dict[str, Any]:
        """Send a request and wait for reply."""
        correlation_id = str(uuid.uuid4())

        # Create future for the reply
        reply_future = asyncio.Future()
        self.pending_requests[correlation_id] = reply_future

        try:
            # Send request
            ok = await self.send_message(
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                payload=payload,
                message_type=MessageType.REQUEST,
                correlation_id=correlation_id,
            )
            if not ok:
                raise TimeoutError("failed to send request")

            # Wait for reply
            reply = await asyncio.wait_for(reply_future, timeout=timeout)
            return reply.payload

        except TimeoutError:
            logger.error(
                "Request timeout", from_agent=from_agent, to_agent=to_agent, topic=topic
            )
            raise
        finally:
            # Clean up
            if correlation_id in self.pending_requests:
                del self.pending_requests[correlation_id]

    async def broadcast_message(
        self, from_agent: str, topic: str, payload: dict[str, Any], priority: int = 5
    ) -> str:
        """Broadcast a message to all agents."""

        return await self.send_message(
            from_agent=from_agent,
            to_agent="broadcast",
            topic=topic,
            payload=payload,
            message_type=MessageType.BROADCAST,
            priority=priority,
        )

    async def multicast_message(
        self,
        from_agent: str,
        to_agents: list[str],
        topic: str,
        payload: dict[str, Any],
        priority: int = 5,
    ) -> list[str]:
        """Send a message to multiple specific agents."""

        message_ids = []

        for to_agent in to_agents:
            message_id = await self.send_message(
                from_agent=from_agent,
                to_agent=to_agent,
                topic=topic,
                payload=payload,
                message_type=MessageType.MULTICAST,
                priority=priority,
            )
            message_ids.append(message_id)

        return message_ids

    async def get_message_history(
        self, agent_name: str, limit: int = 100, since: float | None = None
    ) -> list[Message]:
        """Get message history for an agent."""

        history_key = f"{self.history_prefix}:{agent_name}"

        # Get message IDs from sorted set
        if since:
            message_ids = await self.redis_breaker.call(
                self.redis_client.zrangebyscore,
                history_key, since, "+inf", withscores=False
            )
        else:
            message_ids = await self.redis_breaker.call(
                self.redis_client.zrevrange,
                history_key, 0, limit - 1, withscores=False
            )

        # Retrieve message data
        messages = []
        for message_id in message_ids:
            message_key = f"{self.message_prefix}:{message_id}"
            message_data = await self.redis_breaker.call(self.redis_client.hgetall, message_key)

            if message_data:
                message = self._deserialize_message(message_data)
                messages.append(message)

        return messages

    async def get_message_history_with_circuit_breaker(
        self, agent_name: str, limit: int = 100, since: float | None = None
    ) -> list[Message]:
        """Get message history with circuit breaker protection."""
        try:
            return await self.get_message_history(agent_name, limit, since)
        except CircuitBreakerError as e:
            logger.warning(
                "Circuit breaker open for message history",
                agent=agent_name,
                circuit_breaker=e.service_name,
                state=e.state.value,
            )
            return []  # Return empty list when circuit breaker is open

    async def get_offline_messages(self, agent_name: str) -> list[Message]:
        """Get messages that arrived while agent was offline."""

        queue_key = f"{self.queue_prefix}:{agent_name}"
        offline_messages = []

        # Process all queued messages
        while True:
            message_id = await self.redis_breaker.call(self.redis_client.rpop, queue_key)
            if not message_id:
                break

            message_key = f"{self.message_prefix}:{message_id}"
            message_data = await self.redis_breaker.call(self.redis_client.hgetall, message_key)

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
            else:
                # In tests, pubsub.listen may be an AsyncMock; yield control
                await asyncio.sleep(0)

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

        # Protect Redis operations with circuit breaker
        await self.redis_breaker.call(self.redis_client.hset, message_key, mapping=message_data)
        await self.redis_breaker.call(self.redis_client.expire, message_key, ttl)

    async def _queue_for_offline_delivery(self, message: Message) -> None:
        """Queue message for offline agent."""

        queue_key = f"{self.queue_prefix}:{message.to_agent}"
        # Protect Redis operations with circuit breaker
        await self.redis_breaker.call(self.redis_client.lpush, queue_key, message.id)

        # Set TTL for queue
        await self.redis_breaker.call(self.redis_client.expire, queue_key, 86400)  # 24 hours

        logger.debug(
            "Message queued for offline delivery",
            to_agent=message.to_agent,
            message_id=message.id,
        )

    async def _store_message_history(self, message: Message) -> None:
        """Store message in history for both sender and receiver."""

        # Store for sender
        sender_key = f"{self.history_prefix}:{message.from_agent}"
        await self.redis_breaker.call(self.redis_client.zadd, sender_key, {message.id: message.timestamp})
        await self.redis_breaker.call(self.redis_client.expire, sender_key, 604800)  # 7 days

        # Store for receiver (unless broadcast)
        if message.to_agent != "broadcast":
            receiver_key = f"{self.history_prefix}:{message.to_agent}"
            await self.redis_breaker.call(self.redis_client.zadd, receiver_key, {message.id: message.timestamp})
            await self.redis_breaker.call(self.redis_client.expire, receiver_key, 604800)  # 7 days

    async def _move_to_dlq(self, message: Message, reason: str) -> None:
        """Move message to dead letter queue and track reason counters."""

        dlq_key = f"{self.dlq_prefix}:{message.to_agent}"
        dlq_data = {
            "message_id": message.id,
            "reason": reason,
            "timestamp": time.time(),
        }

        # Protect Redis operations with circuit breaker
        await self.redis_breaker.call(self.redis_client.lpush, dlq_key, json.dumps(dlq_data))
        await self.redis_breaker.call(self.redis_client.expire, dlq_key, 604800)  # 7 days

        logger.warning(
            "Message moved to dead letter queue", message_id=message.id, reason=reason
        )

        # Track DLQ reasons for metrics if feature flag is enabled
        if os.getenv("HIVE_METRICS_FLUSH", "0") == "1":
            self._dlq_reasons[reason] = self._dlq_reasons.get(reason, 0) + 1
            await self._flush_dlq_metrics_if_needed()

    async def _flush_dlq_metrics_if_needed(self) -> None:
        """Flush DLQ metrics if flush interval has passed."""
        current_time = time.time()
        if current_time - self._last_dlq_flush >= self._dlq_flush_interval:
            await self.flush_dlq_metrics()
            self._last_dlq_flush = current_time

    async def flush_dlq_metrics(self) -> None:
        """Flush accumulated DLQ reason counters to database if feature flag is enabled."""
        if not os.getenv("HIVE_METRICS_FLUSH", "0") == "1":
            return

        # Import here to avoid circular imports
        try:
            from .async_db import get_async_database_manager

            db_manager = await get_async_database_manager()
        except Exception:
            logger.debug("Cannot flush DLQ metrics - database manager not available")
            return

        if not self._dlq_reasons:
            return

        try:
            # Convert DLQ reasons to metric format for bulk write
            metrics_to_write = []
            current_time = time.time()

            for reason, count in self._dlq_reasons.items():
                metrics_to_write.append(
                    {
                        "metric_name": f"message_broker_dlq_total",
                        "metric_type": "counter",
                        "value": float(count),
                        "unit": "count",
                        "labels": {
                            "reason": reason,
                        },
                    }
                )

            if metrics_to_write:
                written_count = await db_manager.record_metrics_bulk(metrics_to_write)
                logger.debug(
                    "Flushed DLQ metrics to database",
                    metrics_count=written_count,
                )

                # Clear counters after successful flush
                self._dlq_reasons.clear()

        except Exception as e:
            logger.error("Failed to flush DLQ metrics", error=str(e))

    def get_dlq_counters(self) -> dict[str, int]:
        """Get current DLQ reason counters."""
        return self._dlq_reasons.copy()

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
            "max_retries": message.max_retries,
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
            "max_retries": str(message.max_retries),
        }

        # Optional fields
        if message.expires_at:
            data["expires_at"] = str(message.expires_at)
        if message.reply_to:
            data["reply_to"] = message.reply_to
        if message.correlation_id:
            data["correlation_id"] = message.correlation_id
        if message.idempotency_key:
            data["idempotency_key"] = message.idempotency_key

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
            max_retries=int(data.get("max_retries", 3)),
            idempotency_key=data.get("idempotency_key"),
        )

    def increment_dlq_reason(self, reason: str) -> None:
        """Increment DLQ reason counter."""
        self._dlq_reasons[reason] = self._dlq_reasons.get(reason, 0) + 1

    async def flush_dlq_metrics(self, async_db_manager, agent_uuid: str, agent_name: str) -> None:
        """Flush DLQ metrics to database if feature flag is enabled."""
        import os
        if not os.getenv("HIVE_METRICS_FLUSH", "0") == "1":
            return

        if not async_db_manager or not agent_uuid:
            logger.debug("Cannot flush DLQ metrics - database manager or agent UUID not available")
            return

        try:
            # Convert DLQ reasons to metric format for bulk write
            metrics_to_write = []
            current_time = time.time()

            for reason, count in self._dlq_reasons.items():
                metrics_to_write.append(
                    {
                        "metric_name": "message_broker_dlq_reasons",
                        "metric_type": "counter",
                        "value": float(count),
                        "unit": "count",
                        "agent_id": agent_uuid,
                        "labels": {
                            "reason": reason,
                            "agent_name": agent_name,
                        },
                    }
                )

            if metrics_to_write:
                written_count = await async_db_manager.record_metrics_bulk(metrics_to_write)
                logger.debug(
                    "Flushed DLQ metrics to database",
                    agent=agent_name,
                    metrics_count=written_count,
                )

                # Clear counters after successful flush
                self._dlq_reasons.clear()

        except Exception as e:
            logger.error("Failed to flush DLQ metrics", agent=agent_name, error=str(e))

    async def shutdown(self) -> None:
        """Shutdown the message broker and close connections."""
        try:
            # Cancel background tasks
            if hasattr(self, "_cleanup_task") and self._cleanup_task:
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()

            logger.info("Message broker shutdown complete")
        except Exception as e:
            logger.error(f"Error during message broker shutdown: {str(e)}")


# Global message broker instance
message_broker = MessageBroker()


def get_message_broker() -> MessageBroker:
    """Get a fresh message broker instance with current config."""
    global message_broker
    from .config import get_settings

    current_redis_url = get_settings().redis_url

    # Check if broker needs to be recreated with new config
    if hasattr(message_broker, "redis_client"):
        broker_url = message_broker.redis_client.connection_pool.connection_kwargs
        broker_port = broker_url.get("port", 6379)
        current_port = (
            int(current_redis_url.split(":")[-1]) if ":" in current_redis_url else 6379
        )

        if broker_port != current_port:
            message_broker = MessageBroker()

    return message_broker


def get_message_broker_status() -> dict[str, Any]:
    """Get message broker status including circuit breaker information."""
    broker = get_message_broker()
    return {
        "redis_connected": broker.redis_client is not None,
        "circuit_breaker": broker.redis_breaker.get_status() if hasattr(broker, 'redis_breaker') else None,
        "active_subscriptions": len(broker.subscriptions),
        "pending_requests": len(broker.pending_requests),
        "dlq_metrics": broker._dlq_reasons.copy(),
    }
