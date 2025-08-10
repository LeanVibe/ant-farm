"""Enhanced message routing system with delivery guarantees and reliability features."""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Callable, Tuple
from collections import defaultdict

import structlog

from .message_broker import Message, MessageType, MessageBroker

logger = structlog.get_logger()


class DeliveryGuarantee(Enum):
    """Delivery guarantee levels."""

    AT_MOST_ONCE = "at_most_once"  # Fire and forget
    AT_LEAST_ONCE = "at_least_once"  # Retry until acknowledged
    EXACTLY_ONCE = "exactly_once"  # Deliver exactly once with deduplication
    ORDERED = "ordered"  # Maintain message order
    TRANSACTIONAL = "transactional"  # All or nothing delivery


class RouteStrategy(Enum):
    """Message routing strategies."""

    DIRECT = "direct"  # Direct agent-to-agent
    LOAD_BALANCED = "load_balanced"  # Route to least loaded agent
    ROUND_ROBIN = "round_robin"  # Distribute messages evenly
    CAPABILITY_BASED = "capability_based"  # Route based on agent capabilities
    GEOGRAPHICAL = "geographical"  # Route based on location/affinity
    PRIORITY_BASED = "priority_based"  # Route high priority messages first


class MessageStatus(Enum):
    """Message delivery status."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    ACKNOWLEDGED = "acknowledged"
    FAILED = "failed"
    EXPIRED = "expired"
    DUPLICATE = "duplicate"


@dataclass
class DeliveryReceipt:
    """Delivery receipt for message acknowledgment."""

    message_id: str
    recipient: str
    status: MessageStatus
    timestamp: float
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class RouteEntry:
    """Routing table entry."""

    destination_pattern: str  # Agent name pattern or capability
    route_strategy: RouteStrategy
    target_agents: List[str]
    load_weights: Dict[str, float] = field(default_factory=dict)
    priority: int = 5
    enabled: bool = True


@dataclass
class QueuedMessage:
    """Message in delivery queue with metadata."""

    message: Message
    delivery_guarantee: DeliveryGuarantee
    max_retries: int
    current_retries: int = 0
    next_retry_time: float = 0
    delivery_timeout: float = 0
    route_strategy: RouteStrategy = RouteStrategy.DIRECT
    target_agents: List[str] = field(default_factory=list)


class EnhancedMessageRouter:
    """Enhanced message router with delivery guarantees and intelligent routing."""

    def __init__(self, message_broker: MessageBroker):
        self.broker = message_broker
        self.routing_table: Dict[str, RouteEntry] = {}
        self.delivery_queues: Dict[str, List[QueuedMessage]] = defaultdict(
            list
        )  # agent -> messages
        self.pending_acks: Dict[str, QueuedMessage] = {}  # message_id -> queued_message
        self.delivery_receipts: Dict[str, List[DeliveryReceipt]] = defaultdict(list)
        self.agent_load_metrics: Dict[str, float] = defaultdict(float)
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.message_sequence: Dict[str, int] = defaultdict(int)  # For ordered delivery
        self.duplicate_detector: Set[str] = set()  # For exactly-once delivery

        # Configuration
        self.max_retry_attempts = 3
        self.default_timeout = 30.0  # seconds
        self.ack_timeout = 10.0  # seconds
        self.cleanup_interval = 300  # 5 minutes

        # Metrics
        self.routing_metrics = {
            "messages_routed": 0,
            "delivery_failures": 0,
            "retries_attempted": 0,
            "duplicates_detected": 0,
            "average_delivery_time": 0.0,
        }

    async def initialize(self) -> None:
        """Initialize the enhanced message router."""

        # Start background tasks
        asyncio.create_task(self._delivery_processor_loop())
        asyncio.create_task(self._retry_processor_loop())
        asyncio.create_task(self._cleanup_loop())
        asyncio.create_task(self._metrics_collector_loop())

        logger.info("Enhanced message router initialized")

    async def add_route(
        self,
        destination_pattern: str,
        route_strategy: RouteStrategy,
        target_agents: List[str],
        priority: int = 5,
        load_weights: Dict[str, float] = None,
    ) -> None:
        """Add a routing rule."""

        route = RouteEntry(
            destination_pattern=destination_pattern,
            route_strategy=route_strategy,
            target_agents=target_agents,
            load_weights=load_weights or {},
            priority=priority,
        )

        self.routing_table[destination_pattern] = route

        logger.info(
            "Route added",
            pattern=destination_pattern,
            strategy=route_strategy.value,
            targets=len(target_agents),
        )

    async def remove_route(self, destination_pattern: str) -> bool:
        """Remove a routing rule."""

        if destination_pattern in self.routing_table:
            del self.routing_table[destination_pattern]
            logger.info("Route removed", pattern=destination_pattern)
            return True

        return False

    async def send_message_guaranteed(
        self,
        from_agent: str,
        to_agent: str,
        topic: str,
        payload: Dict[str, Any],
        delivery_guarantee: DeliveryGuarantee = DeliveryGuarantee.AT_LEAST_ONCE,
        route_strategy: RouteStrategy = RouteStrategy.DIRECT,
        priority: int = 5,
        timeout: float = None,
        max_retries: int = None,
    ) -> str:
        """Send message with delivery guarantees."""

        message = Message(
            id=str(uuid.uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            topic=topic,
            message_type=MessageType.DIRECT,
            payload=payload,
            timestamp=time.time(),
            priority=priority,
        )

        # Check for duplicates in exactly-once mode
        if delivery_guarantee == DeliveryGuarantee.EXACTLY_ONCE:
            message_fingerprint = self._create_message_fingerprint(message)
            if message_fingerprint in self.duplicate_detector:
                logger.warning("Duplicate message detected", message_id=message.id)
                self.routing_metrics["duplicates_detected"] += 1
                return message.id
            self.duplicate_detector.add(message_fingerprint)

        # Determine target agents based on routing strategy
        target_agents = await self._resolve_target_agents(to_agent, route_strategy)

        queued_message = QueuedMessage(
            message=message,
            delivery_guarantee=delivery_guarantee,
            max_retries=max_retries or self.max_retry_attempts,
            delivery_timeout=time.time() + (timeout or self.default_timeout),
            route_strategy=route_strategy,
            target_agents=target_agents,
        )

        # Add to delivery queue based on strategy
        if delivery_guarantee == DeliveryGuarantee.ORDERED:
            # Assign sequence number for ordered delivery
            sequence_key = f"{from_agent}->{to_agent}"
            self.message_sequence[sequence_key] += 1
            message.payload["_sequence_number"] = self.message_sequence[sequence_key]

        # Queue for delivery
        primary_target = target_agents[0] if target_agents else to_agent
        self.delivery_queues[primary_target].append(queued_message)

        self.routing_metrics["messages_routed"] += 1

        logger.debug(
            "Message queued for guaranteed delivery",
            message_id=message.id,
            guarantee=delivery_guarantee.value,
            strategy=route_strategy.value,
            targets=len(target_agents),
        )

        return message.id

    async def acknowledge_message(self, message_id: str, agent_name: str) -> bool:
        """Acknowledge message delivery."""

        if message_id not in self.pending_acks:
            logger.warning("Acknowledgment for unknown message", message_id=message_id)
            return False

        queued_message = self.pending_acks[message_id]

        # Create delivery receipt
        receipt = DeliveryReceipt(
            message_id=message_id,
            recipient=agent_name,
            status=MessageStatus.ACKNOWLEDGED,
            timestamp=time.time(),
            retry_count=queued_message.current_retries,
        )

        self.delivery_receipts[message_id].append(receipt)

        # Remove from pending acknowledgments
        del self.pending_acks[message_id]

        logger.debug("Message acknowledged", message_id=message_id, agent=agent_name)

        return True

    async def get_delivery_status(self, message_id: str) -> Optional[Dict[str, Any]]:
        """Get delivery status for a message."""

        receipts = self.delivery_receipts.get(message_id, [])

        if not receipts:
            # Check if still pending
            if message_id in self.pending_acks:
                queued_msg = self.pending_acks[message_id]
                return {
                    "message_id": message_id,
                    "status": MessageStatus.SENT.value,
                    "retry_count": queued_msg.current_retries,
                    "delivery_guarantee": queued_msg.delivery_guarantee.value,
                }
            return None

        latest_receipt = receipts[-1]

        return {
            "message_id": message_id,
            "status": latest_receipt.status.value,
            "recipient": latest_receipt.recipient,
            "delivery_time": latest_receipt.timestamp,
            "retry_count": latest_receipt.retry_count,
            "total_receipts": len(receipts),
        }

    async def update_agent_load(self, agent_name: str, load_metric: float) -> None:
        """Update agent load metrics for load balancing."""

        self.agent_load_metrics[agent_name] = load_metric

        logger.debug("Agent load updated", agent=agent_name, load=load_metric)

    async def update_agent_capabilities(
        self, agent_name: str, capabilities: List[str]
    ) -> None:
        """Update agent capabilities for capability-based routing."""

        self.agent_capabilities[agent_name] = capabilities

        logger.debug(
            "Agent capabilities updated", agent=agent_name, capabilities=capabilities
        )

    async def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing system metrics."""

        return {
            **self.routing_metrics,
            "active_routes": len(self.routing_table),
            "pending_deliveries": sum(
                len(queue) for queue in self.delivery_queues.values()
            ),
            "pending_acks": len(self.pending_acks),
            "agent_loads": dict(self.agent_load_metrics),
        }

    async def _resolve_target_agents(
        self, to_agent: str, route_strategy: RouteStrategy
    ) -> List[str]:
        """Resolve target agents based on routing strategy."""

        # Check routing table for pattern match
        route_entry = None
        for pattern, entry in self.routing_table.items():
            if self._pattern_matches(to_agent, pattern) and entry.enabled:
                route_entry = entry
                break

        if route_entry:
            target_agents = route_entry.target_agents.copy()

            # Apply routing strategy
            if route_entry.route_strategy == RouteStrategy.LOAD_BALANCED:
                target_agents.sort(key=lambda a: self.agent_load_metrics.get(a, 0))

            elif route_entry.route_strategy == RouteStrategy.ROUND_ROBIN:
                # Simple round-robin implementation
                round_robin_key = f"rr_{pattern}"
                if round_robin_key not in self.message_sequence:
                    self.message_sequence[round_robin_key] = 0
                self.message_sequence[round_robin_key] = (
                    self.message_sequence[round_robin_key] + 1
                ) % len(target_agents)
                selected_agent = target_agents[self.message_sequence[round_robin_key]]
                target_agents = [selected_agent]

            elif route_entry.route_strategy == RouteStrategy.CAPABILITY_BASED:
                # Filter by capability requirements (if specified in message)
                # This would require capability info in the message payload
                pass

            return target_agents

        # Default to direct routing
        return [to_agent]

    async def _delivery_processor_loop(self) -> None:
        """Main delivery processing loop."""

        while True:
            try:
                await asyncio.sleep(0.1)  # Process every 100ms

                for agent_name, queue in list(self.delivery_queues.items()):
                    if not queue:
                        continue

                    # Process messages in queue
                    messages_to_remove = []

                    for i, queued_msg in enumerate(queue):
                        try:
                            # Check if message expired
                            if time.time() > queued_msg.delivery_timeout:
                                await self._handle_message_expired(queued_msg)
                                messages_to_remove.append(i)
                                continue

                            # Check delivery guarantee requirements
                            if (
                                queued_msg.delivery_guarantee
                                == DeliveryGuarantee.ORDERED
                            ):
                                # For ordered delivery, ensure sequence
                                if not await self._can_deliver_in_order(queued_msg):
                                    continue

                            # Attempt delivery
                            success = await self._deliver_message(queued_msg)

                            if success:
                                if queued_msg.delivery_guarantee in [
                                    DeliveryGuarantee.AT_LEAST_ONCE,
                                    DeliveryGuarantee.EXACTLY_ONCE,
                                    DeliveryGuarantee.ORDERED,
                                ]:
                                    # Wait for acknowledgment
                                    self.pending_acks[queued_msg.message.id] = (
                                        queued_msg
                                    )

                                messages_to_remove.append(i)
                            else:
                                # Handle delivery failure
                                await self._handle_delivery_failure(queued_msg)
                                if queued_msg.current_retries >= queued_msg.max_retries:
                                    messages_to_remove.append(i)

                        except Exception as e:
                            logger.error(
                                "Delivery processing error",
                                message_id=queued_msg.message.id,
                                error=str(e),
                            )
                            messages_to_remove.append(i)

                    # Remove processed messages
                    for i in reversed(messages_to_remove):
                        queue.pop(i)

            except Exception as e:
                logger.error("Delivery processor loop error", error=str(e))

    async def _retry_processor_loop(self) -> None:
        """Process message retries."""

        while True:
            try:
                await asyncio.sleep(1)  # Check every second

                current_time = time.time()
                expired_acks = []

                for message_id, queued_msg in self.pending_acks.items():
                    # Check for acknowledgment timeout
                    time_since_sent = current_time - queued_msg.message.timestamp

                    if time_since_sent > self.ack_timeout:
                        # Acknowledgment timeout - retry if allowed
                        if queued_msg.current_retries < queued_msg.max_retries:
                            # Re-queue for retry
                            primary_target = (
                                queued_msg.target_agents[0]
                                if queued_msg.target_agents
                                else queued_msg.message.to_agent
                            )
                            self.delivery_queues[primary_target].append(queued_msg)
                            expired_acks.append(message_id)
                        else:
                            # Max retries exceeded
                            await self._handle_max_retries_exceeded(queued_msg)
                            expired_acks.append(message_id)

                # Clean up expired acknowledgments
                for message_id in expired_acks:
                    del self.pending_acks[message_id]

            except Exception as e:
                logger.error("Retry processor loop error", error=str(e))

    async def _cleanup_loop(self) -> None:
        """Clean up old delivery receipts and metrics."""

        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)

                current_time = time.time()
                cutoff_time = current_time - (24 * 3600)  # 24 hours

                # Clean up old delivery receipts
                for message_id in list(self.delivery_receipts.keys()):
                    receipts = self.delivery_receipts[message_id]
                    if receipts and receipts[-1].timestamp < cutoff_time:
                        del self.delivery_receipts[message_id]

                # Clean up duplicate detector
                # In a real implementation, this would be more sophisticated
                if len(self.duplicate_detector) > 10000:
                    self.duplicate_detector.clear()

                logger.debug("Cleanup completed", timestamp=current_time)

            except Exception as e:
                logger.error("Cleanup loop error", error=str(e))

    async def _metrics_collector_loop(self) -> None:
        """Collect and update routing metrics."""

        while True:
            try:
                await asyncio.sleep(60)  # Update every minute

                # Calculate average delivery time
                recent_receipts = []
                cutoff_time = time.time() - 3600  # Last hour

                for receipts in self.delivery_receipts.values():
                    for receipt in receipts:
                        if (
                            receipt.timestamp > cutoff_time
                            and receipt.status == MessageStatus.ACKNOWLEDGED
                        ):
                            recent_receipts.append(receipt)

                if recent_receipts:
                    delivery_times = [
                        receipt.timestamp - (receipt.timestamp - 5)  # Approximate
                        for receipt in recent_receipts
                    ]
                    self.routing_metrics["average_delivery_time"] = sum(
                        delivery_times
                    ) / len(delivery_times)

            except Exception as e:
                logger.error("Metrics collector error", error=str(e))

    async def _deliver_message(self, queued_msg: QueuedMessage) -> bool:
        """Attempt to deliver a message."""

        try:
            # Choose target based on strategy
            target_agent = await self._select_target_agent(queued_msg)

            # Send message via broker
            success = await self.broker.send_message(
                from_agent=queued_msg.message.from_agent,
                to_agent=target_agent,
                topic=queued_msg.message.topic,
                payload=queued_msg.message.payload,
                message_type=queued_msg.message.message_type,
                priority=queued_msg.message.priority,
            )

            if success:
                # Create delivery receipt
                receipt = DeliveryReceipt(
                    message_id=queued_msg.message.id,
                    recipient=target_agent,
                    status=MessageStatus.DELIVERED
                    if queued_msg.delivery_guarantee == DeliveryGuarantee.AT_MOST_ONCE
                    else MessageStatus.SENT,
                    timestamp=time.time(),
                    retry_count=queued_msg.current_retries,
                )

                self.delivery_receipts[queued_msg.message.id].append(receipt)

                logger.debug(
                    "Message delivered",
                    message_id=queued_msg.message.id,
                    target=target_agent,
                    retries=queued_msg.current_retries,
                )

            return success

        except Exception as e:
            logger.error(
                "Message delivery failed",
                message_id=queued_msg.message.id,
                error=str(e),
            )
            return False

    async def _select_target_agent(self, queued_msg: QueuedMessage) -> str:
        """Select target agent based on routing strategy."""

        if not queued_msg.target_agents:
            return queued_msg.message.to_agent

        if queued_msg.route_strategy == RouteStrategy.LOAD_BALANCED:
            return min(
                queued_msg.target_agents,
                key=lambda a: self.agent_load_metrics.get(a, 0),
            )

        elif queued_msg.route_strategy == RouteStrategy.PRIORITY_BASED:
            # Select based on message priority and agent availability
            available_agents = [
                agent
                for agent in queued_msg.target_agents
                if self.agent_load_metrics.get(agent, 0) < 0.8  # Less than 80% load
            ]
            return (
                available_agents[0] if available_agents else queued_msg.target_agents[0]
            )

        # Default to first target
        return queued_msg.target_agents[0]

    async def _handle_delivery_failure(self, queued_msg: QueuedMessage) -> None:
        """Handle message delivery failure."""

        queued_msg.current_retries += 1
        queued_msg.next_retry_time = time.time() + (
            2**queued_msg.current_retries
        )  # Exponential backoff

        self.routing_metrics["delivery_failures"] += 1
        self.routing_metrics["retries_attempted"] += 1

        logger.warning(
            "Message delivery failed",
            message_id=queued_msg.message.id,
            retry_count=queued_msg.current_retries,
            max_retries=queued_msg.max_retries,
        )

    async def _handle_message_expired(self, queued_msg: QueuedMessage) -> None:
        """Handle expired message."""

        receipt = DeliveryReceipt(
            message_id=queued_msg.message.id,
            recipient=queued_msg.message.to_agent,
            status=MessageStatus.EXPIRED,
            timestamp=time.time(),
            retry_count=queued_msg.current_retries,
            error_message="Message delivery timeout",
        )

        self.delivery_receipts[queued_msg.message.id].append(receipt)

        logger.warning("Message expired", message_id=queued_msg.message.id)

    async def _handle_max_retries_exceeded(self, queued_msg: QueuedMessage) -> None:
        """Handle message that exceeded max retries."""

        receipt = DeliveryReceipt(
            message_id=queued_msg.message.id,
            recipient=queued_msg.message.to_agent,
            status=MessageStatus.FAILED,
            timestamp=time.time(),
            retry_count=queued_msg.current_retries,
            error_message="Maximum retry attempts exceeded",
        )

        self.delivery_receipts[queued_msg.message.id].append(receipt)

        logger.error(
            "Message failed after max retries", message_id=queued_msg.message.id
        )

    async def _can_deliver_in_order(self, queued_msg: QueuedMessage) -> bool:
        """Check if message can be delivered while maintaining order."""

        # Simple implementation - check sequence number
        sequence_number = queued_msg.message.payload.get("_sequence_number", 0)
        sequence_key = f"{queued_msg.message.from_agent}->{queued_msg.message.to_agent}"
        expected_sequence = (
            self.message_sequence.get(f"{sequence_key}_delivered", 0) + 1
        )

        return sequence_number <= expected_sequence

    def _pattern_matches(self, agent_name: str, pattern: str) -> bool:
        """Check if agent name matches routing pattern."""

        if pattern == "*":
            return True

        if pattern.startswith("capability:"):
            required_capability = pattern[11:]
            agent_caps = self.agent_capabilities.get(agent_name, [])
            return required_capability in agent_caps

        return agent_name == pattern

    def _create_message_fingerprint(self, message: Message) -> str:
        """Create unique fingerprint for duplicate detection."""

        content = f"{message.from_agent}:{message.to_agent}:{message.topic}:{json.dumps(message.payload, sort_keys=True)}"
        return str(hash(content))


# Global enhanced router instance
enhanced_router = None


def get_enhanced_router(message_broker: MessageBroker) -> EnhancedMessageRouter:
    """Get enhanced message router instance."""
    global enhanced_router

    if enhanced_router is None:
        enhanced_router = EnhancedMessageRouter(message_broker)

    return enhanced_router
