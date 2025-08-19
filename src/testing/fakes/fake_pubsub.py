"""Fake pubsub implementation for deterministic testing."""

import asyncio
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List, Optional


class FakePubSub:
    """Deterministic fake pubsub for testing, replacing AsyncMock patterns."""

    def __init__(
        self, preseed: Optional[Dict[str, List[Dict]]] = None, delay_ms: int = 0
    ):
        """Initialize with optional preseeded messages per channel.

        Args:
            preseed: Dict mapping channel names to lists of messages
            delay_ms: Optional delay in milliseconds between message yields
        """
        self._channels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._subscriptions: set = set()
        self._published_messages: List[Dict[str, Any]] = []
        self.delay_ms = delay_ms

        # Pre-seed messages if provided
        if preseed:
            for channel, messages in preseed.items():
                self._channels[channel].extend(messages)

    def seed_messages(self, channel: str, messages: List[Dict[str, Any]]) -> None:
        """Pre-seed messages for a channel."""
        self._channels[channel].extend(messages)

    async def publish(self, channel: str, message: str) -> int:
        """Publish a message to a channel.

        Returns:
            Number of subscribers (for Redis compatibility)
        """
        msg = {"channel": channel, "data": message, "type": "message"}
        self._channels[channel].append(msg)
        self._published_messages.append(msg)
        return 1 if channel in self._subscriptions else 0

    def publish_sync(self, channel: str, data: Any) -> None:
        """Synchronous publish method for testing convenience."""
        message = {
            "channel": channel,
            "data": str(data) if not isinstance(data, str) else data,
            "type": "message",
        }
        self._channels[channel].append(message)
        self._published_messages.append(message)

    async def subscribe(self, *channels) -> None:
        """Subscribe to one or more channels."""
        for channel in channels:
            self._subscriptions.add(channel)

    async def unsubscribe(self, *channels) -> None:
        """Unsubscribe from one or more channels."""
        for channel in channels:
            self._subscriptions.discard(channel)

    async def listen(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Async generator yielding messages in FIFO order per channel.

        Yields messages with format: {"channel": str, "data": bytes|str, "type": "message"}
        """
        # Yield pre-seeded and published messages for subscribed channels
        for channel in self._subscriptions:
            for message in self._channels[channel]:
                if self.delay_ms > 0:
                    await asyncio.sleep(self.delay_ms / 1000.0)

                yield {
                    "channel": channel,
                    "data": message.get("data", ""),
                    "type": "message",
                }

        # Yield a control message to prevent tests from hanging indefinitely
        yield {"type": "control", "data": "end_of_messages"}

    def get_published_messages(self) -> List[Dict[str, Any]]:
        """Get all published messages for test verification."""
        return self._published_messages.copy()

    def get_subscriptions(self) -> set:
        """Get current subscriptions for test verification."""
        return self._subscriptions.copy()

    def clear(self) -> None:
        """Clear all messages and subscriptions for test cleanup."""
        self._channels.clear()
        self._subscriptions.clear()
        self._published_messages.clear()

    def advance_clock(self, ms: int) -> None:
        """Simulate time advancement for timing-based tests."""
        # This is a placeholder for potential future timing control
        # Could be used to trigger delayed message delivery
        pass
