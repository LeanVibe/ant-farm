"""Fake pubsub implementation for deterministic testing."""

import asyncio
from collections import defaultdict
from typing import Any, AsyncGenerator, Dict, List


class FakePubSub:
    """Deterministic fake pubsub for testing."""

    def __init__(self):
        self._channels: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._subscriptions: set = set()
        self._published_messages: List[Dict[str, Any]] = []

    def seed_messages(self, channel: str, messages: List[Dict[str, Any]]) -> None:
        """Pre-seed messages for a channel."""
        self._channels[channel].extend(messages)

    async def publish(self, channel: str, message: str) -> None:
        """Publish a message to a channel."""
        msg = {"data": message, "type": "message"}
        self._channels[channel].append(msg)
        self._published_messages.append(msg)

    async def subscribe(self, channel: str) -> None:
        """Subscribe to a channel."""
        self._subscriptions.add(channel)

    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        self._subscriptions.discard(channel)

    async def listen(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Async generator yielding pre-seeded messages."""
        # Yield pre-seeded messages first
        for channel in self._subscriptions:
            for message in self._channels[channel]:
                yield message

        # Then yield a control message to prevent tests from hanging
        yield {"type": "control", "data": ""}
