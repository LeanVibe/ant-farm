"""Component isolation testing framework with advanced dependency mocking.

This module provides a comprehensive framework for testing components in complete
isolation from their dependencies. It ensures that each component's behavior can
be validated independently of external systems like Redis, PostgreSQL, file system,
and network calls.

Key Features:
- Automatic dependency injection mocking
- Realistic behavior simulation for external dependencies
- State verification and assertion helpers
- Performance and resource usage tracking
- Error scenario simulation
- Contract validation
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import structlog

logger = structlog.get_logger()


@dataclass
class MockInteraction:
    """Records an interaction with a mocked dependency."""

    timestamp: float
    component: str
    method: str
    args: tuple
    kwargs: dict
    result: Any
    duration: float = 0.0
    exception: Exception | None = None


@dataclass
class ComponentTestConfig:
    """Configuration for component isolation testing."""

    mock_redis: bool = True
    mock_database: bool = True
    mock_file_system: bool = True
    mock_network: bool = True
    mock_cli_tools: bool = True
    record_interactions: bool = True
    validate_contracts: bool = True
    simulate_errors: bool = False
    error_probability: float = 0.0


class MockRedisClient:
    """Advanced Redis client mock with realistic behavior."""

    def __init__(self, record_interactions: bool = True):
        self.data: dict[str, Any] = {}
        self.lists: dict[str, list[str]] = defaultdict(list)
        self.sets: dict[str, set[str]] = defaultdict(set)
        self.hashes: dict[str, dict[str, str]] = defaultdict(dict)
        self.sorted_sets: dict[str, dict[str, float]] = defaultdict(dict)
        self.expirations: dict[str, float] = {}
        self.pubsub_channels: dict[str, list[str]] = defaultdict(list)
        self.pubsub_subscribers: dict[str, list[AsyncMock]] = defaultdict(list)
        self.interactions: list[MockInteraction] = []
        self.record_interactions = record_interactions
        self.connection_pool = MagicMock()
        self.connection_pool.connection_kwargs = {"port": 6381}

    def _record_interaction(
        self, method: str, args: tuple, kwargs: dict, result: Any, duration: float = 0.0
    ):
        """Record interaction for analysis."""
        if self.record_interactions:
            self.interactions.append(
                MockInteraction(
                    timestamp=time.time(),
                    component="redis",
                    method=method,
                    args=args,
                    kwargs=kwargs,
                    result=result,
                    duration=duration,
                )
            )

    async def ping(self) -> bool:
        """Mock Redis ping."""
        start_time = time.time()
        result = True
        duration = time.time() - start_time
        self._record_interaction("ping", (), {}, result, duration)
        return result

    async def get(self, key: str) -> str | None:
        """Mock Redis GET operation."""
        start_time = time.time()

        # Check if key has expired
        if key in self.expirations and time.time() > self.expirations[key]:
            del self.data[key]
            del self.expirations[key]
            result = None
        else:
            result = self.data.get(key)

        duration = time.time() - start_time
        self._record_interaction("get", (key,), {}, result, duration)
        return result

    async def set(self, key: str, value: str, ex: int | None = None) -> bool:
        """Mock Redis SET operation."""
        start_time = time.time()

        self.data[key] = value
        if ex:
            self.expirations[key] = time.time() + ex

        result = True
        duration = time.time() - start_time
        self._record_interaction("set", (key, value), {"ex": ex}, result, duration)
        return result

    async def hset(self, key: str, mapping: dict) -> int:
        """Mock Redis HSET operation."""
        start_time = time.time()

        if key not in self.hashes:
            self.hashes[key] = {}

        fields_added = 0
        for field, value in mapping.items():
            if field not in self.hashes[key]:
                fields_added += 1
            self.hashes[key][field] = str(value)

        duration = time.time() - start_time
        self._record_interaction(
            "hset", (key,), {"mapping": mapping}, fields_added, duration
        )
        return fields_added

    async def hgetall(self, key: str) -> dict:
        """Mock Redis HGETALL operation."""
        start_time = time.time()

        # Check if key has expired
        if key in self.expirations and time.time() > self.expirations[key]:
            if key in self.hashes:
                del self.hashes[key]
            del self.expirations[key]
            result = {}
        else:
            result = self.hashes.get(key, {}).copy()

        duration = time.time() - start_time
        self._record_interaction("hgetall", (key,), {}, result, duration)
        return result

    async def lpush(self, key: str, *values: str) -> int:
        """Mock Redis LPUSH operation."""
        start_time = time.time()

        for value in reversed(values):
            self.lists[key].insert(0, value)

        result = len(self.lists[key])
        duration = time.time() - start_time
        self._record_interaction("lpush", (key, *values), {}, result, duration)
        return result

    async def rpop(self, key: str) -> str | None:
        """Mock Redis RPOP operation."""
        start_time = time.time()

        if key in self.lists and self.lists[key]:
            result = self.lists[key].pop()
        else:
            result = None

        duration = time.time() - start_time
        self._record_interaction("rpop", (key,), {}, result, duration)
        return result

    async def brpop(self, key: str, timeout: int = 1) -> tuple | None:
        """Mock Redis BRPOP operation."""
        start_time = time.time()

        # Simulate blocking behavior
        await asyncio.sleep(0.001)  # Small delay to simulate network

        if key in self.lists and self.lists[key]:
            value = self.lists[key].pop()
            result = (key, value)
        else:
            result = None

        duration = time.time() - start_time
        self._record_interaction(
            "brpop", (key,), {"timeout": timeout}, result, duration
        )
        return result

    async def publish(self, channel: str, message: str) -> int:
        """Mock Redis PUBLISH operation."""
        start_time = time.time()

        # Store published message
        self.pubsub_channels[channel].append(message)

        # Notify subscribers
        subscriber_count = len(self.pubsub_subscribers.get(channel, []))
        for subscriber in self.pubsub_subscribers.get(channel, []):
            # Simulate message delivery
            if hasattr(subscriber, "_deliver_message"):
                await subscriber._deliver_message(channel, message)

        duration = time.time() - start_time
        self._record_interaction(
            "publish", (channel, message), {}, subscriber_count, duration
        )
        return subscriber_count

    async def zadd(self, key: str, mapping: dict) -> int:
        """Mock Redis ZADD operation."""
        start_time = time.time()

        added_count = 0
        for member, score in mapping.items():
            if member not in self.sorted_sets[key]:
                added_count += 1
            self.sorted_sets[key][member] = float(score)

        duration = time.time() - start_time
        self._record_interaction(
            "zadd", (key,), {"mapping": mapping}, added_count, duration
        )
        return added_count

    async def zrange(
        self, key: str, start: int, stop: int, withscores: bool = False
    ) -> list:
        """Mock Redis ZRANGE operation."""
        start_time = time.time()

        if key not in self.sorted_sets:
            result = []
        else:
            # Sort by score
            sorted_items = sorted(self.sorted_sets[key].items(), key=lambda x: x[1])

            # Apply range
            if stop == -1:
                range_items = sorted_items[start:]
            else:
                range_items = sorted_items[start : stop + 1]

            if withscores:
                result = [(member, score) for member, score in range_items]
            else:
                result = [member for member, score in range_items]

        duration = time.time() - start_time
        self._record_interaction(
            "zrange", (key, start, stop), {"withscores": withscores}, result, duration
        )
        return result

    async def expire(self, key: str, seconds: int) -> bool:
        """Mock Redis EXPIRE operation."""
        start_time = time.time()

        if (
            key in self.data
            or key in self.hashes
            or key in self.lists
            or key in self.sorted_sets
        ):
            self.expirations[key] = time.time() + seconds
            result = True
        else:
            result = False

        duration = time.time() - start_time
        self._record_interaction("expire", (key, seconds), {}, result, duration)
        return result

    async def keys(self, pattern: str = "*") -> list[str]:
        """Mock Redis KEYS operation."""
        start_time = time.time()

        # Simple pattern matching (just * for now)
        all_keys = set()
        all_keys.update(self.data.keys())
        all_keys.update(self.hashes.keys())
        all_keys.update(self.lists.keys())
        all_keys.update(self.sorted_sets.keys())

        # Remove expired keys
        current_time = time.time()
        expired_keys = [
            k for k, exp_time in self.expirations.items() if current_time > exp_time
        ]
        for key in expired_keys:
            all_keys.discard(key)
            del self.expirations[key]

        if pattern == "*":
            result = list(all_keys)
        else:
            # Simple pattern matching - can be enhanced
            result = [k for k in all_keys if pattern.replace("*", "") in k]

        duration = time.time() - start_time
        self._record_interaction("keys", (pattern,), {}, result, duration)
        return result

    async def flushdb(self) -> bool:
        """Mock Redis FLUSHDB operation."""
        start_time = time.time()

        self.data.clear()
        self.lists.clear()
        self.sets.clear()
        self.hashes.clear()
        self.sorted_sets.clear()
        self.expirations.clear()
        self.pubsub_channels.clear()

        result = True
        duration = time.time() - start_time
        self._record_interaction("flushdb", (), {}, result, duration)
        return result

    async def aclose(self) -> None:
        """Mock Redis connection close."""
        start_time = time.time()
        duration = time.time() - start_time
        self._record_interaction("aclose", (), {}, None, duration)

    async def close(self) -> None:
        """Mock Redis connection close."""
        await self.aclose()

    def pubsub(self) -> "MockPubSub":
        """Create a mock pub/sub instance."""
        return MockPubSub(self)


class MockPubSub:
    """Mock Redis pub/sub client."""

    def __init__(self, redis_client: MockRedisClient):
        self.redis_client = redis_client
        self.subscribed_channels: set[str] = set()
        self.message_queue: asyncio.Queue = asyncio.Queue()

    async def subscribe(self, *channels: str) -> None:
        """Subscribe to channels."""
        for channel in channels:
            self.subscribed_channels.add(channel)
            if channel not in self.redis_client.pubsub_subscribers:
                self.redis_client.pubsub_subscribers[channel] = []
            self.redis_client.pubsub_subscribers[channel].append(self)

    async def unsubscribe(self, *channels: str) -> None:
        """Unsubscribe from channels."""
        for channel in channels:
            self.subscribed_channels.discard(channel)
            if channel in self.redis_client.pubsub_subscribers:
                try:
                    self.redis_client.pubsub_subscribers[channel].remove(self)
                except ValueError:
                    pass

    async def listen(self):
        """Listen for messages."""
        while True:
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                yield message
            except TimeoutError:
                # Yield control to allow other tasks to run
                await asyncio.sleep(0.001)
                continue

    async def _deliver_message(self, channel: str, data: str):
        """Internal method to deliver messages."""
        if channel in self.subscribed_channels:
            message = {"type": "message", "channel": channel, "data": data}
            await self.message_queue.put(message)


class MockAsyncSession:
    """Mock SQLAlchemy async session."""

    def __init__(self):
        self.data: dict[str, list[dict]] = defaultdict(list)
        self.interactions: list[MockInteraction] = []
        self.transaction_active = False

    def _record_interaction(self, method: str, args: tuple, kwargs: dict, result: Any):
        """Record database interaction."""
        self.interactions.append(
            MockInteraction(
                timestamp=time.time(),
                component="database",
                method=method,
                args=args,
                kwargs=kwargs,
                result=result,
            )
        )

    async def execute(self, query):
        """Mock query execution."""
        start_time = time.time()

        # Create a mock result
        result = MagicMock()
        result.fetchall.return_value = []
        result.fetchone.return_value = None
        result.scalar.return_value = None

        duration = time.time() - start_time
        self._record_interaction("execute", (str(query),), {}, result)
        return result

    async def commit(self):
        """Mock transaction commit."""
        self._record_interaction("commit", (), {}, True)
        self.transaction_active = False

    async def rollback(self):
        """Mock transaction rollback."""
        self._record_interaction("rollback", (), {}, True)
        self.transaction_active = False

    async def close(self):
        """Mock session close."""
        self._record_interaction("close", (), {}, None)

    def begin(self):
        """Mock transaction begin."""
        self.transaction_active = True
        return MockAsyncContextManager(self)

    def add(self, instance):
        """Mock adding an instance."""
        self._record_interaction("add", (instance,), {}, None)

    async def refresh(self, instance):
        """Mock refreshing an instance."""
        self._record_interaction("refresh", (instance,), {}, None)


class MockAsyncContextManager:
    """Mock async context manager for database transactions."""

    def __init__(self, session: MockAsyncSession):
        self.session = session

    async def __aenter__(self):
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.session.rollback()
        else:
            await self.session.commit()


class MockCLITools:
    """Mock CLI tools for testing."""

    def __init__(self):
        self.interactions: list[MockInteraction] = []
        self.available_tools = {
            "opencode": {"name": "OpenCode", "command": "opencode", "available": True},
            "claude": {"name": "Claude CLI", "command": "claude", "available": True},
        }
        self.preferred_tool = "opencode"

    def _record_interaction(self, method: str, args: tuple, kwargs: dict, result: Any):
        """Record CLI interaction."""
        self.interactions.append(
            MockInteraction(
                timestamp=time.time(),
                component="cli_tools",
                method=method,
                args=args,
                kwargs=kwargs,
                result=result,
            )
        )

    async def execute_prompt(self, prompt: str, tool_override: str = None):
        """Mock CLI tool execution."""
        from src.agents.base_agent import ToolResult

        # Simulate processing time
        await asyncio.sleep(0.01)

        result = ToolResult(
            success=True,
            output=f"Mock CLI response to: {prompt[:50]}...",
            tool_used=tool_override or self.preferred_tool,
            execution_time=0.01,
        )

        self._record_interaction(
            "execute_prompt", (prompt,), {"tool_override": tool_override}, result
        )
        return result


class ComponentIsolationTestFramework:
    """Comprehensive framework for component isolation testing."""

    def __init__(self, config: ComponentTestConfig = None):
        self.config = config or ComponentTestConfig()
        self.mock_redis_client: MockRedisClient | None = None
        self.mock_db_session: MockAsyncSession | None = None
        self.mock_cli_tools: MockCLITools | None = None
        self.patches: list[Any] = []
        self.interactions: list[MockInteraction] = []

    @asynccontextmanager
    async def isolate_component(self, component_class, **kwargs):
        """Context manager for isolating a component during testing."""

        # Start all patches
        await self._start_patches()

        try:
            # Create component instance with mocked dependencies
            if "redis_url" in kwargs and self.config.mock_redis:
                kwargs["redis_url"] = "redis://mock:6379"

            # Inject mocked dependencies into constructor if needed
            component = component_class(**kwargs)

            # Replace component dependencies with mocks
            await self._inject_mocks(component)

            yield component

        finally:
            # Stop all patches
            await self._stop_patches()

            # Collect interactions from all mocks
            self._collect_interactions()

    async def _start_patches(self):
        """Start all necessary patches."""

        if self.config.mock_redis:
            self.mock_redis_client = MockRedisClient(self.config.record_interactions)

            # Patch redis module
            redis_patch = patch("redis.asyncio.Redis.from_url")
            mock_redis_from_url = redis_patch.start()
            mock_redis_from_url.return_value = self.mock_redis_client
            self.patches.append(redis_patch)

            # Patch redis imports in various modules
            message_broker_patch = patch("src.core.message_broker.redis.from_url")
            mock_mb_redis = message_broker_patch.start()
            mock_mb_redis.return_value = self.mock_redis_client
            self.patches.append(message_broker_patch)

        if self.config.mock_database:
            self.mock_db_session = MockAsyncSession()

            # Patch SQLAlchemy session creation
            session_patch = patch("sqlalchemy.ext.asyncio.AsyncSession")
            mock_session_class = session_patch.start()
            mock_session_class.return_value = self.mock_db_session
            self.patches.append(session_patch)

            # Patch async database manager
            db_manager_patch = patch("src.core.async_db.get_async_database_manager")
            mock_db_manager = db_manager_patch.start()
            mock_db_manager.return_value = MagicMock()
            self.patches.append(db_manager_patch)

        if self.config.mock_cli_tools:
            self.mock_cli_tools = MockCLITools()

            # Patch CLI tool manager
            cli_patch = patch("src.agents.base_agent.CLIToolManager")
            mock_cli_class = cli_patch.start()
            mock_cli_class.return_value = self.mock_cli_tools
            self.patches.append(cli_patch)

        if self.config.mock_network:
            # Patch HTTP requests
            http_patch = patch("aiohttp.ClientSession")
            mock_http = http_patch.start()
            mock_http.return_value.__aenter__ = AsyncMock()
            mock_http.return_value.__aexit__ = AsyncMock()
            self.patches.append(http_patch)

        if self.config.mock_file_system:
            # Patch file operations
            file_patch = patch("builtins.open")
            mock_file = file_patch.start()
            self.patches.append(file_patch)

    async def _inject_mocks(self, component):
        """Inject mocks into component dependencies."""

        # Inject Redis client if component uses it
        if hasattr(component, "redis_client") and self.mock_redis_client:
            component.redis_client = self.mock_redis_client

        # Inject database session if component uses it
        if hasattr(component, "db_session") and self.mock_db_session:
            component.db_session = self.mock_db_session

        # Inject CLI tools if component uses them
        if hasattr(component, "cli_tools") and self.mock_cli_tools:
            component.cli_tools = self.mock_cli_tools

        # For message broker components
        if hasattr(component, "pubsub") and self.mock_redis_client:
            component.pubsub = self.mock_redis_client.pubsub()

    async def _stop_patches(self):
        """Stop all active patches."""
        for patch_obj in self.patches:
            patch_obj.stop()
        self.patches.clear()

    def _collect_interactions(self):
        """Collect all interactions from mocks."""
        self.interactions.clear()

        if self.mock_redis_client:
            self.interactions.extend(self.mock_redis_client.interactions)

        if self.mock_db_session:
            self.interactions.extend(self.mock_db_session.interactions)

        if self.mock_cli_tools:
            self.interactions.extend(self.mock_cli_tools.interactions)

    def assert_no_external_calls(self):
        """Assert that no external calls were made."""
        external_calls = [
            interaction
            for interaction in self.interactions
            if interaction.component
            in ["redis", "database", "cli_tools", "network", "file_system"]
        ]

        if external_calls:
            call_summary = []
            for call in external_calls:
                call_summary.append(
                    f"{call.component}.{call.method}({call.args}, {call.kwargs})"
                )

            pytest.fail(f"External calls detected: {call_summary}")

    def assert_redis_interactions(self, expected_calls: list[dict[str, Any]]):
        """Assert specific Redis interactions occurred."""
        redis_calls = [
            interaction
            for interaction in self.interactions
            if interaction.component == "redis"
        ]

        assert len(redis_calls) == len(expected_calls), (
            f"Expected {len(expected_calls)} Redis calls, got {len(redis_calls)}"
        )

        for i, (actual, expected) in enumerate(zip(redis_calls, expected_calls, strict=False)):
            assert actual.method == expected["method"], (
                f"Call {i}: expected method {expected['method']}, got {actual.method}"
            )

            if "args" in expected:
                assert actual.args == tuple(expected["args"]), (
                    f"Call {i}: expected args {expected['args']}, got {actual.args}"
                )

    def assert_database_interactions(self, expected_calls: list[dict[str, Any]]):
        """Assert specific database interactions occurred."""
        db_calls = [
            interaction
            for interaction in self.interactions
            if interaction.component == "database"
        ]

        assert len(db_calls) == len(expected_calls), (
            f"Expected {len(expected_calls)} database calls, got {len(db_calls)}"
        )

        for i, (actual, expected) in enumerate(zip(db_calls, expected_calls, strict=False)):
            assert actual.method == expected["method"], (
                f"Call {i}: expected method {expected['method']}, got {actual.method}"
            )

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics from recorded interactions."""
        metrics = {
            "total_interactions": len(self.interactions),
            "components_used": set(i.component for i in self.interactions),
            "total_duration": sum(i.duration for i in self.interactions),
            "average_duration": 0.0,
            "by_component": defaultdict(dict),
        }

        if self.interactions:
            metrics["average_duration"] = metrics["total_duration"] / len(
                self.interactions
            )

        # Group by component
        for interaction in self.interactions:
            component = interaction.component
            if component not in metrics["by_component"]:
                metrics["by_component"][component] = {
                    "call_count": 0,
                    "total_duration": 0.0,
                    "methods": defaultdict(int),
                }

            comp_metrics = metrics["by_component"][component]
            comp_metrics["call_count"] += 1
            comp_metrics["total_duration"] += interaction.duration
            comp_metrics["methods"][interaction.method] += 1

        return dict(metrics)

    def assert_performance_constraints(
        self, max_calls: int = None, max_duration: float = None
    ):
        """Assert performance constraints are met."""
        metrics = self.get_performance_metrics()

        if max_calls is not None:
            assert metrics["total_interactions"] <= max_calls, (
                f"Too many interactions: {metrics['total_interactions']} > {max_calls}"
            )

        if max_duration is not None:
            assert metrics["total_duration"] <= max_duration, (
                f"Total duration too high: {metrics['total_duration']} > {max_duration}"
            )

    def simulate_redis_error(
        self, method: str, error_type: Exception = ConnectionError
    ):
        """Simulate Redis errors for specific methods."""
        if self.mock_redis_client:
            original_method = getattr(self.mock_redis_client, method)

            async def error_method(*args, **kwargs):
                raise error_type(f"Simulated {method} error")

            setattr(self.mock_redis_client, method, error_method)

    def simulate_database_error(self, method: str, error_type: Exception = Exception):
        """Simulate database errors for specific methods."""
        if self.mock_db_session:
            original_method = getattr(self.mock_db_session, method)

            async def error_method(*args, **kwargs):
                raise error_type(f"Simulated {method} error")

            setattr(self.mock_db_session, method, error_method)


# Test helper functions
async def create_isolated_message_broker():
    """Create an isolated MessageBroker for testing."""
    framework = ComponentIsolationTestFramework()

    async with framework.isolate_component(
        lambda: None,  # We'll manually create the broker
        redis_url="redis://mock:6379",
    ) as _:
        from src.core.message_broker import MessageBroker

        # Create broker with mocked Redis
        broker = MessageBroker("redis://mock:6379")
        broker.redis_client = framework.mock_redis_client
        broker.pubsub = framework.mock_redis_client.pubsub()

        return broker, framework


async def create_isolated_enhanced_message_broker():
    """Create an isolated EnhancedMessageBroker for testing."""
    framework = ComponentIsolationTestFramework()

    async with framework.isolate_component(
        lambda: None,  # We'll manually create the broker
        redis_url="redis://mock:6379",
    ) as _:
        from src.core.enhanced_message_broker import EnhancedMessageBroker

        # Create broker with mocked dependencies
        broker = EnhancedMessageBroker("redis://mock:6379")
        broker.redis_client = framework.mock_redis_client
        broker.pubsub = framework.mock_redis_client.pubsub()

        return broker, framework


async def create_isolated_agent(agent_class, **kwargs):
    """Create an isolated agent for testing."""
    framework = ComponentIsolationTestFramework()

    async with framework.isolate_component(agent_class, **kwargs) as agent:
        return agent, framework


# Example usage and validation
if __name__ == "__main__":

    async def test_framework_example():
        """Example of how to use the isolation testing framework."""

        # Test MessageBroker in isolation
        broker, framework = await create_isolated_message_broker()

        # Initialize broker (should only interact with mocked Redis)
        await broker.initialize()

        # Send a message (should only use mocked Redis)
        success = await broker.send_message(
            from_agent="test_agent",
            to_agent="target_agent",
            topic="test_topic",
            payload={"test": "data"},
        )

        # Verify behavior
        assert success is True

        # Check that only expected Redis calls were made
        framework.assert_redis_interactions(
            [
                {"method": "ping"},
                {"method": "hset"},
                {"method": "expire"},
                {"method": "publish"},
            ]
        )

        # Verify no unexpected external calls
        framework.assert_no_external_calls()

        # Check performance
        metrics = framework.get_performance_metrics()
        print(f"Test completed with {metrics['total_interactions']} interactions")

        # Performance constraints
        framework.assert_performance_constraints(max_calls=10, max_duration=1.0)

    # Run the example
    asyncio.run(test_framework_example())
