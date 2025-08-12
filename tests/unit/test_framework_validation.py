"""Test the component isolation framework itself."""

import asyncio

import pytest

from tests.unit.test_component_isolation_framework import (
    ComponentIsolationTestFramework,
    ComponentTestConfig,
    MockAsyncSession,
    MockCLITools,
    MockRedisClient,
)


class TestComponentIsolationFramework:
    """Test the component isolation framework functionality."""

    @pytest.mark.asyncio
    async def test_mock_redis_client_basic_operations(self):
        """Test basic Redis mock operations."""
        mock_redis = MockRedisClient()

        # Test ping
        result = await mock_redis.ping()
        assert result is True

        # Test set/get
        await mock_redis.set("test_key", "test_value")
        value = await mock_redis.get("test_key")
        assert value == "test_value"

        # Test hset/hgetall
        await mock_redis.hset("test_hash", {"field1": "value1", "field2": "value2"})
        hash_data = await mock_redis.hgetall("test_hash")
        assert hash_data["field1"] == "value1"
        assert hash_data["field2"] == "value2"

    @pytest.mark.asyncio
    async def test_mock_redis_list_operations(self):
        """Test Redis list operations."""
        mock_redis = MockRedisClient()

        # Test lpush/rpop
        await mock_redis.lpush("test_list", "item1", "item2", "item3")

        item = await mock_redis.rpop("test_list")
        assert item == "item1"  # FIFO behavior

        item = await mock_redis.rpop("test_list")
        assert item == "item2"

    @pytest.mark.asyncio
    async def test_mock_redis_expiration(self):
        """Test Redis key expiration."""
        mock_redis = MockRedisClient()

        # Set key with expiration
        await mock_redis.set("expiring_key", "value", ex=1)

        # Key should exist initially
        value = await mock_redis.get("expiring_key")
        assert value == "value"

        # Manually trigger expiration
        mock_redis.expirations["expiring_key"] = 0  # Past expiration

        # Key should be gone
        value = await mock_redis.get("expiring_key")
        assert value is None

    @pytest.mark.asyncio
    async def test_mock_redis_pubsub(self):
        """Test Redis pub/sub functionality."""
        mock_redis = MockRedisClient()
        pubsub = mock_redis.pubsub()

        # Subscribe to channel
        await pubsub.subscribe("test_channel")
        assert "test_channel" in pubsub.subscribed_channels

        # Publish message
        subscriber_count = await mock_redis.publish("test_channel", "test_message")
        assert subscriber_count == 1

        # Should have message in queue
        assert not pubsub.message_queue.empty()

    @pytest.mark.asyncio
    async def test_mock_database_session(self):
        """Test mock database session."""
        mock_db = MockAsyncSession()

        # Test basic operations
        result = await mock_db.execute("SELECT * FROM agents")
        assert result is not None

        # Test transaction
        async with mock_db.begin() as conn:
            await conn.execute("INSERT INTO agents VALUES (...)")

        # Check interaction recording
        assert len(mock_db.interactions) > 0
        assert any(i.method == "execute" for i in mock_db.interactions)

    def test_mock_cli_tools(self):
        """Test mock CLI tools."""
        mock_cli = MockCLITools()

        # Check available tools
        assert "opencode" in mock_cli.available_tools
        assert "claude" in mock_cli.available_tools
        assert mock_cli.preferred_tool == "opencode"

        # Tool should be marked as available
        assert mock_cli.available_tools["opencode"]["available"] is True

    @pytest.mark.asyncio
    async def test_framework_configuration(self):
        """Test framework configuration options."""
        config = ComponentTestConfig(
            mock_redis=False, mock_database=True, record_interactions=False
        )

        framework = ComponentIsolationTestFramework(config)

        # Check configuration is applied
        assert framework.config.mock_redis is False
        assert framework.config.mock_database is True
        assert framework.config.record_interactions is False

    @pytest.mark.asyncio
    async def test_interaction_recording(self):
        """Test interaction recording functionality."""
        mock_redis = MockRedisClient(record_interactions=True)

        # Perform operations
        await mock_redis.ping()
        await mock_redis.set("key", "value")
        await mock_redis.get("key")

        # Check interactions were recorded
        assert len(mock_redis.interactions) == 3

        methods = [i.method for i in mock_redis.interactions]
        assert "ping" in methods
        assert "set" in methods
        assert "get" in methods

    @pytest.mark.asyncio
    async def test_performance_metrics(self):
        """Test performance metrics collection."""
        framework = ComponentIsolationTestFramework()

        # Simulate some interactions
        framework.mock_redis_client = MockRedisClient()
        await framework.mock_redis_client.ping()
        await framework.mock_redis_client.set("test", "value")

        framework._collect_interactions()

        # Get metrics
        metrics = framework.get_performance_metrics()

        assert "total_interactions" in metrics
        assert "components_used" in metrics
        assert "total_duration" in metrics
        assert metrics["total_interactions"] >= 2

    @pytest.mark.asyncio
    async def test_error_simulation(self):
        """Test error simulation functionality."""
        framework = ComponentIsolationTestFramework()
        framework.mock_redis_client = MockRedisClient()

        # Simulate Redis error
        framework.simulate_redis_error("get", ConnectionError)

        # Should raise error
        with pytest.raises(ConnectionError):
            await framework.mock_redis_client.get("test_key")

    def test_assertion_helpers(self):
        """Test framework assertion helpers."""
        framework = ComponentIsolationTestFramework()

        # Test successful assertion (no interactions)
        framework.interactions = []
        framework.assert_no_external_calls()  # Should not raise

        # Test failing assertion (has interactions)
        import time

        from tests.unit.test_component_isolation_framework import MockInteraction

        framework.interactions = [
            MockInteraction(
                timestamp=time.time(),
                component="redis",
                method="get",
                args=("key",),
                kwargs={},
                result="value",
            )
        ]

        # Should raise assertion error
        with pytest.raises(Exception):  # pytest.fail raises an exception
            framework.assert_no_external_calls()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
