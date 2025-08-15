"""Tests for metrics wiring functionality."""

import asyncio
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base_agent import BaseAgent, CLIToolType
from src.core.message_broker import Message, MessageBroker, MessageType


class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    async def run(self):
        pass


class TestBaseAgentMetricsFlush:
    """Test cases for BaseAgent metrics flush functionality."""

    @pytest.fixture
    def agent(self):
        """Create a test agent instance."""
        with (
            patch("src.agents.base_agent.get_async_database_manager"),
            patch("src.agents.base_agent.get_context_engine"),
            patch("src.agents.base_agent.task_queue"),
            patch("src.agents.base_agent.message_broker"),
        ):
            agent = TestAgent("test-agent", "test", "testing")
            agent.async_db_manager = AsyncMock()
            agent.agent_uuid = "test-uuid"
            return agent

    def test_flush_metrics_disabled_by_default(self, agent):
        """Test that metrics flush is disabled by default."""
        # Set counters to verify they're not flushed
        agent.cli_tools._counters = {
            "opencode": {
                "calls": 5,
                "success": 3,
                "failure": 2,
                "by_category": {"timeout": 1, "rate_limit": 1},
            }
        }

        # Mock the database manager to verify it's not called
        agent.async_db_manager.record_metrics_bulk = AsyncMock()

        # Flush should not do anything when HIVE_METRICS_FLUSH is not set
        asyncio.run(agent.flush_metrics())

        # Verify no metrics were written
        agent.async_db_manager.record_metrics_bulk.assert_not_called()

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_flush_metrics_with_feature_flag_enabled(self, agent):
        """Test that metrics are flushed when feature flag is enabled."""
        # Set counters
        agent.cli_tools._counters = {
            "opencode": {
                "calls": 5,
                "success": 3,
                "failure": 2,
                "by_category": {"timeout": 1, "rate_limit": 1},
            }
        }

        # Mock the database manager
        agent.async_db_manager.record_metrics_bulk = AsyncMock(return_value=7)

        # Flush metrics
        asyncio.run(agent.flush_metrics())

        # Verify metrics were written
        agent.async_db_manager.record_metrics_bulk.assert_called_once()
        call_args = agent.async_db_manager.record_metrics_bulk.call_args[0][0]

        # Verify we have the expected number of metrics
        assert len(call_args) == 7  # 3 base counters + 2 category counters

        # Verify counter metrics structure
        calls_metric = next(
            (m for m in call_args if m["metric_name"] == "cli_tool_calls_total"), None
        )
        assert calls_metric is not None
        assert calls_metric["value"] == 5.0
        assert calls_metric["labels"]["tool"] == "opencode"

        success_metric = next(
            (m for m in call_args if m["metric_name"] == "cli_tool_calls_success"), None
        )
        assert success_metric is not None
        assert success_metric["value"] == 3.0

        failure_metric = next(
            (m for m in call_args if m["metric_name"] == "cli_tool_calls_failure"), None
        )
        assert failure_metric is not None
        assert failure_metric["value"] == 2.0

        # Verify category metrics
        timeout_metric = next(
            (m for m in call_args if m["labels"].get("category") == "timeout"), None
        )
        assert timeout_metric is not None
        assert timeout_metric["value"] == 1.0

        rate_limit_metric = next(
            (m for m in call_args if m["labels"].get("category") == "rate_limit"), None
        )
        assert rate_limit_metric is not None
        assert rate_limit_metric["value"] == 1.0

        # Verify counters are cleared after flush
        assert len(agent.cli_tools._counters) == 0

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_flush_metrics_handles_missing_database_manager(self, agent):
        """Test that flush handles missing database manager gracefully."""
        agent.async_db_manager = None

        # Set counters
        agent.cli_tools._counters = {
            "opencode": {"calls": 1, "success": 1, "failure": 0, "by_category": {}}
        }

        # Should not raise an exception
        asyncio.run(agent.flush_metrics())

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_flush_metrics_handles_missing_agent_uuid(self, agent):
        """Test that flush handles missing agent UUID gracefully."""
        agent.agent_uuid = None

        # Set counters
        agent.cli_tools._counters = {
            "opencode": {"calls": 1, "success": 1, "failure": 0, "by_category": {}}
        }

        # Should not raise an exception
        asyncio.run(agent.flush_metrics())

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_flush_metrics_handles_database_error(self, agent):
        """Test that flush handles database errors gracefully."""
        # Set counters
        agent.cli_tools._counters = {
            "opencode": {"calls": 1, "success": 1, "failure": 0, "by_category": {}}
        }

        # Mock database manager to raise an exception
        agent.async_db_manager.record_metrics_bulk = AsyncMock(
            side_effect=Exception("DB error")
        )

        # Should not raise an exception
        asyncio.run(agent.flush_metrics())

        # Counters should not be cleared on error
        assert len(agent.cli_tools._counters) == 1


class TestMessageBrokerDLQMetrics:
    """Test cases for MessageBroker DLQ metrics functionality."""

    @pytest.fixture
    def broker(self):
        """Create a test message broker instance."""
        # Create broker with mock Redis client to avoid connection issues
        broker = MessageBroker()
        broker.redis_client = AsyncMock()
        return broker

    def test_move_to_dlq_tracks_reasons_when_feature_flag_disabled(self, broker):
        """Test that DLQ tracking works even when feature flag is disabled."""
        # Create a test message
        message = Message(
            id="test-msg-1",
            from_agent="sender",
            to_agent="receiver",
            topic="test",
            message_type=MessageType.DIRECT,
            payload={},
            timestamp=time.time(),
        )

        # Move to DLQ
        asyncio.run(broker._move_to_dlq(message, "test_reason"))

        # Counters should still be tracked in memory
        counters = broker.get_dlq_counters()
        assert "test_reason" in counters
        assert counters["test_reason"] == 1

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_move_to_dlq_tracks_reasons_when_feature_flag_enabled(self, broker):
        """Test that DLQ tracking works when feature flag is enabled."""
        # Create a test message
        message = Message(
            id="test-msg-1",
            from_agent="sender",
            to_agent="receiver",
            topic="test",
            message_type=MessageType.DIRECT,
            payload={},
            timestamp=time.time(),
        )

        # Move multiple messages to DLQ with same reason
        asyncio.run(broker._move_to_dlq(message, "test_reason"))

        message2 = Message(
            id="test-msg-2",
            from_agent="sender",
            to_agent="receiver",
            topic="test2",
            message_type=MessageType.DIRECT,
            payload={},
            timestamp=time.time(),
        )
        asyncio.run(broker._move_to_dlq(message2, "test_reason"))

        message3 = Message(
            id="test-msg-3",
            from_agent="sender",
            to_agent="receiver",
            topic="test3",
            message_type=MessageType.DIRECT,
            payload={},
            timestamp=time.time(),
        )
        asyncio.run(broker._move_to_dlq(message3, "another_reason"))

        # Check counters
        counters = broker.get_dlq_counters()
        assert counters["test_reason"] == 2
        assert counters["another_reason"] == 1

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_flush_dlq_metrics_disabled_when_no_counters(self, broker):
        """Test that flush does nothing when there are no counters."""
        with patch("src.core.message_broker.get_async_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            # Flush with no counters
            asyncio.run(broker.flush_dlq_metrics())

            # Database manager should not be called
            mock_db_manager.record_metrics_bulk.assert_not_called()

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_flush_dlq_metrics_with_counters(self, broker):
        """Test that DLQ metrics are flushed correctly."""
        # Set up counters
        broker._dlq_reasons = {"test_reason": 3, "another_reason": 1}

        with patch("src.core.message_broker.get_async_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_db_manager.record_metrics_bulk = AsyncMock(return_value=2)
            mock_get_db.return_value = mock_db_manager

            # Flush metrics
            asyncio.run(broker.flush_dlq_metrics())

            # Verify database call
            mock_db_manager.record_metrics_bulk.assert_called_once()
            call_args = mock_db_manager.record_metrics_bulk.call_args[0][0]

            # Verify metrics structure
            assert len(call_args) == 2

            metric1 = call_args[0]
            assert metric1["metric_name"] == "message_broker_dlq_total"
            assert metric1["value"] == 3.0
            assert metric1["labels"]["reason"] == "test_reason"

            metric2 = call_args[1]
            assert metric2["metric_name"] == "message_broker_dlq_total"
            assert metric2["value"] == 1.0
            assert metric2["labels"]["reason"] == "another_reason"

            # Verify counters are cleared
            assert len(broker._dlq_reasons) == 0

    @patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"})
    def test_flush_dlq_metrics_handles_database_error(self, broker):
        """Test that DLQ flush handles database errors gracefully."""
        # Set up counters
        broker._dlq_reasons = {"test_reason": 1}

        with patch("src.core.message_broker.get_async_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_db_manager.record_metrics_bulk = AsyncMock(
                side_effect=Exception("DB error")
            )
            mock_get_db.return_value = mock_db_manager

            # Should not raise an exception
            asyncio.run(broker.flush_dlq_metrics())

            # Counters should not be cleared on error
            assert len(broker._dlq_reasons) == 1

    def test_get_dlq_counters_returns_copy(self, broker):
        """Test that get_dlq_counters returns a copy of the counters."""
        # Set up counters
        broker._dlq_reasons = {"test_reason": 1}

        # Get counters
        counters = broker.get_dlq_counters()

        # Modify returned dict
        counters["new_reason"] = 2

        # Original counters should not be modified
        assert "new_reason" not in broker._dlq_reasons
        assert len(broker._dlq_reasons) == 1
