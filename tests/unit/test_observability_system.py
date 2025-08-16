"""Comprehensive unit tests for the observability system components.

Tests cover metrics collection, event aggregation, time series storage,
alert management, and WebSocket infrastructure.
"""

import asyncio
import pytest
import time
import json
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List

# Import components to test
try:
    from src.core.observability.metrics_collector import (
        MetricsCollector,
        SystemMetrics,
        AgentMetrics,
        TaskMetrics,
        MetricPoint,
        MetricType,
    )
    from src.core.observability.event_aggregator import (
        EventAggregator,
        EventType,
        SystemEvent,
    )
    from src.core.observability.time_series_store import (
        TimeSeriesStore,
        MetricPoint as TSMetricPoint,
        TimeSeriesQuery,
    )
    from src.core.observability.alert_manager import (
        AlertManager,
        AlertRule,
        Alert,
        AlertSeverity,
        AlertState,
    )
    from src.api.websocket.dashboard_handler import (
        DashboardWebSocketManager,
        ClientConnection,
        MessageType,
    )

    HAS_OBSERVABILITY = True
except ImportError:
    HAS_OBSERVABILITY = False


@pytest.mark.skipif(not HAS_OBSERVABILITY, reason="Observability system not available")
class TestMetricsCollector:
    """Test metrics collection functionality."""

    @pytest.fixture
    def metrics_collector(self):
        """Create a metrics collector for testing."""
        return MetricsCollector(collection_interval=1.0)

    @pytest.mark.asyncio
    async def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization."""
        assert metrics_collector.collection_interval == 1.0
        assert metrics_collector.is_collecting is False
        assert len(metrics_collector.metrics_history) == 0

    @pytest.mark.asyncio
    async def test_start_stop_collection(self, metrics_collector):
        """Test starting and stopping metrics collection."""
        await metrics_collector.start_collection()
        assert metrics_collector.is_collecting is True
        assert metrics_collector.collection_task is not None

        await metrics_collector.stop_collection()
        assert metrics_collector.is_collecting is False

    @pytest.mark.asyncio
    async def test_add_metric_point(self, metrics_collector):
        """Test adding metric points."""
        point = MetricPoint(
            name="test_metric",
            value=42.0,
            metric_type=MetricType.GAUGE,
            tags={"source": "test"},
        )

        metrics_collector.add_metric_point(point)
        assert len(metrics_collector.metrics_history) == 1
        assert metrics_collector.metrics_history[0].name == "test_metric"
        assert metrics_collector.metrics_history[0].value == 42.0

    @pytest.mark.asyncio
    async def test_get_recent_metrics(self, metrics_collector):
        """Test retrieving recent metrics."""
        # Add some test metrics
        for i in range(5):
            point = MetricPoint(
                name=f"metric_{i}",
                value=float(i),
                timestamp=time.time() - (10 * i),  # Spread over time
                tags={"test": "true"},
            )
            metrics_collector.add_metric_point(point)

        # Get recent metrics (last 60 seconds)
        recent = metrics_collector.get_recent_metrics(60.0)
        assert len(recent) == 5

        # Get recent metrics with name filter
        recent_filtered = metrics_collector.get_recent_metrics(
            60.0, metric_name_filter="metric_1"
        )
        assert len(recent_filtered) == 1
        assert recent_filtered[0].name == "metric_1"

    @pytest.mark.asyncio
    async def test_get_aggregated_metrics(self, metrics_collector):
        """Test metric aggregation."""
        # Add multiple points for same metric
        for i in range(5):
            point = MetricPoint(
                name="test_metric",
                value=float(i + 1),  # Values 1, 2, 3, 4, 5
                tags={"source": "test"},
            )
            metrics_collector.add_metric_point(point)

        # Test average aggregation
        avg = metrics_collector.get_aggregated_metrics("test_metric", 300, "avg")
        assert avg == 3.0  # (1+2+3+4+5)/5 = 3

        # Test sum aggregation
        total = metrics_collector.get_aggregated_metrics("test_metric", 300, "sum")
        assert total == 15.0  # 1+2+3+4+5 = 15

        # Test min/max aggregation
        minimum = metrics_collector.get_aggregated_metrics("test_metric", 300, "min")
        maximum = metrics_collector.get_aggregated_metrics("test_metric", 300, "max")
        assert minimum == 1.0
        assert maximum == 5.0


@pytest.mark.skipif(not HAS_OBSERVABILITY, reason="Observability system not available")
class TestEventAggregator:
    """Test event aggregation functionality."""

    @pytest.fixture
    def event_aggregator(self):
        """Create an event aggregator for testing."""
        return EventAggregator(max_events=100)

    @pytest.mark.asyncio
    async def test_event_aggregator_initialization(self, event_aggregator):
        """Test event aggregator initialization."""
        assert event_aggregator.max_events == 100
        assert len(event_aggregator.events) == 0
        assert event_aggregator.is_processing is False

    @pytest.mark.asyncio
    async def test_emit_event(self, event_aggregator):
        """Test event emission."""
        event = SystemEvent(
            event_type=EventType.SYSTEM_STARTUP,
            source="test",
            data={"test": "data"},
            severity="info",
        )

        await event_aggregator.emit_event(event)
        assert len(event_aggregator.events) == 1
        assert event_aggregator.events[0].event_type == EventType.SYSTEM_STARTUP

    @pytest.mark.asyncio
    async def test_event_subscription(self, event_aggregator):
        """Test event subscription and notification."""
        received_events = []

        def event_handler(event):
            received_events.append(event)

        # Subscribe to specific event type
        event_aggregator.subscribe(EventType.AGENT_SPAWNED, event_handler)

        # Emit matching event
        event = SystemEvent(
            event_type=EventType.AGENT_SPAWNED,
            source="test",
            data={"agent_name": "test-agent"},
        )

        await event_aggregator.emit_event(event)
        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.AGENT_SPAWNED

    @pytest.mark.asyncio
    async def test_get_recent_events(self, event_aggregator):
        """Test retrieving recent events."""
        # Emit multiple events
        for i in range(5):
            event = SystemEvent(
                event_type=EventType.TASK_COMPLETED,
                source=f"test_{i}",
                data={"task_id": f"task_{i}"},
                timestamp=time.time() - (10 * i),
            )
            await event_aggregator.emit_event(event)

        # Get recent events
        recent = event_aggregator.get_recent_events(60.0)
        assert len(recent) == 5

        # Get events filtered by type
        filtered = event_aggregator.get_recent_events(
            60.0, event_types=[EventType.TASK_COMPLETED]
        )
        assert len(filtered) == 5

        # Get events filtered by severity
        severity_filtered = event_aggregator.get_recent_events(
            60.0, severity_filter="info"
        )
        assert len(severity_filtered) == 5

    @pytest.mark.asyncio
    async def test_convenience_methods(self, event_aggregator):
        """Test convenience event emission methods."""
        # Test agent spawned event
        await event_aggregator.emit_agent_spawned("test-agent", "meta")
        assert len(event_aggregator.events) == 1
        assert event_aggregator.events[0].event_type == EventType.AGENT_SPAWNED

        # Test task completed event
        await event_aggregator.emit_task_completed("task-1", "build", 30.5, "agent-1")
        assert len(event_aggregator.events) == 2
        assert event_aggregator.events[1].event_type == EventType.TASK_COMPLETED

        # Test error event
        await event_aggregator.emit_error("Test error", "test_component")
        assert len(event_aggregator.events) == 3
        assert event_aggregator.events[2].event_type == EventType.ERROR_OCCURRED


@pytest.mark.skipif(not HAS_OBSERVABILITY, reason="Observability system not available")
class TestTimeSeriesStore:
    """Test time series storage functionality."""

    @pytest.fixture
    def ts_store(self):
        """Create a time series store for testing."""
        return TimeSeriesStore(max_points_per_series=100, retention_hours=1)

    @pytest.mark.asyncio
    async def test_store_initialization(self, ts_store):
        """Test time series store initialization."""
        assert ts_store.max_points_per_series == 100
        assert ts_store.retention_hours == 1
        assert len(ts_store.series_data) == 0

    @pytest.mark.asyncio
    async def test_store_single_point(self, ts_store):
        """Test storing a single metric point."""
        point = TSMetricPoint(
            timestamp=time.time(), value=42.0, tags={"source": "test"}
        )

        await ts_store.store_point("test_metric", point)
        assert "test_metric" in ts_store.series_data
        assert len(ts_store.series_data["test_metric"]) == 1

    @pytest.mark.asyncio
    async def test_store_multiple_points(self, ts_store):
        """Test storing multiple metric points."""
        points = []
        for i in range(5):
            point = TSMetricPoint(
                timestamp=time.time() + i, value=float(i), tags={"source": "test"}
            )
            points.append(point)

        await ts_store.store_points("test_metric", points)
        assert len(ts_store.series_data["test_metric"]) == 5

    @pytest.mark.asyncio
    async def test_query_time_series(self, ts_store):
        """Test querying time series data."""
        # Store test data
        current_time = time.time()
        points = []
        for i in range(10):
            point = TSMetricPoint(
                timestamp=current_time + i,
                value=float(i),
                tags={"source": "test", "env": "dev"},
            )
            points.append(point)

        await ts_store.store_points("test_metric", points)

        # Test basic query
        query = TimeSeriesQuery(
            metric_name="test_metric",
            start_time=current_time,
            end_time=current_time + 10,
        )

        results = await ts_store.query(query)
        assert len(results) == 10

        # Test query with tags filter
        query_filtered = TimeSeriesQuery(
            metric_name="test_metric",
            start_time=current_time,
            end_time=current_time + 10,
            tags_filter={"source": "test"},
        )

        results_filtered = await ts_store.query(query_filtered)
        assert len(results_filtered) == 10

    @pytest.mark.asyncio
    async def test_query_with_aggregation(self, ts_store):
        """Test querying with aggregation."""
        # Store test data in 2-second intervals
        current_time = time.time()
        points = []
        for i in range(6):  # 6 points over 6 seconds
            point = TSMetricPoint(
                timestamp=current_time + i,
                value=float(i + 1),  # Values 1,2,3,4,5,6
                tags={"source": "test"},
            )
            points.append(point)

        await ts_store.store_points("test_metric", points)

        # Query with 2-second aggregation intervals
        query = TimeSeriesQuery(
            metric_name="test_metric",
            start_time=current_time,
            end_time=current_time + 6,
            aggregation="avg",
            interval=2.0,
        )

        results = await ts_store.query(query)
        assert len(results) == 3  # 3 buckets of 2 seconds each


@pytest.mark.skipif(not HAS_OBSERVABILITY, reason="Observability system not available")
class TestAlertManager:
    """Test alert management functionality."""

    @pytest.fixture
    def alert_manager(self):
        """Create an alert manager for testing."""
        return AlertManager(check_interval=1.0)

    @pytest.mark.asyncio
    async def test_alert_manager_initialization(self, alert_manager):
        """Test alert manager initialization."""
        assert alert_manager.check_interval == 1.0
        assert len(alert_manager.alert_rules) == 0
        assert len(alert_manager.active_alerts) == 0
        assert alert_manager.is_evaluating is False

    def test_add_alert_rule(self, alert_manager):
        """Test adding alert rules."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test alert rule",
            metric_name="test_metric",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
        )

        alert_manager.add_alert_rule(rule)
        assert "test_rule" in alert_manager.alert_rules
        assert alert_manager.alert_rules["test_rule"].name == "Test Rule"

    def test_enable_disable_rule(self, alert_manager):
        """Test enabling and disabling alert rules."""
        rule = AlertRule(
            rule_id="test_rule",
            name="Test Rule",
            description="A test alert rule",
            metric_name="test_metric",
            condition="gt",
            threshold=100.0,
            severity=AlertSeverity.WARNING,
        )

        alert_manager.add_alert_rule(rule)
        assert alert_manager.alert_rules["test_rule"].enabled is True

        alert_manager.disable_alert_rule("test_rule")
        assert alert_manager.alert_rules["test_rule"].enabled is False

        alert_manager.enable_alert_rule("test_rule")
        assert alert_manager.alert_rules["test_rule"].enabled is True

    def test_alert_rule_validation(self):
        """Test alert rule validation."""
        # Test invalid condition
        with pytest.raises(ValueError):
            AlertRule(
                rule_id="invalid_rule",
                name="Invalid Rule",
                description="Invalid condition",
                metric_name="test_metric",
                condition="invalid",
                threshold=100.0,
            )

        # Test invalid evaluation window
        with pytest.raises(ValueError):
            AlertRule(
                rule_id="invalid_rule",
                name="Invalid Rule",
                description="Invalid window",
                metric_name="test_metric",
                condition="gt",
                threshold=100.0,
                evaluation_window=-1.0,
            )

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, alert_manager):
        """Test acknowledging alerts."""
        # Create a mock alert
        alert = Alert(
            alert_id="test_alert",
            rule_id="test_rule",
            rule_name="Test Rule",
            metric_name="test_metric",
            current_value=150.0,
            threshold=100.0,
            condition="gt",
            severity=AlertSeverity.WARNING,
        )

        alert_manager.active_alerts["test_alert"] = alert

        # Acknowledge the alert
        success = await alert_manager.acknowledge_alert(
            "test_alert", "test_user", "Investigating issue"
        )

        assert success is True
        assert alert.state == AlertState.ACKNOWLEDGED
        assert alert.acknowledged_by == "test_user"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test resolving alerts."""
        # Create a mock alert
        alert = Alert(
            alert_id="test_alert",
            rule_id="test_rule",
            rule_name="Test Rule",
            metric_name="test_metric",
            current_value=150.0,
            threshold=100.0,
            condition="gt",
            severity=AlertSeverity.WARNING,
        )

        alert_manager.active_alerts["test_alert"] = alert

        # Resolve the alert
        await alert_manager.resolve_alert("test_alert", "Issue fixed")

        assert "test_alert" not in alert_manager.active_alerts
        assert len(alert_manager.resolved_alerts) == 1
        assert alert_manager.resolved_alerts[0].state == AlertState.RESOLVED

    def test_create_default_rules(self, alert_manager):
        """Test creating default alert rules."""
        alert_manager.create_default_rules()
        assert len(alert_manager.alert_rules) > 0

        # Check for specific default rules
        assert "high_cpu_usage" in alert_manager.alert_rules
        assert "high_memory_usage" in alert_manager.alert_rules
        assert "low_disk_space" in alert_manager.alert_rules


@pytest.mark.skipif(not HAS_OBSERVABILITY, reason="Observability system not available")
class TestWebSocketManager:
    """Test WebSocket manager functionality."""

    @pytest.fixture
    def websocket_manager(self):
        """Create a WebSocket manager for testing."""
        return DashboardWebSocketManager()

    @pytest.mark.asyncio
    async def test_websocket_manager_initialization(self, websocket_manager):
        """Test WebSocket manager initialization."""
        assert len(websocket_manager.connections) == 0
        assert websocket_manager.is_running is False
        assert websocket_manager.heartbeat_interval == 30.0

    @pytest.mark.asyncio
    async def test_start_stop_manager(self, websocket_manager):
        """Test starting and stopping the WebSocket manager."""
        await websocket_manager.start()
        assert websocket_manager.is_running is True

        await websocket_manager.stop()
        assert websocket_manager.is_running is False

    @pytest.mark.asyncio
    async def test_client_connection_tracking(self, websocket_manager):
        """Test client connection tracking."""
        # Mock WebSocket
        mock_websocket = AsyncMock()
        mock_websocket.accept = AsyncMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.headers = {"user-agent": "test-browser"}

        # Connect client
        with patch.object(websocket_manager, "_initialize_integrations"):
            client_id = await websocket_manager.connect_client(mock_websocket)

        assert client_id in websocket_manager.connections
        assert len(websocket_manager.connections) == 1

        # Disconnect client
        await websocket_manager.disconnect_client(client_id)
        assert len(websocket_manager.connections) == 0

    @pytest.mark.asyncio
    async def test_subscription_management(self, websocket_manager):
        """Test client subscription management."""
        # Create mock client
        client = ClientConnection(client_id="test_client", websocket=AsyncMock())
        client.websocket.send_text = AsyncMock()

        websocket_manager.connections["test_client"] = client

        # Subscribe to topics
        await websocket_manager.subscribe_client(
            "test_client", ["system_events", "alerts"]
        )

        assert "system_events" in client.subscriptions
        assert "alerts" in client.subscriptions

        # Unsubscribe from topics
        await websocket_manager.unsubscribe_client("test_client", ["system_events"])

        assert "system_events" not in client.subscriptions
        assert "alerts" in client.subscriptions

    @pytest.mark.asyncio
    async def test_message_broadcasting(self, websocket_manager):
        """Test message broadcasting to clients."""
        # Create mock clients
        clients = []
        for i in range(3):
            client = ClientConnection(client_id=f"client_{i}", websocket=AsyncMock())
            client.websocket.send_text = AsyncMock()
            clients.append(client)
            websocket_manager.connections[f"client_{i}"] = client

        # Subscribe clients to different topics
        clients[0].subscriptions.add("alerts")
        clients[1].subscriptions.add("alerts")
        clients[2].subscriptions.add("system_events")

        # Broadcast to subscribers
        test_message = {"type": "alert", "data": "test alert"}
        await websocket_manager.broadcast_to_subscribers("alerts", test_message)

        # Check that only subscribed clients received the message
        assert clients[0].websocket.send_text.called
        assert clients[1].websocket.send_text.called
        assert not clients[2].websocket.send_text.called

    @pytest.mark.asyncio
    async def test_client_message_handling(self, websocket_manager):
        """Test handling messages from clients."""
        # Create mock client
        client = ClientConnection(client_id="test_client", websocket=AsyncMock())
        client.websocket.send_text = AsyncMock()
        websocket_manager.connections["test_client"] = client

        # Test subscription message
        subscription_message = json.dumps(
            {
                "type": "subscribe",
                "payload": {"subscriptions": ["system_events", "alerts"]},
            }
        )

        await websocket_manager.handle_client_message(
            "test_client", subscription_message
        )

        assert "system_events" in client.subscriptions
        assert "alerts" in client.subscriptions

        # Test heartbeat message
        heartbeat_message = json.dumps({"type": "heartbeat", "payload": {}})

        old_heartbeat = client.last_heartbeat
        await websocket_manager.handle_client_message("test_client", heartbeat_message)

        assert client.last_heartbeat > old_heartbeat

    def test_connection_statistics(self, websocket_manager):
        """Test connection statistics generation."""
        # Add mock connections
        for i in range(3):
            client = ClientConnection(
                client_id=f"client_{i}",
                websocket=AsyncMock(),
                connected_at=time.time() - (i * 10),  # Different connection times
            )
            if i < 2:
                client.subscriptions.add("alerts")
            if i % 2 == 0:
                client.subscriptions.add("system_events")

            websocket_manager.connections[f"client_{i}"] = client

        stats = websocket_manager.get_connection_statistics()

        assert stats["total_connections"] == 3
        assert stats["healthy_connections"] == 3
        assert "alerts" in stats["subscription_distribution"]
        assert "system_events" in stats["subscription_distribution"]
        assert stats["subscription_distribution"]["alerts"] == 2
        assert stats["subscription_distribution"]["system_events"] == 2


@pytest.mark.skipif(not HAS_OBSERVABILITY, reason="Observability system not available")
class TestIntegration:
    """Integration tests for the observability system."""

    @pytest.mark.asyncio
    async def test_metrics_to_websocket_flow(self):
        """Test end-to-end flow from metrics to WebSocket."""
        # Create components
        metrics_collector = MetricsCollector(collection_interval=0.1)
        event_aggregator = EventAggregator()
        websocket_manager = DashboardWebSocketManager()

        # Start components
        await metrics_collector.start_collection()
        await event_aggregator.start_processing()

        # Mock WebSocket integration
        with patch.object(websocket_manager, "_initialize_integrations"):
            await websocket_manager.start()

        # Add test metrics
        test_point = MetricPoint(
            name="test_cpu_usage",
            value=85.0,
            metric_type=MetricType.GAUGE,
            tags={"source": "system"},
        )
        metrics_collector.add_metric_point(test_point)

        # Emit test event
        test_event = SystemEvent(
            event_type=EventType.AGENT_SPAWNED,
            source="test",
            data={"agent_name": "test-agent"},
        )
        await event_aggregator.emit_event(test_event)

        # Verify metrics were collected
        recent_metrics = metrics_collector.get_recent_metrics(60.0)
        assert len(recent_metrics) == 1
        assert recent_metrics[0].name == "test_cpu_usage"

        # Verify events were aggregated
        recent_events = event_aggregator.get_recent_events(60.0)
        assert len(recent_events) == 1
        assert recent_events[0].event_type == EventType.AGENT_SPAWNED

        # Clean up
        await metrics_collector.stop_collection()
        await event_aggregator.stop_processing()
        await websocket_manager.stop()

    @pytest.mark.asyncio
    async def test_alert_integration_flow(self):
        """Test integration between metrics, time series, and alerts."""
        # Create components
        metrics_collector = MetricsCollector()
        time_series_store = TimeSeriesStore()
        alert_manager = AlertManager(check_interval=0.1)

        # Create alert rule
        rule = AlertRule(
            rule_id="high_cpu",
            name="High CPU Usage",
            description="CPU usage too high",
            metric_name="system_cpu_percent",
            condition="gt",
            threshold=80.0,
            severity=AlertSeverity.WARNING,
            min_duration=0.0,  # Immediate alerting for test
        )
        alert_manager.add_alert_rule(rule)

        # Mock time series query to return high CPU value
        async def mock_query(query):
            return [TSMetricPoint(timestamp=time.time(), value=90.0)]

        with patch.object(time_series_store, "query", mock_query):
            # Start alert evaluation
            await alert_manager.start_evaluation()

            # Wait for evaluation cycle
            await asyncio.sleep(0.2)

            # Check if alert was triggered
            active_alerts = alert_manager.get_active_alerts()
            assert len(active_alerts) > 0

            # Stop evaluation
            await alert_manager.stop_evaluation()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
