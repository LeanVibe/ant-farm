"""
Real-time performance monitoring dashboard for LeanVibe Agent Hive.

Provides real-time metrics collection, alerting, and dashboard capabilities
with WebSocket-based live updates and threshold monitoring.
"""

import asyncio
import json
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import structlog
from fastapi import WebSocket

from .analytics import PerformanceCollector, PerformanceMetric, SystemResource

logger = structlog.get_logger()


@dataclass
class PerformanceAlert:
    """Performance alert data structure."""

    id: str
    metric_name: str
    current_value: float
    threshold_value: float
    threshold_type: str  # 'above', 'below'
    severity: str  # 'critical', 'warning', 'info'
    message: str
    timestamp: float
    acknowledged: bool = False
    auto_resolved: bool = False


@dataclass
class MetricThreshold:
    """Threshold configuration for metrics."""

    metric_name: str
    warning_above: Optional[float] = None
    critical_above: Optional[float] = None
    warning_below: Optional[float] = None
    critical_below: Optional[float] = None
    time_window_seconds: int = 60
    min_samples: int = 3


class RealTimeMonitor:
    """Real-time performance monitoring with alerts and dashboards."""

    def __init__(self, performance_collector: PerformanceCollector):
        self.collector = performance_collector
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        self.websocket_connections: List[WebSocket] = []
        self.metric_thresholds: Dict[str, MetricThreshold] = {}
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Setup default thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Setup default performance thresholds."""
        self.metric_thresholds = {
            "cpu_usage": MetricThreshold(
                metric_name="cpu_usage", warning_above=70.0, critical_above=90.0
            ),
            "memory_usage": MetricThreshold(
                metric_name="memory_usage", warning_above=80.0, critical_above=95.0
            ),
            "disk_usage": MetricThreshold(
                metric_name="disk_usage", warning_above=85.0, critical_above=95.0
            ),
            "api_response_time": MetricThreshold(
                metric_name="api_response_time",
                warning_above=100.0,  # 100ms
                critical_above=500.0,  # 500ms
                time_window_seconds=30,
                min_samples=5,
            ),
            "database_query_time": MetricThreshold(
                metric_name="database_query_time",
                warning_above=50.0,  # 50ms
                critical_above=200.0,  # 200ms
                time_window_seconds=60,
                min_samples=3,
            ),
            "task_queue_depth": MetricThreshold(
                metric_name="task_queue_depth", warning_above=100, critical_above=500
            ),
            "agent_failure_rate": MetricThreshold(
                metric_name="agent_failure_rate",
                warning_above=0.05,  # 5%
                critical_above=0.20,  # 20%
            ),
        }

    async def add_websocket_connection(self, websocket: WebSocket):
        """Add a WebSocket connection for real-time updates."""
        await websocket.accept()
        self.websocket_connections.append(websocket)

        # Send current state to new connection
        await self._send_current_state(websocket)

        logger.info(
            "Real-time monitoring WebSocket connected",
            total_connections=len(self.websocket_connections),
        )

    def remove_websocket_connection(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            logger.info(
                "Real-time monitoring WebSocket disconnected",
                total_connections=len(self.websocket_connections),
            )

    async def _send_current_state(self, websocket: WebSocket):
        """Send current monitoring state to a WebSocket."""
        try:
            state = {
                "type": "monitoring_state",
                "data": {
                    "monitoring_active": self.monitoring_active,
                    "active_alerts": [
                        asdict(alert) for alert in self.active_alerts.values()
                    ],
                    "metrics_summary": self.collector.get_metrics_summary(
                        5
                    ),  # Last 5 minutes
                    "system_health": self.collector.get_system_health_score(),
                    "timestamp": time.time(),
                },
            }
            await websocket.send_text(json.dumps(state))
        except Exception as e:
            logger.error("Failed to send current state to WebSocket", error=str(e))

    async def broadcast_to_websockets(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSockets."""
        if not self.websocket_connections:
            return

        disconnected = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to broadcast to WebSocket", error=str(e))
                disconnected.append(websocket)

        # Remove disconnected clients
        for websocket in disconnected:
            self.remove_websocket_connection(websocket)

    async def check_thresholds(self):
        """Check all metrics against configured thresholds."""
        current_time = time.time()
        new_alerts = []
        resolved_alerts = []

        # Get recent system resources
        recent_resources = [
            r
            for r in self.collector.system_resources
            if current_time - r.timestamp <= 60
        ]

        if recent_resources:
            latest_resource = recent_resources[-1]

            # Check system resource thresholds
            await self._check_resource_thresholds(
                latest_resource, new_alerts, resolved_alerts
            )

        # Check API performance thresholds
        await self._check_api_thresholds(new_alerts, resolved_alerts)

        # Check custom metric thresholds
        await self._check_custom_thresholds(new_alerts, resolved_alerts)

        # Process new alerts
        for alert in new_alerts:
            self.active_alerts[alert.id] = alert
            self.alert_history.append(alert)
            await self._broadcast_alert(alert, "new")

        # Process resolved alerts
        for alert_id in resolved_alerts:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.auto_resolved = True
                del self.active_alerts[alert_id]
                await self._broadcast_alert(alert, "resolved")

    async def _check_resource_thresholds(
        self,
        resource: SystemResource,
        new_alerts: List[PerformanceAlert],
        resolved_alerts: List[str],
    ):
        """Check system resource thresholds."""
        checks = [
            ("cpu_usage", resource.cpu_percent),
            ("memory_usage", resource.memory_percent),
            ("disk_usage", resource.disk_usage_percent),
        ]

        for metric_name, value in checks:
            threshold = self.metric_thresholds.get(metric_name)
            if not threshold:
                continue

            alert_id = f"{metric_name}_threshold"

            # Check if alert should be triggered
            alert_triggered = False
            severity = "info"

            if threshold.critical_above and value >= threshold.critical_above:
                alert_triggered = True
                severity = "critical"
            elif threshold.warning_above and value >= threshold.warning_above:
                alert_triggered = True
                severity = "warning"
            elif threshold.critical_below and value <= threshold.critical_below:
                alert_triggered = True
                severity = "critical"
            elif threshold.warning_below and value <= threshold.warning_below:
                alert_triggered = True
                severity = "warning"

            if alert_triggered and alert_id not in self.active_alerts:
                # Create new alert
                alert = PerformanceAlert(
                    id=alert_id,
                    metric_name=metric_name,
                    current_value=value,
                    threshold_value=threshold.critical_above
                    or threshold.warning_above
                    or 0,
                    threshold_type="above",
                    severity=severity,
                    message=f"{metric_name.replace('_', ' ').title()} is {value:.1f}%",
                    timestamp=time.time(),
                )
                new_alerts.append(alert)

            elif not alert_triggered and alert_id in self.active_alerts:
                # Resolve existing alert
                resolved_alerts.append(alert_id)

    async def _check_api_thresholds(
        self, new_alerts: List[PerformanceAlert], resolved_alerts: List[str]
    ):
        """Check API performance thresholds."""
        threshold = self.metric_thresholds.get("api_response_time")
        if not threshold:
            return

        # Get recent API performance data
        api_performance = self.collector.get_api_performance_summary(
            threshold.time_window_seconds // 60 or 1
        )

        for endpoint, perf in api_performance.items():
            if perf.request_count < threshold.min_samples:
                continue

            alert_id = f"api_response_time_{endpoint.replace('/', '_')}"
            avg_time = perf.avg_response_time

            alert_triggered = False
            severity = "info"

            if threshold.critical_above and avg_time >= threshold.critical_above:
                alert_triggered = True
                severity = "critical"
            elif threshold.warning_above and avg_time >= threshold.warning_above:
                alert_triggered = True
                severity = "warning"

            if alert_triggered and alert_id not in self.active_alerts:
                alert = PerformanceAlert(
                    id=alert_id,
                    metric_name="api_response_time",
                    current_value=avg_time,
                    threshold_value=threshold.critical_above
                    or threshold.warning_above
                    or 0,
                    threshold_type="above",
                    severity=severity,
                    message=f"API endpoint {endpoint} avg response time is {avg_time:.1f}ms",
                    timestamp=time.time(),
                )
                new_alerts.append(alert)

            elif not alert_triggered and alert_id in self.active_alerts:
                resolved_alerts.append(alert_id)

    async def _check_custom_thresholds(
        self, new_alerts: List[PerformanceAlert], resolved_alerts: List[str]
    ):
        """Check custom metric thresholds."""
        # Get recent metrics
        current_time = time.time()
        recent_metrics = [
            m
            for m in self.collector.metrics_history
            if current_time - m.timestamp <= 300  # Last 5 minutes
        ]

        # Group by metric name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.name].append(metric)

        # Check thresholds for metrics with sufficient data
        for metric_name, metrics in metrics_by_name.items():
            threshold = self.metric_thresholds.get(metric_name)
            if not threshold or len(metrics) < threshold.min_samples:
                continue

            # Calculate average value over time window
            window_metrics = [
                m
                for m in metrics
                if current_time - m.timestamp <= threshold.time_window_seconds
            ]

            if len(window_metrics) < threshold.min_samples:
                continue

            avg_value = sum(m.value for m in window_metrics) / len(window_metrics)
            alert_id = f"{metric_name}_custom_threshold"

            alert_triggered = False
            severity = "info"

            if threshold.critical_above and avg_value >= threshold.critical_above:
                alert_triggered = True
                severity = "critical"
            elif threshold.warning_above and avg_value >= threshold.warning_above:
                alert_triggered = True
                severity = "warning"
            elif threshold.critical_below and avg_value <= threshold.critical_below:
                alert_triggered = True
                severity = "critical"
            elif threshold.warning_below and avg_value <= threshold.warning_below:
                alert_triggered = True
                severity = "warning"

            if alert_triggered and alert_id not in self.active_alerts:
                alert = PerformanceAlert(
                    id=alert_id,
                    metric_name=metric_name,
                    current_value=avg_value,
                    threshold_value=(
                        threshold.critical_above
                        or threshold.warning_above
                        or threshold.critical_below
                        or threshold.warning_below
                        or 0
                    ),
                    threshold_type="above"
                    if threshold.critical_above or threshold.warning_above
                    else "below",
                    severity=severity,
                    message=f"{metric_name.replace('_', ' ').title()} is {avg_value:.2f}",
                    timestamp=time.time(),
                )
                new_alerts.append(alert)

            elif not alert_triggered and alert_id in self.active_alerts:
                resolved_alerts.append(alert_id)

    async def _broadcast_alert(self, alert: PerformanceAlert, action: str):
        """Broadcast alert to all connected clients."""
        message = {
            "type": "performance_alert",
            "action": action,  # 'new', 'resolved', 'acknowledged'
            "alert": asdict(alert),
            "timestamp": time.time(),
        }

        await self.broadcast_to_websockets(message)

        # Log the alert
        log_level = logger.error if alert.severity == "critical" else logger.warning
        log_level(
            "Performance alert",
            alert_id=alert.id,
            metric=alert.metric_name,
            value=alert.current_value,
            threshold=alert.threshold_value,
            severity=alert.severity,
            action=action,
        )

    async def start_monitoring(self):
        """Start real-time monitoring."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

        logger.info("Real-time performance monitoring started")

    async def stop_monitoring(self):
        """Stop real-time monitoring."""
        if not self.monitoring_active:
            return

        self.monitoring_active = False

        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Real-time performance monitoring stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        try:
            while self.monitoring_active:
                # Check thresholds
                await self.check_thresholds()

                # Broadcast current metrics to connected clients
                if self.websocket_connections:
                    await self._broadcast_current_metrics()

                # Wait before next check
                await asyncio.sleep(5)  # Check every 5 seconds

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Error in monitoring loop", error=str(e))
            await asyncio.sleep(10)  # Wait before retrying

    async def _broadcast_current_metrics(self):
        """Broadcast current metrics to all connected clients."""
        try:
            # Get current system state
            system_health = self.collector.get_system_health_score()
            metrics_summary = self.collector.get_metrics_summary(1)  # Last minute

            # Get latest system resource
            latest_resource = None
            if self.collector.system_resources:
                latest_resource = asdict(self.collector.system_resources[-1])

            message = {
                "type": "metrics_update",
                "data": {
                    "system_health_score": system_health,
                    "metrics_summary": metrics_summary,
                    "latest_system_resource": latest_resource,
                    "active_alerts_count": len(self.active_alerts),
                    "monitoring_active": self.monitoring_active,
                    "timestamp": time.time(),
                },
            }

            await self.broadcast_to_websockets(message)

        except Exception as e:
            logger.error("Failed to broadcast current metrics", error=str(e))

    def acknowledge_alert(self, alert_id: str, user_id: str = "system") -> bool:
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info("Alert acknowledged", alert_id=alert_id, user_id=user_id)

            # Broadcast acknowledgment
            asyncio.create_task(
                self._broadcast_alert(self.active_alerts[alert_id], "acknowledged")
            )
            return True
        return False

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "monitoring_active": self.monitoring_active,
            "active_alerts": [asdict(alert) for alert in self.active_alerts.values()],
            "alert_history": [
                asdict(alert) for alert in self.alert_history[-50:]
            ],  # Last 50
            "system_health_score": self.collector.get_system_health_score(),
            "metrics_summary": self.collector.get_metrics_summary(60),  # Last hour
            "api_performance": {
                k: v.to_dict()
                for k, v in self.collector.get_api_performance_summary(60).items()
            },
            "performance_alerts": self.collector.get_performance_alerts(),
            "connected_clients": len(self.websocket_connections),
            "thresholds": {
                name: asdict(threshold)
                for name, threshold in self.metric_thresholds.items()
            },
            "timestamp": time.time(),
        }


# Global real-time monitor instance
real_time_monitor = RealTimeMonitor(PerformanceCollector())


async def start_real_time_monitoring():
    """Start the global real-time monitoring system."""
    await real_time_monitor.start_monitoring()


async def stop_real_time_monitoring():
    """Stop the global real-time monitoring system."""
    await real_time_monitor.stop_monitoring()


def get_real_time_monitor() -> RealTimeMonitor:
    """Get the global real-time monitor instance."""
    return real_time_monitor
