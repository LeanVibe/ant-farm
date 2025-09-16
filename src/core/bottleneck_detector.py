"""Bottleneck detection and threshold monitoring for LeanVibe Agent Hive.

This module provides intelligent bottleneck detection including:
- Configurable performance thresholds
- Anomaly detection algorithms
- Resource utilization monitoring
- Predictive bottleneck identification
- Automated threshold adjustments
"""

import asyncio
import logging
import time
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import json

from .metrics_collector import metrics_collector, MetricPoint

logger = logging.getLogger(__name__)


class Severity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status states."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class ThresholdConfig:
    """Configuration for a performance threshold."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    comparison: str = "gt"  # gt, lt, gte, lte
    duration_seconds: int = 60  # How long threshold must be exceeded
    cooldown_seconds: int = 300  # Minimum time between alerts
    enabled: bool = True


@dataclass
class Alert:
    """Represents a performance alert."""
    id: str
    title: str
    description: str
    severity: Severity
    metric_name: str
    current_value: float
    threshold_value: float
    status: AlertStatus
    created_at: float
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyPattern:
    """Represents a detected anomaly pattern."""
    metric_name: str
    pattern_type: str  # "spike", "drop", "trend", "seasonal"
    confidence: float
    description: str
    detected_at: float
    severity: Severity
    metadata: Dict[str, Any] = field(default_factory=dict)


class BottleneckDetector:
    """Intelligent bottleneck detection and alerting system."""

    def __init__(self):
        self.thresholds: Dict[str, ThresholdConfig] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        self.anomalies: List[AnomalyPattern] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        self.is_running = False
        self.monitoring_task: Optional[asyncio.Task] = None

        # Initialize default thresholds
        self._init_default_thresholds()

    def _init_default_thresholds(self):
        """Initialize default performance thresholds."""
        default_thresholds = [
            # System thresholds
            ThresholdConfig("cpu_percent", 70.0, 90.0, "gt", 60, 300),
            ThresholdConfig("memory_percent", 80.0, 95.0, "gt", 60, 300),
            ThresholdConfig("disk_usage_percent", 85.0, 95.0, "gt", 300, 600),

            # Application thresholds
            ThresholdConfig("http_request_duration_seconds", 2.0, 5.0, "gt", 30, 180),
            ThresholdConfig("api_errors_total", 10.0, 50.0, "gt", 60, 300),  # per minute
            ThresholdConfig("websocket_connections", 1000, 2000, "gt", 30, 300),

            # Database thresholds
            ThresholdConfig("db_connections_active", 50, 100, "gt", 60, 300),
            ThresholdConfig("db_query_duration_seconds", 1.0, 3.0, "gt", 30, 180),

            # Message broker thresholds
            ThresholdConfig("broker_queue_depth", 1000, 5000, "gt", 60, 300),
            ThresholdConfig("broker_connections_active", 200, 500, "gt", 60, 300),

            # Agent thresholds
            ThresholdConfig("agent_response_time_seconds", 5.0, 15.0, "gt", 60, 300),
            ThresholdConfig("agents_active", 100, 200, "gt", 120, 600),  # Too many agents
        ]

        for threshold in default_thresholds:
            self.thresholds[threshold.metric_name] = threshold

    def add_threshold(self, threshold: ThresholdConfig):
        """Add a custom performance threshold."""
        self.thresholds[threshold.metric_name] = threshold
        logger.info(f"Added threshold for {threshold.metric_name}")

    def remove_threshold(self, metric_name: str):
        """Remove a performance threshold."""
        if metric_name in self.thresholds:
            del self.thresholds[metric_name]
            logger.info(f"Removed threshold for {metric_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)

    async def start_monitoring(self):
        """Start the bottleneck detection monitoring."""
        if self.is_running:
            return

        self.is_running = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Bottleneck detection started")

    async def stop_monitoring(self):
        """Stop the bottleneck detection monitoring."""
        if not self.is_running:
            return

        self.is_running = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("Bottleneck detection stopped")

    async def _monitoring_loop(self):
        """Main monitoring loop for bottleneck detection."""
        while self.is_running:
            try:
                await self._check_thresholds()
                await self._detect_anomalies()
                await self._cleanup_resolved_alerts()

                await asyncio.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(10)

    async def _check_thresholds(self):
        """Check all configured thresholds against current metrics."""
        for metric_name, threshold in self.thresholds.items():
            if not threshold.enabled:
                continue

            try:
                # Get recent metric values
                recent_points = metrics_collector.get_metric_values(
                    metric_name,
                    limit=max(10, threshold.duration_seconds // 10)
                )

                if not recent_points:
                    continue

                # Check if threshold is exceeded
                current_value = recent_points[-1].value
                exceeded = self._check_threshold_exceeded(
                    threshold, recent_points, time.time()
                )

                if exceeded:
                    await self._create_alert(threshold, current_value, recent_points)
                else:
                    # Check if we should resolve existing alerts
                    alert_key = f"{metric_name}_{threshold.warning_threshold}"
                    if alert_key in self.active_alerts:
                        await self._resolve_alert(alert_key)

            except Exception as e:
                logger.error(f"Error checking threshold for {metric_name}: {str(e)}")

    def _check_threshold_exceeded(self, threshold: ThresholdConfig,
                                points: List[MetricPoint], current_time: float) -> bool:
        """Check if a threshold has been exceeded for the required duration."""
        if len(points) < 2:
            return False

        # Check recent values within the duration window
        window_start = current_time - threshold.duration_seconds
        recent_values = [
            p.value for p in points
            if p.timestamp >= window_start
        ]

        if not recent_values:
            return False

        # Apply comparison operator
        if threshold.comparison == "gt":
            exceeded = all(v > threshold.critical_threshold for v in recent_values)
        elif threshold.comparison == "gte":
            exceeded = all(v >= threshold.critical_threshold for v in recent_values)
        elif threshold.comparison == "lt":
            exceeded = all(v < threshold.critical_threshold for v in recent_values)
        elif threshold.comparison == "lte":
            exceeded = all(v <= threshold.critical_threshold for v in recent_values)
        else:
            logger.warning(f"Unknown comparison operator: {threshold.comparison}")
            return False

        return exceeded

    async def _create_alert(self, threshold: ThresholdConfig, current_value: float,
                          recent_points: List[MetricPoint]):
        """Create a new alert for threshold violation."""
        alert_key = f"{threshold.metric_name}_{threshold.critical_threshold}"

        # Check cooldown period
        if alert_key in self.active_alerts:
            last_alert = self.active_alerts[alert_key]
            if time.time() - last_alert.created_at < threshold.cooldown_seconds:
                return  # Still in cooldown

        # Determine severity
        if current_value >= threshold.critical_threshold:
            severity = Severity.CRITICAL
        elif current_value >= threshold.warning_threshold:
            severity = Severity.HIGH
        else:
            severity = Severity.MEDIUM

        # Create alert
        alert = Alert(
            id=f"alert_{int(time.time())}_{alert_key}",
            title=f"Performance threshold exceeded: {threshold.metric_name}",
            description=f"{threshold.metric_name} is {current_value:.2f}, "
                       f"exceeding threshold of {threshold.critical_threshold}",
            severity=severity,
            metric_name=threshold.metric_name,
            current_value=current_value,
            threshold_value=threshold.critical_threshold,
            status=AlertStatus.ACTIVE,
            created_at=time.time(),
            metadata={
                "comparison": threshold.comparison,
                "duration_seconds": threshold.duration_seconds,
                "recent_values": [p.value for p in recent_points[-5:]]
            }
        )

        self.active_alerts[alert_key] = alert

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {str(e)}")

        logger.warning(f"Alert created: {alert.title} (severity: {severity.value})")

    async def _resolve_alert(self, alert_key: str):
        """Resolve an active alert."""
        if alert_key not in self.active_alerts:
            return

        alert = self.active_alerts[alert_key]
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = time.time()

        # Move to resolved alerts
        self.resolved_alerts.append(alert)
        del self.active_alerts[alert_key]

        # Keep only last 100 resolved alerts
        if len(self.resolved_alerts) > 100:
            self.resolved_alerts = self.resolved_alerts[-100:]

        logger.info(f"Alert resolved: {alert.title}")

    async def _detect_anomalies(self):
        """Detect anomalies in metric patterns."""
        for metric_name in metrics_collector.metrics.keys():
            try:
                points = metrics_collector.get_metric_values(metric_name, limit=50)
                if len(points) < 20:  # Need minimum data points
                    continue

                anomaly = self._detect_metric_anomaly(metric_name, points)
                if anomaly:
                    self.anomalies.append(anomaly)
                    logger.warning(f"Anomaly detected: {anomaly.description}")

                    # Keep only last 50 anomalies
                    if len(self.anomalies) > 50:
                        self.anomalies = self.anomalies[-50:]

            except Exception as e:
                logger.error(f"Error detecting anomalies for {metric_name}: {str(e)}")

    def _detect_metric_anomaly(self, metric_name: str, points: List[MetricPoint]) -> Optional[AnomalyPattern]:
        """Detect anomalies in a metric's time series."""
        if len(points) < 20:
            return None

        values = [p.value for p in points]

        # Calculate statistical measures
        mean = statistics.mean(values)
        stdev = statistics.stdev(values) if len(values) > 1 else 0

        if stdev == 0:
            return None

        # Check for spikes (sudden increases)
        recent_avg = statistics.mean(values[-5:])
        overall_avg = statistics.mean(values[:-5])

        if recent_avg > overall_avg + (3 * stdev):
            confidence = min(1.0, (recent_avg - overall_avg) / (3 * stdev))
            return AnomalyPattern(
                metric_name=metric_name,
                pattern_type="spike",
                confidence=confidence,
                description=f"Sudden spike detected in {metric_name}: "
                           f"{recent_avg:.2f} vs baseline {overall_avg:.2f}",
                detected_at=time.time(),
                severity=Severity.HIGH if confidence > 0.8 else Severity.MEDIUM,
                metadata={
                    "recent_avg": recent_avg,
                    "baseline_avg": overall_avg,
                    "standard_deviation": stdev
                }
            )

        # Check for drops (sudden decreases)
        if recent_avg < overall_avg - (3 * stdev):
            confidence = min(1.0, (overall_avg - recent_avg) / (3 * stdev))
            return AnomalyPattern(
                metric_name=metric_name,
                pattern_type="drop",
                confidence=confidence,
                description=f"Sudden drop detected in {metric_name}: "
                           f"{recent_avg:.2f} vs baseline {overall_avg:.2f}",
                detected_at=time.time(),
                severity=Severity.HIGH if confidence > 0.8 else Severity.MEDIUM,
                metadata={
                    "recent_avg": recent_avg,
                    "baseline_avg": overall_avg,
                    "standard_deviation": stdev
                }
            )

        return None

    async def _cleanup_resolved_alerts(self):
        """Clean up old resolved alerts."""
        cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago

        # Remove old resolved alerts
        self.resolved_alerts = [
            alert for alert in self.resolved_alerts
            if alert.resolved_at and alert.resolved_at > cutoff_time
        ]

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_resolved_alerts(self, limit: int = 50) -> List[Alert]:
        """Get recent resolved alerts."""
        return self.resolved_alerts[-limit:]

    def get_recent_anomalies(self, limit: int = 20) -> List[AnomalyPattern]:
        """Get recent anomaly detections."""
        return self.anomalies[-limit:]

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an active alert."""
        for alert in self.active_alerts.values():
            if alert.id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = time.time()
                logger.info(f"Alert acknowledged: {alert.title}")
                break

    def get_system_bottlenecks(self) -> Dict[str, Any]:
        """Identify current system bottlenecks."""
        bottlenecks = {
            "cpu_bottleneck": False,
            "memory_bottleneck": False,
            "disk_bottleneck": False,
            "network_bottleneck": False,
            "database_bottleneck": False,
            "broker_bottleneck": False,
            "agent_bottleneck": False,
            "recommendations": []
        }

        # Check CPU
        cpu_values = metrics_collector.get_metric_values("cpu_percent", 10)
        if cpu_values and statistics.mean([p.value for p in cpu_values]) > 80:
            bottlenecks["cpu_bottleneck"] = True
            bottlenecks["recommendations"].append("High CPU usage detected - consider scaling or optimizing compute-intensive tasks")

        # Check memory
        mem_values = metrics_collector.get_metric_values("memory_percent", 10)
        if mem_values and statistics.mean([p.value for p in mem_values]) > 85:
            bottlenecks["memory_bottleneck"] = True
            bottlenecks["recommendations"].append("High memory usage detected - check for memory leaks or increase memory allocation")

        # Check disk
        disk_values = metrics_collector.get_metric_values("disk_usage_percent", 10)
        if disk_values and statistics.mean([p.value for p in disk_values]) > 90:
            bottlenecks["disk_bottleneck"] = True
            bottlenecks["recommendations"].append("High disk usage detected - clean up old files or increase disk space")

        # Check database connections
        db_conn_values = metrics_collector.get_metric_values("db_connections_active", 10)
        if db_conn_values and statistics.mean([p.value for p in db_conn_values]) > 80:
            bottlenecks["database_bottleneck"] = True
            bottlenecks["recommendations"].append("High database connection usage - consider connection pooling optimization")

        # Check message broker
        queue_values = metrics_collector.get_metric_values("broker_queue_depth", 10)
        if queue_values and statistics.mean([p.value for p in queue_values]) > 1000:
            bottlenecks["broker_bottleneck"] = True
            bottlenecks["recommendations"].append("High message queue depth - check message processing rate or scale consumers")

        return bottlenecks


# Global bottleneck detector instance
bottleneck_detector = BottleneckDetector()


async def get_bottleneck_detector() -> BottleneckDetector:
    """Get the global bottleneck detector instance."""
    return bottleneck_detector


async def start_bottleneck_detection():
    """Start the global bottleneck detection."""
    await bottleneck_detector.start_monitoring()


async def stop_bottleneck_detection():
    """Stop the global bottleneck detection."""
    await bottleneck_detector.stop_monitoring()