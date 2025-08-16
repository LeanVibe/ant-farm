"""Alert Management System for Real-time Observability.

This module provides threshold-based alerting capabilities for the observability system.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
import json

try:
    import structlog

    logger = structlog.get_logger()
    use_structlog = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    use_structlog = False


def log_info(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.info(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{message} {extra_info}" if extra_info else message)


def log_error(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.error(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.error(f"{message} {extra_info}" if extra_info else message)


def log_warning(message: str, **kwargs):
    """Helper function for logging."""
    if use_structlog:
        logger.warning(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.warning(f"{message} {extra_info}" if extra_info else message)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertState(Enum):
    """Alert states."""

    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


@dataclass
class AlertRule:
    """Defines an alert rule for threshold monitoring."""

    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # "gt", "gte", "lt", "lte", "eq", "ne"
    threshold: Union[int, float]
    severity: AlertSeverity = AlertSeverity.WARNING
    evaluation_window: float = 300.0  # Seconds to evaluate over
    min_duration: float = 60.0  # Minimum duration before triggering alert
    tags_filter: Optional[Dict[str, str]] = None
    enabled: bool = True

    def __post_init__(self):
        """Validate alert rule after initialization."""
        if self.condition not in ["gt", "gte", "lt", "lte", "eq", "ne"]:
            raise ValueError(f"Invalid condition: {self.condition}")

        if self.evaluation_window <= 0:
            raise ValueError("Evaluation window must be positive")

        if self.min_duration < 0:
            raise ValueError("Min duration cannot be negative")


@dataclass
class Alert:
    """Represents an active or resolved alert."""

    alert_id: str
    rule_id: str
    rule_name: str
    metric_name: str
    current_value: Union[int, float]
    threshold: Union[int, float]
    condition: str
    severity: AlertSeverity
    state: AlertState = AlertState.ACTIVE
    triggered_at: float = field(default_factory=time.time)
    resolved_at: Optional[float] = None
    acknowledged_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    message: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "rule_name": self.rule_name,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold": self.threshold,
            "condition": self.condition,
            "severity": self.severity.value,
            "state": self.state.value,
            "triggered_at": self.triggered_at,
            "resolved_at": self.resolved_at,
            "acknowledged_at": self.acknowledged_at,
            "acknowledged_by": self.acknowledged_by,
            "message": self.message,
            "tags": self.tags,
        }


class AlertManager:
    """Manages alert rules and active alerts."""

    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: List[Alert] = []
        self.max_resolved_alerts = 1000

        # Alert handlers
        self.alert_handlers: List[Callable] = []

        # Evaluation state
        self.is_evaluating = False
        self.evaluation_task: Optional[asyncio.Task] = None

        # Rule violation tracking for min_duration
        self.rule_violations: Dict[str, float] = {}

        log_info("AlertManager initialized", check_interval=check_interval)

    async def start_evaluation(self):
        """Start alert rule evaluation loop."""
        if self.is_evaluating:
            log_info("Alert evaluation already running")
            return

        self.is_evaluating = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        log_info("Started alert evaluation")

    async def stop_evaluation(self):
        """Stop alert rule evaluation."""
        self.is_evaluating = False

        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass

        log_info("Stopped alert evaluation")

    async def _evaluation_loop(self):
        """Main alert evaluation loop."""
        while self.is_evaluating:
            try:
                await self.evaluate_all_rules()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Error in alert evaluation loop", error=str(e))
                await asyncio.sleep(min(300, self.check_interval * 2))

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules[rule.rule_id] = rule
        log_info("Alert rule added", rule_id=rule.rule_id, name=rule.name)

    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]

            # Remove any active alerts for this rule
            alerts_to_resolve = [
                alert
                for alert in self.active_alerts.values()
                if alert.rule_id == rule_id
            ]

            for alert in alerts_to_resolve:
                self.resolve_alert(alert.alert_id, "Rule removed")

            log_info("Alert rule removed", rule_id=rule_id)

    def enable_alert_rule(self, rule_id: str):
        """Enable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = True
            log_info("Alert rule enabled", rule_id=rule_id)

    def disable_alert_rule(self, rule_id: str):
        """Disable an alert rule."""
        if rule_id in self.alert_rules:
            self.alert_rules[rule_id].enabled = False

            # Resolve any active alerts for this rule
            alerts_to_resolve = [
                alert
                for alert in self.active_alerts.values()
                if alert.rule_id == rule_id
            ]

            for alert in alerts_to_resolve:
                self.resolve_alert(alert.alert_id, "Rule disabled")

            log_info("Alert rule disabled", rule_id=rule_id)

    def add_alert_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.alert_handlers.append(handler)
        log_info("Alert handler added")

    def remove_alert_handler(self, handler: Callable):
        """Remove an alert handler function."""
        try:
            self.alert_handlers.remove(handler)
            log_info("Alert handler removed")
        except ValueError:
            pass

    async def evaluate_all_rules(self):
        """Evaluate all enabled alert rules."""
        try:
            from .metrics_collector import get_metrics_collector
            from .time_series_store import get_time_series_store, TimeSeriesQuery

            metrics_collector = get_metrics_collector()
            time_series_store = get_time_series_store()

            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue

                try:
                    await self._evaluate_rule(rule, time_series_store)
                except Exception as e:
                    log_error(
                        "Failed to evaluate rule", rule_id=rule.rule_id, error=str(e)
                    )

        except Exception as e:
            log_error("Failed to evaluate alert rules", error=str(e))

    async def _evaluate_rule(self, rule: AlertRule, time_series_store):
        """Evaluate a single alert rule."""
        try:
            # Query time series data
            from .time_series_store import TimeSeriesQuery

            query = TimeSeriesQuery(
                metric_name=rule.metric_name,
                start_time=time.time() - rule.evaluation_window,
                tags_filter=rule.tags_filter,
                aggregation="avg",  # Use average for threshold comparison
            )

            points = await time_series_store.query(query)

            if not points:
                # No data available, clear any violation tracking
                self.rule_violations.pop(rule.rule_id, None)
                return

            # Get current value (latest point or average of recent points)
            current_value = (
                points[-1].value
                if len(points) == 1
                else sum(p.value for p in points) / len(points)
            )

            # Check if threshold is violated
            is_violated = self._check_threshold(
                current_value, rule.condition, rule.threshold
            )

            if is_violated:
                # Track violation start time
                if rule.rule_id not in self.rule_violations:
                    self.rule_violations[rule.rule_id] = time.time()

                # Check if violation has persisted long enough
                violation_duration = time.time() - self.rule_violations[rule.rule_id]

                if violation_duration >= rule.min_duration:
                    # Generate alert if not already active
                    await self._trigger_alert(rule, current_value)
            else:
                # Threshold not violated, clear violation tracking
                self.rule_violations.pop(rule.rule_id, None)

                # Resolve any active alerts for this rule
                await self._resolve_rule_alerts(rule.rule_id, current_value)

        except Exception as e:
            log_error("Failed to evaluate rule", rule_id=rule.rule_id, error=str(e))

    def _check_threshold(
        self, value: Union[int, float], condition: str, threshold: Union[int, float]
    ) -> bool:
        """Check if value violates threshold based on condition."""
        if condition == "gt":
            return value > threshold
        elif condition == "gte":
            return value >= threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "lte":
            return value <= threshold
        elif condition == "eq":
            return value == threshold
        elif condition == "ne":
            return value != threshold
        else:
            return False

    async def _trigger_alert(self, rule: AlertRule, current_value: Union[int, float]):
        """Trigger an alert for a rule violation."""
        # Check if alert already exists for this rule
        existing_alert = None
        for alert in self.active_alerts.values():
            if alert.rule_id == rule.rule_id:
                existing_alert = alert
                break

        if existing_alert:
            # Update existing alert with new value
            existing_alert.current_value = current_value
            return

        # Create new alert
        alert_id = f"{rule.rule_id}_{int(time.time())}"

        # Generate alert message
        message = f"{rule.name}: {rule.metric_name} {current_value} {rule.condition} {rule.threshold}"

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            rule_name=rule.name,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            condition=rule.condition,
            severity=rule.severity,
            message=message,
            tags=rule.tags_filter or {},
        )

        self.active_alerts[alert_id] = alert

        # Notify handlers
        await self._notify_handlers(alert, "triggered")

        log_warning(
            "Alert triggered",
            alert_id=alert_id,
            rule_name=rule.name,
            current_value=current_value,
            threshold=rule.threshold,
            severity=rule.severity.value,
        )

    async def _resolve_rule_alerts(
        self, rule_id: str, current_value: Union[int, float]
    ):
        """Resolve all active alerts for a specific rule."""
        alerts_to_resolve = [
            alert for alert in self.active_alerts.values() if alert.rule_id == rule_id
        ]

        for alert in alerts_to_resolve:
            alert.current_value = current_value
            await self.resolve_alert(alert.alert_id, "Threshold no longer violated")

    async def resolve_alert(self, alert_id: str, resolution_message: str = ""):
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return

        alert = self.active_alerts[alert_id]
        alert.state = AlertState.RESOLVED
        alert.resolved_at = time.time()
        if resolution_message:
            alert.message += f" | Resolved: {resolution_message}"

        # Move to resolved alerts
        self.resolved_alerts.append(alert)
        del self.active_alerts[alert_id]

        # Maintain resolved alerts limit
        if len(self.resolved_alerts) > self.max_resolved_alerts:
            self.resolved_alerts = self.resolved_alerts[-self.max_resolved_alerts :]

        # Notify handlers
        await self._notify_handlers(alert, "resolved")

        log_info(
            "Alert resolved",
            alert_id=alert_id,
            rule_name=alert.rule_name,
            resolution_message=resolution_message,
        )

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str, note: str = ""
    ):
        """Acknowledge an active alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.state = AlertState.ACKNOWLEDGED
        alert.acknowledged_at = time.time()
        alert.acknowledged_by = acknowledged_by

        if note:
            alert.message += f" | Acknowledged: {note}"

        # Notify handlers
        await self._notify_handlers(alert, "acknowledged")

        log_info(
            "Alert acknowledged",
            alert_id=alert_id,
            acknowledged_by=acknowledged_by,
            note=note,
        )
        return True

    async def silence_alert(
        self, alert_id: str, silenced_by: str, duration_minutes: int = 60
    ):
        """Silence an alert for a specified duration."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.state = AlertState.SILENCED

        # Set up automatic unsilencing
        async def unsilence_later():
            await asyncio.sleep(duration_minutes * 60)
            if (
                alert_id in self.active_alerts
                and self.active_alerts[alert_id].state == AlertState.SILENCED
            ):
                self.active_alerts[alert_id].state = AlertState.ACTIVE
                log_info("Alert unsilenced automatically", alert_id=alert_id)

        asyncio.create_task(unsilence_later())

        log_info(
            "Alert silenced",
            alert_id=alert_id,
            silenced_by=silenced_by,
            duration_minutes=duration_minutes,
        )
        return True

    async def _notify_handlers(self, alert: Alert, action: str):
        """Notify all alert handlers."""
        for handler in self.alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert, action)
                else:
                    handler(alert, action)
            except Exception as e:
                log_error("Alert handler error", error=str(e), alert_id=alert.alert_id)

    def get_active_alerts(
        self, severity: Optional[AlertSeverity] = None
    ) -> List[Alert]:
        """Get all active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())

        if severity:
            alerts = [alert for alert in alerts if alert.severity == severity]

        return sorted(alerts, key=lambda a: a.triggered_at, reverse=True)

    def get_resolved_alerts(self, limit: int = 100) -> List[Alert]:
        """Get recent resolved alerts."""
        return self.resolved_alerts[-limit:] if limit else self.resolved_alerts

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        active_count = len(self.active_alerts)
        resolved_count = len(self.resolved_alerts)

        # Count by severity
        severity_counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        for alert in self.active_alerts.values():
            severity_counts[alert.severity.value] += 1

        # Count by state
        state_counts = {"active": 0, "acknowledged": 0, "silenced": 0}
        for alert in self.active_alerts.values():
            if alert.state == AlertState.ACTIVE:
                state_counts["active"] += 1
            elif alert.state == AlertState.ACKNOWLEDGED:
                state_counts["acknowledged"] += 1
            elif alert.state == AlertState.SILENCED:
                state_counts["silenced"] += 1

        return {
            "active_alerts": active_count,
            "resolved_alerts": resolved_count,
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "severity_distribution": severity_counts,
            "state_distribution": state_counts,
            "handlers_count": len(self.alert_handlers),
        }

    def create_default_rules(self):
        """Create default alert rules for common metrics."""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="System CPU usage is above 80%",
                metric_name="system_cpu_percent",
                condition="gt",
                threshold=80.0,
                severity=AlertSeverity.WARNING,
                evaluation_window=300.0,
                min_duration=120.0,
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="System memory usage is above 85%",
                metric_name="system_memory_percent",
                condition="gt",
                threshold=85.0,
                severity=AlertSeverity.WARNING,
                evaluation_window=300.0,
                min_duration=120.0,
            ),
            AlertRule(
                rule_id="critical_memory_usage",
                name="Critical Memory Usage",
                description="System memory usage is above 95%",
                metric_name="system_memory_percent",
                condition="gt",
                threshold=95.0,
                severity=AlertSeverity.CRITICAL,
                evaluation_window=60.0,
                min_duration=30.0,
            ),
            AlertRule(
                rule_id="low_disk_space",
                name="Low Disk Space",
                description="Disk usage is above 90%",
                metric_name="system_disk_usage_percent",
                condition="gt",
                threshold=90.0,
                severity=AlertSeverity.ERROR,
                evaluation_window=300.0,
                min_duration=300.0,
            ),
            AlertRule(
                rule_id="agent_health_degraded",
                name="Agent Health Degraded",
                description="Agent health score is below 0.5",
                metric_name="agent_health_score",
                condition="lt",
                threshold=0.5,
                severity=AlertSeverity.WARNING,
                evaluation_window=300.0,
                min_duration=180.0,
            ),
        ]

        for rule in default_rules:
            self.add_alert_rule(rule)

        log_info("Default alert rules created", count=len(default_rules))


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _global_alert_manager

    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()

    return _global_alert_manager


async def start_alert_evaluation():
    """Start global alert evaluation."""
    manager = get_alert_manager()
    await manager.start_evaluation()


async def stop_alert_evaluation():
    """Stop global alert evaluation."""
    global _global_alert_manager

    if _global_alert_manager:
        await _global_alert_manager.stop_evaluation()
