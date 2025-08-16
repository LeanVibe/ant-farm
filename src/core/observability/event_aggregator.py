"""Event Aggregation System for Real-time Observability.

This module provides event collection, processing, and streaming capabilities
for the observability system.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Set
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


class EventType(Enum):
    """Types of system events."""

    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    AGENT_SPAWNED = "agent_spawned"
    AGENT_STOPPED = "agent_stopped"
    AGENT_HEALTH_CHANGE = "agent_health_change"
    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    METRIC_THRESHOLD_EXCEEDED = "metric_threshold_exceeded"
    ALERT_TRIGGERED = "alert_triggered"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_OCCURRED = "error_occurred"
    USER_ACTION = "user_action"
    WEBSOCKET_CONNECTED = "websocket_connected"
    WEBSOCKET_DISCONNECTED = "websocket_disconnected"


@dataclass
class SystemEvent:
    """Represents a system event."""

    event_type: EventType
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"  # info, warning, error, critical
    correlation_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return {
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "source": self.source,
            "data": self.data,
            "severity": self.severity,
            "correlation_id": self.correlation_id,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SystemEvent":
        """Create event from dictionary."""
        return cls(
            event_type=EventType(data["event_type"]),
            timestamp=data["timestamp"],
            source=data["source"],
            data=data["data"],
            severity=data.get("severity", "info"),
            correlation_id=data.get("correlation_id"),
            tags=data.get("tags", {}),
        )


class EventAggregator:
    """Aggregates and processes system events for real-time observability."""

    def __init__(self, max_events: int = 10000):
        self.max_events = max_events
        self.events: List[SystemEvent] = []
        self.event_subscribers: Dict[EventType, List[Callable]] = {}
        self.global_subscribers: List[Callable] = []

        # Event processing state
        self.is_processing = False
        self.processing_task: Optional[asyncio.Task] = None

        # Event statistics
        self.event_counts: Dict[EventType, int] = {}
        self.event_rates: Dict[EventType, float] = {}
        self.last_rate_calculation = time.time()

        log_info("EventAggregator initialized", max_events=max_events)

    async def start_processing(self):
        """Start event processing loop."""
        if self.is_processing:
            log_info("Event processing already running")
            return

        self.is_processing = True
        self.processing_task = asyncio.create_task(self._processing_loop())
        log_info("Started event processing")

    async def stop_processing(self):
        """Stop event processing."""
        self.is_processing = False

        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass

        log_info("Stopped event processing")

    async def _processing_loop(self):
        """Main event processing loop."""
        while self.is_processing:
            try:
                await self._calculate_event_rates()
                await self._cleanup_old_events()
                await asyncio.sleep(30)  # Process every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Error in event processing loop", error=str(e))
                await asyncio.sleep(60)

    async def emit_event(self, event: SystemEvent):
        """Emit a system event."""
        try:
            # Add to events list
            self.events.append(event)

            # Update event counts
            self.event_counts[event.event_type] = (
                self.event_counts.get(event.event_type, 0) + 1
            )

            # Maintain events list size
            if len(self.events) > self.max_events:
                # Remove oldest 10% when limit exceeded
                remove_count = self.max_events // 10
                self.events = self.events[remove_count:]

            # Notify subscribers
            await self._notify_subscribers(event)

            log_info(
                "Event emitted",
                event_type=event.event_type.value,
                source=event.source,
                severity=event.severity,
            )

        except Exception as e:
            log_error(
                "Failed to emit event", error=str(e), event_type=event.event_type.value
            )

    async def _notify_subscribers(self, event: SystemEvent):
        """Notify event subscribers."""
        try:
            # Notify type-specific subscribers
            type_subscribers = self.event_subscribers.get(event.event_type, [])
            for subscriber in type_subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event)
                    else:
                        subscriber(event)
                except Exception as e:
                    log_error(
                        "Subscriber error",
                        error=str(e),
                        event_type=event.event_type.value,
                    )

            # Notify global subscribers
            for subscriber in self.global_subscribers:
                try:
                    if asyncio.iscoroutinefunction(subscriber):
                        await subscriber(event)
                    else:
                        subscriber(event)
                except Exception as e:
                    log_error("Global subscriber error", error=str(e))

        except Exception as e:
            log_error("Failed to notify subscribers", error=str(e))

    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribe to specific event type."""
        if event_type not in self.event_subscribers:
            self.event_subscribers[event_type] = []
        self.event_subscribers[event_type].append(callback)

        log_info("Event subscription added", event_type=event_type.value)

    def subscribe_global(self, callback: Callable):
        """Subscribe to all events."""
        self.global_subscribers.append(callback)
        log_info("Global event subscription added")

    def unsubscribe(self, event_type: EventType, callback: Callable):
        """Unsubscribe from specific event type."""
        if event_type in self.event_subscribers:
            try:
                self.event_subscribers[event_type].remove(callback)
            except ValueError:
                pass

    def unsubscribe_global(self, callback: Callable):
        """Unsubscribe from all events."""
        try:
            self.global_subscribers.remove(callback)
        except ValueError:
            pass

    async def _calculate_event_rates(self):
        """Calculate event rates for monitoring."""
        try:
            current_time = time.time()
            time_window = current_time - self.last_rate_calculation

            if time_window > 0:
                # Calculate rates per minute
                for event_type, count in self.event_counts.items():
                    rate_per_minute = (count / time_window) * 60
                    self.event_rates[event_type] = rate_per_minute

                # Reset counts for next calculation
                self.event_counts = {}
                self.last_rate_calculation = current_time

        except Exception as e:
            log_error("Failed to calculate event rates", error=str(e))

    async def _cleanup_old_events(self):
        """Remove old events to prevent memory issues."""
        try:
            # Keep events from last 24 hours
            cutoff_time = time.time() - (24 * 3600)

            original_count = len(self.events)
            self.events = [
                event for event in self.events if event.timestamp > cutoff_time
            ]

            removed_count = original_count - len(self.events)
            if removed_count > 0:
                log_info(
                    "Cleaned up old events",
                    removed_count=removed_count,
                    remaining_count=len(self.events),
                )

        except Exception as e:
            log_error("Failed to cleanup old events", error=str(e))

    def get_recent_events(
        self,
        time_window_seconds: float = 300,
        event_types: Optional[List[EventType]] = None,
        severity_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
    ) -> List[SystemEvent]:
        """Get recent events with filtering."""
        cutoff_time = time.time() - time_window_seconds

        filtered_events = []
        for event in self.events:
            # Time filter
            if event.timestamp < cutoff_time:
                continue

            # Event type filter
            if event_types and event.event_type not in event_types:
                continue

            # Severity filter
            if severity_filter and event.severity != severity_filter:
                continue

            # Source filter
            if source_filter and source_filter not in event.source:
                continue

            filtered_events.append(event)

        # Sort by timestamp, newest first
        return sorted(filtered_events, key=lambda e: e.timestamp, reverse=True)

    def get_event_statistics(self) -> Dict[str, Any]:
        """Get event statistics for monitoring."""
        total_events = len(self.events)

        # Event type distribution
        type_counts = {}
        severity_counts = {}
        source_counts = {}

        for event in self.events:
            # Count by type
            type_key = event.event_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1

            # Count by severity
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1

            # Count by source
            source_counts[event.source] = source_counts.get(event.source, 0) + 1

        # Recent event rates
        recent_events = self.get_recent_events(300)  # Last 5 minutes
        recent_rate = len(recent_events) / 5.0  # Events per minute

        return {
            "total_events": total_events,
            "recent_events_5min": len(recent_events),
            "recent_rate_per_minute": recent_rate,
            "event_type_distribution": type_counts,
            "severity_distribution": severity_counts,
            "source_distribution": source_counts,
            "event_rates": {k.value: v for k, v in self.event_rates.items()},
            "subscribers": {
                "type_specific": {
                    k.value: len(v) for k, v in self.event_subscribers.items()
                },
                "global": len(self.global_subscribers),
            },
        }

    async def emit_system_startup(self, source: str = "system"):
        """Convenience method to emit system startup event."""
        await self.emit_event(
            SystemEvent(
                event_type=EventType.SYSTEM_STARTUP,
                source=source,
                data={"startup_time": time.time()},
                severity="info",
            )
        )

    async def emit_agent_spawned(
        self, agent_name: str, agent_type: str, source: str = "orchestrator"
    ):
        """Convenience method to emit agent spawned event."""
        await self.emit_event(
            SystemEvent(
                event_type=EventType.AGENT_SPAWNED,
                source=source,
                data={"agent_name": agent_name, "agent_type": agent_type},
                severity="info",
                tags={"agent_name": agent_name, "agent_type": agent_type},
            )
        )

    async def emit_agent_health_change(
        self,
        agent_name: str,
        old_status: str,
        new_status: str,
        source: str = "health_monitor",
    ):
        """Convenience method to emit agent health change event."""
        severity = (
            "warning"
            if new_status in ["degraded", "unhealthy"]
            else "error"
            if new_status == "critical"
            else "info"
        )

        await self.emit_event(
            SystemEvent(
                event_type=EventType.AGENT_HEALTH_CHANGE,
                source=source,
                data={
                    "agent_name": agent_name,
                    "old_status": old_status,
                    "new_status": new_status,
                },
                severity=severity,
                tags={"agent_name": agent_name, "health_status": new_status},
            )
        )

    async def emit_task_completed(
        self,
        task_id: str,
        task_type: str,
        duration: float,
        agent_name: Optional[str] = None,
        source: str = "task_queue",
    ):
        """Convenience method to emit task completed event."""
        await self.emit_event(
            SystemEvent(
                event_type=EventType.TASK_COMPLETED,
                source=source,
                data={
                    "task_id": task_id,
                    "task_type": task_type,
                    "duration_seconds": duration,
                    "agent_name": agent_name,
                },
                severity="info",
                tags={"task_type": task_type, "agent_name": agent_name or "unknown"},
            )
        )

    async def emit_error(
        self,
        error_message: str,
        error_source: str,
        error_details: Optional[Dict[str, Any]] = None,
        source: str = "system",
    ):
        """Convenience method to emit error event."""
        data = {"error_message": error_message, "error_source": error_source}
        if error_details:
            data.update(error_details)

        await self.emit_event(
            SystemEvent(
                event_type=EventType.ERROR_OCCURRED,
                source=source,
                data=data,
                severity="error",
                tags={"error_source": error_source},
            )
        )

    async def emit_websocket_event(
        self,
        action: str,
        client_id: str,
        event_data: Optional[Dict[str, Any]] = None,
        source: str = "websocket_manager",
    ):
        """Convenience method to emit WebSocket events."""
        event_type = (
            EventType.WEBSOCKET_CONNECTED
            if action == "connected"
            else EventType.WEBSOCKET_DISCONNECTED
        )

        data = {"action": action, "client_id": client_id}
        if event_data:
            data.update(event_data)

        await self.emit_event(
            SystemEvent(
                event_type=event_type,
                source=source,
                data=data,
                severity="info",
                tags={"client_id": client_id, "action": action},
            )
        )


# Global event aggregator instance
_global_event_aggregator: Optional[EventAggregator] = None


def get_event_aggregator() -> EventAggregator:
    """Get the global event aggregator instance."""
    global _global_event_aggregator

    if _global_event_aggregator is None:
        _global_event_aggregator = EventAggregator()

    return _global_event_aggregator


async def start_event_processing():
    """Start global event processing."""
    aggregator = get_event_aggregator()
    await aggregator.start_processing()


async def stop_event_processing():
    """Stop global event processing."""
    global _global_event_aggregator

    if _global_event_aggregator:
        await _global_event_aggregator.stop_processing()
