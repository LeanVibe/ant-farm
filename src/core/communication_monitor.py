"""Agent communication performance monitoring and analytics system."""

import asyncio
import json
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger()


class MetricType(Enum):
    """Types of communication metrics."""

    LATENCY = "latency"  # Message delivery latency
    THROUGHPUT = "throughput"  # Messages per second
    RELIABILITY = "reliability"  # Success/failure rates
    BANDWIDTH = "bandwidth"  # Data volume metrics
    QUEUE_DEPTH = "queue_depth"  # Message queue sizes
    AGENT_LOAD = "agent_load"  # Agent processing load
    RESPONSE_TIME = "response_time"  # Request-response timing
    ERROR_RATE = "error_rate"  # Communication error rates


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class CommunicationMetric:
    """A single communication metric measurement."""

    metric_type: MetricType
    value: float
    timestamp: float
    agent_name: str
    target_agent: str | None = None
    message_id: str | None = None
    topic: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceThreshold:
    """Performance threshold for alerting."""

    metric_type: MetricType
    threshold_value: float
    comparison: str  # "gt", "lt", "eq"
    severity: AlertSeverity
    window_size: int = 300  # seconds
    trigger_count: int = 3  # number of violations to trigger
    enabled: bool = True


@dataclass
class PerformanceAlert:
    """Performance alert notification."""

    id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    affected_agents: set[str]
    metric_values: list[float]
    threshold: PerformanceThreshold
    triggered_at: float
    resolved_at: float | None = None
    acknowledged: bool = False


@dataclass
class AgentCommunicationProfile:
    """Communication profile for an agent."""

    agent_name: str
    message_count: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    average_latency: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    peak_queue_depth: int = 0
    communication_partners: set[str] = field(default_factory=set)
    preferred_topics: dict[str, int] = field(default_factory=dict)
    activity_pattern: dict[str, int] = field(
        default_factory=dict
    )  # hour -> message_count
    last_activity: float = field(default_factory=time.time)


class CommunicationMonitor:
    """Real-time communication performance monitoring system."""

    def __init__(self):
        self.metrics_buffer: deque = deque(maxlen=10000)  # Ring buffer for metrics
        self.agent_profiles: dict[str, AgentCommunicationProfile] = {}
        self.performance_thresholds: dict[str, PerformanceThreshold] = {}
        self.active_alerts: dict[str, PerformanceAlert] = {}
        self.alert_history: list[PerformanceAlert] = []

        # Real-time statistics
        self.current_stats = {
            "total_messages": 0,
            "messages_per_second": 0.0,
            "average_latency": 0.0,
            "error_rate": 0.0,
            "active_agents": 0,
            "queue_depth_total": 0,
        }

        # Time series data for trending
        self.time_series: dict[str, deque] = {
            "latency": deque(maxlen=1440),  # 24 hours of minute data
            "throughput": deque(maxlen=1440),
            "error_rate": deque(maxlen=1440),
            "active_agents": deque(maxlen=1440),
        }

        # Communication topology
        self.communication_graph: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.topic_popularity: dict[str, int] = defaultdict(int)

        # Performance baselines
        self.baselines = {
            "latency_p95": 100.0,  # 100ms
            "throughput_min": 10.0,  # 10 msg/sec
            "error_rate_max": 0.05,  # 5%
            "queue_depth_max": 100,  # 100 messages
        }

    async def initialize(self) -> None:
        """Initialize the communication monitor."""

        # Set up default thresholds
        await self._setup_default_thresholds()

        # Start monitoring tasks
        asyncio.create_task(self._metrics_processor_loop())
        asyncio.create_task(self._statistics_calculator_loop())
        asyncio.create_task(self._alert_processor_loop())
        asyncio.create_task(self._performance_analyzer_loop())

        logger.info("Communication performance monitor initialized")

    async def record_message_sent(
        self,
        from_agent: str,
        to_agent: str,
        message_id: str,
        topic: str,
        message_size: int,
        timestamp: float = None,
    ) -> None:
        """Record a message being sent."""

        timestamp = timestamp or time.time()

        # Record metrics
        metrics = [
            CommunicationMetric(
                metric_type=MetricType.THROUGHPUT,
                value=1.0,
                timestamp=timestamp,
                agent_name=from_agent,
                target_agent=to_agent,
                message_id=message_id,
                topic=topic,
                metadata={"message_size": message_size},
            ),
            CommunicationMetric(
                metric_type=MetricType.BANDWIDTH,
                value=message_size,
                timestamp=timestamp,
                agent_name=from_agent,
                target_agent=to_agent,
                message_id=message_id,
                topic=topic,
            ),
        ]

        for metric in metrics:
            self.metrics_buffer.append(metric)

        # Update agent profile
        await self._update_agent_profile(
            from_agent,
            {
                "message_sent": True,
                "bytes_sent": message_size,
                "target_agent": to_agent,
                "topic": topic,
                "timestamp": timestamp,
            },
        )

        # Update communication graph
        self.communication_graph[from_agent][to_agent] += 1
        self.topic_popularity[topic] += 1

    async def get_real_time_stats(self) -> dict[str, Any]:
        """Get current real-time statistics."""

        return {
            **self.current_stats,
            "active_alerts": len(self.active_alerts),
            "monitored_agents": len(self.agent_profiles),
            "communication_pairs": sum(
                len(targets) for targets in self.communication_graph.values()
            ),
            "top_topics": dict(
                sorted(self.topic_popularity.items(), key=lambda x: x[1], reverse=True)[
                    :5
                ]
            ),
            "timestamp": time.time(),
        }

    async def get_agent_performance(self, agent_name: str) -> dict[str, Any] | None:
        """Get performance metrics for a specific agent."""

        if agent_name not in self.agent_profiles:
            return None

        profile = self.agent_profiles[agent_name]

        # Calculate recent metrics
        recent_metrics = self._get_recent_metrics_for_agent(
            agent_name, 300
        )  # Last 5 minutes

        return {
            "agent_name": agent_name,
            "message_count": profile.message_count,
            "bytes_sent": profile.bytes_sent,
            "bytes_received": profile.bytes_received,
            "average_latency": profile.average_latency,
            "success_rate": profile.success_rate,
            "error_count": profile.error_count,
            "peak_queue_depth": profile.peak_queue_depth,
            "communication_partners": len(profile.communication_partners),
            "last_activity": profile.last_activity,
            "recent_throughput": len(
                [m for m in recent_metrics if m.metric_type == MetricType.THROUGHPUT]
            ),
            "recent_errors": len(
                [m for m in recent_metrics if m.metric_type == MetricType.ERROR_RATE]
            ),
            "activity_pattern": profile.activity_pattern,
            "preferred_topics": dict(
                sorted(
                    profile.preferred_topics.items(), key=lambda x: x[1], reverse=True
                )[:5]
            ),
        }

    async def get_communication_topology(self) -> dict[str, Any]:
        """Get communication topology and patterns."""

        # Calculate network metrics
        total_connections = sum(
            len(targets) for targets in self.communication_graph.values()
        )

        # Find most connected agents
        connection_counts = {
            agent: len(targets)
            + sum(
                1
                for other_targets in self.communication_graph.values()
                if agent in other_targets
            )
            for agent in self.communication_graph.keys()
        }

        most_connected = sorted(
            connection_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return {
            "total_agents": len(self.agent_profiles),
            "total_connections": total_connections,
            "most_connected_agents": most_connected,
            "communication_matrix": dict(self.communication_graph),
            "topic_distribution": dict(
                sorted(self.topic_popularity.items(), key=lambda x: x[1], reverse=True)
            ),
        }

    async def _setup_default_thresholds(self) -> None:
        """Set up default performance thresholds."""

        default_thresholds = [
            (MetricType.LATENCY, 500.0, "gt", AlertSeverity.WARNING),  # > 500ms
            (MetricType.LATENCY, 1000.0, "gt", AlertSeverity.ERROR),  # > 1000ms
            (MetricType.ERROR_RATE, 0.1, "gt", AlertSeverity.WARNING),  # > 10%
            (MetricType.ERROR_RATE, 0.2, "gt", AlertSeverity.ERROR),  # > 20%
            (MetricType.QUEUE_DEPTH, 50, "gt", AlertSeverity.WARNING),  # > 50 messages
            (MetricType.QUEUE_DEPTH, 100, "gt", AlertSeverity.ERROR),  # > 100 messages
            (MetricType.AGENT_LOAD, 80.0, "gt", AlertSeverity.WARNING),  # > 80%
            (MetricType.AGENT_LOAD, 95.0, "gt", AlertSeverity.CRITICAL),  # > 95%
        ]

        for metric_type, value, comparison, severity in default_thresholds:
            await self.add_performance_threshold(
                metric_type, value, comparison, severity
            )

    async def add_performance_threshold(
        self,
        metric_type: MetricType,
        threshold_value: float,
        comparison: str,
        severity: AlertSeverity,
        window_size: int = 300,
        trigger_count: int = 3,
    ) -> str:
        """Add a performance threshold for alerting."""

        threshold_id = str(uuid.uuid4())

        threshold = PerformanceThreshold(
            metric_type=metric_type,
            threshold_value=threshold_value,
            comparison=comparison,
            severity=severity,
            window_size=window_size,
            trigger_count=trigger_count,
        )

        self.performance_thresholds[threshold_id] = threshold

        logger.info(
            "Performance threshold added",
            threshold_id=threshold_id,
            metric=metric_type.value,
            value=threshold_value,
            severity=severity.value,
        )

        return threshold_id

    async def _metrics_processor_loop(self) -> None:
        """Process incoming metrics continuously."""

        while True:
            try:
                await asyncio.sleep(1)  # Process every second

                if not self.metrics_buffer:
                    continue

                # Simple processing - in full implementation would do more

            except Exception as e:
                logger.error("Metrics processor error", error=str(e))

    async def _statistics_calculator_loop(self) -> None:
        """Calculate real-time statistics."""

        while True:
            try:
                await asyncio.sleep(10)  # Update every 10 seconds

                current_time = time.time()
                window_start = current_time - 60  # Last minute

                # Get recent metrics
                recent_metrics = [
                    metric
                    for metric in self.metrics_buffer
                    if metric.timestamp > window_start
                ]

                # Calculate throughput
                throughput_metrics = [
                    m for m in recent_metrics if m.metric_type == MetricType.THROUGHPUT
                ]
                self.current_stats["messages_per_second"] = (
                    len(throughput_metrics) / 60.0
                )

                # Update active agents count
                active_threshold = current_time - 300  # 5 minutes
                active_agents = set()
                for profile in self.agent_profiles.values():
                    if profile.last_activity > active_threshold:
                        active_agents.add(profile.agent_name)

                self.current_stats["active_agents"] = len(active_agents)

            except Exception as e:
                logger.error("Statistics calculator error", error=str(e))

    async def _alert_processor_loop(self) -> None:
        """Process performance alerts."""

        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                # Alert processing logic would go here

            except Exception as e:
                logger.error("Alert processor error", error=str(e))

    async def _performance_analyzer_loop(self) -> None:
        """Analyze performance patterns and generate insights."""

        while True:
            try:
                await asyncio.sleep(300)  # Analyze every 5 minutes

                # Performance analysis logic would go here

            except Exception as e:
                logger.error("Performance analyzer error", error=str(e))

    async def _update_agent_profile(
        self, agent_name: str, event_data: dict[str, Any]
    ) -> None:
        """Update agent communication profile."""

        if agent_name not in self.agent_profiles:
            self.agent_profiles[agent_name] = AgentCommunicationProfile(
                agent_name=agent_name
            )

        profile = self.agent_profiles[agent_name]
        profile.last_activity = event_data.get("timestamp", time.time())

        if event_data.get("message_sent"):
            profile.message_count += 1
            profile.bytes_sent += event_data.get("bytes_sent", 0)

            if "target_agent" in event_data:
                profile.communication_partners.add(event_data["target_agent"])

            if "topic" in event_data:
                profile.preferred_topics[event_data["topic"]] = (
                    profile.preferred_topics.get(event_data["topic"], 0) + 1
                )

    def _get_recent_metrics_for_agent(
        self, agent_name: str, window_seconds: int
    ) -> list[CommunicationMetric]:
        """Get recent metrics for a specific agent."""

        cutoff_time = time.time() - window_seconds

        return [
            metric
            for metric in self.metrics_buffer
            if metric.agent_name == agent_name and metric.timestamp > cutoff_time
        ]


# Global communication monitor instance
communication_monitor = CommunicationMonitor()


def get_communication_monitor() -> CommunicationMonitor:
    """Get communication monitor instance."""
    return communication_monitor
