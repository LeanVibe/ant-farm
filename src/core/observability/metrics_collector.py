"""Enhanced Metrics Collection System for Real-time Observability.

This module provides comprehensive metrics collection capabilities that integrate
with existing monitoring infrastructure while adding real-time capabilities.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union

try:
    import structlog

    logger = structlog.get_logger()
    use_structlog = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    use_structlog = False

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# Import existing health system
try:
    from ..agent_health import HealthMonitorRegistry, get_health_monitor_registry

    HAS_HEALTH_SYSTEM = True
except ImportError:
    HAS_HEALTH_SYSTEM = False

# Import existing autonomous dashboard
try:
    from ..monitoring.autonomous_dashboard import (
        AutonomousDashboard,
        MetricsCollector as ExistingMetricsCollector,
    )

    HAS_AUTONOMOUS_DASHBOARD = True
except ImportError:
    HAS_AUTONOMOUS_DASHBOARD = False


def log_info(message: str, **kwargs):
    """Helper function for logging that handles both structlog and standard logging."""
    if use_structlog:
        logger.info(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{message} {extra_info}" if extra_info else message)


def log_error(message: str, **kwargs):
    """Helper function for logging that handles both structlog and standard logging."""
    if use_structlog:
        logger.error(message, **kwargs)
    else:
        extra_info = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.error(f"{message} {extra_info}" if extra_info else message)


class MetricType(Enum):
    """Types of metrics that can be collected."""

    COUNTER = "counter"  # Always increasing values
    GAUGE = "gauge"  # Point-in-time values
    HISTOGRAM = "histogram"  # Distribution of values
    RATE = "rate"  # Rate of change over time


@dataclass
class SystemMetrics:
    """System-level performance metrics."""

    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_available_gb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    load_average: List[float] = field(default_factory=list)
    process_count: int = 0
    thread_count: int = 0


@dataclass
class AgentMetrics:
    """Agent-specific performance metrics."""

    agent_name: str
    timestamp: float = field(default_factory=time.time)
    status: str = "unknown"
    uptime_seconds: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_task_duration: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_percent: float = 0.0
    error_rate: float = 0.0
    response_time_avg: float = 0.0
    health_score: float = 0.0
    circuit_breaker_state: str = "closed"


@dataclass
class TaskMetrics:
    """Task execution performance metrics."""

    task_id: str
    task_type: str
    timestamp: float = field(default_factory=time.time)
    status: str = "unknown"
    duration_seconds: float = 0.0
    queue_time_seconds: float = 0.0
    agent_name: Optional[str] = None
    priority: str = "normal"
    retry_count: int = 0
    error_message: Optional[str] = None


@dataclass
class MetricPoint:
    """Individual metric data point."""

    name: str
    value: Union[int, float]
    timestamp: float = field(default_factory=time.time)
    metric_type: MetricType = MetricType.GAUGE
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Enhanced metrics collector that integrates with existing systems."""

    def __init__(self, collection_interval: float = 10.0):
        self.collection_interval = collection_interval
        self.metrics_history: List[MetricPoint] = []
        self.max_history_size = 10000

        # Integration points
        self.health_registry = None
        self.autonomous_dashboard = None
        self.existing_collector = None

        # Collection state
        self.is_collecting = False
        self.collection_task: Optional[asyncio.Task] = None

        # Initialize integrations
        self._initialize_integrations()

        log_info(
            "MetricsCollector initialized",
            collection_interval=collection_interval,
            has_health_system=HAS_HEALTH_SYSTEM,
            has_autonomous_dashboard=HAS_AUTONOMOUS_DASHBOARD,
            has_psutil=HAS_PSUTIL,
        )

    def _initialize_integrations(self):
        """Initialize integrations with existing monitoring systems."""
        try:
            if HAS_HEALTH_SYSTEM:
                self.health_registry = get_health_monitor_registry()

            if HAS_AUTONOMOUS_DASHBOARD:
                from pathlib import Path

                self.autonomous_dashboard = AutonomousDashboard(Path("."))
                self.existing_collector = ExistingMetricsCollector(Path("."))

        except Exception as e:
            log_error("Failed to initialize integrations", error=str(e))

    async def start_collection(self):
        """Start continuous metrics collection."""
        if self.is_collecting:
            log_info("Metrics collection already running")
            return

        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        log_info("Started metrics collection")

    async def stop_collection(self):
        """Stop metrics collection."""
        self.is_collecting = False

        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        log_info("Stopped metrics collection")

    async def _collection_loop(self):
        """Main collection loop."""
        while self.is_collecting:
            try:
                await self.collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Error in metrics collection loop", error=str(e))
                await asyncio.sleep(min(60.0, self.collection_interval * 2))

    async def collect_all_metrics(self):
        """Collect all types of metrics."""
        collection_start = time.time()

        try:
            # Collect system metrics
            system_metrics = await self.collect_system_metrics()
            if system_metrics:
                self._store_system_metrics(system_metrics)

            # Collect agent metrics
            agent_metrics = await self.collect_agent_metrics()
            for metrics in agent_metrics:
                self._store_agent_metrics(metrics)

            # Collect task metrics
            task_metrics = await self.collect_task_metrics()
            for metrics in task_metrics:
                self._store_task_metrics(metrics)

            collection_duration = time.time() - collection_start

            # Store collection performance metric
            self.add_metric_point(
                MetricPoint(
                    name="metrics_collection_duration",
                    value=collection_duration,
                    metric_type=MetricType.GAUGE,
                    tags={"source": "metrics_collector"},
                )
            )

        except Exception as e:
            log_error("Failed to collect metrics", error=str(e))

    async def collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect system-level metrics."""
        if not HAS_PSUTIL:
            return None

        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)

            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)

            # Disk metrics
            disk = psutil.disk_usage("/")
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024**3)

            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv

            # Load average (Unix only)
            load_avg = []
            try:
                load_avg = list(psutil.getloadavg())
            except:
                pass

            # Process metrics
            process_count = len(psutil.pids())

            # Thread count
            thread_count = 0
            try:
                for proc in psutil.process_iter(["num_threads"]):
                    thread_count += proc.info.get("num_threads", 0)
            except:
                pass

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_available_gb=memory_available_gb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                network_bytes_sent=network_bytes_sent,
                network_bytes_recv=network_bytes_recv,
                load_average=load_avg,
                process_count=process_count,
                thread_count=thread_count,
            )

        except Exception as e:
            log_error("Failed to collect system metrics", error=str(e))
            return None

    async def collect_agent_metrics(self) -> List[AgentMetrics]:
        """Collect metrics for all agents."""
        agent_metrics = []

        try:
            if self.health_registry:
                # Get metrics from health monitoring system
                health_report = await self.health_registry.get_system_health_report()

                for agent_name, report in health_report.get(
                    "agent_reports", {}
                ).items():
                    metrics = AgentMetrics(
                        agent_name=agent_name,
                        status=report.get("current_status", "unknown"),
                        uptime_seconds=report.get("uptime", 0.0),
                        tasks_completed=report.get("metrics", {}).get(
                            "success_count", 0
                        ),
                        tasks_failed=report.get("metrics", {}).get("failure_count", 0),
                        memory_usage_mb=report.get("metrics", {}).get(
                            "memory_usage", 0.0
                        ),
                        cpu_percent=report.get("metrics", {}).get("cpu_usage", 0.0),
                        error_rate=report.get("metrics", {}).get("error_rate", 0.0),
                        response_time_avg=report.get("metrics", {}).get(
                            "avg_response_time", 0.0
                        ),
                        health_score=1.0
                        if report.get("current_status") == "healthy"
                        else 0.5,
                    )
                    agent_metrics.append(metrics)

            # Integrate with existing autonomous dashboard if available
            if self.autonomous_dashboard:
                dashboard_data = self.autonomous_dashboard.get_current_dashboard_data()

                if dashboard_data.get("status") == "active":
                    # Add autonomous metrics as a special agent
                    autonomous_metrics = AgentMetrics(
                        agent_name="autonomous_system",
                        status="active",
                        uptime_seconds=dashboard_data.get("session_duration_hours", 0)
                        * 3600,
                        health_score=dashboard_data.get("current_autonomy_score", 0)
                        / 100,
                        error_rate=1.0 - dashboard_data.get("quality_score", 1.0),
                    )
                    agent_metrics.append(autonomous_metrics)

        except Exception as e:
            log_error("Failed to collect agent metrics", error=str(e))

        return agent_metrics

    async def collect_task_metrics(self) -> List[TaskMetrics]:
        """Collect task execution metrics."""
        task_metrics = []

        try:
            # This would integrate with the task queue system
            # For now, return empty list as we don't have direct access
            # In a full implementation, this would query the task queue
            pass

        except Exception as e:
            log_error("Failed to collect task metrics", error=str(e))

        return task_metrics

    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics as metric points."""
        base_tags = {"source": "system", "type": "system_metrics"}
        timestamp = metrics.timestamp

        system_metric_points = [
            MetricPoint(
                "system_cpu_percent",
                metrics.cpu_percent,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "system_memory_percent",
                metrics.memory_percent,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "system_memory_available_gb",
                metrics.memory_available_gb,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "system_disk_usage_percent",
                metrics.disk_usage_percent,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "system_disk_free_gb",
                metrics.disk_free_gb,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "system_network_bytes_sent",
                metrics.network_bytes_sent,
                timestamp,
                MetricType.COUNTER,
                base_tags,
            ),
            MetricPoint(
                "system_network_bytes_recv",
                metrics.network_bytes_recv,
                timestamp,
                MetricType.COUNTER,
                base_tags,
            ),
            MetricPoint(
                "system_process_count",
                metrics.process_count,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "system_thread_count",
                metrics.thread_count,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
        ]

        # Add load average if available
        for i, load in enumerate(metrics.load_average):
            system_metric_points.append(
                MetricPoint(
                    f"system_load_avg_{i + 1}min",
                    load,
                    timestamp,
                    MetricType.GAUGE,
                    base_tags,
                )
            )

        for point in system_metric_points:
            self.add_metric_point(point)

    def _store_agent_metrics(self, metrics: AgentMetrics):
        """Store agent metrics as metric points."""
        base_tags = {
            "source": "agent",
            "agent_name": metrics.agent_name,
            "type": "agent_metrics",
        }
        timestamp = metrics.timestamp

        agent_metric_points = [
            MetricPoint(
                "agent_uptime_seconds",
                metrics.uptime_seconds,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "agent_tasks_completed",
                metrics.tasks_completed,
                timestamp,
                MetricType.COUNTER,
                base_tags,
            ),
            MetricPoint(
                "agent_tasks_failed",
                metrics.tasks_failed,
                timestamp,
                MetricType.COUNTER,
                base_tags,
            ),
            MetricPoint(
                "agent_avg_task_duration",
                metrics.avg_task_duration,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "agent_memory_usage_mb",
                metrics.memory_usage_mb,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "agent_cpu_percent",
                metrics.cpu_percent,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "agent_error_rate",
                metrics.error_rate,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "agent_response_time_avg",
                metrics.response_time_avg,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
            MetricPoint(
                "agent_health_score",
                metrics.health_score,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
        ]

        # Add status as a metric (1 for active, 0 for inactive)
        status_value = 1.0 if metrics.status in ["healthy", "active"] else 0.0
        agent_metric_points.append(
            MetricPoint(
                "agent_status",
                status_value,
                timestamp,
                MetricType.GAUGE,
                {**base_tags, "status": metrics.status},
            )
        )

        for point in agent_metric_points:
            self.add_metric_point(point)

    def _store_task_metrics(self, metrics: TaskMetrics):
        """Store task metrics as metric points."""
        base_tags = {
            "source": "task",
            "task_type": metrics.task_type,
            "task_id": metrics.task_id,
            "type": "task_metrics",
        }
        if metrics.agent_name:
            base_tags["agent_name"] = metrics.agent_name

        timestamp = metrics.timestamp

        task_metric_points = [
            MetricPoint(
                "task_duration_seconds",
                metrics.duration_seconds,
                timestamp,
                MetricType.HISTOGRAM,
                base_tags,
            ),
            MetricPoint(
                "task_queue_time_seconds",
                metrics.queue_time_seconds,
                timestamp,
                MetricType.HISTOGRAM,
                base_tags,
            ),
            MetricPoint(
                "task_retry_count",
                metrics.retry_count,
                timestamp,
                MetricType.GAUGE,
                base_tags,
            ),
        ]

        # Add status as a metric
        status_value = 1.0 if metrics.status == "completed" else 0.0
        task_metric_points.append(
            MetricPoint(
                "task_success",
                status_value,
                timestamp,
                MetricType.GAUGE,
                {**base_tags, "status": metrics.status},
            )
        )

        for point in task_metric_points:
            self.add_metric_point(point)

    def add_metric_point(self, point: MetricPoint):
        """Add a metric point to the collection."""
        self.metrics_history.append(point)

        # Maintain history size limit
        if len(self.metrics_history) > self.max_history_size:
            # Remove oldest 10% when limit exceeded
            remove_count = self.max_history_size // 10
            self.metrics_history = self.metrics_history[remove_count:]

    def get_recent_metrics(
        self,
        time_window_seconds: float = 300,
        metric_name_filter: Optional[str] = None,
        tags_filter: Optional[Dict[str, str]] = None,
    ) -> List[MetricPoint]:
        """Get recent metrics within specified time window."""
        cutoff_time = time.time() - time_window_seconds

        filtered_metrics = []
        for point in self.metrics_history:
            # Time filter
            if point.timestamp < cutoff_time:
                continue

            # Name filter
            if metric_name_filter and metric_name_filter not in point.name:
                continue

            # Tags filter
            if tags_filter:
                if not all(point.tags.get(k) == v for k, v in tags_filter.items()):
                    continue

            filtered_metrics.append(point)

        return filtered_metrics

    def get_aggregated_metrics(
        self,
        metric_name: str,
        time_window_seconds: float = 300,
        aggregation: str = "avg",
    ) -> Optional[float]:
        """Get aggregated metric value over time window."""
        recent_metrics = self.get_recent_metrics(
            time_window_seconds=time_window_seconds, metric_name_filter=metric_name
        )

        if not recent_metrics:
            return None

        values = [point.value for point in recent_metrics]

        if aggregation == "avg":
            return sum(values) / len(values)
        elif aggregation == "sum":
            return sum(values)
        elif aggregation == "min":
            return min(values)
        elif aggregation == "max":
            return max(values)
        elif aggregation == "count":
            return len(values)
        else:
            return None

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary for dashboard."""
        now = time.time()

        # System metrics summary
        system_summary = {}
        if HAS_PSUTIL:
            system_cpu = self.get_aggregated_metrics("system_cpu_percent", 60)
            system_memory = self.get_aggregated_metrics("system_memory_percent", 60)

            system_summary = {
                "cpu_percent": system_cpu or 0.0,
                "memory_percent": system_memory or 0.0,
                "disk_usage_percent": self.get_aggregated_metrics(
                    "system_disk_usage_percent", 60
                )
                or 0.0,
                "process_count": self.get_aggregated_metrics("system_process_count", 60)
                or 0,
                "load_avg_1min": self.get_aggregated_metrics("system_load_avg_1min", 60)
                or 0.0,
            }

        # Agent metrics summary
        agent_metrics = self.get_recent_metrics(300, tags_filter={"source": "agent"})
        active_agents = len(
            set(
                point.tags.get("agent_name")
                for point in agent_metrics
                if point.tags.get("agent_name")
            )
        )

        # Task metrics summary
        task_metrics = self.get_recent_metrics(300, tags_filter={"source": "task"})
        total_tasks = len(
            set(
                point.tags.get("task_id")
                for point in task_metrics
                if point.tags.get("task_id")
            )
        )

        return {
            "timestamp": now,
            "collection_interval": self.collection_interval,
            "total_metric_points": len(self.metrics_history),
            "system_metrics": system_summary,
            "agent_metrics": {
                "active_agents": active_agents,
                "total_agent_metrics": len(
                    [p for p in agent_metrics if "agent_" in p.name]
                ),
            },
            "task_metrics": {
                "recent_tasks": total_tasks,
                "total_task_metrics": len(
                    [p for p in task_metrics if "task_" in p.name]
                ),
            },
            "integrations": {
                "health_system_available": HAS_HEALTH_SYSTEM,
                "autonomous_dashboard_available": HAS_AUTONOMOUS_DASHBOARD,
                "psutil_available": HAS_PSUTIL,
            },
        }


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _global_metrics_collector

    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()

    return _global_metrics_collector


async def start_metrics_collection():
    """Start global metrics collection."""
    collector = get_metrics_collector()
    await collector.start_collection()


async def stop_metrics_collection():
    """Stop global metrics collection."""
    global _global_metrics_collector

    if _global_metrics_collector:
        await _global_metrics_collector.stop_collection()
