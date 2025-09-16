"""Real-time metrics collection and monitoring for LeanVibe Agent Hive.

This module provides comprehensive performance monitoring capabilities including:
- System resource metrics (CPU, memory, disk, network)
- Application performance metrics (response times, throughput, errors)
- Database connection and query metrics
- Message broker queue and throughput metrics
- Agent performance and health metrics
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque
import psutil
import json

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric measurement point."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """Time series data for a specific metric."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """Central metrics collection and storage service."""

    def __init__(self, redis_client=None, collection_interval: float = 10.0):
        self.redis_client = redis_client
        self.collection_interval = collection_interval
        self.metrics: Dict[str, MetricSeries] = {}
        self.is_running = False
        self.collection_task: Optional[asyncio.Task] = None

        # Initialize metric series
        self._init_metric_series()

    def _init_metric_series(self):
        """Initialize all metric series."""
        # System metrics
        system_metrics = [
            "cpu_percent", "memory_percent", "memory_used_mb", "disk_usage_percent",
            "network_bytes_sent", "network_bytes_recv", "load_average_1m", "load_average_5m"
        ]

        # Application metrics
        app_metrics = [
            "http_requests_total", "http_request_duration_seconds", "http_requests_active",
            "websocket_connections", "api_errors_total", "cache_hit_ratio"
        ]

        # Database metrics
        db_metrics = [
            "db_connections_active", "db_connections_idle", "db_query_duration_seconds",
            "db_transactions_total", "db_connection_pool_size"
        ]

        # Message broker metrics
        broker_metrics = [
            "broker_messages_published", "broker_messages_consumed", "broker_queue_depth",
            "broker_connections_active", "broker_publish_rate", "broker_consume_rate"
        ]

        # Agent metrics
        agent_metrics = [
            "agents_active", "agent_response_time_seconds", "agent_tasks_completed",
            "agent_tasks_failed", "agent_memory_usage_mb", "agent_cpu_percent"
        ]

        all_metrics = system_metrics + app_metrics + db_metrics + broker_metrics + agent_metrics

        for metric_name in all_metrics:
            self.metrics[metric_name] = MetricSeries(name=metric_name)

    async def start_collection(self):
        """Start the metrics collection loop."""
        if self.is_running:
            return

        self.is_running = True
        self.collection_task = asyncio.create_task(self._collection_loop())
        logger.info(f"Metrics collection started with {self.collection_interval}s interval")

    async def stop_collection(self):
        """Stop the metrics collection loop."""
        if not self.is_running:
            return

        self.is_running = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics collection stopped")

    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self.is_running:
            try:
                await self._collect_all_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _collect_all_metrics(self):
        """Collect all types of metrics."""
        # System metrics
        await self._collect_system_metrics()

        # Application metrics (collected via callbacks)
        await self._collect_application_metrics()

        # Database metrics
        await self._collect_database_metrics()

        # Message broker metrics
        await self._collect_broker_metrics()

        # Agent metrics
        await self._collect_agent_metrics()

        # Store to Redis if available
        if self.redis_client:
            await self._store_metrics_to_redis()

    async def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self._record_metric("cpu_percent", cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            self._record_metric("memory_percent", memory.percent)
            self._record_metric("memory_used_mb", memory.used / 1024 / 1024)

            # Disk usage
            disk = psutil.disk_usage('/')
            self._record_metric("disk_usage_percent", disk.percent)

            # Network I/O
            network = psutil.net_io_counters()
            self._record_metric("network_bytes_sent", network.bytes_sent)
            self._record_metric("network_bytes_recv", network.bytes_recv)

            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()
                self._record_metric("load_average_1m", load_avg[0])
                self._record_metric("load_average_5m", load_avg[1])
            except AttributeError:
                # Windows doesn't have load average
                pass

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")

    async def _collect_application_metrics(self):
        """Collect application-level metrics."""
        # These will be updated by the application via record_metric calls
        # For now, just ensure the series exist
        pass

    async def _collect_database_metrics(self):
        """Collect database performance metrics."""
        try:
            # This will be enhanced when we have database connection monitoring
            # For now, placeholder for future database metrics
            pass
        except Exception as e:
            logger.error(f"Failed to collect database metrics: {str(e)}")

    async def _collect_broker_metrics(self):
        """Collect message broker metrics."""
        try:
            # This will be enhanced when we have broker monitoring
            # For now, placeholder for future broker metrics
            pass
        except Exception as e:
            logger.error(f"Failed to collect broker metrics: {str(e)}")

    async def _collect_agent_metrics(self):
        """Collect agent performance metrics."""
        try:
            # This will be enhanced when we have agent monitoring
            # For now, placeholder for future agent metrics
            pass
        except Exception as e:
            logger.error(f"Failed to collect agent metrics: {str(e)}")

    def _record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name)

        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels or {}
        )

        self.metrics[name].points.append(point)

    async def _store_metrics_to_redis(self):
        """Store recent metrics to Redis for persistence."""
        if not self.redis_client:
            return

        try:
            metrics_data = {}
            for name, series in self.metrics.items():
                if series.points:
                    # Store last 100 points for each metric
                    recent_points = list(series.points)[-100:]
                    metrics_data[name] = {
                        "points": [
                            {"timestamp": p.timestamp, "value": p.value, "labels": p.labels}
                            for p in recent_points
                        ],
                        "metadata": series.metadata
                    }

            # Store with 1 hour expiration
            await self.redis_client.setex(
                "system_metrics",
                3600,
                json.dumps(metrics_data)
            )

        except Exception as e:
            logger.error(f"Failed to store metrics to Redis: {str(e)}")

    def record_http_request(self, method: str, path: str, duration: float, status_code: int):
        """Record HTTP request metrics."""
        self._record_metric("http_requests_total", 1, {"method": method, "path": path, "status": str(status_code)})
        self._record_metric("http_request_duration_seconds", duration, {"method": method, "path": path})

    def record_api_error(self, error_type: str, endpoint: str):
        """Record API error metrics."""
        self._record_metric("api_errors_total", 1, {"type": error_type, "endpoint": endpoint})

    def record_websocket_connection(self, action: str):
        """Record WebSocket connection metrics."""
        self._record_metric("websocket_connections", 1 if action == "connect" else -1)

    def record_cache_hit(self, hit: bool):
        """Record cache performance metrics."""
        # This will be used to calculate hit ratio
        self._record_metric("cache_hits" if hit else "cache_misses", 1)

    def get_metric_values(self, name: str, limit: int = 100) -> List[MetricPoint]:
        """Get recent values for a specific metric."""
        if name not in self.metrics:
            return []

        return list(self.metrics[name].points)[-limit:]

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {}
        for name, series in self.metrics.items():
            if series.points:
                values = [p.value for p in series.points]
                summary[name] = {
                    "count": len(values),
                    "latest": values[-1] if values else None,
                    "average": sum(values) / len(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                }
        return summary

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)."""
        try:
            scores = []

            # CPU health (lower is better, invert scale)
            cpu_values = [p.value for p in self.get_metric_values("cpu_percent", 10)]
            if cpu_values:
                avg_cpu = sum(cpu_values) / len(cpu_values)
                cpu_score = max(0, 100 - avg_cpu)  # 100% CPU = 0 score
                scores.append(cpu_score)

            # Memory health
            mem_values = [p.value for p in self.get_metric_values("memory_percent", 10)]
            if mem_values:
                avg_mem = sum(mem_values) / len(mem_values)
                mem_score = max(0, 100 - avg_mem)
                scores.append(mem_score)

            # Error rate health
            error_values = [p.value for p in self.get_metric_values("api_errors_total", 60)]  # Last minute
            if error_values:
                error_rate = sum(error_values) / 60  # Errors per second
                error_score = max(0, 100 - (error_rate * 100))  # High error rate = low score
                scores.append(error_score)

            if scores:
                return sum(scores) / len(scores)
            else:
                return 50.0  # Default neutral score

        except Exception as e:
            logger.error(f"Failed to calculate health score: {str(e)}")
            return 50.0


# Global metrics collector instance
metrics_collector = MetricsCollector()


async def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return metrics_collector


async def start_metrics_collection():
    """Start the global metrics collection."""
    await metrics_collector.start_collection()


async def stop_metrics_collection():
    """Stop the global metrics collection."""
    await metrics_collector.stop_collection()