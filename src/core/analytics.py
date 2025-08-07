"""Performance monitoring and analytics system for LeanVibe Agent Hive."""

import asyncio
import json
import statistics
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import structlog

logger = structlog.get_logger()


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""

    name: str
    value: float
    unit: str
    timestamp: float
    tags: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SystemResource:
    """System resource utilization snapshot."""

    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_used_gb: float
    disk_free_gb: float
    network_bytes_sent: int
    network_bytes_recv: int
    load_average: list[float]
    process_count: int
    thread_count: int
    timestamp: float


@dataclass
class AgentPerformance:
    """Agent-specific performance metrics."""

    agent_id: str
    cpu_usage: float
    memory_usage_mb: float
    tasks_completed: int
    tasks_failed: int
    average_task_duration: float
    last_heartbeat: float
    response_times: list[float]
    error_rate: float
    uptime: float
    load_factor: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class APIPerformance:
    """API endpoint performance metrics."""

    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_minute: float
    error_rate: float
    status_code_distribution: dict[int, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PerformanceCollector:
    """Collects and aggregates performance metrics."""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.system_resources: deque = deque(maxlen=max_history)
        self.agent_metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.api_metrics: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        self.request_times: dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.error_counts: dict[str, int] = defaultdict(int)
        self.request_counts: dict[str, int] = defaultdict(int)

        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 70.0,
            "cpu_critical": 90.0,
            "memory_warning": 80.0,
            "memory_critical": 95.0,
            "response_time_warning": 1000.0,  # ms
            "response_time_critical": 5000.0,  # ms
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.10,  # 10%
        }

        # Start background collection
        self.collection_task: asyncio.Task | None = None
        self.is_collecting = False

    async def start_collection(self, interval: float = 30.0):
        """Start background metric collection."""
        if self.is_collecting:
            return

        self.is_collecting = True
        self.collection_task = asyncio.create_task(self._collection_loop(interval))
        logger.info("Performance collection started", interval=interval)

    async def stop_collection(self):
        """Stop background metric collection."""
        if not self.is_collecting:
            return

        self.is_collecting = False
        if self.collection_task:
            self.collection_task.cancel()
            try:
                await self.collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Performance collection stopped")

    async def _collection_loop(self, interval: float):
        """Background loop for collecting metrics."""
        while self.is_collecting:
            try:
                await self.collect_system_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in performance collection", error=str(e))
                await asyncio.sleep(interval)

    async def collect_system_metrics(self) -> SystemResource:
        """Collect current system resource metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Disk usage for root partition
            disk = psutil.disk_usage("/")

            # Network stats
            network = psutil.net_io_counters()

            # Load average (Unix-like systems)
            try:
                load_avg = list(psutil.getloadavg())
            except (AttributeError, OSError):
                load_avg = [0.0, 0.0, 0.0]

            # Process counts
            process_count = len(psutil.pids())
            thread_count = sum(
                p.num_threads()
                for p in psutil.process_iter(["num_threads"])
                if p.info["num_threads"]
            )

            resource_snapshot = SystemResource(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                load_average=load_avg,
                process_count=process_count,
                thread_count=thread_count,
                timestamp=time.time(),
            )

            self.system_resources.append(resource_snapshot)

            # Create individual metrics
            await self._create_system_metrics(resource_snapshot)

            return resource_snapshot

        except Exception as e:
            logger.error("Failed to collect system metrics", error=str(e))
            raise

    async def _create_system_metrics(self, resource: SystemResource):
        """Create individual metrics from system resource snapshot."""
        timestamp = resource.timestamp

        metrics = [
            PerformanceMetric(
                "cpu_usage", resource.cpu_percent, "%", timestamp, {"type": "system"}
            ),
            PerformanceMetric(
                "memory_usage",
                resource.memory_percent,
                "%",
                timestamp,
                {"type": "system"},
            ),
            PerformanceMetric(
                "memory_used",
                resource.memory_used_mb,
                "MB",
                timestamp,
                {"type": "system"},
            ),
            PerformanceMetric(
                "disk_usage",
                resource.disk_usage_percent,
                "%",
                timestamp,
                {"type": "system"},
            ),
            PerformanceMetric(
                "network_sent",
                resource.network_bytes_sent,
                "bytes",
                timestamp,
                {"type": "network"},
            ),
            PerformanceMetric(
                "network_recv",
                resource.network_bytes_recv,
                "bytes",
                timestamp,
                {"type": "network"},
            ),
            PerformanceMetric(
                "process_count",
                resource.process_count,
                "count",
                timestamp,
                {"type": "system"},
            ),
            PerformanceMetric(
                "thread_count",
                resource.thread_count,
                "count",
                timestamp,
                {"type": "system"},
            ),
        ]

        for i, load in enumerate(resource.load_average):
            metrics.append(
                PerformanceMetric(
                    f"load_avg_{i + 1}m", load, "load", timestamp, {"type": "system"}
                )
            )

        for metric in metrics:
            self.metrics_history.append(metric)

    def record_api_request(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        response_time: float,
        error: str | None = None,
    ):
        """Record API request performance data."""
        request_key = f"{method}:{endpoint}"
        timestamp = time.time()

        # Record response time
        self.request_times[request_key].append(response_time)

        # Update counters
        self.request_counts[request_key] += 1
        if status_code >= 400 or error:
            self.error_counts[request_key] += 1

        # Create performance metric
        metric = PerformanceMetric(
            name="api_response_time",
            value=response_time,
            unit="ms",
            timestamp=timestamp,
            tags={
                "endpoint": endpoint,
                "method": method,
                "status_code": str(status_code),
                "error": "true" if error else "false",
            },
        )

        self.metrics_history.append(metric)

    def record_agent_performance(self, agent_id: str, performance: AgentPerformance):
        """Record agent performance metrics."""
        self.agent_metrics[agent_id].append(performance)

        # Create individual metrics
        timestamp = time.time()
        metrics = [
            PerformanceMetric(
                "agent_cpu_usage",
                performance.cpu_usage,
                "%",
                timestamp,
                {"agent_id": agent_id},
            ),
            PerformanceMetric(
                "agent_memory_usage",
                performance.memory_usage_mb,
                "MB",
                timestamp,
                {"agent_id": agent_id},
            ),
            PerformanceMetric(
                "agent_task_duration",
                performance.average_task_duration,
                "s",
                timestamp,
                {"agent_id": agent_id},
            ),
            PerformanceMetric(
                "agent_error_rate",
                performance.error_rate,
                "rate",
                timestamp,
                {"agent_id": agent_id},
            ),
            PerformanceMetric(
                "agent_load_factor",
                performance.load_factor,
                "factor",
                timestamp,
                {"agent_id": agent_id},
            ),
        ]

        for metric in metrics:
            self.metrics_history.append(metric)

    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-1)."""
        if not self.system_resources:
            return 1.0

        latest = self.system_resources[-1]

        # Score components (0-1, where 1 is best)
        cpu_score = max(0, (100 - latest.cpu_percent) / 100)
        memory_score = max(0, (100 - latest.memory_percent) / 100)
        disk_score = max(0, (100 - latest.disk_usage_percent) / 100)

        # Weight the scores
        health_score = cpu_score * 0.4 + memory_score * 0.4 + disk_score * 0.2

        return min(1.0, max(0.0, health_score))

    def get_api_performance_summary(
        self, time_window_minutes: int = 60
    ) -> dict[str, APIPerformance]:
        """Get API performance summary for the specified time window."""
        cutoff_time = time.time() - (time_window_minutes * 60)
        summary = {}

        for endpoint_key, response_times in self.request_times.items():
            if not response_times:
                continue

            # Filter by time window (approximate)
            recent_times = list(response_times)[-100:]  # Last 100 requests as proxy

            if not recent_times:
                continue

            method, endpoint = endpoint_key.split(":", 1)
            total_requests = self.request_counts[endpoint_key]
            failed_requests = self.error_counts[endpoint_key]
            successful_requests = total_requests - failed_requests

            # Calculate statistics
            avg_response_time = statistics.mean(recent_times)
            p95_response_time = (
                statistics.quantiles(recent_times, n=20)[18]
                if len(recent_times) >= 20
                else avg_response_time
            )
            p99_response_time = (
                statistics.quantiles(recent_times, n=100)[98]
                if len(recent_times) >= 100
                else avg_response_time
            )

            error_rate = failed_requests / max(total_requests, 1)
            requests_per_minute = total_requests / max(time_window_minutes, 1)

            performance = APIPerformance(
                endpoint=endpoint,
                method=method,
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time=avg_response_time,
                p95_response_time=p95_response_time,
                p99_response_time=p99_response_time,
                requests_per_minute=requests_per_minute,
                error_rate=error_rate,
                status_code_distribution={},  # Would need more detailed tracking
            )

            summary[endpoint_key] = performance

        return summary

    def get_performance_alerts(self) -> list[dict[str, Any]]:
        """Get current performance alerts based on thresholds."""
        alerts = []

        if not self.system_resources:
            return alerts

        latest = self.system_resources[-1]

        # CPU alerts
        if latest.cpu_percent >= self.thresholds["cpu_critical"]:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "cpu_usage",
                    "value": latest.cpu_percent,
                    "threshold": self.thresholds["cpu_critical"],
                    "message": f"Critical CPU usage: {latest.cpu_percent:.1f}%",
                }
            )
        elif latest.cpu_percent >= self.thresholds["cpu_warning"]:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "cpu_usage",
                    "value": latest.cpu_percent,
                    "threshold": self.thresholds["cpu_warning"],
                    "message": f"High CPU usage: {latest.cpu_percent:.1f}%",
                }
            )

        # Memory alerts
        if latest.memory_percent >= self.thresholds["memory_critical"]:
            alerts.append(
                {
                    "type": "critical",
                    "metric": "memory_usage",
                    "value": latest.memory_percent,
                    "threshold": self.thresholds["memory_critical"],
                    "message": f"Critical memory usage: {latest.memory_percent:.1f}%",
                }
            )
        elif latest.memory_percent >= self.thresholds["memory_warning"]:
            alerts.append(
                {
                    "type": "warning",
                    "metric": "memory_usage",
                    "value": latest.memory_percent,
                    "threshold": self.thresholds["memory_warning"],
                    "message": f"High memory usage: {latest.memory_percent:.1f}%",
                }
            )

        # API performance alerts
        api_summary = self.get_api_performance_summary(5)  # Last 5 minutes
        for endpoint_key, perf in api_summary.items():
            if perf.error_rate >= self.thresholds["error_rate_critical"]:
                alerts.append(
                    {
                        "type": "critical",
                        "metric": "api_error_rate",
                        "value": perf.error_rate,
                        "threshold": self.thresholds["error_rate_critical"],
                        "message": f"Critical error rate for {endpoint_key}: {perf.error_rate:.1%}",
                    }
                )
            elif perf.error_rate >= self.thresholds["error_rate_warning"]:
                alerts.append(
                    {
                        "type": "warning",
                        "metric": "api_error_rate",
                        "value": perf.error_rate,
                        "threshold": self.thresholds["error_rate_warning"],
                        "message": f"High error rate for {endpoint_key}: {perf.error_rate:.1%}",
                    }
                )

            if perf.average_response_time >= self.thresholds["response_time_critical"]:
                alerts.append(
                    {
                        "type": "critical",
                        "metric": "api_response_time",
                        "value": perf.average_response_time,
                        "threshold": self.thresholds["response_time_critical"],
                        "message": f"Critical response time for {endpoint_key}: {perf.average_response_time:.0f}ms",
                    }
                )
            elif perf.average_response_time >= self.thresholds["response_time_warning"]:
                alerts.append(
                    {
                        "type": "warning",
                        "metric": "api_response_time",
                        "value": perf.average_response_time,
                        "threshold": self.thresholds["response_time_warning"],
                        "message": f"Slow response time for {endpoint_key}: {perf.average_response_time:.0f}ms",
                    }
                )

        return alerts

    def get_metrics_summary(self, time_window_minutes: int = 60) -> dict[str, Any]:
        """Get comprehensive metrics summary."""
        cutoff_time = time.time() - (time_window_minutes * 60)

        # Filter recent metrics
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]

        # Group by metric name
        metrics_by_name = defaultdict(list)
        for metric in recent_metrics:
            metrics_by_name[metric.name].append(metric.value)

        # Calculate statistics
        summary = {}
        for name, values in metrics_by_name.items():
            if values:
                summary[name] = {
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
                }

        return {
            "time_window_minutes": time_window_minutes,
            "metrics_count": len(recent_metrics),
            "metrics": summary,
            "system_health_score": self.get_system_health_score(),
            "alerts": self.get_performance_alerts(),
            "api_performance": {
                k: v.to_dict()
                for k, v in self.get_api_performance_summary(
                    time_window_minutes
                ).items()
            },
            "last_updated": time.time(),
        }

    def export_metrics(self, format: str = "json", time_window_hours: int = 24) -> str:
        """Export metrics in specified format."""
        cutoff_time = time.time() - (time_window_hours * 3600)

        # Filter metrics
        recent_metrics = [m for m in self.metrics_history if m.timestamp >= cutoff_time]
        recent_resources = [
            r for r in self.system_resources if r.timestamp >= cutoff_time
        ]

        if format.lower() == "json":
            export_data = {
                "export_timestamp": time.time(),
                "time_window_hours": time_window_hours,
                "metrics": [m.to_dict() for m in recent_metrics],
                "system_resources": [asdict(r) for r in recent_resources],
                "agent_metrics": {
                    agent_id: [perf.to_dict() for perf in performances]
                    for agent_id, performances in self.agent_metrics.items()
                },
                "summary": self.get_metrics_summary(time_window_hours * 60),
            }
            return json.dumps(export_data, indent=2)

        elif format.lower() == "csv":
            # Simple CSV export of metrics
            lines = ["timestamp,metric_name,value,unit,tags"]
            for metric in recent_metrics:
                tags_str = ";".join(f"{k}={v}" for k, v in metric.tags.items())
                lines.append(
                    f"{metric.timestamp},{metric.name},{metric.value},{metric.unit},{tags_str}"
                )
            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")


# Global performance collector instance
performance_collector = PerformanceCollector()


class PerformanceMiddleware:
    """FastAPI middleware for automatic performance monitoring."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start_time = time.time()
        status_code = 500
        error = None

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            error = str(e)
            raise
        finally:
            # Record the request
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            endpoint = scope.get("path", "unknown")
            method = scope.get("method", "unknown")

            performance_collector.record_api_request(
                endpoint=endpoint,
                method=method,
                status_code=status_code,
                response_time=response_time,
                error=error,
            )


async def start_performance_monitoring():
    """Start the performance monitoring system."""
    await performance_collector.start_collection()


async def stop_performance_monitoring():
    """Stop the performance monitoring system."""
    await performance_collector.stop_collection()


def get_performance_collector() -> PerformanceCollector:
    """Get the global performance collector instance."""
    return performance_collector
