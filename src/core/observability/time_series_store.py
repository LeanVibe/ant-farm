"""Time Series Storage System for Observability Metrics.

This module provides efficient storage and retrieval of time series data
for metrics and observability purposes.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json
from collections import defaultdict, deque

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


@dataclass
class MetricPoint:
    """Individual metric data point for time series."""

    timestamp: float
    value: Union[int, float]
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {"timestamp": self.timestamp, "value": self.value, "tags": self.tags}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricPoint":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"], value=data["value"], tags=data.get("tags", {})
        )


@dataclass
class TimeSeriesQuery:
    """Query parameters for time series data."""

    metric_name: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    tags_filter: Optional[Dict[str, str]] = None
    aggregation: Optional[str] = None  # avg, sum, min, max, count
    interval: Optional[float] = None  # Grouping interval in seconds
    limit: Optional[int] = None

    def __post_init__(self):
        """Set default values after initialization."""
        if self.end_time is None:
            self.end_time = time.time()
        if self.start_time is None:
            self.start_time = self.end_time - 3600  # Default to last hour


class TimeSeriesStore:
    """In-memory time series storage with efficient querying."""

    def __init__(self, max_points_per_series: int = 10000, retention_hours: int = 24):
        self.max_points_per_series = max_points_per_series
        self.retention_hours = retention_hours

        # Storage: metric_name -> list of MetricPoint
        self.series_data: Dict[str, deque[MetricPoint]] = defaultdict(
            lambda: deque(maxlen=max_points_per_series)
        )

        # Indexes for efficient querying
        self.tag_indexes: Dict[str, Dict[str, set]] = defaultdict(
            lambda: defaultdict(set)
        )

        # Cleanup state
        self.is_cleaning = False
        self.cleanup_task: Optional[asyncio.Task] = None

        log_info(
            "TimeSeriesStore initialized",
            max_points_per_series=max_points_per_series,
            retention_hours=retention_hours,
        )

    async def start_cleanup(self):
        """Start automatic cleanup of old data."""
        if self.is_cleaning:
            log_info("Time series cleanup already running")
            return

        self.is_cleaning = True
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        log_info("Started time series cleanup")

    async def stop_cleanup(self):
        """Stop automatic cleanup."""
        self.is_cleaning = False

        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass

        log_info("Stopped time series cleanup")

    async def _cleanup_loop(self):
        """Automatic cleanup loop."""
        while self.is_cleaning:
            try:
                await self.cleanup_old_data()
                await asyncio.sleep(3600)  # Cleanup every hour
            except asyncio.CancelledError:
                break
            except Exception as e:
                log_error("Error in cleanup loop", error=str(e))
                await asyncio.sleep(300)  # Wait 5 minutes on error

    async def store_point(self, metric_name: str, point: MetricPoint):
        """Store a single metric point."""
        try:
            # Add to series data
            self.series_data[metric_name].append(point)

            # Update tag indexes
            for tag_key, tag_value in point.tags.items():
                self.tag_indexes[tag_key][tag_value].add(metric_name)

        except Exception as e:
            log_error(
                "Failed to store metric point", metric_name=metric_name, error=str(e)
            )

    async def store_points(self, metric_name: str, points: List[MetricPoint]):
        """Store multiple metric points efficiently."""
        try:
            series = self.series_data[metric_name]

            for point in points:
                series.append(point)

                # Update tag indexes
                for tag_key, tag_value in point.tags.items():
                    self.tag_indexes[tag_key][tag_value].add(metric_name)

        except Exception as e:
            log_error(
                "Failed to store metric points",
                metric_name=metric_name,
                count=len(points),
                error=str(e),
            )

    async def query(self, query: TimeSeriesQuery) -> List[MetricPoint]:
        """Query time series data."""
        try:
            # Get base series
            if query.metric_name not in self.series_data:
                return []

            points = list(self.series_data[query.metric_name])

            # Apply time range filter
            if query.start_time is not None or query.end_time is not None:
                points = [
                    p
                    for p in points
                    if (query.start_time is None or p.timestamp >= query.start_time)
                    and (query.end_time is None or p.timestamp <= query.end_time)
                ]

            # Apply tags filter
            if query.tags_filter:
                points = [
                    p
                    for p in points
                    if all(p.tags.get(k) == v for k, v in query.tags_filter.items())
                ]

            # Sort by timestamp
            points.sort(key=lambda p: p.timestamp)

            # Apply aggregation if specified
            if query.aggregation and query.interval:
                points = self._aggregate_points(
                    points, query.aggregation, query.interval
                )

            # Apply limit
            if query.limit:
                points = points[: query.limit]

            return points

        except Exception as e:
            log_error(
                "Failed to query time series",
                metric_name=query.metric_name,
                error=str(e),
            )
            return []

    def _aggregate_points(
        self, points: List[MetricPoint], aggregation: str, interval: float
    ) -> List[MetricPoint]:
        """Aggregate points into time intervals."""
        if not points:
            return []

        aggregated = []
        current_bucket_start = None
        current_bucket_points = []

        for point in points:
            # Calculate bucket start time
            bucket_start = (point.timestamp // interval) * interval

            if current_bucket_start is None:
                current_bucket_start = bucket_start

            if bucket_start != current_bucket_start:
                # Process current bucket
                if current_bucket_points:
                    agg_point = self._create_aggregated_point(
                        current_bucket_points, aggregation, current_bucket_start
                    )
                    aggregated.append(agg_point)

                # Start new bucket
                current_bucket_start = bucket_start
                current_bucket_points = []

            current_bucket_points.append(point)

        # Process final bucket
        if current_bucket_points:
            agg_point = self._create_aggregated_point(
                current_bucket_points, aggregation, current_bucket_start
            )
            aggregated.append(agg_point)

        return aggregated

    def _create_aggregated_point(
        self, points: List[MetricPoint], aggregation: str, timestamp: float
    ) -> MetricPoint:
        """Create an aggregated point from a list of points."""
        values = [p.value for p in points]

        if aggregation == "avg":
            agg_value = sum(values) / len(values)
        elif aggregation == "sum":
            agg_value = sum(values)
        elif aggregation == "min":
            agg_value = min(values)
        elif aggregation == "max":
            agg_value = max(values)
        elif aggregation == "count":
            agg_value = len(values)
        else:
            agg_value = sum(values) / len(values)  # Default to average

        # Merge tags from all points
        merged_tags = {}
        for point in points:
            merged_tags.update(point.tags)

        return MetricPoint(timestamp=timestamp, value=agg_value, tags=merged_tags)

    async def get_metric_names(self) -> List[str]:
        """Get all available metric names."""
        return list(self.series_data.keys())

    async def get_tag_values(self, tag_key: str) -> List[str]:
        """Get all available values for a specific tag key."""
        if tag_key in self.tag_indexes:
            return list(self.tag_indexes[tag_key].keys())
        return []

    async def get_metrics_with_tag(self, tag_key: str, tag_value: str) -> List[str]:
        """Get metric names that have a specific tag value."""
        if tag_key in self.tag_indexes and tag_value in self.tag_indexes[tag_key]:
            return list(self.tag_indexes[tag_key][tag_value])
        return []

    async def cleanup_old_data(self):
        """Remove data older than retention period."""
        try:
            cutoff_time = time.time() - (self.retention_hours * 3600)
            cleaned_series = 0
            cleaned_points = 0

            for metric_name, points in self.series_data.items():
                original_count = len(points)

                # Remove old points (deque doesn't support slice assignment, so we rebuild)
                filtered_points = deque(
                    (p for p in points if p.timestamp > cutoff_time),
                    maxlen=self.max_points_per_series,
                )

                self.series_data[metric_name] = filtered_points
                removed_count = original_count - len(filtered_points)

                if removed_count > 0:
                    cleaned_series += 1
                    cleaned_points += removed_count

            # Clean up tag indexes by removing references to deleted metrics
            await self._cleanup_tag_indexes()

            if cleaned_points > 0:
                log_info(
                    "Cleaned up old time series data",
                    cleaned_series=cleaned_series,
                    cleaned_points=cleaned_points,
                    cutoff_hours=self.retention_hours,
                )

        except Exception as e:
            log_error("Failed to cleanup old data", error=str(e))

    async def _cleanup_tag_indexes(self):
        """Clean up tag indexes by removing references to non-existent metrics."""
        try:
            existing_metrics = set(self.series_data.keys())

            for tag_key in list(self.tag_indexes.keys()):
                for tag_value in list(self.tag_indexes[tag_key].keys()):
                    # Filter out non-existent metrics
                    valid_metrics = (
                        self.tag_indexes[tag_key][tag_value] & existing_metrics
                    )

                    if valid_metrics:
                        self.tag_indexes[tag_key][tag_value] = valid_metrics
                    else:
                        # Remove empty tag value
                        del self.tag_indexes[tag_key][tag_value]

                # Remove empty tag keys
                if not self.tag_indexes[tag_key]:
                    del self.tag_indexes[tag_key]

        except Exception as e:
            log_error("Failed to cleanup tag indexes", error=str(e))

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        total_points = sum(len(points) for points in self.series_data.values())
        total_series = len(self.series_data)

        # Calculate size distribution
        series_sizes = [len(points) for points in self.series_data.values()]
        avg_series_size = sum(series_sizes) / len(series_sizes) if series_sizes else 0

        # Tag statistics
        total_tag_keys = len(self.tag_indexes)
        total_tag_values = sum(len(values) for values in self.tag_indexes.values())

        return {
            "total_series": total_series,
            "total_points": total_points,
            "avg_points_per_series": avg_series_size,
            "max_points_per_series": self.max_points_per_series,
            "retention_hours": self.retention_hours,
            "tag_statistics": {
                "total_tag_keys": total_tag_keys,
                "total_tag_values": total_tag_values,
                "avg_values_per_key": total_tag_values / total_tag_keys
                if total_tag_keys > 0
                else 0,
            },
        }

    async def export_data(self, format: str = "json") -> str:
        """Export all time series data."""
        try:
            if format.lower() == "json":
                export_data = {}
                for metric_name, points in self.series_data.items():
                    export_data[metric_name] = [p.to_dict() for p in points]
                return json.dumps(export_data, indent=2)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        except Exception as e:
            log_error("Failed to export data", format=format, error=str(e))
            return ""

    async def import_data(self, data: str, format: str = "json"):
        """Import time series data."""
        try:
            if format.lower() == "json":
                import_data = json.loads(data)

                for metric_name, points_data in import_data.items():
                    points = [MetricPoint.from_dict(p) for p in points_data]
                    await self.store_points(metric_name, points)

                log_info(
                    "Imported time series data",
                    metrics_count=len(import_data),
                    format=format,
                )
            else:
                raise ValueError(f"Unsupported import format: {format}")

        except Exception as e:
            log_error("Failed to import data", format=format, error=str(e))


# Global time series store instance
_global_time_series_store: Optional[TimeSeriesStore] = None


def get_time_series_store() -> TimeSeriesStore:
    """Get the global time series store instance."""
    global _global_time_series_store

    if _global_time_series_store is None:
        _global_time_series_store = TimeSeriesStore()

    return _global_time_series_store


async def start_time_series_cleanup():
    """Start global time series cleanup."""
    store = get_time_series_store()
    await store.start_cleanup()


async def stop_time_series_cleanup():
    """Stop global time series cleanup."""
    global _global_time_series_store

    if _global_time_series_store:
        await _global_time_series_store.stop_cleanup()
