"""Real-time observability system for LeanVibe Agent Hive.

This module provides comprehensive real-time monitoring and observability including:
- Metrics collection and aggregation
- Event streaming and processing
- Time series storage and retrieval
- Alert management and notifications
- Real-time dashboard data feeds
"""

try:
    from .metrics_collector import (
        MetricsCollector,
        SystemMetrics,
        AgentMetrics,
        TaskMetrics,
        MetricType,
        get_metrics_collector,
        start_metrics_collection,
        stop_metrics_collection,
    )
    from .event_aggregator import (
        EventAggregator,
        EventType,
        SystemEvent,
        get_event_aggregator,
        start_event_processing,
        stop_event_processing,
    )
    from .time_series_store import (
        TimeSeriesStore,
        MetricPoint,
        TimeSeriesQuery,
        get_time_series_store,
        start_time_series_cleanup,
        stop_time_series_cleanup,
    )
    from .alert_manager import (
        AlertManager,
        Alert,
        AlertRule,
        AlertSeverity,
        AlertState,
        get_alert_manager,
        start_alert_evaluation,
        stop_alert_evaluation,
    )

    __all__ = [
        "MetricsCollector",
        "SystemMetrics",
        "AgentMetrics",
        "TaskMetrics",
        "MetricType",
        "get_metrics_collector",
        "start_metrics_collection",
        "stop_metrics_collection",
        "EventAggregator",
        "EventType",
        "SystemEvent",
        "get_event_aggregator",
        "start_event_processing",
        "stop_event_processing",
        "TimeSeriesStore",
        "MetricPoint",
        "TimeSeriesQuery",
        "get_time_series_store",
        "start_time_series_cleanup",
        "stop_time_series_cleanup",
        "AlertManager",
        "Alert",
        "AlertRule",
        "AlertSeverity",
        "AlertState",
        "get_alert_manager",
        "start_alert_evaluation",
        "stop_alert_evaluation",
    ]

except ImportError as e:
    # Graceful degradation if dependencies are missing
    print(f"Warning: Observability system components not available: {e}")
    __all__ = []
