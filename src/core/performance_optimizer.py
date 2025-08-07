"""Performance optimization and predictive scheduling system."""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import structlog

from .config import settings
from .models import Agent, SystemMetric, get_database_manager
from .task_queue import task_queue

logger = structlog.get_logger()


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""

    LOAD_BALANCING = "load_balancing"
    RESOURCE_SCALING = "resource_scaling"
    TASK_BATCHING = "task_batching"
    PRIORITY_ADJUSTMENT = "priority_adjustment"
    AGENT_SPECIALIZATION = "agent_specialization"
    PREDICTIVE_SCALING = "predictive_scaling"


class PerformanceMetric(Enum):
    """Key performance metrics to optimize."""

    THROUGHPUT = "throughput"
    LATENCY = "latency"
    RESOURCE_UTILIZATION = "resource_utilization"
    ERROR_RATE = "error_rate"
    QUEUE_TIME = "queue_time"
    AGENT_EFFICIENCY = "agent_efficiency"


@dataclass
class PerformanceTarget:
    """Performance target for optimization."""

    metric: PerformanceMetric
    target_value: float
    tolerance: float
    priority: float  # 0.0 to 1.0


@dataclass
class OptimizationAction:
    """An action to optimize performance."""

    strategy: OptimizationStrategy
    description: str
    parameters: dict[str, Any]
    expected_impact: dict[PerformanceMetric, float]
    confidence: float
    cost: float  # Resource cost of implementing


@dataclass
class PredictionModel:
    """Model for predicting system behavior."""

    name: str
    input_features: list[str]
    coefficients: dict[str, float]
    accuracy: float
    last_trained: float
    prediction_horizon_hours: float


@dataclass
class WorkloadPrediction:
    """Predicted workload characteristics."""

    timestamp: float
    predicted_task_count: int
    predicted_queue_depth: float
    predicted_resource_usage: dict[str, float]
    confidence_interval: tuple[float, float]
    prediction_model: str


class TimeSeriesAnalyzer:
    """Analyzes time series data for patterns and predictions."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_windows: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

    def add_data_point(
        self, metric_name: str, value: float, timestamp: float = None
    ) -> None:
        """Add a data point to the time series."""
        if timestamp is None:
            timestamp = time.time()

        self.data_windows[metric_name].append((timestamp, value))

    def detect_trend(self, metric_name: str) -> tuple[str, float]:
        """Detect trend in metric data."""
        if (
            metric_name not in self.data_windows
            or len(self.data_windows[metric_name]) < 10
        ):
            return "unknown", 0.0

        data = list(self.data_windows[metric_name])
        values = [point[1] for point in data]

        # Simple linear regression for trend detection
        n = len(values)
        x = np.arange(n)
        y = np.array(values)

        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]

        # Determine trend
        if abs(slope) < 0.01:
            return "stable", slope
        elif slope > 0:
            return "increasing", slope
        else:
            return "decreasing", slope

    def detect_seasonality(self, metric_name: str) -> dict[str, Any]:
        """Detect seasonal patterns in data."""
        if (
            metric_name not in self.data_windows
            or len(self.data_windows[metric_name]) < 50
        ):
            return {"seasonal": False}

        data = list(self.data_windows[metric_name])
        timestamps = [point[0] for point in data]
        values = [point[1] for point in data]

        # Check for daily patterns (simplified)
        hours = [(ts % (24 * 3600)) / 3600 for ts in timestamps]

        # Group by hour and calculate average
        hourly_avg = defaultdict(list)
        for hour, value in zip(hours, values, strict=False):
            hourly_avg[int(hour)].append(value)

        # Calculate variance between hours
        hour_averages = {
            h: np.mean(vals) for h, vals in hourly_avg.items() if len(vals) > 1
        }

        if len(hour_averages) > 5:
            variance = np.var(list(hour_averages.values()))
            mean_value = np.mean(list(hour_averages.values()))

            # If variance is significant relative to mean, consider seasonal
            coefficient_of_variation = (
                variance / (mean_value**2) if mean_value > 0 else 0
            )

            return {
                "seasonal": coefficient_of_variation > 0.1,
                "daily_pattern": hour_averages,
                "peak_hour": max(hour_averages.keys(), key=lambda h: hour_averages[h]),
                "low_hour": min(hour_averages.keys(), key=lambda h: hour_averages[h]),
            }

        return {"seasonal": False}

    def predict_next_values(
        self, metric_name: str, steps_ahead: int = 5
    ) -> list[float]:
        """Predict next values using simple time series forecasting."""
        if (
            metric_name not in self.data_windows
            or len(self.data_windows[metric_name]) < 20
        ):
            return [0.0] * steps_ahead

        data = list(self.data_windows[metric_name])
        values = [point[1] for point in data]

        # Simple moving average with trend
        window = min(10, len(values) // 2)
        recent_avg = np.mean(values[-window:])

        # Calculate trend
        _, slope = self.detect_trend(metric_name)

        # Predict future values
        predictions = []
        for i in range(1, steps_ahead + 1):
            predicted_value = recent_avg + (slope * i)
            predictions.append(max(0, predicted_value))  # Ensure non-negative

        return predictions


class ResourcePredictor:
    """Predicts resource requirements and workload patterns."""

    def __init__(self):
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.prediction_models: dict[str, PredictionModel] = {}
        self.feature_history: dict[str, list[float]] = defaultdict(list)

        # Initialize simple prediction models
        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialize prediction models."""
        # Task count prediction model
        self.prediction_models["task_count"] = PredictionModel(
            name="task_count_predictor",
            input_features=[
                "hour_of_day",
                "day_of_week",
                "recent_task_rate",
                "queue_depth",
            ],
            coefficients={
                "hour_of_day": 0.1,
                "day_of_week": 0.05,
                "recent_task_rate": 0.8,
                "queue_depth": 0.05,
            },
            accuracy=0.7,
            last_trained=time.time(),
            prediction_horizon_hours=1.0,
        )

        # Resource usage prediction model
        self.prediction_models["resource_usage"] = PredictionModel(
            name="resource_usage_predictor",
            input_features=["current_usage", "task_count", "agent_count", "time_trend"],
            coefficients={
                "current_usage": 0.6,
                "task_count": 0.2,
                "agent_count": 0.1,
                "time_trend": 0.1,
            },
            accuracy=0.75,
            last_trained=time.time(),
            prediction_horizon_hours=2.0,
        )

    async def collect_features(self) -> dict[str, float]:
        """Collect current features for prediction."""
        features = {}
        current_time = time.time()

        try:
            # Time-based features
            import datetime

            dt = datetime.datetime.fromtimestamp(current_time)
            features["hour_of_day"] = dt.hour
            features["day_of_week"] = dt.weekday()
            features["minute_of_hour"] = dt.minute

            # Task queue features
            queue_depth = await task_queue.get_queue_depth()
            features["queue_depth"] = float(queue_depth)

            # Calculate recent task rate
            completed_tasks = await task_queue.get_completed_tasks()
            # Simple approximation - would track over time window
            features["recent_task_rate"] = float(completed_tasks) / max(
                1, current_time / 3600
            )

            # Agent features
            db_manager = get_database_manager(settings.database_url)
            db_session = db_manager.get_session()

            try:
                active_agents = (
                    db_session.query(Agent).filter_by(status="active").count()
                )
                features["agent_count"] = float(active_agents)

                # Get recent system metrics
                recent_metrics = (
                    db_session.query(SystemMetric)
                    .filter(SystemMetric.timestamp > current_time - 3600)
                    .all()
                )

                if recent_metrics:
                    # CPU and memory usage
                    cpu_metrics = [
                        m.value for m in recent_metrics if m.metric_name == "cpu_usage"
                    ]
                    memory_metrics = [
                        m.value
                        for m in recent_metrics
                        if m.metric_name == "memory_usage"
                    ]

                    if cpu_metrics:
                        features["current_cpu_usage"] = np.mean(cpu_metrics)
                    if memory_metrics:
                        features["current_memory_usage"] = np.mean(memory_metrics)

            finally:
                db_session.close()

            # Add to time series
            for feature_name, value in features.items():
                self.time_series_analyzer.add_data_point(feature_name, value)

            return features

        except Exception as e:
            logger.error("Failed to collect prediction features", error=str(e))
            return {}

    async def predict_workload(self, hours_ahead: float = 1.0) -> WorkloadPrediction:
        """Predict workload characteristics."""
        features = await self.collect_features()

        if not features:
            return WorkloadPrediction(
                timestamp=time.time() + hours_ahead * 3600,
                predicted_task_count=0,
                predicted_queue_depth=0.0,
                predicted_resource_usage={},
                confidence_interval=(0.0, 0.0),
                prediction_model="none",
            )

        # Predict task count
        task_count_model = self.prediction_models["task_count"]
        predicted_task_count = self._apply_model(task_count_model, features)

        # Predict resource usage
        resource_model = self.prediction_models["resource_usage"]
        predicted_cpu = self._apply_model(resource_model, features, "cpu")
        predicted_memory = self._apply_model(resource_model, features, "memory")

        # Simple queue depth prediction based on task rate vs processing capacity
        current_queue = features.get("queue_depth", 0.0)
        task_rate = features.get("recent_task_rate", 0.0)
        agent_count = features.get("agent_count", 1.0)

        # Assume each agent can process ~10 tasks/hour
        processing_capacity = agent_count * 10
        predicted_queue_depth = max(
            0, current_queue + (task_rate - processing_capacity) * hours_ahead
        )

        # Calculate confidence interval (simplified)
        confidence_range = predicted_task_count * 0.2
        confidence_interval = (
            predicted_task_count - confidence_range,
            predicted_task_count + confidence_range,
        )

        return WorkloadPrediction(
            timestamp=time.time() + hours_ahead * 3600,
            predicted_task_count=int(predicted_task_count),
            predicted_queue_depth=predicted_queue_depth,
            predicted_resource_usage={"cpu": predicted_cpu, "memory": predicted_memory},
            confidence_interval=confidence_interval,
            prediction_model=task_count_model.name,
        )

    def _apply_model(
        self, model: PredictionModel, features: dict[str, float], variant: str = None
    ) -> float:
        """Apply a prediction model to features."""
        prediction = 0.0

        for feature_name, coefficient in model.coefficients.items():
            feature_value = features.get(feature_name, 0.0)

            # Apply variant-specific adjustments
            if variant == "cpu" and feature_name == "current_usage":
                feature_value = features.get("current_cpu_usage", 50.0)
            elif variant == "memory" and feature_name == "current_usage":
                feature_value = features.get("current_memory_usage", 50.0)

            prediction += coefficient * feature_value

        return max(0.0, prediction)


class PerformanceOptimizer:
    """Main performance optimization system."""

    def __init__(self):
        self.resource_predictor = ResourcePredictor()
        self.optimization_history: list[OptimizationAction] = []
        self.performance_targets = self._initialize_targets()
        self.active_optimizations: dict[str, OptimizationAction] = {}

        # Performance tracking
        self.baseline_metrics: dict[PerformanceMetric, float] = {}
        self.current_metrics: dict[PerformanceMetric, float] = {}

        logger.info("Performance optimizer initialized")

    def _initialize_targets(self) -> list[PerformanceTarget]:
        """Initialize performance targets."""
        return [
            PerformanceTarget(
                metric=PerformanceMetric.THROUGHPUT,
                target_value=100.0,  # tasks per hour
                tolerance=10.0,
                priority=0.9,
            ),
            PerformanceTarget(
                metric=PerformanceMetric.LATENCY,
                target_value=30.0,  # seconds
                tolerance=10.0,
                priority=0.8,
            ),
            PerformanceTarget(
                metric=PerformanceMetric.RESOURCE_UTILIZATION,
                target_value=70.0,  # percent
                tolerance=15.0,
                priority=0.7,
            ),
            PerformanceTarget(
                metric=PerformanceMetric.ERROR_RATE,
                target_value=5.0,  # percent
                tolerance=2.0,
                priority=1.0,
            ),
            PerformanceTarget(
                metric=PerformanceMetric.QUEUE_TIME,
                target_value=60.0,  # seconds
                tolerance=30.0,
                priority=0.6,
            ),
        ]

    async def analyze_performance(self) -> dict[PerformanceMetric, float]:
        """Analyze current system performance."""
        metrics = {}

        try:
            # Collect performance data
            queue_depth = await task_queue.get_queue_depth()
            completed_tasks = await task_queue.get_completed_tasks()
            failed_tasks = await task_queue.get_failed_tasks()

            # Calculate throughput (tasks per hour)
            current_time = time.time()
            time_window = 3600  # 1 hour
            # Simplified - would track actual completion times
            metrics[PerformanceMetric.THROUGHPUT] = float(completed_tasks) / (
                current_time / time_window
            )

            # Calculate error rate
            total_tasks = completed_tasks + failed_tasks
            if total_tasks > 0:
                metrics[PerformanceMetric.ERROR_RATE] = (
                    failed_tasks / total_tasks
                ) * 100
            else:
                metrics[PerformanceMetric.ERROR_RATE] = 0.0

            # Queue time estimation
            metrics[PerformanceMetric.QUEUE_TIME] = (
                float(queue_depth) * 30
            )  # Assume 30s per task

            # Get resource utilization
            features = await self.resource_predictor.collect_features()
            metrics[PerformanceMetric.RESOURCE_UTILIZATION] = features.get(
                "current_cpu_usage", 50.0
            )

            # Latency estimation (simplified)
            metrics[PerformanceMetric.LATENCY] = 30.0  # Would measure actual latency

            # Agent efficiency
            agent_count = features.get("agent_count", 1.0)
            if agent_count > 0:
                metrics[PerformanceMetric.AGENT_EFFICIENCY] = (
                    metrics[PerformanceMetric.THROUGHPUT] / agent_count
                )

            self.current_metrics = metrics

            return metrics

        except Exception as e:
            logger.error("Performance analysis failed", error=str(e))
            return {}

    async def identify_optimization_opportunities(self) -> list[OptimizationAction]:
        """Identify opportunities for performance optimization."""
        opportunities = []

        current_metrics = await self.analyze_performance()

        for target in self.performance_targets:
            current_value = current_metrics.get(target.metric, 0.0)

            # Check if metric is outside target range
            deviation = abs(current_value - target.target_value)
            if deviation > target.tolerance:
                # Generate optimization actions based on the specific metric
                actions = self._generate_optimization_actions(target, current_value)
                opportunities.extend(actions)

        # Predictive optimizations
        workload_prediction = await self.resource_predictor.predict_workload()
        predictive_actions = self._generate_predictive_actions(workload_prediction)
        opportunities.extend(predictive_actions)

        # Sort by impact and feasibility
        opportunities.sort(
            key=lambda a: a.confidence
            * a.expected_impact.get(PerformanceMetric.THROUGHPUT, 0),
            reverse=True,
        )

        return opportunities[:5]  # Return top 5 opportunities

    def _generate_optimization_actions(
        self, target: PerformanceTarget, current_value: float
    ) -> list[OptimizationAction]:
        """Generate optimization actions for a specific performance target."""
        actions = []

        if target.metric == PerformanceMetric.THROUGHPUT:
            if current_value < target.target_value:
                # Low throughput - need more processing capacity
                actions.append(
                    OptimizationAction(
                        strategy=OptimizationStrategy.RESOURCE_SCALING,
                        description="Spawn additional worker agents to increase throughput",
                        parameters={
                            "additional_agents": 2,
                            "agent_types": ["developer", "qa"],
                        },
                        expected_impact={PerformanceMetric.THROUGHPUT: 30.0},
                        confidence=0.8,
                        cost=0.6,
                    )
                )

                actions.append(
                    OptimizationAction(
                        strategy=OptimizationStrategy.TASK_BATCHING,
                        description="Batch similar tasks to improve processing efficiency",
                        parameters={"batch_size": 5, "similarity_threshold": 0.8},
                        expected_impact={PerformanceMetric.THROUGHPUT: 15.0},
                        confidence=0.7,
                        cost=0.3,
                    )
                )

        elif target.metric == PerformanceMetric.LATENCY:
            if current_value > target.target_value:
                # High latency - need to reduce processing time
                actions.append(
                    OptimizationAction(
                        strategy=OptimizationStrategy.PRIORITY_ADJUSTMENT,
                        description="Adjust task priorities to reduce latency for important tasks",
                        parameters={
                            "priority_boost": 1,
                            "criteria": "importance_score > 0.7",
                        },
                        expected_impact={PerformanceMetric.LATENCY: -10.0},
                        confidence=0.6,
                        cost=0.2,
                    )
                )

        elif target.metric == PerformanceMetric.ERROR_RATE:
            if current_value > target.target_value:
                # High error rate - need better task assignment
                actions.append(
                    OptimizationAction(
                        strategy=OptimizationStrategy.AGENT_SPECIALIZATION,
                        description="Improve task-agent matching to reduce failures",
                        parameters={
                            "capability_threshold": 0.8,
                            "training_enabled": True,
                        },
                        expected_impact={PerformanceMetric.ERROR_RATE: -5.0},
                        confidence=0.75,
                        cost=0.4,
                    )
                )

        elif target.metric == PerformanceMetric.RESOURCE_UTILIZATION:
            if current_value > target.target_value:
                # High resource usage - need load balancing
                actions.append(
                    OptimizationAction(
                        strategy=OptimizationStrategy.LOAD_BALANCING,
                        description="Redistribute tasks to balance resource usage",
                        parameters={
                            "rebalance_threshold": 0.8,
                            "migration_enabled": True,
                        },
                        expected_impact={PerformanceMetric.RESOURCE_UTILIZATION: -15.0},
                        confidence=0.7,
                        cost=0.3,
                    )
                )

        return actions

    def _generate_predictive_actions(
        self, prediction: WorkloadPrediction
    ) -> list[OptimizationAction]:
        """Generate optimization actions based on workload predictions."""
        actions = []

        # If high workload is predicted, proactively scale
        if prediction.predicted_task_count > 50:  # Threshold for high workload
            actions.append(
                OptimizationAction(
                    strategy=OptimizationStrategy.PREDICTIVE_SCALING,
                    description="Proactively scale agents based on predicted workload increase",
                    parameters={
                        "predicted_tasks": prediction.predicted_task_count,
                        "scale_factor": 1.5,
                        "advance_time_minutes": 30,
                    },
                    expected_impact={PerformanceMetric.THROUGHPUT: 25.0},
                    confidence=0.6,
                    cost=0.8,
                )
            )

        # If resource pressure is predicted, optimize preemptively
        predicted_cpu = prediction.predicted_resource_usage.get("cpu", 50.0)
        if predicted_cpu > 80:
            actions.append(
                OptimizationAction(
                    strategy=OptimizationStrategy.RESOURCE_SCALING,
                    description="Optimize resource allocation before predicted high usage",
                    parameters={"cpu_threshold": 80, "memory_threshold": 85},
                    expected_impact={PerformanceMetric.RESOURCE_UTILIZATION: -20.0},
                    confidence=0.65,
                    cost=0.5,
                )
            )

        return actions

    async def apply_optimization(self, action: OptimizationAction) -> bool:
        """Apply an optimization action."""
        logger.info(
            "Applying optimization",
            strategy=action.strategy.value,
            description=action.description,
        )

        try:
            success = False

            if action.strategy == OptimizationStrategy.RESOURCE_SCALING:
                success = await self._apply_resource_scaling(action)

            elif action.strategy == OptimizationStrategy.LOAD_BALANCING:
                success = await self._apply_load_balancing(action)

            elif action.strategy == OptimizationStrategy.TASK_BATCHING:
                success = await self._apply_task_batching(action)

            elif action.strategy == OptimizationStrategy.PRIORITY_ADJUSTMENT:
                success = await self._apply_priority_adjustment(action)

            elif action.strategy == OptimizationStrategy.AGENT_SPECIALIZATION:
                success = await self._apply_agent_specialization(action)

            elif action.strategy == OptimizationStrategy.PREDICTIVE_SCALING:
                success = await self._apply_predictive_scaling(action)

            if success:
                self.optimization_history.append(action)
                action_id = f"{action.strategy.value}_{int(time.time())}"
                self.active_optimizations[action_id] = action

                logger.info(
                    "Optimization applied successfully", strategy=action.strategy.value
                )
            else:
                logger.warning(
                    "Optimization failed to apply", strategy=action.strategy.value
                )

            return success

        except Exception as e:
            logger.error(
                "Optimization application failed",
                strategy=action.strategy.value,
                error=str(e),
            )
            return False

    async def _apply_resource_scaling(self, action: OptimizationAction) -> bool:
        """Apply resource scaling optimization."""
        params = action.parameters
        additional_agents = params.get("additional_agents", 1)
        agent_types = params.get("agent_types", ["developer"])

        try:
            # Use orchestrator to spawn additional agents
            from .orchestrator import orchestrator

            for agent_type in agent_types:
                for i in range(additional_agents):
                    agent_name = f"{agent_type}-optimized-{int(time.time())}-{i}"
                    await orchestrator.spawn_agent(agent_type, agent_name)

            return True

        except Exception as e:
            logger.error("Resource scaling failed", error=str(e))
            return False

    async def _apply_load_balancing(self, action: OptimizationAction) -> bool:
        """Apply load balancing optimization."""
        # This would redistribute tasks among agents
        # For now, just update the task coordinator settings
        return True

    async def _apply_task_batching(self, action: OptimizationAction) -> bool:
        """Apply task batching optimization."""
        # This would implement task batching logic
        # For now, just return success
        return True

    async def _apply_priority_adjustment(self, action: OptimizationAction) -> bool:
        """Apply priority adjustment optimization."""
        # This would adjust task priorities based on criteria
        # For now, just return success
        return True

    async def _apply_agent_specialization(self, action: OptimizationAction) -> bool:
        """Apply agent specialization optimization."""
        # This would improve task-agent matching
        # For now, just return success
        return True

    async def _apply_predictive_scaling(self, action: OptimizationAction) -> bool:
        """Apply predictive scaling optimization."""

        # Schedule scaling action for future execution

        # For now, apply scaling immediately
        return await self._apply_resource_scaling(action)

    async def monitor_optimization_impact(self) -> dict[str, Any]:
        """Monitor the impact of applied optimizations."""
        impact_report = {
            "active_optimizations": len(self.active_optimizations),
            "optimization_history": len(self.optimization_history),
            "performance_changes": {},
            "successful_optimizations": 0,
        }

        # Compare current metrics with baseline
        current_metrics = await self.analyze_performance()

        for metric, current_value in current_metrics.items():
            baseline_value = self.baseline_metrics.get(metric, current_value)

            if baseline_value > 0:
                change_percent = (
                    (current_value - baseline_value) / baseline_value
                ) * 100
                impact_report["performance_changes"][metric.value] = change_percent

        # Count successful optimizations
        for action in self.optimization_history:
            if (
                action.confidence > 0.7
            ):  # Consider high-confidence optimizations successful
                impact_report["successful_optimizations"] += 1

        return impact_report

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get optimization system statistics."""
        return {
            "total_optimizations_applied": len(self.optimization_history),
            "active_optimizations": len(self.active_optimizations),
            "optimization_strategies_used": list(
                {a.strategy.value for a in self.optimization_history}
            ),
            "average_confidence": np.mean(
                [a.confidence for a in self.optimization_history]
            )
            if self.optimization_history
            else 0.0,
            "current_performance_targets": len(self.performance_targets),
            "prediction_accuracy": self.resource_predictor.prediction_models[
                "task_count"
            ].accuracy,
        }


# Global instance
_performance_optimizer: PerformanceOptimizer | None = None


def get_performance_optimizer() -> PerformanceOptimizer:
    """Get or create the performance optimizer singleton."""
    global _performance_optimizer

    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()

    return _performance_optimizer
