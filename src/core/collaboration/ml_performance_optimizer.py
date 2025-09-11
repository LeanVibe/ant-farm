"""
ML-Based Performance Optimization System.

This module uses machine learning techniques to analyze agent performance patterns,
predict optimal task assignments, and continuously improve system efficiency through
adaptive learning and predictive optimization.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from ..ai.model_orchestrator import model_orchestrator
from ..message_broker import MessageType, message_broker

logger = structlog.get_logger()


class OptimizationGoal(Enum):
    """Types of performance optimization goals."""

    THROUGHPUT_MAXIMIZATION = "throughput_maximization"
    LATENCY_MINIMIZATION = "latency_minimization"
    RESOURCE_EFFICIENCY = "resource_efficiency"
    QUALITY_MAXIMIZATION = "quality_maximization"
    COST_OPTIMIZATION = "cost_optimization"
    ADAPTIVE_SCALING = "adaptive_scaling"


class PerformanceMetric(Enum):
    """Performance metrics that can be optimized."""

    TASK_COMPLETION_TIME = "task_completion_time"
    RESOURCE_UTILIZATION = "resource_utilization"
    ERROR_RATE = "error_rate"
    QUALITY_SCORE = "quality_score"
    THROUGHPUT = "throughput"
    COST_EFFICIENCY = "cost_efficiency"
    SCALABILITY_INDEX = "scalability_index"


class LearningModel(Enum):
    """Types of ML models used for optimization."""

    LINEAR_REGRESSION = "linear_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    CLUSTERING = "clustering"


@dataclass
class PerformanceDataPoint:
    """A single performance measurement data point."""

    timestamp: float
    agent_id: str
    task_type: str
    metrics: Dict[str, float]
    context: Dict[str, Any]
    outcome: str  # "success", "failure", "partial"
    performance_score: float


@dataclass
class OptimizationModel:
    """An ML model for performance optimization."""

    id: str
    model_type: LearningModel
    target_metric: PerformanceMetric
    optimization_goal: OptimizationGoal
    features: List[str]
    training_data: List[PerformanceDataPoint] = field(default_factory=list)
    model_parameters: Dict[str, Any] = field(default_factory=dict)
    performance_history: List[Dict[str, Any]] = field(default_factory=dict)
    accuracy_score: float = 0.0
    last_trained: float = 0.0
    is_active: bool = True


@dataclass
class OptimizationRecommendation:
    """A recommendation from the ML optimization system."""

    id: str
    model_id: str
    recommendation_type: str
    target_entity: str  # agent_id, task_type, or system
    recommendation: Dict[str, Any]
    confidence_score: float
    expected_impact: Dict[str, float]
    implementation_complexity: str  # "low", "medium", "high"
    generated_at: float = field(default_factory=time.time)
    implemented: bool = False
    implementation_result: Optional[Dict[str, Any]] = None


class MLPerformanceOptimizer:
    """ML-based performance optimization system."""

    def __init__(self):
        self.models: Dict[str, OptimizationModel] = {}
        self.performance_data: List[PerformanceDataPoint] = []
        self.recommendations: Dict[str, OptimizationRecommendation] = {}
        self.optimization_goals: Dict[str, OptimizationGoal] = {}

        # Learning configuration
        self.data_retention_days = 30
        self.min_training_samples = 100
        self.retraining_interval_hours = 24
        self.confidence_threshold = 0.7

        # Performance tracking
        self.metrics = {
            "models_trained": 0,
            "recommendations_generated": 0,
            "recommendations_implemented": 0,
            "performance_improvements": 0.0,
            "prediction_accuracy": 0.0,
        }

    async def initialize(self):
        """Initialize the ML performance optimization system."""
        await self._load_existing_models()
        await self._initialize_default_models()
        logger.info("ML performance optimization system initialized")

    async def _load_existing_models(self):
        """Load existing ML models from storage."""
        # This would load from persistent storage
        pass

    async def _initialize_default_models(self):
        """Initialize default optimization models."""

        # Task assignment optimization model
        await self.create_optimization_model(
            model_type=LearningModel.RANDOM_FOREST,
            target_metric=PerformanceMetric.TASK_COMPLETION_TIME,
            optimization_goal=OptimizationGoal.LATENCY_MINIMIZATION,
            features=[
                "agent_experience",
                "task_complexity",
                "agent_workload",
                "task_type_similarity",
                "time_of_day",
                "agent_specialization_match"
            ],
            model_parameters={
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5
            }
        )

        # Resource utilization optimization model
        await self.create_optimization_model(
            model_type=LearningModel.GRADIENT_BOOSTING,
            target_metric=PerformanceMetric.RESOURCE_UTILIZATION,
            optimization_goal=OptimizationGoal.RESOURCE_EFFICIENCY,
            features=[
                "cpu_usage",
                "memory_usage",
                "task_queue_length",
                "agent_count",
                "task_complexity_distribution",
                "time_patterns"
            ],
            model_parameters={
                "n_estimators": 200,
                "learning_rate": 0.1,
                "max_depth": 6
            }
        )

        # Quality optimization model
        await self.create_optimization_model(
            model_type=LearningModel.NEURAL_NETWORK,
            target_metric=PerformanceMetric.QUALITY_SCORE,
            optimization_goal=OptimizationGoal.QUALITY_MAXIMIZATION,
            features=[
                "agent_experience",
                "task_complexity",
                "review_feedback",
                "error_patterns",
                "code_complexity_metrics",
                "testing_coverage"
            ],
            model_parameters={
                "hidden_layers": [64, 32],
                "activation": "relu",
                "learning_rate": 0.001,
                "epochs": 100
            }
        )

    async def create_optimization_model(
        self,
        model_type: LearningModel,
        target_metric: PerformanceMetric,
        optimization_goal: OptimizationGoal,
        features: List[str],
        model_parameters: Dict[str, Any] = None,
    ) -> str:
        """Create a new optimization model."""

        model_id = f"model_{uuid.uuid4().hex[:12]}"

        model = OptimizationModel(
            id=model_id,
            model_type=model_type,
            target_metric=target_metric,
            optimization_goal=optimization_goal,
            features=features,
            model_parameters=model_parameters or {},
        )

        self.models[model_id] = model

        logger.info(
            "Optimization model created",
            model_id=model_id,
            type=model_type.value,
            target=target_metric.value,
            goal=optimization_goal.value,
        )

        return model_id

    async def record_performance_data(
        self,
        agent_id: str,
        task_type: str,
        metrics: Dict[str, float],
        context: Dict[str, Any],
        outcome: str,
        performance_score: float,
    ):
        """Record a performance data point."""

        data_point = PerformanceDataPoint(
            timestamp=time.time(),
            agent_id=agent_id,
            task_type=task_type,
            metrics=metrics,
            context=context,
            outcome=outcome,
            performance_score=performance_score,
        )

        self.performance_data.append(data_point)

        # Maintain data retention limits
        await self._maintain_data_retention()

        # Trigger model retraining if needed
        await self._check_retraining_needed()

        logger.debug(
            "Performance data recorded",
            agent_id=agent_id,
            task_type=task_type,
            outcome=outcome,
            score=performance_score,
        )

    async def _maintain_data_retention(self):
        """Maintain data retention limits."""
        cutoff_time = time.time() - (self.data_retention_days * 24 * 3600)

        # Remove old data points
        self.performance_data = [
            dp for dp in self.performance_data
            if dp.timestamp > cutoff_time
        ]

        # Limit total data points
        if len(self.performance_data) > 10000:
            self.performance_data = self.performance_data[-5000:]

    async def _check_retraining_needed(self):
        """Check if any models need retraining."""
        current_time = time.time()

        for model in self.models.values():
            if not model.is_active:
                continue

            time_since_training = current_time - model.last_trained
            has_new_data = len(model.training_data) < len([
                dp for dp in self.performance_data
                if dp.timestamp > model.last_trained
            ])

            if (time_since_training > (self.retraining_interval_hours * 3600)
                and has_new_data and len(self.performance_data) >= self.min_training_samples):
                await self._retrain_model(model.id)

    async def _retrain_model(self, model_id: str):
        """Retrain a specific model with new data."""
        if model_id not in self.models:
            return

        model = self.models[model_id]

        # Prepare training data
        training_data = [
            dp for dp in self.performance_data
            if dp.timestamp > model.last_trained
        ]

        if len(training_data) < self.min_training_samples:
            return

        # Update model training data
        model.training_data.extend(training_data)
        model.last_trained = time.time()

        # Simulate model training (in practice, this would use actual ML libraries)
        await self._train_model(model)

        # Update model performance
        model.accuracy_score = await self._evaluate_model_performance(model)

        logger.info(
            "Model retrained",
            model_id=model_id,
            training_samples=len(training_data),
            accuracy=model.accuracy_score,
        )

    async def _train_model(self, model: OptimizationModel):
        """Train the ML model (simplified implementation)."""
        # This would use actual ML libraries like scikit-learn, TensorFlow, etc.
        # For now, we'll simulate training

        # Simulate training time
        await asyncio.sleep(0.1)

        # Update model parameters based on training
        if model.model_type == LearningModel.RANDOM_FOREST:
            model.model_parameters["feature_importance"] = {
                feature: 0.1 for feature in model.features
            }
        elif model.model_type == LearningModel.GRADIENT_BOOSTING:
            model.model_parameters["feature_importance"] = {
                feature: 0.15 for feature in model.features
            }
        elif model.model_type == LearningModel.NEURAL_NETWORK:
            model.model_parameters["weights"] = "trained_weights_placeholder"

    async def _evaluate_model_performance(self, model: OptimizationModel) -> float:
        """Evaluate model performance (simplified)."""
        # This would use proper cross-validation
        # For now, return a simulated accuracy score
        base_accuracy = 0.75
        improvement = min(0.15, len(model.training_data) / 1000 * 0.1)
        return base_accuracy + improvement

    async def generate_optimization_recommendations(
        self,
        target_entity: str = None,
        max_recommendations: int = 5,
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations using trained models."""

        recommendations = []

        for model in self.models.values():
            if not model.is_active or model.accuracy_score < self.confidence_threshold:
                continue

            # Generate recommendations based on model type
            model_recommendations = await self._generate_model_recommendations(
                model, target_entity
            )
            recommendations.extend(model_recommendations)

        # Sort by expected impact and confidence
        recommendations.sort(
            key=lambda r: (r.expected_impact.get("improvement", 0) * r.confidence_score),
            reverse=True
        )

        # Limit number of recommendations
        recommendations = recommendations[:max_recommendations]

        # Store recommendations
        for rec in recommendations:
            self.recommendations[rec.id] = rec
            self.metrics["recommendations_generated"] += 1

        return recommendations

    async def _generate_model_recommendations(
        self,
        model: OptimizationModel,
        target_entity: str = None,
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations from a specific model."""

        recommendations = []

        if model.target_metric == PerformanceMetric.TASK_COMPLETION_TIME:
            recommendations.extend(await self._generate_task_assignment_recommendations(model, target_entity))
        elif model.target_metric == PerformanceMetric.RESOURCE_UTILIZATION:
            recommendations.extend(await self._generate_resource_optimization_recommendations(model, target_entity))
        elif model.target_metric == PerformanceMetric.QUALITY_SCORE:
            recommendations.extend(await self._generate_quality_improvement_recommendations(model, target_entity))

        return recommendations

    async def _generate_task_assignment_recommendations(
        self,
        model: OptimizationModel,
        target_entity: str = None,
    ) -> List[OptimizationRecommendation]:
        """Generate task assignment optimization recommendations."""

        recommendations = []

        # Analyze recent performance data
        recent_data = [
            dp for dp in self.performance_data
            if time.time() - dp.timestamp < 3600 * 24  # Last 24 hours
        ]

        if len(recent_data) < 10:
            return recommendations

        # Find inefficient task assignments
        inefficient_assignments = []
        for dp in recent_data:
            if dp.metrics.get("completion_time", 0) > 300:  # Over 5 minutes
                inefficient_assignments.append(dp)

        if inefficient_assignments:
            # Recommend agent specialization improvements
            recommendation = OptimizationRecommendation(
                id=f"rec_{uuid.uuid4().hex[:8]}",
                model_id=model.id,
                recommendation_type="agent_specialization",
                target_entity=target_entity or "task_coordinator",
                recommendation={
                    "action": "improve_agent_specialization",
                    "details": f"Found {len(inefficient_assignments)} slow task completions",
                    "suggested_changes": {
                        "increase_specialized_training": True,
                        "adjust_task_routing": True,
                        "implement_load_balancing": True,
                    }
                },
                confidence_score=0.8,
                expected_impact={
                    "completion_time_reduction": 0.25,
                    "throughput_increase": 0.15,
                },
                implementation_complexity="medium",
            )
            recommendations.append(recommendation)

        return recommendations

    async def _generate_resource_optimization_recommendations(
        self,
        model: OptimizationModel,
        target_entity: str = None,
    ) -> List[OptimizationRecommendation]:
        """Generate resource optimization recommendations."""

        recommendations = []

        # Analyze resource utilization patterns
        recent_data = [
            dp for dp in self.performance_data
            if time.time() - dp.timestamp < 3600 * 24
        ]

        # Calculate average resource utilization
        total_cpu = sum(dp.metrics.get("cpu_usage", 0) for dp in recent_data)
        total_memory = sum(dp.metrics.get("memory_usage", 0) for dp in recent_data)
        avg_cpu = total_cpu / len(recent_data) if recent_data else 0
        avg_memory = total_memory / len(recent_data) if recent_data else 0

        if avg_cpu > 0.8:  # High CPU usage
            recommendation = OptimizationRecommendation(
                id=f"rec_{uuid.uuid4().hex[:8]}",
                model_id=model.id,
                recommendation_type="resource_scaling",
                target_entity=target_entity or "resource_manager",
                recommendation={
                    "action": "scale_resources",
                    "details": f"High CPU utilization detected: {avg_cpu:.2%}",
                    "suggested_changes": {
                        "increase_cpu_allocation": True,
                        "implement_horizontal_scaling": True,
                        "optimize_task_scheduling": True,
                    }
                },
                confidence_score=0.85,
                expected_impact={
                    "cpu_utilization_reduction": 0.2,
                    "performance_improvement": 0.15,
                },
                implementation_complexity="high",
            )
            recommendations.append(recommendation)

        if avg_memory > 0.85:  # High memory usage
            recommendation = OptimizationRecommendation(
                id=f"rec_{uuid.uuid4().hex[:8]}",
                model_id=model.id,
                recommendation_type="memory_optimization",
                target_entity=target_entity or "resource_manager",
                recommendation={
                    "action": "optimize_memory_usage",
                    "details": f"High memory utilization detected: {avg_memory:.2%}",
                    "suggested_changes": {
                        "implement_memory_pooling": True,
                        "add_memory_profiling": True,
                        "optimize_data_structures": True,
                    }
                },
                confidence_score=0.8,
                expected_impact={
                    "memory_usage_reduction": 0.25,
                    "stability_improvement": 0.1,
                },
                implementation_complexity="medium",
            )
            recommendations.append(recommendation)

        return recommendations

    async def _generate_quality_improvement_recommendations(
        self,
        model: OptimizationModel,
        target_entity: str = None,
    ) -> List[OptimizationRecommendation]:
        """Generate quality improvement recommendations."""

        recommendations = []

        # Analyze quality patterns
        recent_data = [
            dp for dp in self.performance_data
            if time.time() - dp.timestamp < 3600 * 24
        ]

        # Calculate quality metrics
        quality_scores = [dp.performance_score for dp in recent_data if dp.outcome == "success"]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0

        if avg_quality < 0.7:  # Low quality threshold
            recommendation = OptimizationRecommendation(
                id=f"rec_{uuid.uuid4().hex[:8]}",
                model_id=model.id,
                recommendation_type="quality_improvement",
                target_entity=target_entity or "qa_system",
                recommendation={
                    "action": "enhance_quality_processes",
                    "details": f"Quality score below threshold: {avg_quality:.2f}",
                    "suggested_changes": {
                        "increase_code_reviews": True,
                        "add_automated_testing": True,
                        "implement_quality_gates": True,
                        "enhance_training_programs": True,
                    }
                },
                confidence_score=0.75,
                expected_impact={
                    "quality_score_improvement": 0.3,
                    "error_rate_reduction": 0.2,
                },
                implementation_complexity="high",
            )
            recommendations.append(recommendation)

        return recommendations

    async def implement_recommendation(
        self,
        recommendation_id: str,
        implementation_details: Dict[str, Any] = None,
    ) -> bool:
        """Implement an optimization recommendation."""

        if recommendation_id not in self.recommendations:
            return False

        recommendation = self.recommendations[recommendation_id]

        # Mark as implemented
        recommendation.implemented = True
        recommendation.implementation_result = implementation_details or {}

        self.metrics["recommendations_implemented"] += 1

        # Notify relevant systems
        await message_broker.send_message(
            from_agent="ml_optimizer",
            to_agent=recommendation.target_entity,
            topic="optimization_recommendation",
            payload={
                "recommendation_id": recommendation_id,
                "recommendation": recommendation.recommendation,
                "expected_impact": recommendation.expected_impact,
                "implementation_details": implementation_details,
            },
            message_type=MessageType.NOTIFICATION,
        )

        logger.info(
            "Recommendation implemented",
            recommendation_id=recommendation_id,
            type=recommendation.recommendation_type,
            target=recommendation.target_entity,
        )

        return True

    async def predict_optimal_assignment(
        self,
        task_type: str,
        available_agents: List[str],
        context: Dict[str, Any] = None,
    ) -> Optional[Tuple[str, float]]:
        """Predict the optimal agent for a task using ML models."""

        if not available_agents:
            return None

        # Use the task assignment optimization model
        assignment_model = None
        for model in self.models.values():
            if (model.target_metric == PerformanceMetric.TASK_COMPLETION_TIME
                and model.is_active):
                assignment_model = model
                break

        if not assignment_model:
            # Fallback to simple selection
            return (available_agents[0], 0.5)

        # Simulate prediction (in practice, this would use the trained model)
        # For now, return the first agent with a confidence score
        predicted_agent = available_agents[0]
        confidence = assignment_model.accuracy_score

        return (predicted_agent, confidence)

    async def analyze_performance_trends(
        self,
        time_window_hours: float = 24,
    ) -> Dict[str, Any]:
        """Analyze performance trends over time."""

        cutoff_time = time.time() - (time_window_hours * 3600)

        relevant_data = [
            dp for dp in self.performance_data
            if dp.timestamp > cutoff_time
        ]

        if not relevant_data:
            return {"error": "No data available for analysis"}

        # Calculate trends
        trends = {
            "total_measurements": len(relevant_data),
            "time_window_hours": time_window_hours,
            "performance_distribution": {},
            "agent_performance": {},
            "task_type_performance": {},
        }

        # Performance distribution
        outcomes = {}
        for dp in relevant_data:
            outcomes[dp.outcome] = outcomes.get(dp.outcome, 0) + 1

        trends["performance_distribution"] = {
            outcome: count / len(relevant_data)
            for outcome, count in outcomes.items()
        }

        # Agent performance
        agent_stats = {}
        for dp in relevant_data:
            if dp.agent_id not in agent_stats:
                agent_stats[dp.agent_id] = {
                    "tasks_completed": 0,
                    "avg_performance": 0.0,
                    "success_rate": 0.0,
                }

            agent_stats[dp.agent_id]["tasks_completed"] += 1
            agent_stats[dp.agent_id]["avg_performance"] += dp.performance_score

        for agent_id, stats in agent_stats.items():
            stats["avg_performance"] /= stats["tasks_completed"]
            successful_tasks = sum(
                1 for dp in relevant_data
                if dp.agent_id == agent_id and dp.outcome == "success"
            )
            stats["success_rate"] = successful_tasks / stats["tasks_completed"]

        trends["agent_performance"] = agent_stats

        # Task type performance
        task_stats = {}
        for dp in relevant_data:
            if dp.task_type not in task_stats:
                task_stats[dp.task_type] = {
                    "count": 0,
                    "avg_completion_time": 0.0,
                    "success_rate": 0.0,
                }

            task_stats[dp.task_type]["count"] += 1
            task_stats[dp.task_type]["avg_completion_time"] += dp.metrics.get("completion_time", 0)

        for task_type, stats in task_stats.items():
            stats["avg_completion_time"] /= stats["count"]
            successful_tasks = sum(
                1 for dp in relevant_data
                if dp.task_type == task_type and dp.outcome == "success"
            )
            stats["success_rate"] = successful_tasks / stats["count"]

        trends["task_type_performance"] = task_stats

        return trends

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide ML optimization metrics."""

        total_models = len(self.models)
        active_models = sum(1 for model in self.models.values() if model.is_active)
        total_data_points = len(self.performance_data)

        # Calculate average model accuracy
        accuracies = [model.accuracy_score for model in self.models.values() if model.is_active]
        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0

        # Calculate recommendation success rate
        total_recommendations = len(self.recommendations)
        implemented_recommendations = sum(
            1 for rec in self.recommendations.values() if rec.implemented
        )
        success_rate = implemented_recommendations / total_recommendations if total_recommendations > 0 else 0.0

        return {
            **self.metrics,
            "total_models": total_models,
            "active_models": active_models,
            "total_data_points": total_data_points,
            "average_model_accuracy": avg_accuracy,
            "recommendation_success_rate": success_rate,
            "data_retention_days": self.data_retention_days,
            "min_training_samples": self.min_training_samples,
        }

    async def export_model_data(
        self,
        model_id: str = None,
        export_path: str = None,
    ) -> str:
        """Export model data for analysis or backup."""

        if model_id and model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")

        # Prepare export data
        export_data = {
            "export_timestamp": time.time(),
            "system_metrics": await self.get_system_metrics(),
        }

        if model_id:
            # Export specific model
            model = self.models[model_id]
            export_data["models"] = {
                model_id: {
                    "id": model.id,
                    "type": model.model_type.value,
                    "target_metric": model.target_metric.value,
                    "optimization_goal": model.optimization_goal.value,
                    "features": model.features,
                    "parameters": model.model_parameters,
                    "accuracy": model.accuracy_score,
                    "training_samples": len(model.training_data),
                    "last_trained": model.last_trained,
                    "is_active": model.is_active,
                }
            }
        else:
            # Export all models
            export_data["models"] = {}
            for mid, model in self.models.items():
                export_data["models"][mid] = {
                    "id": model.id,
                    "type": model.model_type.value,
                    "target_metric": model.target_metric.value,
                    "optimization_goal": model.optimization_goal.value,
                    "features": model.features,
                    "parameters": model.model_parameters,
                    "accuracy": model.accuracy_score,
                    "training_samples": len(model.training_data),
                    "last_trained": model.last_trained,
                    "is_active": model.is_active,
                }

        # Export recent performance data (last 7 days)
        cutoff_time = time.time() - (7 * 24 * 3600)
        recent_data = [
            dp for dp in self.performance_data
            if dp.timestamp > cutoff_time
        ]

        export_data["performance_data"] = [
            {
                "timestamp": dp.timestamp,
                "agent_id": dp.agent_id,
                "task_type": dp.task_type,
                "metrics": dp.metrics,
                "context": dp.context,
                "outcome": dp.outcome,
                "performance_score": dp.performance_score,
            }
            for dp in recent_data
        ]

        # Export to file or return as JSON
        if export_path:
            import json
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(
                "ML optimization data exported",
                export_path=export_path,
                models_exported=len(export_data["models"]),
                data_points=len(export_data["performance_data"]),
            )

            return export_path
        else:
            return json.dumps(export_data, indent=2, default=str)


# Global instance
ml_performance_optimizer = MLPerformanceOptimizer()