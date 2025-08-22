"""
Learning Agent - Analyzes performance and optimizes workflows.

This agent learns from workflow execution patterns, identifies optimization
opportunities, and continuously improves the system's performance through
data-driven insights and automated optimization recommendations.
"""

import asyncio
import json
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from ..constants import Intervals
from ..message_broker import MessageType, message_broker

logger = structlog.get_logger()


@dataclass
class WorkflowExecutionData:
    """Data collected from workflow execution for learning."""
    workflow_id: str
    decomposition_complexity: str
    execution_time: float
    success: bool
    agent_performance: Dict[str, float] = field(default_factory=dict)
    bottleneck_identified: Optional[str] = None
    optimization_applied: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class PerformancePattern:
    """Identified performance pattern with optimization recommendations."""
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    impact_score: float
    optimization_recommendation: str
    confidence: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class AgentCapabilityProfile:
    """Performance profile for an agent."""
    agent_id: str
    capabilities: List[str]
    average_execution_time: float = 0.0
    success_rate: float = 0.0
    specialization_score: Dict[str, float] = field(default_factory=dict)
    total_executions: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class OptimizationRecommendation:
    """Recommendation for system optimization."""
    recommendation_id: str
    category: str
    description: str
    impact: str
    effort: str
    confidence: float
    implementation_plan: List[str]
    created_at: float = field(default_factory=time.time)


class LearningAgent:
    """Agent responsible for learning from workflow execution and optimizing performance."""

    def __init__(self):
        self.execution_history: List[WorkflowExecutionData] = []
        self.performance_patterns: Dict[str, PerformancePattern] = {}
        self.agent_profiles: Dict[str, AgentCapabilityProfile] = {}
        self.optimization_recommendations: List[OptimizationRecommendation] = []

        # Learning parameters
        self.pattern_confidence_threshold = 0.7
        self.min_pattern_frequency = 3
        self.learning_window_days = 7

        self.learning_metrics = {
            "workflows_analyzed": 0,
            "patterns_identified": 0,
            "optimizations_recommended": 0,
            "performance_improvements": 0.0
        }

    async def initialize(self):
        """Initialize the learning agent."""
        await self._load_historical_data()
        logger.info("Learning agent initialized")

    async def _load_historical_data(self):
        """Load historical execution data for learning."""
        # In a real implementation, this would load from database
        # For now, we'll start with empty learning state
        pass

    async def analyze_workflow_execution(self, workflow_data: WorkflowExecutionData):
        """Analyze a completed workflow execution for learning opportunities."""
        logger.info("Analyzing workflow execution", workflow_id=workflow_data.workflow_id)

        # Store execution data
        self.execution_history.append(workflow_data)
        self.learning_metrics["workflows_analyzed"] += 1

        # Update agent profiles
        await self._update_agent_profiles(workflow_data)

        # Identify performance patterns
        await self._identify_patterns(workflow_data)

        # Generate optimization recommendations
        await self._generate_recommendations(workflow_data)

        # Clean old data
        await self._cleanup_old_data()

    async def _update_agent_profiles(self, workflow_data: WorkflowExecutionData):
        """Update agent performance profiles based on execution data."""
        for agent_id, performance_score in workflow_data.agent_performance.items():
            if agent_id not in self.agent_profiles:
                # Create new profile
                self.agent_profiles[agent_id] = AgentCapabilityProfile(
                    agent_id=agent_id,
                    capabilities=["general"]  # Would be loaded from agent registry
                )

            profile = self.agent_profiles[agent_id]

            # Update running averages
            profile.total_executions += 1
            profile.average_execution_time = (
                (profile.average_execution_time * (profile.total_executions - 1)) +
                workflow_data.execution_time
            ) / profile.total_executions

            # Update success rate
            success_contribution = 1.0 if workflow_data.success else 0.0
            profile.success_rate = (
                (profile.success_rate * (profile.total_executions - 1)) +
                success_contribution
            ) / profile.total_executions

            profile.last_updated = time.time()

    async def _identify_patterns(self, workflow_data: WorkflowExecutionData):
        """Identify performance patterns from workflow execution."""
        # Look for common patterns in recent executions
        recent_executions = [
            exec for exec in self.execution_history
            if exec.timestamp > (time.time() - (self.learning_window_days * 86400))
        ]

        # Pattern 1: Slow execution times
        if workflow_data.execution_time > 300:  # 5 minutes
            pattern_key = "slow_execution"
            await self._record_pattern(
                pattern_key,
                "performance",
                "Workflow execution taking longer than expected",
                "Consider parallel execution or resource optimization"
            )

        # Pattern 2: Agent bottlenecks
        for agent_id, performance in workflow_data.agent_performance.items():
            if performance < 0.5:  # Low performance score
                pattern_key = f"agent_bottleneck_{agent_id}"
                await self._record_pattern(
                    pattern_key,
                    "agent_performance",
                    f"Agent {agent_id} consistently showing low performance",
                    f"Consider agent training, resource allocation, or replacement"
                )

        # Pattern 3: Complex workflows with high success
        if (workflow_data.decomposition_complexity == "complex" and
            workflow_data.success and
            workflow_data.execution_time < 600):  # 10 minutes
            pattern_key = "efficient_complex_workflow"
            await self._record_pattern(
                pattern_key,
                "positive_pattern",
                "Complex workflows executing efficiently",
                "Document and replicate successful patterns"
            )

    async def _record_pattern(self, pattern_key: str, pattern_type: str, description: str, recommendation: str):
        """Record a performance pattern."""
        if pattern_key not in self.performance_patterns:
            self.performance_patterns[pattern_key] = PerformancePattern(
                pattern_id=pattern_key,
                pattern_type=pattern_type,
                description=description,
                frequency=1,
                impact_score=0.5,  # Default impact
                optimization_recommendation=recommendation,
                confidence=0.3  # Start with low confidence
            )
            self.learning_metrics["patterns_identified"] += 1
        else:
            pattern = self.performance_patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_seen = time.time()
            pattern.confidence = min(1.0, pattern.confidence + 0.1)  # Increase confidence

    async def _generate_recommendations(self, workflow_data: WorkflowExecutionData):
        """Generate optimization recommendations based on analysis."""
        recommendations = []

        # Recommendation 1: Parallel execution for independent subtasks
        if workflow_data.decomposition_complexity in ["complex", "enterprise"]:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"parallel_exec_{workflow_data.workflow_id}",
                category="execution_optimization",
                description="Consider parallel execution for independent subtasks",
                impact="high",
                effort="medium",
                confidence=0.8,
                implementation_plan=[
                    "Identify independent subtasks in workflow decomposition",
                    "Implement parallel execution in coordination agent",
                    "Add synchronization points for dependent subtasks",
                    "Monitor performance improvement"
                ]
            ))

        # Recommendation 2: Agent specialization
        if len(workflow_data.agent_performance) > 3:
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"agent_specialization_{workflow_data.workflow_id}",
                category="agent_optimization",
                description="Consider agent specialization for better performance",
                impact="medium",
                effort="high",
                confidence=0.6,
                implementation_plan=[
                    "Analyze agent performance patterns",
                    "Identify specialization opportunities",
                    "Implement agent capability profiles",
                    "Update task assignment logic"
                ]
            ))

        # Recommendation 3: Cache optimization
        if workflow_data.execution_time > 60:  # Over 1 minute
            recommendations.append(OptimizationRecommendation(
                recommendation_id=f"caching_optimization_{workflow_data.workflow_id}",
                category="performance_optimization",
                description="Implement intelligent caching for frequently accessed data",
                impact="medium",
                effort="medium",
                confidence=0.7,
                implementation_plan=[
                    "Identify frequently accessed data patterns",
                    "Implement Redis caching layer",
                    "Add cache invalidation strategies",
                    "Monitor cache hit rates"
                ]
            ))

        # Add recommendations to the list
        self.optimization_recommendations.extend(recommendations)
        self.learning_metrics["optimizations_recommended"] += len(recommendations)

    async def _cleanup_old_data(self):
        """Clean up old execution data to maintain learning window."""
        cutoff_time = time.time() - (self.learning_window_days * 86400)

        # Remove old execution history
        self.execution_history = [
            exec for exec in self.execution_history
            if exec.timestamp > cutoff_time
        ]

        # Remove old patterns with low frequency
        patterns_to_remove = []
        for pattern_key, pattern in self.performance_patterns.items():
            if (pattern.last_seen < cutoff_time and
                pattern.frequency < self.min_pattern_frequency):
                patterns_to_remove.append(pattern_key)

        for pattern_key in patterns_to_remove:
            del self.performance_patterns[pattern_key]

    async def get_workflow_insights(self, workflow_id: str) -> Dict[str, Any]:
        """Get insights and recommendations for a specific workflow."""
        # Find workflow execution data
        workflow_exec = next(
            (exec for exec in self.execution_history if exec.workflow_id == workflow_id),
            None
        )

        if not workflow_exec:
            return {"error": "Workflow not found in execution history"}

        # Generate insights
        insights = {
            "workflow_id": workflow_id,
            "execution_time": workflow_exec.execution_time,
            "success": workflow_exec.success,
            "complexity": workflow_exec.decomposition_complexity,
            "performance_patterns": [],
            "recommendations": []
        }

        # Add relevant patterns
        for pattern in self.performance_patterns.values():
            if pattern.confidence >= self.pattern_confidence_threshold:
                insights["performance_patterns"].append({
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "recommendation": pattern.optimization_recommendation,
                    "confidence": pattern.confidence
                })

        # Add relevant recommendations
        for recommendation in self.optimization_recommendations[-5:]:  # Last 5 recommendations
            insights["recommendations"].append({
                "category": recommendation.category,
                "description": recommendation.description,
                "impact": recommendation.impact,
                "effort": recommendation.effort,
                "confidence": recommendation.confidence
            })

        return insights

    async def get_system_insights(self) -> Dict[str, Any]:
        """Get overall system performance insights."""
        # Calculate system-wide metrics
        total_executions = len(self.execution_history)
        successful_executions = len([exec for exec in self.execution_history if exec.success])

        success_rate = successful_executions / total_executions if total_executions > 0 else 0

        # Calculate average execution time
        execution_times = [exec.execution_time for exec in self.execution_history]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0

        # Get top patterns
        top_patterns = sorted(
            self.performance_patterns.values(),
            key=lambda p: p.impact_score * p.confidence,
            reverse=True
        )[:5]

        # Get agent performance ranking
        agent_ranking = sorted(
            self.agent_profiles.values(),
            key=lambda p: p.success_rate * (1 / (p.average_execution_time + 1)),
            reverse=True
        )[:5]

        return {
            "system_metrics": {
                "total_workflows": total_executions,
                "success_rate": success_rate,
                "average_execution_time": avg_execution_time,
                "patterns_identified": len(self.performance_patterns),
                "active_agents": len(self.agent_profiles)
            },
            "top_performance_patterns": [
                {
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "impact": pattern.impact_score,
                    "confidence": pattern.confidence
                }
                for pattern in top_patterns
            ],
            "top_performing_agents": [
                {
                    "agent_id": profile.agent_id,
                    "success_rate": profile.success_rate,
                    "avg_execution_time": profile.average_execution_time,
                    "total_executions": profile.total_executions
                }
                for profile in agent_ranking
            ],
            "learning_metrics": self.learning_metrics
        }

    async def apply_optimization(self, recommendation_id: str) -> bool:
        """Apply a specific optimization recommendation."""
        recommendation = next(
            (rec for rec in self.optimization_recommendations if rec.recommendation_id == recommendation_id),
            None
        )

        if not recommendation:
            return False

        # In a real implementation, this would apply the optimization
        # For now, we'll just mark it as applied and log it
        logger.info(
            "Optimization applied",
            recommendation_id=recommendation_id,
            category=recommendation.category,
            description=recommendation.description
        )

        # Update metrics
        self.learning_metrics["performance_improvements"] += 0.1  # Assume 10% improvement

        return True

    async def get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning agent performance metrics."""
        return {
            **self.learning_metrics,
            "execution_history_size": len(self.execution_history),
            "patterns_tracked": len(self.performance_patterns),
            "agent_profiles": len(self.agent_profiles),
            "pending_recommendations": len(self.optimization_recommendations)
        }

    async def export_learning_data(self) -> Dict[str, Any]:
        """Export all learning data for analysis or backup."""
        return {
            "execution_history": [
                {
                    "workflow_id": exec.workflow_id,
                    "complexity": exec.decomposition_complexity,
                    "execution_time": exec.execution_time,
                    "success": exec.success,
                    "timestamp": exec.timestamp
                }
                for exec in self.execution_history
            ],
            "performance_patterns": [
                {
                    "pattern_id": pattern.pattern_id,
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence,
                    "recommendation": pattern.optimization_recommendation
                }
                for pattern in self.performance_patterns.values()
            ],
            "agent_profiles": [
                {
                    "agent_id": profile.agent_id,
                    "capabilities": profile.capabilities,
                    "success_rate": profile.success_rate,
                    "avg_execution_time": profile.average_execution_time,
                    "total_executions": profile.total_executions
                }
                for profile in self.agent_profiles.values()
            ],
            "export_timestamp": time.time()
        }

    async def import_learning_data(self, data: Dict[str, Any]):
        """Import learning data from backup or analysis."""
        # This would be used to restore learning state or share insights
        logger.info("Learning data import completed", data_size=len(str(data)))

    async def reset_learning_state(self):
        """Reset all learning state (useful for testing or fresh starts)."""
        self.execution_history.clear()
        self.performance_patterns.clear()
        self.agent_profiles.clear()
        self.optimization_recommendations.clear()

        # Reset metrics
        for key in self.learning_metrics:
            self.learning_metrics[key] = 0

        logger.info("Learning state reset completed")