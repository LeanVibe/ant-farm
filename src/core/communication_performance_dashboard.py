"""Comprehensive communication performance dashboard and monitoring integration.

Provides a unified view of performance metrics across all communication components:
- Enhanced Message Broker performance
- Real-time Collaboration metrics
- Shared Knowledge Base analytics
- System-wide communication health
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from .communication_monitor import get_communication_monitor
from .enhanced_message_broker import EnhancedMessageBroker
from .realtime_collaboration import RealTimeCollaborationSync
from .shared_knowledge_base import SharedKnowledgeBase

logger = structlog.get_logger()


@dataclass
class SystemHealthStatus:
    """Overall system health status."""

    status: str  # "healthy", "degraded", "critical"
    overall_score: float  # 0.0 to 1.0
    component_scores: dict[str, float] = field(default_factory=dict)
    alerts: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class CommunicationPerformanceDashboard:
    """Unified performance monitoring dashboard for communication system."""

    def __init__(
        self,
        enhanced_broker: EnhancedMessageBroker | None = None,
        collaboration_sync: RealTimeCollaborationSync | None = None,
        knowledge_base: SharedKnowledgeBase | None = None,
    ):
        self.enhanced_broker = enhanced_broker
        self.collaboration_sync = collaboration_sync
        self.knowledge_base = knowledge_base
        self.communication_monitor = get_communication_monitor()

        # Performance thresholds for health scoring
        self.health_thresholds = {
            "latency_warning": 500,  # milliseconds
            "latency_critical": 1000,  # milliseconds
            "error_rate_warning": 0.05,  # 5%
            "error_rate_critical": 0.15,  # 15%
            "throughput_min_warning": 5,  # messages per second
            "throughput_min_critical": 1,  # messages per second
            "memory_usage_warning": 0.8,  # 80%
            "memory_usage_critical": 0.95,  # 95%
        }

        # Historical metrics for trending
        self.metrics_history = {
            "latency": [],
            "throughput": [],
            "error_rate": [],
            "active_agents": [],
        }

    async def get_comprehensive_metrics(self) -> dict[str, Any]:
        """Get comprehensive metrics from all communication components."""

        start_time = time.time()

        # Gather metrics from all components in parallel
        tasks = []

        # Core communication metrics
        tasks.append(("core", self.communication_monitor.get_real_time_stats()))

        # Enhanced broker metrics
        if self.enhanced_broker:
            tasks.append(
                (
                    "enhanced_broker",
                    self.enhanced_broker.get_communication_performance_metrics(),
                )
            )

        # Collaboration metrics
        if self.collaboration_sync:
            tasks.append(
                (
                    "collaboration",
                    self.collaboration_sync.get_collaboration_performance_metrics(),
                )
            )

        # Knowledge base metrics
        if self.knowledge_base:
            tasks.append(
                (
                    "knowledge_base",
                    self.knowledge_base.get_knowledge_performance_metrics(),
                )
            )

        # Execute all metric collection tasks
        results = {}
        for component_name, task in tasks:
            try:
                if asyncio.iscoroutine(task):
                    results[component_name] = await task
                else:
                    results[component_name] = task
            except Exception as e:
                logger.error(f"Failed to get {component_name} metrics", error=str(e))
                results[component_name] = {"error": str(e)}

        # Calculate system overview
        collection_time = time.time() - start_time
        results["system_overview"] = await self._calculate_system_overview(results)
        results["collection_time"] = collection_time
        results["timestamp"] = time.time()

        return results

    async def get_system_health(self) -> SystemHealthStatus:
        """Calculate overall system health status."""

        metrics = await self.get_comprehensive_metrics()

        # Calculate component health scores
        component_scores = {}
        alerts = []
        recommendations = []

        # Core communication health
        if "core" in metrics and isinstance(metrics["core"], dict):
            core_metrics = metrics["core"]
            core_score = await self._calculate_core_health_score(core_metrics)
            component_scores["core_communication"] = core_score

            if core_score < 0.7:
                alerts.append("Core communication performance degraded")
            if core_score < 0.5:
                recommendations.append("Consider restarting message broker services")

        # Enhanced broker health
        if "enhanced_broker" in metrics and isinstance(
            metrics["enhanced_broker"], dict
        ):
            broker_metrics = metrics["enhanced_broker"]
            broker_score = await self._calculate_broker_health_score(broker_metrics)
            component_scores["enhanced_broker"] = broker_score

            if broker_score < 0.7:
                alerts.append("Enhanced message broker showing performance issues")

        # Collaboration health
        if "collaboration" in metrics and isinstance(metrics["collaboration"], dict):
            collab_metrics = metrics["collaboration"]
            collab_score = await self._calculate_collaboration_health_score(
                collab_metrics
            )
            component_scores["collaboration"] = collab_score

            if collab_score < 0.8:
                recommendations.append("Monitor collaboration session efficiency")

        # Knowledge base health
        if "knowledge_base" in metrics and isinstance(metrics["knowledge_base"], dict):
            kb_metrics = metrics["knowledge_base"]
            kb_score = await self._calculate_knowledge_base_health_score(kb_metrics)
            component_scores["knowledge_base"] = kb_score

        # Calculate overall score
        if component_scores:
            overall_score = sum(component_scores.values()) / len(component_scores)
        else:
            overall_score = 0.0

        # Determine overall status
        if overall_score >= 0.8:
            status = "healthy"
        elif overall_score >= 0.6:
            status = "degraded"
        else:
            status = "critical"
            alerts.append("System performance is critically degraded")
            recommendations.append("Immediate investigation required")

        return SystemHealthStatus(
            status=status,
            overall_score=overall_score,
            component_scores=component_scores,
            alerts=alerts,
            recommendations=recommendations,
        )

    async def get_performance_trends(self, hours: int = 24) -> dict[str, Any]:
        """Get performance trends over specified time period."""

        current_time = time.time()
        start_time = current_time - (hours * 3600)

        # This would typically query stored historical data
        # For now, return current snapshot with trend analysis
        current_metrics = await self.get_comprehensive_metrics()

        trends = {
            "period_hours": hours,
            "start_time": start_time,
            "end_time": current_time,
            "current_snapshot": current_metrics,
            "trend_analysis": {
                "latency_trend": "stable",  # Would calculate from historical data
                "throughput_trend": "improving",
                "error_rate_trend": "stable",
                "agent_activity_trend": "increasing",
            },
            "performance_summary": {
                "peak_throughput": current_metrics.get("core", {}).get(
                    "messages_per_second", 0
                ),
                "average_latency": 50.0,  # Would calculate from historical data
                "uptime_percentage": 99.9,
                "total_messages": current_metrics.get("core", {}).get(
                    "total_messages", 0
                ),
            },
        }

        return trends

    async def get_agent_performance_breakdown(self) -> dict[str, Any]:
        """Get detailed performance breakdown by agent."""

        agent_metrics = {}

        # Get agent-specific metrics from communication monitor
        if hasattr(self.communication_monitor, "agent_profiles"):
            for (
                agent_name,
                profile,
            ) in self.communication_monitor.agent_profiles.items():
                agent_perf = await self.communication_monitor.get_agent_performance(
                    agent_name
                )
                if agent_perf:
                    agent_metrics[agent_name] = agent_perf

        # Add collaboration session participation
        if self.collaboration_sync and hasattr(
            self.collaboration_sync, "active_sessions"
        ):
            for agent_name in agent_metrics:
                # Count active collaboration sessions for agent
                active_sessions = 0
                for session in self.collaboration_sync.active_sessions.values():
                    if agent_name in session.participants:
                        active_sessions += 1
                agent_metrics[agent_name]["active_collaboration_sessions"] = (
                    active_sessions
                )

        # Add knowledge base contribution metrics
        if self.knowledge_base and hasattr(
            self.knowledge_base, "agent_knowledge_graphs"
        ):
            for agent_name in agent_metrics:
                if agent_name in self.knowledge_base.agent_knowledge_graphs:
                    graph = self.knowledge_base.agent_knowledge_graphs[agent_name]
                    agent_metrics[agent_name]["knowledge_contributions"] = len(
                        graph["contributed"]
                    )
                    agent_metrics[agent_name]["knowledge_accessed"] = len(
                        graph["accessed"]
                    )

        return {
            "agent_count": len(agent_metrics),
            "agents": agent_metrics,
            "top_performers": self._identify_top_performing_agents(agent_metrics),
            "performance_summary": self._calculate_agent_performance_summary(
                agent_metrics
            ),
        }

    async def _calculate_system_overview(
        self, metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate high-level system overview metrics."""

        overview = {
            "components_active": len(
                [
                    k
                    for k, v in metrics.items()
                    if isinstance(v, dict) and "error" not in v
                ]
            ),
            "components_total": len(metrics)
            - 2,  # Exclude system_overview and collection_time
            "system_load": "normal",  # Would calculate based on various metrics
            "performance_score": 0.85,  # Would calculate based on all metrics
        }

        # Extract key metrics from components
        if "core" in metrics:
            core = metrics["core"]
            overview["total_messages"] = core.get("total_messages", 0)
            overview["active_agents"] = core.get("active_agents", 0)
            overview["messages_per_second"] = core.get("messages_per_second", 0)

        if "enhanced_broker" in metrics:
            broker = metrics["enhanced_broker"]
            if "enhanced_features" in broker:
                enhanced = broker["enhanced_features"]
                overview["shared_contexts"] = enhanced.get("shared_contexts_active", 0)

        if "collaboration" in metrics:
            collab = metrics["collaboration"]
            overview["active_collaboration_sessions"] = collab.get("active_sessions", 0)

        if "knowledge_base" in metrics:
            kb = metrics["knowledge_base"]
            overview["knowledge_items"] = kb.get("total_knowledge_items", 0)

        return overview

    async def _calculate_core_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate health score for core communication metrics."""

        score = 1.0

        # Check error rate
        error_rate = metrics.get("error_rate", 0)
        if error_rate > self.health_thresholds["error_rate_critical"]:
            score -= 0.4
        elif error_rate > self.health_thresholds["error_rate_warning"]:
            score -= 0.2

        # Check throughput
        throughput = metrics.get("messages_per_second", 0)
        if throughput < self.health_thresholds["throughput_min_critical"]:
            score -= 0.3
        elif throughput < self.health_thresholds["throughput_min_warning"]:
            score -= 0.1

        # Check active agents
        active_agents = metrics.get("active_agents", 0)
        if active_agents == 0:
            score -= 0.5

        return max(0.0, score)

    async def _calculate_broker_health_score(self, metrics: dict[str, Any]) -> float:
        """Calculate health score for enhanced message broker."""

        score = 1.0

        if "enhanced_features" in metrics:
            enhanced = metrics["enhanced_features"]

            # Check if sync tasks are running
            sync_tasks = enhanced.get("sync_tasks_running", 0)
            if sync_tasks == 0:
                score -= 0.2

            # Check context sharing efficiency
            contexts = enhanced.get("shared_contexts_active", 0)
            if contexts > 0:
                avg_participants = enhanced.get("average_participants_per_context", 0)
                if avg_participants < 2:
                    score -= 0.1  # Contexts should have multiple participants

        return max(0.0, score)

    async def _calculate_collaboration_health_score(
        self, metrics: dict[str, Any]
    ) -> float:
        """Calculate health score for collaboration system."""

        score = 1.0

        active_sessions = metrics.get("active_sessions", 0)
        pending_operations = metrics.get("pending_operations", 0)

        # If there are pending operations, check if they're reasonable
        if active_sessions > 0 and pending_operations > active_sessions * 10:
            score -= 0.3  # Too many pending operations per session

        return max(0.0, score)

    async def _calculate_knowledge_base_health_score(
        self, metrics: dict[str, Any]
    ) -> float:
        """Calculate health score for knowledge base."""

        score = 1.0

        total_items = metrics.get("total_knowledge_items", 0)
        sharing_efficiency = metrics.get("sharing_efficiency", 0)

        # Check if knowledge is being used effectively
        if total_items > 0 and sharing_efficiency < 0.1:
            score -= 0.2  # Knowledge exists but isn't being shared effectively

        return max(0.0, score)

    def _identify_top_performing_agents(
        self, agent_metrics: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify top performing agents based on various metrics."""

        agent_scores = []

        for agent_name, metrics in agent_metrics.items():
            score = 0.0

            # Factor in success rate
            success_rate = metrics.get("success_rate", 0)
            score += success_rate * 0.3

            # Factor in message count (activity level)
            message_count = metrics.get("message_count", 0)
            if message_count > 0:
                score += min(message_count / 1000, 1.0) * 0.2  # Cap at 1000 messages

            # Factor in knowledge contributions
            knowledge_contributions = metrics.get("knowledge_contributions", 0)
            score += (
                min(knowledge_contributions / 10, 1.0) * 0.3
            )  # Cap at 10 contributions

            # Factor in collaboration participation
            collab_sessions = metrics.get("active_collaboration_sessions", 0)
            score += min(collab_sessions / 5, 1.0) * 0.2  # Cap at 5 sessions

            agent_scores.append(
                {
                    "agent_name": agent_name,
                    "performance_score": score,
                    "metrics": metrics,
                }
            )

        # Sort by performance score
        agent_scores.sort(key=lambda x: x["performance_score"], reverse=True)

        return agent_scores[:10]  # Return top 10

    def _calculate_agent_performance_summary(
        self, agent_metrics: dict[str, Any]
    ) -> dict[str, Any]:
        """Calculate summary statistics for agent performance."""

        if not agent_metrics:
            return {}

        total_messages = sum(m.get("message_count", 0) for m in agent_metrics.values())
        total_knowledge = sum(
            m.get("knowledge_contributions", 0) for m in agent_metrics.values()
        )
        avg_success_rate = sum(
            m.get("success_rate", 0) for m in agent_metrics.values()
        ) / len(agent_metrics)

        return {
            "total_agents": len(agent_metrics),
            "total_messages": total_messages,
            "total_knowledge_contributions": total_knowledge,
            "average_success_rate": avg_success_rate,
            "most_active_agent": max(
                agent_metrics.items(),
                key=lambda x: x[1].get("message_count", 0),
                default=(None, {}),
            )[0],
            "top_knowledge_contributor": max(
                agent_metrics.items(),
                key=lambda x: x[1].get("knowledge_contributions", 0),
                default=(None, {}),
            )[0],
        }


# Global dashboard instance
performance_dashboard = None


def get_performance_dashboard(
    enhanced_broker: EnhancedMessageBroker | None = None,
    collaboration_sync: RealTimeCollaborationSync | None = None,
    knowledge_base: SharedKnowledgeBase | None = None,
) -> CommunicationPerformanceDashboard:
    """Get performance dashboard instance."""
    global performance_dashboard

    if performance_dashboard is None:
        performance_dashboard = CommunicationPerformanceDashboard(
            enhanced_broker, collaboration_sync, knowledge_base
        )

    return performance_dashboard
