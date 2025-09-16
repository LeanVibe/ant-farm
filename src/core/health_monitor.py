# Health Monitor - Monitors agent health and system metrics

import asyncio
import time
from typing import Dict, List, Optional, Any

import structlog

from .interfaces import AgentInfo, AgentStatus, IMessageBroker, IHealthMonitor

logger = structlog.get_logger()


class HealthMonitor(IHealthMonitor):
    """Monitors agent health, system metrics, and provides health reporting."""

    def __init__(
        self,
        message_broker: IMessageBroker,
        heartbeat_interval: int = 30,
        health_check_timeout: int = 10
    ):
        self.message_broker = message_broker
        self.heartbeat_interval = heartbeat_interval
        self.health_check_timeout = health_check_timeout

        self.running = False
        self.last_health_check = time.time()
        self._health_scores: Dict[str, float] = {}
        self._background_tasks: List[asyncio.Task] = []

    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self.running:
            return

        self.running = True
        logger.info("Health monitor starting")

        # Start background monitoring tasks
        self._background_tasks.append(asyncio.create_task(self._heartbeat_monitor()))
        self._background_tasks.append(asyncio.create_task(self._health_check_loop()))

        logger.info("Health monitor background tasks started")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.running:
            return

        self.running = False
        logger.info("Health monitor stopping")

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            try:
                await asyncio.wait(self._background_tasks, timeout=2.0)
            except Exception:
                pass

        self._background_tasks.clear()
        logger.info("Health monitor stopped")

    async def check_agent_health(self, agent_name: str) -> Dict[str, Any]:
        """Check specific agent health."""
        try:
            # Send ping message to agent
            result = await self.message_broker.send_message_with_result(
                from_agent="health_monitor",
                to_agent=agent_name,
                topic="health_check",
                payload={"timestamp": time.time()},
                message_type="request"
            )

            if result and result.get("success"):
                response_time = time.time() - result.get("timestamp", time.time())
                health_score = max(0.0, min(1.0, 1.0 - (response_time / self.health_check_timeout)))

                self._health_scores[agent_name] = health_score

                return {
                    "agent_name": agent_name,
                    "status": "healthy",
                    "response_time": response_time,
                    "health_score": health_score,
                    "last_check": time.time()
                }
            else:
                # Agent is unresponsive
                self._health_scores[agent_name] = 0.0
                return {
                    "agent_name": agent_name,
                    "status": "unhealthy",
                    "error": "No response to health check",
                    "health_score": 0.0,
                    "last_check": time.time()
                }

        except Exception as e:
            logger.error("Health check failed", agent_name=agent_name, error=str(e))
            self._health_scores[agent_name] = 0.0
            return {
                "agent_name": agent_name,
                "status": "error",
                "error": str(e),
                "health_score": 0.0,
                "last_check": time.time()
            }

    async def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        try:
            current_time = time.time()

            # Calculate overall system health score
            health_scores = list(self._health_scores.values())
            if health_scores:
                avg_health_score = sum(health_scores) / len(health_scores)
            else:
                avg_health_score = 1.0  # Default healthy if no agents

            # System metrics
            system_metrics = {
                "overall_health_score": avg_health_score,
                "total_agents": len(self._health_scores),
                "healthy_agents": sum(1 for score in health_scores if score > 0.8),
                "unhealthy_agents": sum(1 for score in health_scores if score <= 0.8),
                "last_health_check": self.last_health_check,
                "monitoring_uptime": current_time - self.last_health_check if self.running else 0
            }

            # Agent-specific health details
            agent_health_details = {}
            for agent_name, score in self._health_scores.items():
                agent_health_details[agent_name] = {
                    "health_score": score,
                    "status": "healthy" if score > 0.8 else "unhealthy",
                    "last_updated": current_time
                }

            return {
                "timestamp": current_time,
                "system_metrics": system_metrics,
                "agent_health": agent_health_details,
                "recommendations": self._generate_health_recommendations(system_metrics)
            }

        except Exception as e:
            logger.error("Failed to generate health report", error=str(e))
            return {
                "timestamp": time.time(),
                "error": str(e),
                "system_metrics": {"overall_health_score": 0.0}
            }

    async def get_health_score(self) -> float:
        """Get overall system health score (0-100)."""
        try:
            report = await self.get_system_health_report()
            return report.get("system_metrics", {}).get("overall_health_score", 0.0) * 100
        except Exception as e:
            logger.error("Failed to get health score", error=str(e))
            return 0.0

    async def _heartbeat_monitor(self) -> None:
        """Monitor agent heartbeats and detect unresponsive agents."""
        while self.running:
            try:
                current_time = time.time()

                # Check for agents that haven't sent heartbeats recently
                unresponsive_agents = []
                for agent_name, last_check in self._health_scores.items():
                    # If we haven't checked this agent recently, mark for health check
                    if current_time - last_check > self.heartbeat_interval * 2:
                        unresponsive_agents.append(agent_name)

                # Perform health checks on unresponsive agents
                for agent_name in unresponsive_agents:
                    health_result = await self.check_agent_health(agent_name)
                    if health_result["status"] != "healthy":
                        logger.warning(
                            "Agent unresponsive",
                            agent_name=agent_name,
                            health_score=health_result["health_score"]
                        )

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error("Heartbeat monitor error", error=str(e))
                await asyncio.sleep(5)

    async def _health_check_loop(self) -> None:
        """Periodic comprehensive health check loop."""
        while self.running:
            try:
                self.last_health_check = time.time()

                # Perform comprehensive health checks
                # This could include database connectivity, message broker health, etc.
                await self._perform_system_health_checks()

                # Update health scores
                await self._update_health_scores()

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                logger.error("Health check loop error", error=str(e))
                await asyncio.sleep(5)

    async def _perform_system_health_checks(self) -> None:
        """Perform comprehensive system health checks."""
        try:
            # Check message broker connectivity
            broker_healthy = await self._check_message_broker_health()

            # Check database connectivity (would need database manager)
            # db_healthy = await self._check_database_health()

            # Log system health
            logger.debug(
                "System health check completed",
                message_broker_healthy=broker_healthy,
                timestamp=time.time()
            )

        except Exception as e:
            logger.error("System health check failed", error=str(e))

    async def _check_message_broker_health(self) -> bool:
        """Check message broker health."""
        try:
            # Simple health check - try to send a test message
            test_result = await self.message_broker.send_message(
                from_agent="health_monitor",
                to_agent="health_monitor",
                topic="health_test",
                payload={"test": True}
            )
            return test_result
        except Exception:
            return False

    async def _update_health_scores(self) -> None:
        """Update health scores based on recent activity."""
        try:
            current_time = time.time()

            # Decay health scores for agents we haven't checked recently
            for agent_name in list(self._health_scores.keys()):
                last_check_time = self._health_scores.get(f"{agent_name}_last_check", current_time)
                time_since_check = current_time - last_check_time

                if time_since_check > self.heartbeat_interval * 3:
                    # Decay health score
                    current_score = self._health_scores.get(agent_name, 1.0)
                    decayed_score = max(0.0, current_score * 0.9)  # 10% decay
                    self._health_scores[agent_name] = decayed_score

                    if decayed_score < 0.5:
                        logger.warning(
                            "Agent health decaying",
                            agent_name=agent_name,
                            health_score=decayed_score
                        )

        except Exception as e:
            logger.error("Failed to update health scores", error=str(e))

    def _generate_health_recommendations(self, system_metrics: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on system metrics."""
        recommendations = []

        health_score = system_metrics.get("overall_health_score", 1.0)
        unhealthy_count = system_metrics.get("unhealthy_agents", 0)
        total_agents = system_metrics.get("total_agents", 0)

        if health_score < 0.8:
            recommendations.append("Overall system health is degraded")

        if unhealthy_count > 0:
            recommendations.append(f"{unhealthy_count} out of {total_agents} agents are unhealthy")

        if health_score < 0.5:
            recommendations.append("Consider restarting unhealthy agents")
            recommendations.append("Check system resources (CPU, memory, network)")

        if not recommendations:
            recommendations.append("System health is good")

        return recommendations