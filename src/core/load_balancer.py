"""
Intelligent agent load balancing system for LeanVibe Agent Hive.

Provides dynamic task assignment based on agent capabilities, current load,
resource utilization, and historical performance metrics.
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from .enums import AgentStatus, TaskPriority
from .models import Agent, Task
from .orchestrator import get_orchestrator
from .task_queue import task_queue

logger = structlog.get_logger()


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""

    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_WEIGHTED = "performance_weighted"
    HYBRID_INTELLIGENT = "hybrid_intelligent"


@dataclass
class AgentLoad:
    """Agent load and capacity metrics."""

    agent_id: str
    active_tasks: int = 0
    max_concurrent_tasks: int = 5
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    avg_task_completion_time: float = 0.0
    success_rate: float = 1.0
    last_heartbeat: float = 0.0
    capabilities: Set[str] = field(default_factory=set)
    preferred_task_types: Set[str] = field(default_factory=set)

    @property
    def load_factor(self) -> float:
        """Calculate overall load factor (0.0 = no load, 1.0 = fully loaded)."""
        task_load = self.active_tasks / max(self.max_concurrent_tasks, 1)
        resource_load = (self.cpu_usage_percent + self.memory_usage_percent) / 200
        return min((task_load + resource_load) / 2, 1.0)

    @property
    def efficiency_score(self) -> float:
        """Calculate agent efficiency score based on performance metrics."""
        # Combine success rate with inverse completion time
        time_factor = 1.0 / max(self.avg_task_completion_time, 0.1)
        # Normalize time factor (assuming 60s is baseline)
        time_score = min(time_factor / (1.0 / 60.0), 1.0)

        return (self.success_rate * 0.7) + (time_score * 0.3)

    @property
    def is_available(self) -> bool:
        """Check if agent is available for new tasks."""
        if self.active_tasks >= self.max_concurrent_tasks:
            return False
        if time.time() - self.last_heartbeat > 300:  # 5 minutes
            return False
        if self.cpu_usage_percent > 90 or self.memory_usage_percent > 90:
            return False
        return True


@dataclass
class TaskRequirement:
    """Task requirements for load balancing."""

    task_id: str
    task_type: str
    priority: TaskPriority
    required_capabilities: Set[str] = field(default_factory=set)
    estimated_duration: float = 60.0  # seconds
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    preferred_agents: Set[str] = field(default_factory=set)
    excluded_agents: Set[str] = field(default_factory=set)


class IntelligentLoadBalancer:
    """Intelligent load balancer for agent task assignment."""

    def __init__(self):
        self.agent_loads: Dict[str, AgentLoad] = {}
        self.task_history: deque = deque(maxlen=1000)
        self.assignment_history: deque = deque(maxlen=500)
        self.performance_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.strategy = LoadBalancingStrategy.HYBRID_INTELLIGENT

        # Performance tuning parameters
        self.capability_weight = 0.4
        self.load_weight = 0.3
        self.performance_weight = 0.3

        # Update tracking
        self.last_metrics_update = 0
        self.metrics_update_interval = 30  # seconds

    async def update_agent_metrics(self):
        """Update agent load and performance metrics."""
        current_time = time.time()

        if current_time - self.last_metrics_update < self.metrics_update_interval:
            return

        try:
            orchestrator = await get_orchestrator()
            agents = await orchestrator.registry.list_agents()

            for agent in agents:
                await self._update_agent_load(agent)

            self.last_metrics_update = current_time

        except Exception as e:
            logger.error("Failed to update agent metrics", error=str(e))

    async def _update_agent_load(self, agent: Agent):
        """Update load metrics for a specific agent."""
        agent_id = str(agent.id)

        # Get or create agent load tracker
        if agent_id not in self.agent_loads:
            self.agent_loads[agent_id] = AgentLoad(
                agent_id=agent_id,
                capabilities=set(agent.capabilities or []),
                last_heartbeat=agent.last_heartbeat or 0,
            )

        load = self.agent_loads[agent_id]

        # Update basic metrics
        load.last_heartbeat = agent.last_heartbeat or 0
        load.capabilities = set(agent.capabilities or [])

        # Count active tasks
        active_tasks = await task_queue.count_tasks_for_agent(
            agent_id, statuses=["running", "assigned"]
        )
        load.active_tasks = active_tasks

        # Update performance metrics from history
        await self._update_performance_metrics(agent_id, load)

        # Get system resource usage if available
        await self._update_resource_usage(agent_id, load)

    async def _update_performance_metrics(self, agent_id: str, load: AgentLoad):
        """Update agent performance metrics from task history."""
        # Get recent completed tasks for this agent
        recent_tasks = await task_queue.get_agent_task_history(
            agent_id, limit=50, statuses=["completed", "failed"]
        )

        if not recent_tasks:
            return

        # Calculate success rate
        completed_tasks = [t for t in recent_tasks if t.status == "completed"]
        load.success_rate = len(completed_tasks) / len(recent_tasks)

        # Calculate average completion time
        if completed_tasks:
            completion_times = []
            for task in completed_tasks:
                if task.started_at and task.completed_at:
                    duration = task.completed_at - task.started_at
                    completion_times.append(duration)

            if completion_times:
                load.avg_task_completion_time = sum(completion_times) / len(
                    completion_times
                )

        # Update preferred task types based on success patterns
        task_type_success = defaultdict(list)
        for task in recent_tasks:
            success = 1.0 if task.status == "completed" else 0.0
            task_type_success[task.task_type].append(success)

        # Identify task types with >80% success rate
        load.preferred_task_types = {
            task_type
            for task_type, successes in task_type_success.items()
            if sum(successes) / len(successes) > 0.8 and len(successes) >= 3
        }

    async def _update_resource_usage(self, agent_id: str, load: AgentLoad):
        """Update agent resource usage metrics."""
        try:
            # This would integrate with system monitoring
            # For now, we'll use mock data or estimates based on active tasks
            base_cpu = min(load.active_tasks * 15, 80)  # Estimate 15% CPU per task
            base_memory = min(
                load.active_tasks * 10, 70
            )  # Estimate 10% memory per task

            load.cpu_usage_percent = base_cpu
            load.memory_usage_percent = base_memory

        except Exception as e:
            logger.warning(
                "Failed to update resource usage", agent_id=agent_id, error=str(e)
            )

    async def assign_task(self, task: Task) -> Optional[str]:
        """Assign a task to the best available agent."""
        await self.update_agent_metrics()

        # Create task requirement
        requirement = TaskRequirement(
            task_id=task.id,
            task_type=task.task_type,
            priority=task.priority,
            required_capabilities=self._extract_required_capabilities(task),
            estimated_duration=self._estimate_task_duration(task),
        )

        # Find the best agent
        best_agent_id = await self._find_best_agent(requirement)

        if best_agent_id:
            # Record assignment
            self.assignment_history.append(
                {
                    "task_id": task.id,
                    "agent_id": best_agent_id,
                    "timestamp": time.time(),
                    "strategy": self.strategy.value,
                    "load_factor": self.agent_loads[best_agent_id].load_factor,
                }
            )

            # Update agent load
            self.agent_loads[best_agent_id].active_tasks += 1

            logger.info(
                "Task assigned",
                task_id=task.id,
                agent_id=best_agent_id,
                strategy=self.strategy.value,
                load_factor=self.agent_loads[best_agent_id].load_factor,
            )

        return best_agent_id

    def _extract_required_capabilities(self, task: Task) -> Set[str]:
        """Extract required capabilities from task."""
        capabilities = set()

        # Map task types to required capabilities
        capability_mapping = {
            "code_generation": {"coding", "python", "javascript"},
            "code_review": {"coding", "analysis", "security"},
            "testing": {"testing", "qa", "automation"},
            "deployment": {"devops", "docker", "kubernetes"},
            "documentation": {"writing", "markdown"},
            "architecture": {"design", "system_design"},
            "debugging": {"debugging", "analysis", "troubleshooting"},
            "self_improvement": {"meta_programming", "analysis", "optimization"},
        }

        return capability_mapping.get(task.task_type, set())

    def _estimate_task_duration(self, task: Task) -> float:
        """Estimate task duration based on type and complexity."""
        # Base durations by task type (in seconds)
        base_durations = {
            "code_generation": 300,  # 5 minutes
            "code_review": 180,  # 3 minutes
            "testing": 240,  # 4 minutes
            "deployment": 600,  # 10 minutes
            "documentation": 180,  # 3 minutes
            "architecture": 900,  # 15 minutes
            "debugging": 420,  # 7 minutes
            "self_improvement": 1200,  # 20 minutes
        }

        base_duration = base_durations.get(task.task_type, 300)

        # Adjust based on priority
        priority_multipliers = {
            TaskPriority.LOW: 0.8,
            TaskPriority.NORMAL: 1.0,
            TaskPriority.HIGH: 1.3,
            TaskPriority.CRITICAL: 1.5,
        }

        multiplier = priority_multipliers.get(task.priority, 1.0)

        # Adjust based on description length (complexity proxy)
        if len(task.description) > 500:
            multiplier *= 1.5
        elif len(task.description) > 200:
            multiplier *= 1.2

        return base_duration * multiplier

    async def _find_best_agent(self, requirement: TaskRequirement) -> Optional[str]:
        """Find the best agent for a task requirement."""
        available_agents = [
            (agent_id, load)
            for agent_id, load in self.agent_loads.items()
            if load.is_available and agent_id not in requirement.excluded_agents
        ]

        if not available_agents:
            logger.warning("No available agents for task", task_id=requirement.task_id)
            return None

        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_assignment(available_agents)

        elif self.strategy == LoadBalancingStrategy.LEAST_LOADED:
            return self._least_loaded_assignment(available_agents)

        elif self.strategy == LoadBalancingStrategy.CAPABILITY_BASED:
            return self._capability_based_assignment(available_agents, requirement)

        elif self.strategy == LoadBalancingStrategy.PERFORMANCE_WEIGHTED:
            return self._performance_weighted_assignment(available_agents, requirement)

        elif self.strategy == LoadBalancingStrategy.HYBRID_INTELLIGENT:
            return self._hybrid_intelligent_assignment(available_agents, requirement)

        else:
            # Default to least loaded
            return self._least_loaded_assignment(available_agents)

    def _round_robin_assignment(
        self, available_agents: List[Tuple[str, AgentLoad]]
    ) -> str:
        """Simple round-robin assignment."""
        if not hasattr(self, "_round_robin_index"):
            self._round_robin_index = 0

        agent_id = available_agents[self._round_robin_index % len(available_agents)][0]
        self._round_robin_index += 1

        return agent_id

    def _least_loaded_assignment(
        self, available_agents: List[Tuple[str, AgentLoad]]
    ) -> str:
        """Assign to the least loaded agent."""
        return min(available_agents, key=lambda x: x[1].load_factor)[0]

    def _capability_based_assignment(
        self,
        available_agents: List[Tuple[str, AgentLoad]],
        requirement: TaskRequirement,
    ) -> str:
        """Assign based on capability matching."""
        scored_agents = []

        for agent_id, load in available_agents:
            # Calculate capability match score
            capability_overlap = len(
                requirement.required_capabilities & load.capabilities
            )
            total_required = len(requirement.required_capabilities)

            if total_required == 0:
                capability_score = 1.0
            else:
                capability_score = capability_overlap / total_required

            # Boost score for preferred task types
            if requirement.task_type in load.preferred_task_types:
                capability_score += 0.3

            scored_agents.append((agent_id, capability_score, load.load_factor))

        # Sort by capability score (desc), then by load (asc)
        scored_agents.sort(key=lambda x: (-x[1], x[2]))

        return scored_agents[0][0] if scored_agents else available_agents[0][0]

    def _performance_weighted_assignment(
        self,
        available_agents: List[Tuple[str, AgentLoad]],
        requirement: TaskRequirement,
    ) -> str:
        """Assign based on performance metrics."""
        scored_agents = []

        for agent_id, load in available_agents:
            # Calculate composite score
            performance_score = load.efficiency_score
            load_penalty = load.load_factor  # Higher load = penalty

            composite_score = performance_score * (1 - load_penalty * 0.5)
            scored_agents.append((agent_id, composite_score))

        # Sort by composite score (descending)
        scored_agents.sort(key=lambda x: -x[1])

        return scored_agents[0][0]

    def _hybrid_intelligent_assignment(
        self,
        available_agents: List[Tuple[str, AgentLoad]],
        requirement: TaskRequirement,
    ) -> str:
        """Intelligent assignment using weighted scoring."""
        scored_agents = []

        for agent_id, load in available_agents:
            # Capability score
            capability_overlap = len(
                requirement.required_capabilities & load.capabilities
            )
            total_required = max(len(requirement.required_capabilities), 1)
            capability_score = capability_overlap / total_required

            # Preferred task type bonus
            if requirement.task_type in load.preferred_task_types:
                capability_score += 0.3

            # Load score (inverted - lower load is better)
            load_score = 1.0 - load.load_factor

            # Performance score
            performance_score = load.efficiency_score

            # Priority adjustment
            priority_weights = {
                TaskPriority.LOW: 0.8,
                TaskPriority.NORMAL: 1.0,
                TaskPriority.HIGH: 1.2,
                TaskPriority.CRITICAL: 1.5,
            }
            priority_weight = priority_weights.get(requirement.priority, 1.0)

            # Composite score
            composite_score = (
                capability_score * self.capability_weight
                + load_score * self.load_weight
                + performance_score * self.performance_weight
            ) * priority_weight

            scored_agents.append(
                (
                    agent_id,
                    composite_score,
                    {
                        "capability_score": capability_score,
                        "load_score": load_score,
                        "performance_score": performance_score,
                        "composite_score": composite_score,
                    },
                )
            )

        # Sort by composite score (descending)
        scored_agents.sort(key=lambda x: -x[1])

        if scored_agents:
            best_agent = scored_agents[0]
            logger.debug(
                "Agent selection scoring",
                task_id=requirement.task_id,
                selected_agent=best_agent[0],
                scores=best_agent[2],
            )
            return best_agent[0]

        return available_agents[0][0] if available_agents else None

    async def task_completed(
        self, task_id: str, agent_id: str, success: bool, duration: float
    ):
        """Record task completion for load balancing optimization."""
        if agent_id in self.agent_loads:
            # Update active task count
            self.agent_loads[agent_id].active_tasks = max(
                0, self.agent_loads[agent_id].active_tasks - 1
            )

            # Record performance metrics
            self.performance_metrics[agent_id].append(
                {
                    "task_id": task_id,
                    "success": success,
                    "duration": duration,
                    "timestamp": time.time(),
                }
            )

        # Record in task history
        self.task_history.append(
            {
                "task_id": task_id,
                "agent_id": agent_id,
                "success": success,
                "duration": duration,
                "completed_at": time.time(),
            }
        )

    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics and metrics."""
        return {
            "strategy": self.strategy.value,
            "active_agents": len(
                [load for load in self.agent_loads.values() if load.is_available]
            ),
            "total_agents": len(self.agent_loads),
            "agent_loads": {
                agent_id: {
                    "load_factor": load.load_factor,
                    "active_tasks": load.active_tasks,
                    "efficiency_score": load.efficiency_score,
                    "success_rate": load.success_rate,
                    "is_available": load.is_available,
                }
                for agent_id, load in self.agent_loads.items()
            },
            "recent_assignments": list(self.assignment_history)[-20:],
            "performance_summary": self._calculate_performance_summary(),
            "timestamp": time.time(),
        }

    def _calculate_performance_summary(self) -> Dict[str, float]:
        """Calculate overall load balancing performance summary."""
        if not self.task_history:
            return {}

        recent_tasks = [
            task
            for task in self.task_history
            if time.time() - task["completed_at"] <= 3600  # Last hour
        ]

        if not recent_tasks:
            return {}

        total_tasks = len(recent_tasks)
        successful_tasks = len([t for t in recent_tasks if t["success"]])
        average_duration = sum(t["duration"] for t in recent_tasks) / total_tasks

        # Calculate load distribution evenness (lower is better)
        agent_task_counts = defaultdict(int)
        for task in recent_tasks:
            agent_task_counts[task["agent_id"]] += 1

        if len(agent_task_counts) > 1:
            task_counts = list(agent_task_counts.values())
            load_variance = sum(
                (count - (total_tasks / len(agent_task_counts))) ** 2
                for count in task_counts
            ) / len(task_counts)
        else:
            load_variance = 0.0

        return {
            "total_tasks_last_hour": total_tasks,
            "success_rate": successful_tasks / total_tasks if total_tasks > 0 else 0,
            "average_duration_seconds": average_duration,
            "load_distribution_variance": load_variance,
            "active_agents_utilization": len(agent_task_counts)
            / max(len(self.agent_loads), 1),
        }


# Global load balancer instance
intelligent_load_balancer = IntelligentLoadBalancer()


async def assign_task_intelligently(task: Task) -> Optional[str]:
    """Assign a task using the intelligent load balancer."""
    return await intelligent_load_balancer.assign_task(task)


def get_load_balancer() -> IntelligentLoadBalancer:
    """Get the global load balancer instance."""
    return intelligent_load_balancer
