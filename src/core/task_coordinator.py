"""Intelligent task assignment and coordination system for LeanVibe Agent Hive 2.0."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

from .config import settings
from .models import Agent as AgentModel
from .models import get_database_manager
from .task_queue import Task, TaskPriority, TaskStatus, task_queue

logger = structlog.get_logger()


class AssignmentStrategy(Enum):
    """Task assignment strategies."""

    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    PERFORMANCE_BASED = "performance_based"
    ROUND_ROBIN = "round_robin"
    SPECIALIZED = "specialized"


@dataclass
class AgentCapability:
    """Agent capability assessment."""

    agent_name: str
    task_type: str
    proficiency: float  # 0.0 to 1.0
    success_rate: float  # 0.0 to 1.0
    avg_execution_time: float  # seconds
    current_load: float  # 0.0 to 1.0
    availability: bool


@dataclass
class TaskAssignment:
    """Task assignment record."""

    task_id: str
    agent_name: str
    assigned_at: float
    strategy_used: AssignmentStrategy
    confidence_score: float  # 0.0 to 1.0
    estimated_completion_time: float


@dataclass
class AgentWorkload:
    """Agent workload tracking."""

    agent_name: str
    current_tasks: int
    max_capacity: int
    avg_task_time: float
    last_activity: float
    efficiency_score: float


class TaskCoordinator:
    """Intelligent task assignment and coordination system."""

    def __init__(self):
        self.agent_capabilities: dict[str, dict[str, AgentCapability]] = {}
        self.agent_workloads: dict[str, AgentWorkload] = {}
        self.task_assignments: dict[str, TaskAssignment] = {}
        self.coordination_active = False

        # Performance tracking
        self.assignment_history: list[TaskAssignment] = []
        self.success_rates: dict[str, float] = {}
        self.efficiency_metrics: dict[str, float] = {}

        # Configuration
        self.max_assignments_per_cycle = 10
        self.capability_learning_rate = 0.1
        self.performance_window = 3600  # 1 hour

        logger.info("Task coordinator initialized")

    async def initialize(self) -> None:
        """Initialize the task coordination system."""
        logger.info("Initializing task coordination system")

        # Start coordination loop
        self.coordination_active = True
        asyncio.create_task(self._coordination_loop())

        # Initialize agent capabilities from database
        await self._initialize_agent_capabilities()

        logger.info("Task coordination system initialized")

    async def shutdown(self) -> None:
        """Shutdown the coordination system."""
        logger.info("Shutting down task coordination system")
        self.coordination_active = False

    async def _coordination_loop(self) -> None:
        """Main coordination loop."""
        while self.coordination_active:
            try:
                # Update agent status and workloads
                await self._update_agent_status()

                # Process pending task assignments
                await self._process_pending_assignments()

                # Optimize current assignments
                await self._optimize_assignments()

                # Update capability assessments
                await self._update_capability_assessments()

                # Monitor task progress
                await self._monitor_task_progress()

                await asyncio.sleep(5)  # 5-second coordination cycle

            except Exception as e:
                logger.error("Coordination loop error", error=str(e))
                await asyncio.sleep(30)

    async def _initialize_agent_capabilities(self) -> None:
        """Initialize agent capabilities from historical data."""
        try:
            db_manager = get_database_manager(settings.database_url)
            db_session = db_manager.get_session()

            try:
                agents = db_session.query(AgentModel).filter_by(status="active").all()

                for agent in agents:
                    agent_name = agent.name
                    agent_type = agent.type

                    # Initialize default capabilities based on agent type
                    self.agent_capabilities[agent_name] = {}

                    if agent_type == "meta":
                        self._add_agent_capability(
                            agent_name, "system_analysis", 0.9, 0.85, 45.0
                        )
                        self._add_agent_capability(
                            agent_name, "coordination", 0.95, 0.9, 30.0
                        )
                        self._add_agent_capability(
                            agent_name, "planning", 0.8, 0.8, 60.0
                        )

                    elif agent_type == "developer":
                        self._add_agent_capability(
                            agent_name, "development", 0.85, 0.8, 120.0
                        )
                        self._add_agent_capability(
                            agent_name, "coding", 0.9, 0.85, 90.0
                        )
                        self._add_agent_capability(
                            agent_name, "implementation", 0.8, 0.75, 150.0
                        )
                        self._add_agent_capability(
                            agent_name, "refactoring", 0.7, 0.7, 180.0
                        )

                    elif agent_type == "qa":
                        self._add_agent_capability(
                            agent_name, "testing", 0.9, 0.85, 60.0
                        )
                        self._add_agent_capability(
                            agent_name, "validation", 0.85, 0.8, 45.0
                        )
                        self._add_agent_capability(agent_name, "qa", 0.95, 0.9, 75.0)
                        self._add_agent_capability(
                            agent_name, "debugging", 0.8, 0.75, 90.0
                        )

                    elif agent_type == "architect":
                        self._add_agent_capability(
                            agent_name, "architecture", 0.9, 0.85, 180.0
                        )
                        self._add_agent_capability(
                            agent_name, "design", 0.85, 0.8, 120.0
                        )
                        self._add_agent_capability(
                            agent_name, "planning", 0.8, 0.75, 90.0
                        )
                        self._add_agent_capability(
                            agent_name, "analysis", 0.75, 0.7, 60.0
                        )

                    elif agent_type == "research":
                        self._add_agent_capability(
                            agent_name, "research", 0.9, 0.85, 300.0
                        )
                        self._add_agent_capability(
                            agent_name, "analysis", 0.8, 0.75, 180.0
                        )
                        self._add_agent_capability(
                            agent_name, "learning", 0.85, 0.8, 240.0
                        )

                    # Initialize workload tracking
                    self.agent_workloads[agent_name] = AgentWorkload(
                        agent_name=agent_name,
                        current_tasks=0,
                        max_capacity=self._get_agent_max_capacity(agent_type),
                        avg_task_time=120.0,  # Default 2 minutes
                        last_activity=time.time(),
                        efficiency_score=0.7,
                    )

                logger.info(
                    "Agent capabilities initialized",
                    agent_count=len(self.agent_capabilities),
                )

            finally:
                db_session.close()

        except Exception as e:
            logger.error("Failed to initialize agent capabilities", error=str(e))

    def _add_agent_capability(
        self,
        agent_name: str,
        task_type: str,
        proficiency: float,
        success_rate: float,
        avg_time: float,
    ) -> None:
        """Add capability for an agent."""
        capability = AgentCapability(
            agent_name=agent_name,
            task_type=task_type,
            proficiency=proficiency,
            success_rate=success_rate,
            avg_execution_time=avg_time,
            current_load=0.0,
            availability=True,
        )

        if agent_name not in self.agent_capabilities:
            self.agent_capabilities[agent_name] = {}

        self.agent_capabilities[agent_name][task_type] = capability

    def _get_agent_max_capacity(self, agent_type: str) -> int:
        """Get maximum concurrent task capacity for agent type."""
        capacities = {"meta": 3, "developer": 2, "qa": 4, "architect": 2, "research": 1}
        return capacities.get(agent_type, 2)

    async def _update_agent_status(self) -> None:
        """Update agent status and availability."""
        try:
            db_manager = get_database_manager(settings.database_url)
            db_session = db_manager.get_session()

            try:
                agents = db_session.query(AgentModel).filter_by(status="active").all()
                current_time = time.time()

                for agent in agents:
                    agent_name = agent.name

                    # Check if agent is responsive
                    if agent.last_heartbeat:
                        time_since_heartbeat = current_time - agent.last_heartbeat
                        availability = time_since_heartbeat < 300  # 5 minutes
                    else:
                        availability = False

                    # Update capability availability
                    if agent_name in self.agent_capabilities:
                        for capability in self.agent_capabilities[agent_name].values():
                            capability.availability = availability

                    # Update workload info
                    if agent_name in self.agent_workloads:
                        workload = self.agent_workloads[agent_name]
                        if agent.last_heartbeat:
                            workload.last_activity = agent.last_heartbeat

            finally:
                db_session.close()

        except Exception as e:
            logger.error("Failed to update agent status", error=str(e))

    async def _process_pending_assignments(self) -> None:
        """Process pending task assignments."""
        try:
            # Get unassigned tasks
            unassigned_tasks = await task_queue.get_unassigned_tasks()

            if not unassigned_tasks:
                return

            # Limit assignments per cycle
            tasks_to_assign = unassigned_tasks[: self.max_assignments_per_cycle]

            for task in tasks_to_assign:
                assignment = await self._assign_task(task)
                if assignment:
                    self.task_assignments[task.id] = assignment
                    logger.info(
                        "Task assigned",
                        task_id=task.id,
                        agent=assignment.agent_name,
                        strategy=assignment.strategy_used.value,
                        confidence=assignment.confidence_score,
                    )
                else:
                    logger.warning("Failed to assign task", task_id=task.id)

        except Exception as e:
            logger.error("Failed to process pending assignments", error=str(e))

    async def _assign_task(self, task: Task) -> TaskAssignment | None:
        """Assign a task to the best available agent."""

        # Determine assignment strategy
        strategy = self._select_assignment_strategy(task)

        # Find best agent for the task
        best_agent = await self._find_best_agent(task, strategy)

        if not best_agent:
            return None

        # Calculate confidence score
        confidence = self._calculate_assignment_confidence(task, best_agent, strategy)

        # Estimate completion time
        estimated_time = self._estimate_completion_time(task, best_agent)

        # Update task with assignment
        await task_queue.assign_task(task.id, best_agent)

        # Update workload
        if best_agent in self.agent_workloads:
            self.agent_workloads[best_agent].current_tasks += 1

        assignment = TaskAssignment(
            task_id=task.id,
            agent_name=best_agent,
            assigned_at=time.time(),
            strategy_used=strategy,
            confidence_score=confidence,
            estimated_completion_time=estimated_time,
        )

        return assignment

    def _select_assignment_strategy(self, task: Task) -> AssignmentStrategy:
        """Select the best assignment strategy for a task."""

        # High priority tasks use performance-based assignment
        if task.priority == TaskPriority.HIGH:
            return AssignmentStrategy.PERFORMANCE_BASED

        # Tasks with specific types use capability-based assignment
        if task.task_type in ["development", "testing", "architecture", "research"]:
            return AssignmentStrategy.CAPABILITY_BASED

        # System tasks use specialized assignment
        if task.task_type in ["system_analysis", "coordination", "monitoring"]:
            return AssignmentStrategy.SPECIALIZED

        # Default to load-balanced assignment
        return AssignmentStrategy.LOAD_BALANCED

    async def _find_best_agent(
        self, task: Task, strategy: AssignmentStrategy
    ) -> str | None:
        """Find the best agent for a task using the specified strategy."""

        available_agents = self._get_available_agents()

        if not available_agents:
            return None

        if strategy == AssignmentStrategy.CAPABILITY_BASED:
            return self._find_best_agent_by_capability(task, available_agents)

        elif strategy == AssignmentStrategy.PERFORMANCE_BASED:
            return self._find_best_agent_by_performance(task, available_agents)

        elif strategy == AssignmentStrategy.LOAD_BALANCED:
            return self._find_best_agent_by_load(task, available_agents)

        elif strategy == AssignmentStrategy.SPECIALIZED:
            return self._find_specialized_agent(task, available_agents)

        elif strategy == AssignmentStrategy.ROUND_ROBIN:
            return self._find_agent_round_robin(task, available_agents)

        else:
            return available_agents[0] if available_agents else None

    def _get_available_agents(self) -> list[str]:
        """Get list of available agents."""
        available = []

        for agent_name, workload in self.agent_workloads.items():
            # Check if agent has capacity
            if workload.current_tasks < workload.max_capacity:
                # Check if any capabilities are available
                if agent_name in self.agent_capabilities:
                    if any(
                        cap.availability
                        for cap in self.agent_capabilities[agent_name].values()
                    ):
                        available.append(agent_name)

        return available

    def _find_best_agent_by_capability(
        self, task: Task, agents: list[str]
    ) -> str | None:
        """Find best agent based on capability match."""
        best_agent = None
        best_score = 0.0

        for agent_name in agents:
            if agent_name not in self.agent_capabilities:
                continue

            capabilities = self.agent_capabilities[agent_name]

            # Find matching capability
            if task.task_type in capabilities:
                capability = capabilities[task.task_type]
                if capability.availability:
                    # Score based on proficiency and success rate
                    score = capability.proficiency * 0.6 + capability.success_rate * 0.4

                    if score > best_score:
                        best_score = score
                        best_agent = agent_name

        return best_agent

    def _find_best_agent_by_performance(
        self, task: Task, agents: list[str]
    ) -> str | None:
        """Find best agent based on historical performance."""
        best_agent = None
        best_score = 0.0

        for agent_name in agents:
            # Get historical performance
            performance_score = self.efficiency_metrics.get(agent_name, 0.5)
            success_rate = self.success_rates.get(agent_name, 0.5)

            # Combine performance metrics
            score = performance_score * 0.6 + success_rate * 0.4

            # Factor in current load
            if agent_name in self.agent_workloads:
                workload = self.agent_workloads[agent_name]
                load_factor = 1.0 - (workload.current_tasks / workload.max_capacity)
                score *= load_factor

            if score > best_score:
                best_score = score
                best_agent = agent_name

        return best_agent

    def _find_best_agent_by_load(self, task: Task, agents: list[str]) -> str | None:
        """Find agent with lowest current load."""
        best_agent = None
        lowest_load = float("inf")

        for agent_name in agents:
            if agent_name in self.agent_workloads:
                workload = self.agent_workloads[agent_name]
                load_ratio = workload.current_tasks / workload.max_capacity

                if load_ratio < lowest_load:
                    lowest_load = load_ratio
                    best_agent = agent_name

        return best_agent

    def _find_specialized_agent(self, task: Task, agents: list[str]) -> str | None:
        """Find specialized agent for specific task types."""

        # Mapping of task types to preferred agent types
        specializations = {
            "system_analysis": ["meta"],
            "coordination": ["meta"],
            "monitoring": ["meta"],
            "development": ["developer"],
            "coding": ["developer"],
            "implementation": ["developer"],
            "testing": ["qa"],
            "validation": ["qa"],
            "qa": ["qa"],
            "architecture": ["architect"],
            "design": ["architect"],
            "planning": ["architect"],
            "research": ["research"],
            "learning": ["research"],
        }

        preferred_types = specializations.get(task.task_type, [])

        # Find agents of preferred types
        for agent_name in agents:
            try:
                db_manager = get_database_manager(settings.database_url)
                db_session = db_manager.get_session()

                try:
                    agent = (
                        db_session.query(AgentModel).filter_by(name=agent_name).first()
                    )
                    if agent and agent.type in preferred_types:
                        return agent_name
                finally:
                    db_session.close()

            except Exception:
                continue

        # Fallback to capability-based selection
        return self._find_best_agent_by_capability(task, agents)

    def _find_agent_round_robin(self, task: Task, agents: list[str]) -> str | None:
        """Find agent using round-robin assignment."""
        if not agents:
            return None

        # Simple round-robin based on task count
        min_tasks = min(
            self.agent_workloads[agent].current_tasks
            for agent in agents
            if agent in self.agent_workloads
        )

        for agent_name in agents:
            if agent_name in self.agent_workloads:
                if self.agent_workloads[agent_name].current_tasks == min_tasks:
                    return agent_name

        return agents[0]

    def _calculate_assignment_confidence(
        self, task: Task, agent_name: str, strategy: AssignmentStrategy
    ) -> float:
        """Calculate confidence score for an assignment."""

        confidence = 0.5  # Base confidence

        # Factor in agent capability
        if agent_name in self.agent_capabilities:
            capabilities = self.agent_capabilities[agent_name]
            if task.task_type in capabilities:
                capability = capabilities[task.task_type]
                confidence += capability.proficiency * 0.3
                confidence += capability.success_rate * 0.2

        # Factor in agent workload
        if agent_name in self.agent_workloads:
            workload = self.agent_workloads[agent_name]
            load_factor = 1.0 - (workload.current_tasks / workload.max_capacity)
            confidence += load_factor * 0.2

        # Factor in strategy appropriateness
        strategy_bonus = {
            AssignmentStrategy.SPECIALIZED: 0.2,
            AssignmentStrategy.CAPABILITY_BASED: 0.15,
            AssignmentStrategy.PERFORMANCE_BASED: 0.1,
            AssignmentStrategy.LOAD_BALANCED: 0.05,
            AssignmentStrategy.ROUND_ROBIN: 0.0,
        }

        confidence += strategy_bonus.get(strategy, 0.0)

        return min(1.0, confidence)

    def _estimate_completion_time(self, task: Task, agent_name: str) -> float:
        """Estimate task completion time."""

        base_time = 300.0  # 5 minutes default

        # Use capability-based estimation if available
        if (
            agent_name in self.agent_capabilities
            and task.task_type in self.agent_capabilities[agent_name]
        ):
            capability = self.agent_capabilities[agent_name][task.task_type]
            base_time = capability.avg_execution_time

        # Factor in task priority (high priority may take longer due to care)
        if task.priority == TaskPriority.HIGH:
            base_time *= 1.2
        elif task.priority == TaskPriority.LOW:
            base_time *= 0.8

        # Factor in current agent load
        if agent_name in self.agent_workloads:
            workload = self.agent_workloads[agent_name]
            load_factor = 1.0 + (workload.current_tasks * 0.2)
            base_time *= load_factor

        return base_time

    async def _optimize_assignments(self) -> None:
        """Optimize current task assignments."""
        # This would implement assignment optimization logic
        # For now, we'll just check for stuck tasks

        current_time = time.time()
        stuck_threshold = 3600  # 1 hour

        for assignment in self.task_assignments.values():
            time_since_assignment = current_time - assignment.assigned_at

            if time_since_assignment > stuck_threshold:
                # Check if task is still running
                task = await task_queue.get_task(assignment.task_id)

                if task and task.status == TaskStatus.IN_PROGRESS:
                    logger.warning(
                        "Potentially stuck task detected",
                        task_id=assignment.task_id,
                        agent=assignment.agent_name,
                        duration=time_since_assignment,
                    )

                    # Could implement reassignment logic here

    async def _update_capability_assessments(self) -> None:
        """Update agent capability assessments based on performance."""

        # Get completed tasks from recent window
        recent_tasks = await self._get_recent_completed_tasks()

        for task in recent_tasks:
            if task.assigned_to and task.assigned_to in self.agent_capabilities:
                agent_name = task.assigned_to
                task_type = task.task_type

                # Calculate performance metrics
                success = task.status == TaskStatus.COMPLETED
                execution_time = (task.completed_at or time.time()) - (
                    task.started_at or task.created_at
                )

                # Update capability if it exists
                if task_type in self.agent_capabilities[agent_name]:
                    capability = self.agent_capabilities[agent_name][task_type]

                    # Update success rate
                    learning_rate = self.capability_learning_rate
                    if success:
                        capability.success_rate = (
                            capability.success_rate * (1 - learning_rate)
                            + 1.0 * learning_rate
                        )
                    else:
                        capability.success_rate = (
                            capability.success_rate * (1 - learning_rate)
                            + 0.0 * learning_rate
                        )

                    # Update average execution time
                    capability.avg_execution_time = (
                        capability.avg_execution_time * (1 - learning_rate)
                        + execution_time * learning_rate
                    )

    async def _monitor_task_progress(self) -> None:
        """Monitor progress of assigned tasks."""

        # Update workload counts based on current tasks
        for agent_name in self.agent_workloads:
            current_tasks = await task_queue.get_agent_active_task_count(agent_name)
            self.agent_workloads[agent_name].current_tasks = current_tasks

    async def _get_recent_completed_tasks(self) -> list[Task]:
        """Get recently completed tasks for performance analysis."""
        # This would query the task queue for recently completed tasks
        # For now, return empty list
        return []

    async def get_coordination_status(self) -> dict[str, Any]:
        """Get current coordination system status."""

        total_agents = len(self.agent_capabilities)
        available_agents = len(self._get_available_agents())
        total_assignments = len(self.task_assignments)

        # Calculate average confidence
        if self.task_assignments:
            avg_confidence = sum(
                a.confidence_score for a in self.task_assignments.values()
            ) / len(self.task_assignments)
        else:
            avg_confidence = 0.0

        # Get workload distribution
        workload_distribution = {}
        for agent_name, workload in self.agent_workloads.items():
            workload_distribution[agent_name] = {
                "current_tasks": workload.current_tasks,
                "max_capacity": workload.max_capacity,
                "utilization": workload.current_tasks / workload.max_capacity,
                "efficiency": workload.efficiency_score,
            }

        return {
            "coordination_active": self.coordination_active,
            "total_agents": total_agents,
            "available_agents": available_agents,
            "total_assignments": total_assignments,
            "avg_confidence": avg_confidence,
            "workload_distribution": workload_distribution,
            "strategy_usage": self._get_strategy_usage_stats(),
        }

    def _get_strategy_usage_stats(self) -> dict[str, int]:
        """Get statistics on assignment strategy usage."""
        strategy_counts = {}

        for assignment in self.task_assignments.values():
            strategy = assignment.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        return strategy_counts


# Global instance
task_coordinator = TaskCoordinator()
