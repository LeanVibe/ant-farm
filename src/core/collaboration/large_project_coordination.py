"""
Advanced multi-agent coordination for large, complex projects.

This module extends the basic coordination system with capabilities specifically
designed for large-scale development projects requiring sophisticated coordination.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import structlog

from ..agent_coordination import CollaborationCoordinator, CollaborationType, TaskPhase
from ..constants import Intervals, Thresholds
from ..message_broker import MessageType, message_broker
from ..models import get_database_manager

logger = structlog.get_logger()


class ProjectScale(Enum):
    """Scale classification for projects."""

    SMALL = "small"  # 1-3 agents, <100 files
    MEDIUM = "medium"  # 4-8 agents, 100-1000 files
    LARGE = "large"  # 9-20 agents, 1000-10000 files
    MASSIVE = "massive"  # 20+ agents, 10000+ files


class ProjectPhase(Enum):
    """Development phases for large projects."""

    INITIATION = "initiation"
    PLANNING = "planning"
    DESIGN = "design"
    DEVELOPMENT = "development"
    INTEGRATION = "integration"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MAINTENANCE = "maintenance"


class ResourceType(Enum):
    """Types of computational resources."""

    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    CLI_TOOLS = "cli_tools"


@dataclass
class ProjectWorkspace:
    """Shared workspace for large project coordination."""

    id: str
    name: str
    description: str
    scale: ProjectScale
    phase: ProjectPhase
    root_path: Path

    # Agent coordination
    lead_agent: str
    participating_agents: Set[str] = field(default_factory=set)
    agent_roles: Dict[str, List[str]] = field(default_factory=dict)

    # Task and dependency management
    task_graph: Dict[str, Set[str]] = field(
        default_factory=dict
    )  # task_id -> dependencies
    completed_tasks: Set[str] = field(default_factory=set)
    active_tasks: Dict[str, str] = field(default_factory=dict)  # task_id -> agent_id

    # Resource management
    resource_pools: Dict[ResourceType, Dict[str, Any]] = field(default_factory=dict)
    resource_allocations: Dict[str, Dict[ResourceType, float]] = field(
        default_factory=dict
    )

    # Progress tracking
    milestones: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    progress_metrics: Dict[str, float] = field(default_factory=dict)

    # Collaboration features
    shared_context: Dict[str, Any] = field(default_factory=dict)
    communication_channels: List[str] = field(default_factory=list)
    conflict_resolution_rules: Dict[str, str] = field(default_factory=dict)

    # Metadata
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    priority: int = 5


@dataclass
class ResourcePool:
    """Manages computational resources for multi-agent coordination."""

    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    allocations: Dict[str, float] = field(
        default_factory=dict
    )  # agent_id -> allocated_amount
    queue: List[Dict[str, Any]] = field(
        default_factory=list
    )  # pending allocation requests

    def allocate(self, agent_id: str, amount: float) -> bool:
        """Allocate resources to an agent."""
        if self.available_capacity >= amount:
            self.allocations[agent_id] = self.allocations.get(agent_id, 0) + amount
            self.available_capacity -= amount
            return True
        return False

    def deallocate(self, agent_id: str, amount: float = None) -> float:
        """Deallocate resources from an agent."""
        current_allocation = self.allocations.get(agent_id, 0)
        amount_to_free = amount if amount is not None else current_allocation
        amount_to_free = min(amount_to_free, current_allocation)

        self.allocations[agent_id] = current_allocation - amount_to_free
        if self.allocations[agent_id] <= 0:
            del self.allocations[agent_id]

        self.available_capacity += amount_to_free
        return amount_to_free


@dataclass
class TaskDependencyGraph:
    """Manages complex task dependencies for large projects."""

    dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    reverse_dependencies: Dict[str, Set[str]] = field(default_factory=dict)
    task_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def add_task(
        self,
        task_id: str,
        dependencies: List[str] = None,
        metadata: Dict[str, Any] = None,
    ):
        """Add a task to the dependency graph."""
        dependencies = dependencies or []
        self.dependencies[task_id] = set(dependencies)
        self.task_metadata[task_id] = metadata or {}

        # Update reverse dependencies
        for dep in dependencies:
            if dep not in self.reverse_dependencies:
                self.reverse_dependencies[dep] = set()
            self.reverse_dependencies[dep].add(task_id)

    def get_ready_tasks(self, completed_tasks: Set[str]) -> List[str]:
        """Get tasks that are ready to execute (all dependencies completed)."""
        ready_tasks = []
        for task_id, deps in self.dependencies.items():
            if task_id not in completed_tasks and deps.issubset(completed_tasks):
                ready_tasks.append(task_id)
        return ready_tasks

    def get_critical_path(self) -> List[str]:
        """Calculate the critical path through the task graph."""
        # Simplified critical path calculation
        # In a real implementation, this would use more sophisticated algorithms
        visited = set()
        path = []

        def dfs_longest_path(task_id: str, current_path: List[str]) -> List[str]:
            if task_id in visited:
                return current_path

            visited.add(task_id)
            current_path.append(task_id)

            longest_path = current_path.copy()
            for dependent in self.reverse_dependencies.get(task_id, []):
                candidate_path = dfs_longest_path(dependent, current_path.copy())
                if len(candidate_path) > len(longest_path):
                    longest_path = candidate_path

            return longest_path

        # Find tasks with no dependencies (start nodes)
        start_tasks = [
            task_id for task_id, deps in self.dependencies.items() if not deps
        ]

        for start_task in start_tasks:
            candidate_path = dfs_longest_path(start_task, [])
            if len(candidate_path) > len(path):
                path = candidate_path

        return path


class LargeProjectCoordinator:
    """Advanced coordinator for large, complex multi-agent projects."""

    def __init__(self, base_coordinator: CollaborationCoordinator):
        self.base_coordinator = base_coordinator
        self.active_projects: Dict[str, ProjectWorkspace] = {}
        self.resource_pools: Dict[ResourceType, ResourcePool] = {}
        self.dependency_graphs: Dict[str, TaskDependencyGraph] = {}

        # Initialize resource pools
        self._initialize_resource_pools()

        # Performance monitoring
        self.coordination_metrics = {
            "projects_coordinated": 0,
            "average_completion_time": 0.0,
            "resource_utilization": {},
            "conflict_resolutions": 0,
        }

    def _initialize_resource_pools(self):
        """Initialize resource pools based on system capabilities."""
        self.resource_pools = {
            ResourceType.CPU: ResourcePool(ResourceType.CPU, 100.0, 100.0),
            ResourceType.MEMORY: ResourcePool(ResourceType.MEMORY, 100.0, 100.0),
            ResourceType.STORAGE: ResourcePool(ResourceType.STORAGE, 100.0, 100.0),
            ResourceType.NETWORK: ResourcePool(ResourceType.NETWORK, 100.0, 100.0),
            ResourceType.CLI_TOOLS: ResourcePool(
                ResourceType.CLI_TOOLS, 10.0, 10.0
            ),  # Max 10 concurrent CLI tool sessions
        }

    async def create_project_workspace(
        self,
        name: str,
        description: str,
        scale: ProjectScale,
        lead_agent: str,
        root_path: str = None,
    ) -> str:
        """Create a new project workspace for large project coordination."""
        project_id = f"project_{uuid.uuid4().hex[:8]}"

        workspace = ProjectWorkspace(
            id=project_id,
            name=name,
            description=description,
            scale=scale,
            phase=ProjectPhase.INITIATION,
            root_path=Path(root_path)
            if root_path
            else Path.cwd() / f"projects/{project_id}",
            lead_agent=lead_agent,
        )

        # Create physical workspace directory
        workspace.root_path.mkdir(parents=True, exist_ok=True)

        # Initialize dependency graph
        self.dependency_graphs[project_id] = TaskDependencyGraph()

        # Set up communication channels
        workspace.communication_channels = [
            f"project_{project_id}_general",
            f"project_{project_id}_technical",
            f"project_{project_id}_coordination",
        ]

        self.active_projects[project_id] = workspace

        logger.info(
            "Created large project workspace",
            project_id=project_id,
            name=name,
            scale=scale.value,
            lead_agent=lead_agent,
        )

        # Notify agents
        await message_broker.publish(
            MessageType.SYSTEM_EVENT,
            {
                "event": "project_workspace_created",
                "project_id": project_id,
                "workspace": {
                    "name": name,
                    "description": description,
                    "scale": scale.value,
                    "lead_agent": lead_agent,
                    "root_path": str(workspace.root_path),
                },
            },
        )

        return project_id

    async def join_project(
        self, project_id: str, agent_id: str, roles: List[str] = None
    ) -> bool:
        """Add an agent to a project workspace."""
        if project_id not in self.active_projects:
            logger.warning(
                "Attempted to join non-existent project", project_id=project_id
            )
            return False

        workspace = self.active_projects[project_id]
        workspace.participating_agents.add(agent_id)
        workspace.agent_roles[agent_id] = roles or ["contributor"]
        workspace.updated_at = time.time()

        # Allocate base resources for the agent
        await self._allocate_base_resources(project_id, agent_id)

        logger.info(
            "Agent joined project",
            project_id=project_id,
            agent_id=agent_id,
            roles=roles,
        )

        # Notify other agents
        await message_broker.publish(
            MessageType.AGENT_EVENT,
            {
                "event": "agent_joined_project",
                "project_id": project_id,
                "agent_id": agent_id,
                "roles": roles,
            },
        )

        return True

    async def _allocate_base_resources(self, project_id: str, agent_id: str):
        """Allocate base resources for an agent joining a project."""
        workspace = self.active_projects[project_id]

        # Base resource allocation based on project scale
        base_allocations = {
            ProjectScale.SMALL: {
                ResourceType.CPU: 20.0,
                ResourceType.MEMORY: 20.0,
                ResourceType.CLI_TOOLS: 2.0,
            },
            ProjectScale.MEDIUM: {
                ResourceType.CPU: 15.0,
                ResourceType.MEMORY: 15.0,
                ResourceType.CLI_TOOLS: 1.5,
            },
            ProjectScale.LARGE: {
                ResourceType.CPU: 10.0,
                ResourceType.MEMORY: 10.0,
                ResourceType.CLI_TOOLS: 1.0,
            },
            ProjectScale.MASSIVE: {
                ResourceType.CPU: 5.0,
                ResourceType.MEMORY: 5.0,
                ResourceType.CLI_TOOLS: 0.5,
            },
        }

        allocations = base_allocations.get(
            workspace.scale, base_allocations[ProjectScale.MEDIUM]
        )
        agent_allocations = {}

        for resource_type, amount in allocations.items():
            if self.resource_pools[resource_type].allocate(agent_id, amount):
                agent_allocations[resource_type] = amount
            else:
                logger.warning(
                    "Could not allocate resource",
                    resource_type=resource_type.value,
                    amount=amount,
                    agent_id=agent_id,
                )

        workspace.resource_allocations[agent_id] = agent_allocations

    async def decompose_large_task(
        self,
        project_id: str,
        task_description: str,
        estimated_complexity: int,
        target_agents: List[str] = None,
    ) -> Dict[str, Any]:
        """Decompose a large task into coordinated sub-tasks."""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")

        workspace = self.active_projects[project_id]

        # Use AI-assisted task decomposition
        decomposition_prompt = f"""
        Decompose this large development task into coordinated sub-tasks:
        
        Task: {task_description}
        Project Scale: {workspace.scale.value}
        Available Agents: {list(workspace.participating_agents)}
        Agent Roles: {workspace.agent_roles}
        Estimated Complexity: {estimated_complexity}/10
        
        Create a detailed breakdown with:
        1. Sub-tasks with clear dependencies
        2. Agent assignments based on roles/capabilities
        3. Resource requirements
        4. Time estimates
        5. Integration points
        
        Format as JSON with task hierarchy and dependencies.
        """

        # This would typically use a CLI tool for AI assistance
        # For now, we'll create a structured decomposition

        task_id = f"task_{uuid.uuid4().hex[:8]}"
        sub_tasks = []

        # Basic decomposition based on complexity and scale
        if estimated_complexity >= 8:  # Very complex task
            sub_tasks = [
                {
                    "id": f"{task_id}_planning",
                    "type": "planning",
                    "agent_type": "architect",
                },
                {
                    "id": f"{task_id}_design",
                    "type": "design",
                    "agent_type": "architect",
                    "depends_on": [f"{task_id}_planning"],
                },
                {
                    "id": f"{task_id}_implementation",
                    "type": "development",
                    "agent_type": "developer",
                    "depends_on": [f"{task_id}_design"],
                },
                {
                    "id": f"{task_id}_testing",
                    "type": "testing",
                    "agent_type": "qa",
                    "depends_on": [f"{task_id}_implementation"],
                },
                {
                    "id": f"{task_id}_integration",
                    "type": "integration",
                    "agent_type": "devops",
                    "depends_on": [f"{task_id}_testing"],
                },
                {
                    "id": f"{task_id}_review",
                    "type": "review",
                    "agent_type": "meta",
                    "depends_on": [f"{task_id}_integration"],
                },
            ]
        elif estimated_complexity >= 5:  # Medium complexity
            sub_tasks = [
                {
                    "id": f"{task_id}_design",
                    "type": "design",
                    "agent_type": "architect",
                },
                {
                    "id": f"{task_id}_implementation",
                    "type": "development",
                    "agent_type": "developer",
                    "depends_on": [f"{task_id}_design"],
                },
                {
                    "id": f"{task_id}_testing",
                    "type": "testing",
                    "agent_type": "qa",
                    "depends_on": [f"{task_id}_implementation"],
                },
                {
                    "id": f"{task_id}_review",
                    "type": "review",
                    "agent_type": "meta",
                    "depends_on": [f"{task_id}_testing"],
                },
            ]
        else:  # Simple task
            sub_tasks = [
                {
                    "id": f"{task_id}_implementation",
                    "type": "development",
                    "agent_type": "developer",
                },
                {
                    "id": f"{task_id}_review",
                    "type": "review",
                    "agent_type": "qa",
                    "depends_on": [f"{task_id}_implementation"],
                },
            ]

        # Add tasks to dependency graph
        dependency_graph = self.dependency_graphs[project_id]
        for sub_task in sub_tasks:
            dependency_graph.add_task(
                sub_task["id"],
                sub_task.get("depends_on", []),
                {
                    "type": sub_task["type"],
                    "agent_type": sub_task["agent_type"],
                    "parent_task": task_id,
                    "description": f"{sub_task['type']} for: {task_description}",
                },
            )

        # Assign tasks to available agents
        assignments = await self._assign_tasks_to_agents(
            project_id, sub_tasks, target_agents
        )

        logger.info(
            "Decomposed large task",
            project_id=project_id,
            task_id=task_id,
            sub_task_count=len(sub_tasks),
            assignments=len(assignments),
        )

        return {
            "task_id": task_id,
            "sub_tasks": sub_tasks,
            "assignments": assignments,
            "dependency_graph": dependency_graph.dependencies,
            "critical_path": dependency_graph.get_critical_path(),
        }

    async def _assign_tasks_to_agents(
        self,
        project_id: str,
        sub_tasks: List[Dict[str, Any]],
        target_agents: List[str] = None,
    ) -> Dict[str, str]:
        """Assign sub-tasks to appropriate agents."""
        workspace = self.active_projects[project_id]
        assignments = {}

        available_agents = target_agents or list(workspace.participating_agents)

        for sub_task in sub_tasks:
            task_id = sub_task["id"]
            required_agent_type = sub_task.get("agent_type", "developer")

            # Find best agent for this task type
            best_agent = None
            best_score = 0

            for agent_id in available_agents:
                if agent_id in workspace.agent_roles:
                    agent_roles = workspace.agent_roles[agent_id]

                    # Simple scoring based on role match
                    score = 0
                    if required_agent_type in agent_roles:
                        score += 10
                    if "contributor" in agent_roles:
                        score += 1

                    # Consider current workload (simplified)
                    current_tasks = sum(
                        1
                        for t_id, assigned_agent in workspace.active_tasks.items()
                        if assigned_agent == agent_id
                    )
                    score -= current_tasks * 2  # Penalty for high workload

                    if score > best_score:
                        best_score = score
                        best_agent = agent_id

            if best_agent:
                assignments[task_id] = best_agent
                workspace.active_tasks[task_id] = best_agent

        return assignments

    async def monitor_project_progress(self, project_id: str) -> Dict[str, Any]:
        """Monitor and report on project progress."""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")

        workspace = self.active_projects[project_id]
        dependency_graph = self.dependency_graphs[project_id]

        # Calculate progress metrics
        total_tasks = len(dependency_graph.dependencies)
        completed_tasks = len(workspace.completed_tasks)
        completion_percentage = (
            (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        )

        # Resource utilization
        resource_utilization = {}
        for resource_type, pool in self.resource_pools.items():
            utilization = (
                (
                    (pool.total_capacity - pool.available_capacity)
                    / pool.total_capacity
                    * 100
                )
                if pool.total_capacity > 0
                else 0
            )
            resource_utilization[resource_type.value] = utilization

        # Critical path analysis
        critical_path = dependency_graph.get_critical_path()
        critical_path_completion = (
            sum(1 for task_id in critical_path if task_id in workspace.completed_tasks)
            / len(critical_path)
            * 100
            if critical_path
            else 0
        )

        # Agent activity
        agent_activity = {}
        for agent_id in workspace.participating_agents:
            active_task_count = sum(
                1
                for assigned_agent in workspace.active_tasks.values()
                if assigned_agent == agent_id
            )
            agent_activity[agent_id] = {
                "active_tasks": active_task_count,
                "roles": workspace.agent_roles.get(agent_id, []),
                "resource_allocation": workspace.resource_allocations.get(agent_id, {}),
            }

        progress_report = {
            "project_id": project_id,
            "completion_percentage": completion_percentage,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "active_tasks": len(workspace.active_tasks),
            "critical_path_completion": critical_path_completion,
            "resource_utilization": resource_utilization,
            "agent_activity": agent_activity,
            "phase": workspace.phase.value,
            "participating_agents": len(workspace.participating_agents),
        }

        # Update workspace metrics
        workspace.progress_metrics = progress_report
        workspace.updated_at = time.time()

        return progress_report

    async def handle_conflict_resolution(
        self,
        project_id: str,
        conflict_type: str,
        involved_agents: List[str],
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle conflicts in large project coordination."""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")

        workspace = self.active_projects[project_id]
        conflict_id = f"conflict_{uuid.uuid4().hex[:8]}"

        logger.info(
            "Handling project conflict",
            project_id=project_id,
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            involved_agents=involved_agents,
        )

        # Apply conflict resolution rules
        resolution_strategy = workspace.conflict_resolution_rules.get(
            conflict_type, "escalate_to_lead"
        )

        resolution_result = {
            "conflict_id": conflict_id,
            "resolution_strategy": resolution_strategy,
            "timestamp": time.time(),
            "involved_agents": involved_agents,
            "context": context,
        }

        if resolution_strategy == "escalate_to_lead":
            # Notify lead agent
            await message_broker.publish(
                MessageType.AGENT_COMMUNICATION,
                {
                    "to": workspace.lead_agent,
                    "from": "large_project_coordinator",
                    "subject": f"Conflict Resolution Required: {conflict_type}",
                    "content": {
                        "conflict_id": conflict_id,
                        "project_id": project_id,
                        "conflict_type": conflict_type,
                        "involved_agents": involved_agents,
                        "context": context,
                    },
                },
            )
            resolution_result["action"] = "escalated_to_lead"

        elif resolution_strategy == "automatic_merge":
            # Implement automatic conflict resolution
            resolution_result["action"] = "automatic_resolution_attempted"

        elif resolution_strategy == "vote":
            # Initiate voting among involved agents
            resolution_result["action"] = "voting_initiated"

        # Track conflict resolution metrics
        self.coordination_metrics["conflict_resolutions"] += 1

        return resolution_result

    async def get_project_status(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a large project."""
        if project_id not in self.active_projects:
            raise ValueError(f"Project {project_id} not found")

        workspace = self.active_projects[project_id]
        progress_report = await self.monitor_project_progress(project_id)

        return {
            "workspace": {
                "id": workspace.id,
                "name": workspace.name,
                "description": workspace.description,
                "scale": workspace.scale.value,
                "phase": workspace.phase.value,
                "lead_agent": workspace.lead_agent,
                "participating_agents": list(workspace.participating_agents),
                "created_at": workspace.created_at,
                "updated_at": workspace.updated_at,
            },
            "progress": progress_report,
            "resource_pools": {
                rt.value: {
                    "total_capacity": pool.total_capacity,
                    "available_capacity": pool.available_capacity,
                    "utilization_percentage": (
                        (pool.total_capacity - pool.available_capacity)
                        / pool.total_capacity
                        * 100
                    )
                    if pool.total_capacity > 0
                    else 0,
                }
                for rt, pool in self.resource_pools.items()
            },
            "coordination_metrics": self.coordination_metrics,
        }


# Global instance
large_project_coordinator = None


async def get_large_project_coordinator() -> LargeProjectCoordinator:
    """Get or create the global large project coordinator."""
    global large_project_coordinator

    if large_project_coordinator is None:
        from .agent_coordination import CollaborationCoordinator

        base_coordinator = CollaborationCoordinator(message_broker)
        large_project_coordinator = LargeProjectCoordinator(base_coordinator)

        logger.info("Initialized large project coordinator")

    return large_project_coordinator
