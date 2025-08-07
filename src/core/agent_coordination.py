"""Advanced inter-agent coordination and collaboration protocols."""

import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import structlog

from .message_broker import MessageBroker, MessageType, message_broker
from .orchestrator import orchestrator

logger = structlog.get_logger()


class CollaborationType(Enum):
    """Types of agent collaboration patterns."""

    SEQUENTIAL = "sequential"  # Agents work in sequence, one after another
    PARALLEL = "parallel"  # Agents work in parallel on different sub-tasks
    PIPELINE = "pipeline"  # Agents form a processing pipeline
    CONSENSUS = "consensus"  # Agents need to reach consensus
    COMPETITIVE = "competitive"  # Agents compete for best solution
    DELEGATION = "delegation"  # One agent delegates to others


class TaskPhase(Enum):
    """Phases of collaborative task execution."""

    PLANNING = "planning"
    ASSIGNMENT = "assignment"
    EXECUTION = "execution"
    REVIEW = "review"
    COMPLETION = "completion"
    FAILED = "failed"


@dataclass
class CollaborationContext:
    """Context for a collaborative task execution."""

    id: str
    title: str
    description: str
    collaboration_type: CollaborationType
    phase: TaskPhase
    coordinator_agent: str
    participating_agents: list[str] = field(default_factory=list)
    sub_tasks: dict[str, dict[str, Any]] = field(default_factory=dict)
    dependencies: dict[str, list[str]] = field(default_factory=dict)
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    deadline: float | None = None
    priority: int = 5


@dataclass
class AgentCapabilityMap:
    """Maps agent capabilities to task requirements."""

    agent_name: str
    capabilities: list[str]
    specializations: list[str]
    load_factor: float
    availability: bool
    performance_metrics: dict[str, float] = field(default_factory=dict)


class TaskDecomposer:
    """Decomposes complex tasks into sub-tasks for different agents."""

    def __init__(self):
        self.decomposition_strategies = {
            CollaborationType.SEQUENTIAL: self._decompose_sequential,
            CollaborationType.PARALLEL: self._decompose_parallel,
            CollaborationType.PIPELINE: self._decompose_pipeline,
            CollaborationType.CONSENSUS: self._decompose_consensus,
            CollaborationType.COMPETITIVE: self._decompose_competitive,
            CollaborationType.DELEGATION: self._decompose_delegation,
        }

    async def decompose_task(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose a task based on collaboration type and available agents."""

        strategy = self.decomposition_strategies.get(context.collaboration_type)
        if not strategy:
            raise ValueError(
                f"Unknown collaboration type: {context.collaboration_type}"
            )

        return await strategy(context, available_agents)

    async def _decompose_sequential(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for sequential execution."""

        sub_tasks = {}

        # Example sequential decomposition for code development
        required_capabilities = context.metadata.get("required_capabilities", [])

        if any(
            cap in ["code_generation", "system_design", "development"]
            for cap in required_capabilities
        ):
            sequence = [
                (
                    "requirements_analysis",
                    ["system_design", "architecture_planning", "analysis"],
                    "Analyze and refine requirements",
                ),
                (
                    "design",
                    ["system_design", "architecture_planning"],
                    "Create system design and architecture",
                ),
                (
                    "implementation",
                    ["code_generation", "testing", "debugging"],
                    "Implement the code",
                ),
                (
                    "testing",
                    ["testing", "quality_assurance"],
                    "Write and run comprehensive tests",
                ),
                (
                    "review",
                    ["system_design", "testing", "quality_assurance"],
                    "Review and validate implementation",
                ),
            ]

            for i, (task_name, required_caps, description) in enumerate(sequence):
                suitable_agents = [
                    agent
                    for agent in available_agents
                    if any(cap in agent.capabilities for cap in required_caps)
                ]

                if suitable_agents:
                    # Select best agent based on load and capabilities
                    selected_agent = min(suitable_agents, key=lambda a: a.load_factor)

                    sub_tasks[f"seq_{i}_{task_name}"] = {
                        "description": description,
                        "assigned_agent": selected_agent.agent_name,
                        "required_capabilities": required_caps,
                        "depends_on": [f"seq_{i - 1}_{sequence[i - 1][0]}"]
                        if i > 0
                        else [],
                        "estimated_duration": 300,  # 5 minutes default
                        "priority": context.priority,
                    }
        else:
            # Generic sequential decomposition
            if available_agents:
                # Create a simple sequential task chain
                generic_tasks = [
                    ("analysis", "Analyze the task requirements"),
                    ("planning", "Create execution plan"),
                    ("implementation", "Execute the main task"),
                    ("validation", "Validate and test results"),
                ]

                for i, (task_name, description) in enumerate(generic_tasks):
                    # Round-robin assignment to available agents
                    selected_agent = available_agents[i % len(available_agents)]

                    sub_tasks[f"seq_{i}_{task_name}"] = {
                        "description": description,
                        "assigned_agent": selected_agent.agent_name,
                        "required_capabilities": selected_agent.capabilities,
                        "depends_on": [f"seq_{i - 1}_{generic_tasks[i - 1][0]}"]
                        if i > 0
                        else [],
                        "estimated_duration": 180,
                        "priority": context.priority,
                    }

        return sub_tasks

    async def _decompose_parallel(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for parallel execution."""

        sub_tasks = {}

        # Example parallel decomposition
        if "system_optimization" in context.metadata.get("task_type", ""):
            parallel_tasks = [
                (
                    "performance_analysis",
                    ["deployment", "monitoring"],
                    "Analyze system performance",
                ),
                (
                    "security_audit",
                    ["testing", "quality_assurance"],
                    "Conduct security audit",
                ),
                ("code_review", ["system_design", "analysis"], "Review code quality"),
                (
                    "documentation_update",
                    ["code_generation", "testing"],
                    "Update documentation",
                ),
            ]

            for task_name, required_caps, description in parallel_tasks:
                suitable_agents = [
                    agent
                    for agent in available_agents
                    if any(cap in agent.capabilities for cap in required_caps)
                ]

                if suitable_agents:
                    selected_agent = min(suitable_agents, key=lambda a: a.load_factor)

                    sub_tasks[task_name] = {
                        "description": description,
                        "assigned_agent": selected_agent.agent_name,
                        "required_capabilities": required_caps,
                        "depends_on": [],  # No dependencies for parallel tasks
                        "estimated_duration": 180,
                        "priority": context.priority,
                    }
        else:
            # Generic parallel decomposition - create tasks for each available agent
            if available_agents:
                base_tasks = [
                    ("component_a", "Work on component A"),
                    ("component_b", "Work on component B"),
                    ("component_c", "Work on component C"),
                    ("integration", "Handle integration tasks"),
                ]

                for i, agent in enumerate(available_agents[: len(base_tasks)]):
                    task_name, description = base_tasks[i % len(base_tasks)]

                    sub_tasks[f"parallel_{task_name}_{i}"] = {
                        "description": f"{description} - assigned to {agent.agent_name}",
                        "assigned_agent": agent.agent_name,
                        "required_capabilities": agent.capabilities,
                        "depends_on": [],  # No dependencies for parallel tasks
                        "estimated_duration": 200,
                        "priority": context.priority,
                    }

        return sub_tasks

    async def _decompose_parallel(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for parallel execution."""

        sub_tasks = {}

        # Example parallel decomposition
        if "system_optimization" in context.metadata.get("task_type", ""):
            parallel_tasks = [
                (
                    "performance_analysis",
                    ["deployment", "infrastructure", "monitoring"],
                    "Analyze system performance",
                ),
                (
                    "security_audit",
                    ["testing", "quality_assurance"],
                    "Conduct security audit",
                ),
                (
                    "code_review",
                    ["system_design", "architecture_planning", "analysis"],
                    "Review code quality",
                ),
                (
                    "documentation_update",
                    ["code_generation", "testing", "debugging"],
                    "Update documentation",
                ),
            ]

            for task_name, required_caps, description in parallel_tasks:
                suitable_agents = [
                    agent
                    for agent in available_agents
                    if any(cap in agent.capabilities for cap in required_caps)
                ]

                if suitable_agents:
                    selected_agent = min(suitable_agents, key=lambda a: a.load_factor)

                    sub_tasks[task_name] = {
                        "description": description,
                        "assigned_agent": selected_agent.agent_name,
                        "required_capabilities": required_caps,
                        "depends_on": [],  # No dependencies for parallel tasks
                        "estimated_duration": 180,
                        "priority": context.priority,
                    }

        return sub_tasks

    async def _decompose_pipeline(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for pipeline execution."""

        sub_tasks = {}

        # Example CI/CD pipeline
        pipeline_stages = [
            ("source_preparation", ["developer"], "Prepare source code"),
            ("build", ["devops"], "Build and compile"),
            ("unit_tests", ["qa"], "Run unit tests"),
            ("integration_tests", ["qa"], "Run integration tests"),
            ("deployment", ["devops"], "Deploy to environment"),
            ("validation", ["qa"], "Validate deployment"),
        ]

        for i, (stage_name, required_caps, description) in enumerate(pipeline_stages):
            suitable_agents = [
                agent
                for agent in available_agents
                if any(cap in agent.capabilities for cap in required_caps)
            ]

            if suitable_agents:
                selected_agent = min(suitable_agents, key=lambda a: a.load_factor)

                sub_tasks[f"pipeline_{i}_{stage_name}"] = {
                    "description": description,
                    "assigned_agent": selected_agent.agent_name,
                    "required_capabilities": required_caps,
                    "depends_on": [f"pipeline_{i - 1}_{pipeline_stages[i - 1][0]}"]
                    if i > 0
                    else [],
                    "estimated_duration": 120,
                    "priority": context.priority,
                    "pipeline_stage": i,
                }

        return sub_tasks

    async def _decompose_consensus(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for consensus-based execution."""

        sub_tasks = {}

        # Select multiple agents with similar capabilities
        required_caps = context.metadata.get("required_capabilities", [])

        if required_caps:
            suitable_agents = [
                agent
                for agent in available_agents
                if any(cap in agent.capabilities for cap in required_caps)
            ][:3]  # Limit to 3 agents for consensus
        else:
            # Use all available agents if no specific requirements
            suitable_agents = available_agents[:3]

        for i, agent in enumerate(suitable_agents):
            sub_tasks[f"consensus_proposal_{i}"] = {
                "description": f"Generate solution proposal for {context.title}",
                "assigned_agent": agent.agent_name,
                "required_capabilities": agent.capabilities,
                "depends_on": [],
                "estimated_duration": 240,
                "priority": context.priority,
                "consensus_role": "proposer",
            }

        # Add consensus evaluation task
        if suitable_agents:
            coordinator = suitable_agents[0]  # First agent as coordinator
            sub_tasks["consensus_evaluation"] = {
                "description": "Evaluate proposals and reach consensus",
                "assigned_agent": coordinator.agent_name,
                "required_capabilities": ["analysis", "system_design"],
                "depends_on": [
                    f"consensus_proposal_{i}" for i in range(len(suitable_agents))
                ],
                "estimated_duration": 180,
                "priority": context.priority,
                "consensus_role": "evaluator",
            }

        return sub_tasks

    async def _decompose_competitive(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for competitive execution."""

        sub_tasks = {}

        # Select multiple agents to compete
        required_caps = context.metadata.get("required_capabilities", [])

        if required_caps:
            suitable_agents = [
                agent
                for agent in available_agents
                if any(cap in agent.capabilities for cap in required_caps)
            ][:3]  # Limit to 3 competing agents
        else:
            # Use all available agents if no specific requirements
            suitable_agents = available_agents[:3]

        for i, agent in enumerate(suitable_agents):
            sub_tasks[f"competitive_solution_{i}"] = {
                "description": f"Develop competing solution for {context.title}",
                "assigned_agent": agent.agent_name,
                "required_capabilities": agent.capabilities,
                "depends_on": [],
                "estimated_duration": 300,
                "priority": context.priority,
                "competitive_round": 1,
            }

        # Add evaluation task
        evaluator_agents = [
            agent
            for agent in available_agents
            if any(
                cap in ["testing", "quality_assurance", "analysis", "system_design"]
                for cap in agent.capabilities
            )
        ]

        if not evaluator_agents:
            evaluator_agents = available_agents  # Fallback to any agent

        if evaluator_agents:
            evaluator = evaluator_agents[0]
            sub_tasks["solution_evaluation"] = {
                "description": "Evaluate competing solutions and select best",
                "assigned_agent": evaluator.agent_name,
                "required_capabilities": ["testing", "analysis"],
                "depends_on": [
                    f"competitive_solution_{i}" for i in range(len(suitable_agents))
                ],
                "estimated_duration": 120,
                "priority": context.priority,
                "evaluation_criteria": context.metadata.get(
                    "evaluation_criteria", ["performance", "quality", "maintainability"]
                ),
            }

        return sub_tasks

    async def _decompose_competitive(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for competitive execution."""

        sub_tasks = {}

        # Select multiple agents to compete
        required_caps = context.metadata.get("required_capabilities", [])
        suitable_agents = [
            agent
            for agent in available_agents
            if any(cap in agent.capabilities for cap in required_caps)
        ][:3]  # Limit to 3 competing agents

        for i, agent in enumerate(suitable_agents):
            sub_tasks[f"competitive_solution_{i}"] = {
                "description": f"Develop competing solution for {context.title}",
                "assigned_agent": agent.agent_name,
                "required_capabilities": required_caps,
                "depends_on": [],
                "estimated_duration": 300,
                "priority": context.priority,
                "competitive_round": 1,
            }

        # Add evaluation task
        evaluator_agents = [
            agent
            for agent in available_agents
            if "evaluation" in agent.capabilities or "qa" in agent.capabilities
        ]

        if evaluator_agents:
            evaluator = evaluator_agents[0]
            sub_tasks["solution_evaluation"] = {
                "description": "Evaluate competing solutions and select best",
                "assigned_agent": evaluator.agent_name,
                "required_capabilities": ["evaluation", "analysis"],
                "depends_on": [
                    f"competitive_solution_{i}" for i in range(len(suitable_agents))
                ],
                "estimated_duration": 120,
                "priority": context.priority,
                "evaluation_criteria": context.metadata.get("evaluation_criteria", []),
            }

        return sub_tasks

    async def _decompose_delegation(
        self, context: CollaborationContext, available_agents: list[AgentCapabilityMap]
    ) -> dict[str, dict[str, Any]]:
        """Decompose task for delegation pattern."""

        sub_tasks = {}

        # Coordinator delegates to specialists
        coordinator = context.coordinator_agent

        # Example delegation for system maintenance
        delegation_tasks = [
            (
                "database_optimization",
                ["devops", "database"],
                "Optimize database performance",
            ),
            ("security_patches", ["devops", "security"], "Apply security patches"),
            ("code_cleanup", ["developer"], "Clean up and refactor code"),
            ("monitoring_setup", ["devops", "monitoring"], "Set up monitoring"),
        ]

        for task_name, required_caps, description in delegation_tasks:
            suitable_agents = [
                agent
                for agent in available_agents
                if any(cap in agent.capabilities for cap in required_caps)
                and agent.agent_name != coordinator
            ]

            if suitable_agents:
                selected_agent = min(suitable_agents, key=lambda a: a.load_factor)

                sub_tasks[task_name] = {
                    "description": description,
                    "assigned_agent": selected_agent.agent_name,
                    "required_capabilities": required_caps,
                    "depends_on": [],
                    "estimated_duration": 200,
                    "priority": context.priority,
                    "delegated_by": coordinator,
                }

        return sub_tasks


class CollaborationCoordinator:
    """Coordinates complex multi-agent collaborations."""

    def __init__(self, message_broker: MessageBroker):
        self.message_broker = message_broker
        self.active_collaborations: dict[str, CollaborationContext] = {}
        self.task_decomposer = TaskDecomposer()
        self.coordination_handlers = {
            "collaboration_request": self._handle_collaboration_request,
            "sub_task_completed": self._handle_sub_task_completed,
            "agent_status_update": self._handle_agent_status_update,
            "collaboration_update": self._handle_collaboration_update,
        }

    async def initialize(self):
        """Initialize the coordination system."""
        # Register message handlers
        for topic in self.coordination_handlers:
            await self.message_broker.start_listening(
                "coordination_system", self._create_handler(topic)
            )

        logger.info("Collaboration coordinator initialized")

    def _create_handler(self, topic: str):
        """Create a message handler for a specific topic."""

        async def handler(message):
            handler_func = self.coordination_handlers.get(topic)
            if handler_func:
                return await handler_func(message)
            return None

        return handler

    async def start_collaboration(
        self,
        title: str,
        description: str,
        collaboration_type: CollaborationType,
        coordinator_agent: str,
        required_capabilities: list[str] = None,
        deadline: datetime | None = None,
        priority: int = 5,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Start a new collaborative task."""

        context = CollaborationContext(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            collaboration_type=collaboration_type,
            phase=TaskPhase.PLANNING,
            coordinator_agent=coordinator_agent,
            deadline=deadline.timestamp() if deadline else None,
            priority=priority,
            metadata=metadata or {},
        )

        if required_capabilities:
            context.metadata["required_capabilities"] = required_capabilities

        # Get available agents
        available_agents = await self._get_available_agents()

        # Decompose task into sub-tasks
        sub_tasks = await self.task_decomposer.decompose_task(context, available_agents)
        context.sub_tasks = sub_tasks

        # Extract participating agents
        context.participating_agents = list(
            set(
                task["assigned_agent"]
                for task in sub_tasks.values()
                if "assigned_agent" in task
            )
        )

        # Store collaboration context
        self.active_collaborations[context.id] = context

        # Transition to assignment phase
        context.phase = TaskPhase.ASSIGNMENT

        # Notify participating agents
        await self._notify_collaboration_start(context)

        logger.info(
            "Collaboration started",
            collaboration_id=context.id,
            type=collaboration_type.value,
            participants=len(context.participating_agents),
        )

        return context.id

    async def _get_available_agents(self) -> list[AgentCapabilityMap]:
        """Get list of available agents with their capabilities."""

        # This would typically query the orchestrator
        # For now, simulate with basic agent info
        available_agents = []

        # Get agents from orchestrator (simplified)
        agents = await orchestrator.registry.list_agents()

        for agent in agents:
            if agent.status.value in ["idle", "active"]:
                capability_map = AgentCapabilityMap(
                    agent_name=agent.name,
                    capabilities=agent.capabilities,
                    specializations=agent.capabilities,  # Simplified
                    load_factor=agent.load_factor,
                    availability=True,
                )
                available_agents.append(capability_map)

        return available_agents

    async def _notify_collaboration_start(self, context: CollaborationContext):
        """Notify all participating agents about the collaboration."""

        for agent_name in context.participating_agents:
            await self.message_broker.send_message(
                from_agent="coordination_system",
                to_agent=agent_name,
                topic="collaboration_invitation",
                payload={
                    "collaboration_id": context.id,
                    "title": context.title,
                    "description": context.description,
                    "collaboration_type": context.collaboration_type.value,
                    "your_tasks": [
                        task
                        for task_id, task in context.sub_tasks.items()
                        if task.get("assigned_agent") == agent_name
                    ],
                    "coordinator": context.coordinator_agent,
                    "deadline": context.deadline,
                    "priority": context.priority,
                },
                message_type=MessageType.NOTIFICATION,
            )

    async def _handle_collaboration_request(self, message):
        """Handle requests to start new collaborations."""

        payload = message.payload

        collaboration_id = await self.start_collaboration(
            title=payload["title"],
            description=payload["description"],
            collaboration_type=CollaborationType(payload["collaboration_type"]),
            coordinator_agent=message.from_agent,
            required_capabilities=payload.get("required_capabilities"),
            deadline=datetime.fromisoformat(payload["deadline"])
            if payload.get("deadline")
            else None,
            priority=payload.get("priority", 5),
            metadata=payload.get("metadata", {}),
        )

        return {"collaboration_id": collaboration_id, "status": "started"}

    async def _handle_sub_task_completed(self, message):
        """Handle completion of sub-tasks."""

        payload = message.payload
        collaboration_id = payload["collaboration_id"]
        task_id = payload["task_id"]
        result = payload["result"]

        if collaboration_id not in self.active_collaborations:
            logger.warning("Unknown collaboration", collaboration_id=collaboration_id)
            return {"error": "Unknown collaboration"}

        context = self.active_collaborations[collaboration_id]

        # Store result
        context.results[task_id] = {
            "result": result,
            "completed_by": message.from_agent,
            "completed_at": time.time(),
        }

        # Check if all dependencies are met for other tasks
        await self._check_task_dependencies(context)

        # Check if collaboration is complete
        if len(context.results) == len(context.sub_tasks):
            await self._complete_collaboration(context)

        return {"status": "acknowledged"}

    async def _check_task_dependencies(self, context: CollaborationContext):
        """Check and notify agents when their task dependencies are met."""

        completed_tasks = set(context.results.keys())

        for task_id, task in context.sub_tasks.items():
            if task_id in completed_tasks:
                continue  # Already completed

            dependencies = task.get("depends_on", [])
            if all(dep in completed_tasks for dep in dependencies):
                # Dependencies met, notify assigned agent
                assigned_agent = task.get("assigned_agent")
                if assigned_agent:
                    await self.message_broker.send_message(
                        from_agent="coordination_system",
                        to_agent=assigned_agent,
                        topic="task_ready",
                        payload={
                            "collaboration_id": context.id,
                            "task_id": task_id,
                            "task": task,
                            "dependency_results": {
                                dep: context.results[dep] for dep in dependencies
                            },
                        },
                        message_type=MessageType.NOTIFICATION,
                    )

    async def _complete_collaboration(self, context: CollaborationContext):
        """Complete a collaboration and notify all participants."""

        context.phase = TaskPhase.COMPLETION

        # Aggregate results
        final_result = {
            "collaboration_id": context.id,
            "title": context.title,
            "completed_at": time.time(),
            "duration": time.time() - context.created_at,
            "participating_agents": context.participating_agents,
            "sub_task_results": context.results,
            "success": True,
        }

        # Notify coordinator
        await self.message_broker.send_message(
            from_agent="coordination_system",
            to_agent=context.coordinator_agent,
            topic="collaboration_completed",
            payload=final_result,
            message_type=MessageType.NOTIFICATION,
        )

        # Notify all participants
        for agent_name in context.participating_agents:
            await self.message_broker.send_message(
                from_agent="coordination_system",
                to_agent=agent_name,
                topic="collaboration_completed",
                payload=final_result,
                message_type=MessageType.NOTIFICATION,
            )

        # Clean up
        del self.active_collaborations[context.id]

        logger.info(
            "Collaboration completed",
            collaboration_id=context.id,
            duration=final_result["duration"],
            participants=len(context.participating_agents),
        )

    async def _handle_agent_status_update(self, message):
        """Handle agent status updates that might affect collaborations."""

        payload = message.payload
        agent_name = message.from_agent
        status = payload["status"]

        if status in ["error", "stopped", "stopping"]:
            # Agent became unavailable, handle gracefully
            await self._handle_agent_unavailable(agent_name)

        return {"status": "acknowledged"}

    async def _handle_agent_unavailable(self, agent_name: str):
        """Handle when an agent becomes unavailable during collaboration."""

        affected_collaborations = [
            context
            for context in self.active_collaborations.values()
            if agent_name in context.participating_agents
        ]

        for context in affected_collaborations:
            # Find tasks assigned to this agent
            incomplete_tasks = [
                task_id
                for task_id, task in context.sub_tasks.items()
                if task.get("assigned_agent") == agent_name
                and task_id not in context.results
            ]

            if incomplete_tasks:
                # Try to reassign tasks to other available agents
                available_agents = await self._get_available_agents()

                for task_id in incomplete_tasks:
                    task = context.sub_tasks[task_id]
                    required_caps = task.get("required_capabilities", [])

                    suitable_agents = [
                        agent
                        for agent in available_agents
                        if any(cap in agent.capabilities for cap in required_caps)
                        and agent.agent_name != agent_name
                    ]

                    if suitable_agents:
                        # Reassign to best available agent
                        new_agent = min(suitable_agents, key=lambda a: a.load_factor)
                        task["assigned_agent"] = new_agent.agent_name

                        # Notify new agent
                        await self.message_broker.send_message(
                            from_agent="coordination_system",
                            to_agent=new_agent.agent_name,
                            topic="task_reassigned",
                            payload={
                                "collaboration_id": context.id,
                                "task_id": task_id,
                                "task": task,
                                "reason": f"Original agent {agent_name} became unavailable",
                            },
                            message_type=MessageType.NOTIFICATION,
                        )

                        logger.info(
                            "Task reassigned",
                            collaboration_id=context.id,
                            task_id=task_id,
                            from_agent=agent_name,
                            to_agent=new_agent.agent_name,
                        )
                    else:
                        # No suitable replacement, mark collaboration as failed
                        context.phase = TaskPhase.FAILED
                        await self._fail_collaboration(
                            context,
                            f"Agent {agent_name} unavailable and no replacement found",
                        )

    async def _fail_collaboration(self, context: CollaborationContext, reason: str):
        """Fail a collaboration and notify participants."""

        context.phase = TaskPhase.FAILED

        failure_result = {
            "collaboration_id": context.id,
            "title": context.title,
            "failed_at": time.time(),
            "reason": reason,
            "participating_agents": context.participating_agents,
            "completed_tasks": len(context.results),
            "total_tasks": len(context.sub_tasks),
            "success": False,
        }

        # Notify coordinator
        await self.message_broker.send_message(
            from_agent="coordination_system",
            to_agent=context.coordinator_agent,
            topic="collaboration_failed",
            payload=failure_result,
            message_type=MessageType.NOTIFICATION,
        )

        # Notify participants
        for agent_name in context.participating_agents:
            await self.message_broker.send_message(
                from_agent="coordination_system",
                to_agent=agent_name,
                topic="collaboration_failed",
                payload=failure_result,
                message_type=MessageType.NOTIFICATION,
            )

        logger.error("Collaboration failed", collaboration_id=context.id, reason=reason)

    async def _handle_collaboration_update(self, message):
        """Handle updates to ongoing collaborations."""

        payload = message.payload
        collaboration_id = payload["collaboration_id"]

        if collaboration_id not in self.active_collaborations:
            return {"error": "Unknown collaboration"}

        context = self.active_collaborations[collaboration_id]

        # Handle different types of updates
        update_type = payload.get("update_type")

        if update_type == "priority_change":
            context.priority = payload["new_priority"]
        elif update_type == "deadline_change":
            context.deadline = payload["new_deadline"]
        elif update_type == "add_agent":
            new_agent = payload["agent_name"]
            if new_agent not in context.participating_agents:
                context.participating_agents.append(new_agent)

        return {"status": "updated"}

    async def get_collaboration_status(self, collaboration_id: str) -> dict[str, Any]:
        """Get status of a collaboration."""

        if collaboration_id not in self.active_collaborations:
            return {"error": "Collaboration not found"}

        context = self.active_collaborations[collaboration_id]

        return {
            "id": context.id,
            "title": context.title,
            "phase": context.phase.value,
            "collaboration_type": context.collaboration_type.value,
            "coordinator": context.coordinator_agent,
            "participating_agents": context.participating_agents,
            "total_tasks": len(context.sub_tasks),
            "completed_tasks": len(context.results),
            "progress": len(context.results) / len(context.sub_tasks)
            if context.sub_tasks
            else 0,
            "created_at": context.created_at,
            "deadline": context.deadline,
            "priority": context.priority,
        }


# Global coordination instance
coordination_system = CollaborationCoordinator(message_broker)
