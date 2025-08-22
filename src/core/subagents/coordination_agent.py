"""
Coordination Agent - Manages subagent communication and synchronization.

This agent coordinates the execution of complex multi-agent workflows, handling
communication between subagents, managing state synchronization, and ensuring
proper execution order and error handling.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import structlog

from ..constants import Intervals
from ..message_broker import MessageType, message_broker
from .task_decomposition_agent import SubTask, DecompositionResult

logger = structlog.get_logger()


class CoordinationState(Enum):
    """States of the coordination process."""
    INITIALIZING = "initializing"
    EXECUTING = "executing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"


class SubagentStatus(Enum):
    """Status of individual subagents."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    ROLLED_BACK = "rolled_back"


@dataclass
class SubagentExecution:
    """Tracks the execution of a subagent."""
    subtask_id: str
    agent_id: Optional[str] = None
    status: SubagentStatus = SubagentStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class CoordinationContext:
    """Context for coordinating subagent execution."""
    workflow_id: str
    decomposition: DecompositionResult
    execution_plan: List[SubagentExecution] = field(default_factory=list)
    state: CoordinationState = CoordinationState.INITIALIZING
    start_time: float = field(default_factory=time.time)
    current_step: int = 0
    completed_subtasks: Set[str] = field(default_factory=set)
    failed_subtasks: Set[str] = field(default_factory=set)
    rollback_actions: List[Dict[str, Any]] = field(default_factory=list)


class CoordinationAgent:
    """Agent responsible for coordinating complex multi-agent workflows."""

    def __init__(self):
        self.active_coordinations: Dict[str, CoordinationContext] = {}
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        self.coordination_metrics = {
            "workflows_started": 0,
            "workflows_completed": 0,
            "workflows_failed": 0,
            "subtasks_executed": 0,
            "coordination_efficiency": 0.0
        }

    async def initialize(self):
        """Initialize the coordination agent."""
        await self._load_agent_registry()
        logger.info("Coordination agent initialized")

    async def _load_agent_registry(self):
        """Load available agents and their capabilities."""
        # This would typically query the agent registry
        # For now, we'll use a mock registry
        self.agent_registry = {
            "architect_agent": {
                "capabilities": ["architecture", "design", "system_thinking"],
                "status": "available",
                "specializations": ["web_apps", "microservices", "data_processing"]
            },
            "backend_agent": {
                "capabilities": ["coding", "development", "api_design"],
                "status": "available",
                "specializations": ["python", "fastapi", "postgresql"]
            },
            "frontend_agent": {
                "capabilities": ["ui_development", "responsive_design"],
                "status": "available",
                "specializations": ["react", "typescript", "pwa"]
            },
            "qa_agent": {
                "capabilities": ["testing", "quality_assurance", "debugging"],
                "status": "available",
                "specializations": ["unit_tests", "integration_tests", "performance_tests"]
            },
            "devops_agent": {
                "capabilities": ["deployment", "infrastructure", "monitoring"],
                "status": "available",
                "specializations": ["docker", "kubernetes", "ci_cd"]
            }
        }

    async def coordinate_workflow(self, decomposition: DecompositionResult) -> str:
        """Start coordinating a complex workflow based on task decomposition."""
        workflow_id = f"workflow_{uuid.uuid4().hex[:12]}"

        # Create coordination context
        context = CoordinationContext(
            workflow_id=workflow_id,
            decomposition=decomposition
        )

        # Initialize execution plan
        await self._create_execution_plan(context)

        # Store coordination context
        self.active_coordinations[workflow_id] = context

        # Start execution
        asyncio.create_task(self._execute_workflow(context))

        self.coordination_metrics["workflows_started"] += 1

        logger.info(
            "Workflow coordination started",
            workflow_id=workflow_id,
            subtask_count=len(decomposition.subtasks),
            complexity=decomposition.complexity.value
        )

        return workflow_id

    async def _create_execution_plan(self, context: CoordinationContext):
        """Create an optimized execution plan for the subtasks."""
        decomposition = context.decomposition

        # Create execution entries for each subtask
        for subtask in decomposition.subtasks:
            execution = SubagentExecution(subtask_id=subtask.id)
            context.execution_plan.append(execution)

        # Optimize execution order based on dependencies
        await self._optimize_execution_order(context)

    async def _optimize_execution_order(self, context: CoordinationContext):
        """Optimize the execution order based on dependencies and resource availability."""
        # Simple topological sort for now
        # In a more advanced implementation, this would consider:
        # - Agent availability and load
        # - Resource constraints
        # - Estimated execution times
        # - Parallel execution opportunities

        # For now, we'll execute in the order provided by the decomposition
        # with basic dependency checking
        pass

    async def _execute_workflow(self, context: CoordinationContext):
        """Execute the workflow by coordinating subagents."""
        context.state = CoordinationState.EXECUTING

        try:
            while context.current_step < len(context.execution_plan):
                execution = context.execution_plan[context.current_step]

                # Check if dependencies are satisfied
                if await self._check_dependencies(context, execution):
                    # Assign agent and execute
                    await self._assign_and_execute_subtask(context, execution)
                    context.current_step += 1
                else:
                    # Wait for dependencies
                    context.state = CoordinationState.WAITING
                    await asyncio.sleep(1)  # Brief wait before checking again

            # All subtasks completed
            context.state = CoordinationState.COMPLETED
            self.coordination_metrics["workflows_completed"] += 1

            logger.info(
                "Workflow completed successfully",
                workflow_id=context.workflow_id,
                completed_subtasks=len(context.completed_subtasks)
            )

        except Exception as e:
            context.state = CoordinationState.ERROR
            self.coordination_metrics["workflows_failed"] += 1

            logger.error(
                "Workflow execution failed",
                workflow_id=context.workflow_id,
                error=str(e)
            )

            # Initiate rollback
            await self._rollback_workflow(context)

    async def _check_dependencies(self, context: CoordinationContext, execution: SubagentExecution) -> bool:
        """Check if all dependencies for this execution are satisfied."""
        # Find the subtask
        subtask = next(
            (st for st in context.decomposition.subtasks if st.id == execution.subtask_id),
            None
        )

        if not subtask:
            return False

        # Check if all dependencies are completed
        return all(dep_id in context.completed_subtasks for dep_id in subtask.dependencies)

    async def _assign_and_execute_subtask(self, context: CoordinationContext, execution: SubagentExecution):
        """Assign an appropriate agent and execute the subtask."""
        # Find the subtask
        subtask = next(
            (st for st in context.decomposition.subtasks if st.id == execution.subtask_id),
            None
        )

        if not subtask:
            raise ValueError(f"Subtask {execution.subtask_id} not found")

        # Find suitable agent
        agent_id = await self._find_suitable_agent(subtask)

        if not agent_id:
            raise ValueError(f"No suitable agent found for subtask {execution.subtask_id}")

        # Assign agent
        execution.agent_id = agent_id
        execution.status = SubagentStatus.ASSIGNED
        execution.start_time = time.time()

        # Execute subtask
        await self._execute_subtask(context, execution, subtask)

    async def _find_suitable_agent(self, subtask: SubTask) -> Optional[str]:
        """Find an agent suitable for executing the given subtask."""
        best_agent = None
        best_score = 0

        for agent_id, agent_info in self.agent_registry.items():
            if agent_info["status"] != "available":
                continue

            # Calculate match score based on capabilities
            score = 0
            agent_capabilities = set(agent_info["capabilities"])

            for requirement in subtask.agent_requirements:
                if requirement in agent_capabilities:
                    score += 1

            # Bonus for specialization match
            agent_specializations = set(agent_info.get("specializations", []))
            for requirement in subtask.agent_requirements:
                if requirement in agent_specializations:
                    score += 2

            if score > best_score:
                best_score = score
                best_agent = agent_id

        return best_agent

    async def _execute_subtask(self, context: CoordinationContext, execution: SubagentExecution, subtask: SubTask):
        """Execute a subtask with proper error handling and retries."""
        execution.status = SubagentStatus.EXECUTING

        while execution.retry_count <= execution.max_retries:
            try:
                # Send subtask to assigned agent
                result = await self._send_subtask_to_agent(
                    execution.agent_id,
                    subtask,
                    context.workflow_id
                )

                if result.get("success", False):
                    execution.status = SubagentStatus.COMPLETED
                    execution.result = result
                    execution.end_time = time.time()

                    context.completed_subtasks.add(subtask.id)
                    self.coordination_metrics["subtasks_executed"] += 1

                    logger.info(
                        "Subtask completed successfully",
                        workflow_id=context.workflow_id,
                        subtask_id=subtask.id,
                        agent_id=execution.agent_id
                    )

                    return
                else:
                    raise Exception(f"Subtask execution failed: {result.get('error', 'Unknown error')}")

            except Exception as e:
                execution.retry_count += 1
                execution.error = str(e)

                if execution.retry_count > execution.max_retries:
                    execution.status = SubagentStatus.FAILED
                    context.failed_subtasks.add(subtask.id)

                    logger.error(
                        "Subtask failed after max retries",
                        workflow_id=context.workflow_id,
                        subtask_id=subtask.id,
                        agent_id=execution.agent_id,
                        error=str(e),
                        retry_count=execution.retry_count
                    )

                    raise
                else:
                    execution.status = SubagentStatus.RETRYING
                    logger.warning(
                        "Subtask execution failed, retrying",
                        workflow_id=context.workflow_id,
                        subtask_id=subtask.id,
                        agent_id=execution.agent_id,
                        error=str(e),
                        retry_count=execution.retry_count
                    )

                    # Wait before retry
                    await asyncio.sleep(Intervals.TASK_COORDINATION_CYCLE)

    async def _send_subtask_to_agent(self, agent_id: str, subtask: SubTask, workflow_id: str) -> Dict[str, Any]:
        """Send a subtask to the assigned agent for execution."""
        message_payload = {
            "workflow_id": workflow_id,
            "subtask": {
                "id": subtask.id,
                "type": subtask.type.value,
                "title": subtask.title,
                "description": subtask.description,
                "context_requirements": subtask.context_requirements,
                "estimated_effort": subtask.estimated_effort
            },
            "coordination_context": {
                "timestamp": time.time(),
                "coordinator": "coordination_agent"
            }
        }

        try:
            # Send message to agent
            result = await message_broker.send_message_with_result(
                from_agent="coordination_agent",
                to_agent=agent_id,
                topic="execute_subtask",
                payload=message_payload
            )

            if result.success:
                # Wait for agent response (in a real implementation, this would be async)
                # For now, simulate successful execution
                return {
                    "success": True,
                    "result": "Subtask executed successfully",
                    "execution_time": subtask.estimated_effort * 60  # Convert to seconds
                }
            else:
                return {
                    "success": False,
                    "error": result.details or "Agent execution failed"
                }

        except Exception as e:
            return {
                "success": False,
                "error": f"Communication error: {str(e)}"
            }

    async def _rollback_workflow(self, context: CoordinationContext):
        """Rollback a failed workflow."""
        context.state = CoordinationState.ROLLING_BACK

        logger.info(
            "Starting workflow rollback",
            workflow_id=context.workflow_id
        )

        # Rollback in reverse order
        for execution in reversed(context.execution_plan):
            if execution.status == SubagentStatus.COMPLETED:
                try:
                    await self._rollback_subtask(context, execution)
                    execution.status = SubagentStatus.ROLLED_BACK
                except Exception as e:
                    logger.error(
                        "Rollback failed for subtask",
                        workflow_id=context.workflow_id,
                        subtask_id=execution.subtask_id,
                        error=str(e)
                    )

        context.state = CoordinationState.ERROR

    async def _rollback_subtask(self, context: CoordinationContext, execution: SubagentExecution):
        """Rollback a specific subtask."""
        # Send rollback message to the agent that executed the subtask
        rollback_payload = {
            "workflow_id": context.workflow_id,
            "subtask_id": execution.subtask_id,
            "rollback_reason": "workflow_failure"
        }

        await message_broker.send_message(
            from_agent="coordination_agent",
            to_agent=execution.agent_id,
            topic="rollback_subtask",
            payload=rollback_payload,
            message_type=MessageType.NOTIFICATION
        )

    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow."""
        context = self.active_coordinations.get(workflow_id)
        if not context:
            return None

        return {
            "workflow_id": workflow_id,
            "state": context.state.value,
            "progress": f"{len(context.completed_subtasks)}/{len(context.decomposition.subtasks)}",
            "completed_subtasks": list(context.completed_subtasks),
            "failed_subtasks": list(context.failed_subtasks),
            "current_step": context.current_step,
            "start_time": context.start_time,
            "duration": time.time() - context.start_time
        }

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        context = self.active_coordinations.get(workflow_id)
        if not context:
            return False

        if context.state not in [CoordinationState.EXECUTING, CoordinationState.WAITING]:
            return False

        # Mark workflow as cancelled
        context.state = CoordinationState.ERROR

        # Cancel all running subtasks
        for execution in context.execution_plan:
            if execution.status in [SubagentStatus.ASSIGNED, SubagentStatus.EXECUTING]:
                execution.status = SubagentStatus.FAILED
                execution.error = "Workflow cancelled"

        logger.info("Workflow cancelled", workflow_id=workflow_id)
        return True

    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get coordination performance metrics."""
        total_workflows = (
            self.coordination_metrics["workflows_completed"] +
            self.coordination_metrics["workflows_failed"]
        )

        if total_workflows > 0:
            success_rate = (
                self.coordination_metrics["workflows_completed"] / total_workflows
            )
            self.coordination_metrics["coordination_efficiency"] = success_rate

        return {
            **self.coordination_metrics,
            "active_workflows": len(self.active_coordinations),
            "available_agents": len([a for a in self.agent_registry.values() if a["status"] == "available"])
        }

    async def update_agent_status(self, agent_id: str, status: str, capabilities: Optional[List[str]] = None):
        """Update the status and capabilities of an agent."""
        if agent_id not in self.agent_registry:
            self.agent_registry[agent_id] = {
                "capabilities": capabilities or [],
                "status": status,
                "specializations": []
            }
        else:
            self.agent_registry[agent_id]["status"] = status
            if capabilities:
                self.agent_registry[agent_id]["capabilities"] = capabilities

        logger.info(
            "Agent status updated",
            agent_id=agent_id,
            status=status,
            capabilities=capabilities
        )