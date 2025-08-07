"""FastAPI server for LeanVibe Agent Hive 2.0 - Agent management and coordination."""

import sys
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Handle both module and direct execution imports
try:
    from ..core.config import settings
    from ..core.message_broker import MessageType, message_broker
    from ..core.models import SystemMetric, get_database_manager
    from ..core.orchestrator import get_orchestrator
    from ..core.task_queue import Task, TaskPriority, task_queue
except ImportError:
    # Direct execution - add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from core.config import settings
    from core.message_broker import MessageType, message_broker
    from core.models import SystemMetric, get_database_manager
    from core.orchestrator import get_orchestrator
    from core.task_queue import Task, TaskPriority, task_queue

logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="LeanVibe Agent Hive 2.0",
    description="Multi-agent autonomous development system",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for API
class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"


class AgentInfo(BaseModel):
    name: str
    type: str
    role: str
    status: AgentStatus
    capabilities: dict[str, Any]
    last_heartbeat: float | None = None
    tasks_completed: int = 0
    tasks_failed: int = 0
    uptime: float = 0.0


class TaskCreate(BaseModel):
    title: str
    description: str
    type: str
    priority: TaskPriority = TaskPriority.NORMAL
    assigned_to: str | None = None
    dependencies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskInfo(BaseModel):
    id: str
    title: str
    description: str
    type: str
    status: str
    priority: TaskPriority
    assigned_to: str | None = None
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class SystemStatus(BaseModel):
    health_score: float
    active_agents: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    queue_depth: int
    uptime: float
    last_analysis: float | None = None


class MessageSend(BaseModel):
    to_agent: str
    topic: str
    content: dict[str, Any]
    message_type: MessageType = MessageType.DIRECT


class APIResponse(BaseModel):
    success: bool
    data: Any | None = None
    error: str | None = None
    timestamp: float = Field(default_factory=time.time)
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting LeanVibe Agent Hive API server")

    try:
        # Initialize message broker
        await message_broker.initialize()

        # Initialize orchestrator
        from pathlib import Path

        orch = await get_orchestrator(
            "postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive", Path(".")
        )
        await orch.start()

        logger.info("API server started successfully")

    except Exception as e:
        logger.error("Failed to start API server", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down LeanVibe Agent Hive API server")

    try:
        orch = await get_orchestrator(
            "postgresql://hive_user:hive_pass@localhost:5433/leanvibe_hive", Path(".")
        )
        await orch.stop()
        await message_broker.shutdown()

        logger.info("API server shut down successfully")

    except Exception as e:
        logger.error("Error during API server shutdown", error=str(e))


# Health check endpoints
@app.get("/health", response_model=APIResponse)
async def health_check():
    """Basic health check."""
    return APIResponse(
        success=True, data={"status": "healthy", "service": "agent-hive-api"}
    )


@app.get("/api/v1/status", response_model=APIResponse)
async def get_system_status():
    """Get comprehensive system status."""
    try:
        # Get basic metrics
        total_tasks = await task_queue.get_total_tasks()
        completed_tasks = await task_queue.get_completed_tasks()
        failed_tasks = await task_queue.get_failed_tasks()
        queue_depth = await task_queue.get_queue_depth()

        # Get active agents
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        active_agents = await orchestrator.get_active_agent_count()

        # Calculate health score (simplified)
        health_score = 1.0
        if total_tasks > 0:
            success_rate = completed_tasks / total_tasks
            health_score = success_rate * 0.8 + (0.2 if active_agents > 0 else 0.0)

        status = SystemStatus(
            health_score=health_score,
            active_agents=active_agents,
            total_tasks=total_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            queue_depth=queue_depth,
            uptime=time.time() - startup_time,
            last_analysis=None,  # Would get from meta-agent
        )

        return APIResponse(success=True, data=status.dict())

    except Exception as e:
        logger.error("Failed to get system status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Agent management endpoints
@app.get("/api/v1/agents", response_model=APIResponse)
async def list_agents():
    """List all agents."""
    try:
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        agents = await orchestrator.list_agents()
        agent_info = []

        for agent in agents:
            info = AgentInfo(
                name=agent.name,
                type=agent.type,
                role=agent.role,
                status=AgentStatus(agent.status),
                capabilities=agent.capabilities or {},
                last_heartbeat=agent.last_heartbeat,
                uptime=time.time() - agent.created_at if agent.created_at else 0.0,
            )
            agent_info.append(info.dict())

        return APIResponse(success=True, data=agent_info)

    except Exception as e:
        logger.error("Failed to list agents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/agents/{agent_name}", response_model=APIResponse)
async def get_agent(agent_name: str):
    """Get specific agent information."""
    try:
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        agent = await orchestrator.get_agent(agent_name)
        if not agent:
            raise HTTPException(status_code=404, detail="Agent not found")

        info = AgentInfo(
            name=agent.name,
            type=agent.type,
            role=agent.role,
            status=AgentStatus(agent.status),
            capabilities=agent.capabilities or {},
            last_heartbeat=agent.last_heartbeat,
            uptime=time.time() - agent.created_at if agent.created_at else 0.0,
        )

        return APIResponse(success=True, data=info.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get agent", agent_name=agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/agents", response_model=APIResponse)
async def spawn_agent(
    agent_type: str,
    agent_name: str | None = None,
    background_tasks: BackgroundTasks = None,
):
    """Spawn a new agent."""
    try:
        if not agent_name:
            agent_name = f"{agent_type}-{int(time.time())}"

        # Spawn agent asynchronously
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        session_name = await orchestrator.spawn_agent(agent_type, agent_name)

        return APIResponse(
            success=True,
            data={
                "agent_name": agent_name,
                "agent_type": agent_type,
                "session_name": session_name,
                "status": "spawning",
            },
        )

    except Exception as e:
        logger.error("Failed to spawn agent", agent_type=agent_type, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/agents/{agent_name}/stop", response_model=APIResponse)
async def stop_agent(agent_name: str):
    """Stop a specific agent."""
    try:
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        success = await orchestrator.stop_agent(agent_name)

        if success:
            return APIResponse(
                success=True, data={"agent_name": agent_name, "status": "stopping"}
            )
        else:
            raise HTTPException(status_code=404, detail="Agent not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to stop agent", agent_name=agent_name, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/agents/{agent_name}/health", response_model=APIResponse)
async def check_agent_health(agent_name: str):
    """Check agent health."""
    try:
        # Send health check message to agent
        response = await message_broker.send_message(
            from_agent="api-server",
            to_agent=agent_name,
            topic="health_check",
            payload={"timestamp": time.time()},
            message_type=MessageType.DIRECT,
        )

        return APIResponse(
            success=True, data={"message_id": response, "health_check": "requested"}
        )

    except Exception as e:
        logger.error(
            "Failed to check agent health", agent_name=agent_name, error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


# Task management endpoints
@app.get("/api/v1/tasks", response_model=APIResponse)
async def list_tasks(status: str | None = None, assigned_to: str | None = None):
    """List tasks with optional filtering."""
    try:
        tasks = await task_queue.list_tasks(status=status, assigned_to=assigned_to)
        task_info = []

        for task in tasks:
            info = TaskInfo(
                id=task.id,
                title=task.title,
                description=task.description,
                type=task.task_type,
                status=task.status,
                priority=task.priority,
                assigned_to=task.assigned_to,
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                result=task.result,
                error=task.error,
            )
            task_info.append(info.dict())

        return APIResponse(success=True, data=task_info)

    except Exception as e:
        logger.error("Failed to list tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/tasks", response_model=APIResponse)
async def create_task(task_create: TaskCreate):
    """Create a new task."""
    try:
        task = Task(
            id=str(uuid.uuid4()),
            title=task_create.title,
            description=task_create.description,
            type=task_create.type,
            priority=task_create.priority,
            assigned_to=task_create.assigned_to,
            dependencies=task_create.dependencies,
            metadata=task_create.metadata,
            status="pending",
            created_at=time.time(),
        )

        task_id = await task_queue.add_task(task)

        return APIResponse(
            success=True,
            data={"task_id": task_id, "title": task.title, "status": task.status.value},
        )

    except Exception as e:
        logger.error("Failed to create task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/tasks/{task_id}", response_model=APIResponse)
async def get_task(task_id: str):
    """Get specific task information."""
    try:
        task = await task_queue.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        info = TaskInfo(
            id=task.id,
            title=task.title,
            description=task.description,
            type=task.task_type,
            status=task.status,
            priority=task.priority,
            assigned_to=task.assigned_to,
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result,
            error=task.error,
        )

        return APIResponse(success=True, data=info.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/tasks/{task_id}/cancel", response_model=APIResponse)
async def cancel_task(task_id: str):
    """Cancel a task."""
    try:
        success = await task_queue.cancel_task(task_id)

        if success:
            return APIResponse(
                success=True, data={"task_id": task_id, "status": "cancelled"}
            )
        else:
            raise HTTPException(status_code=404, detail="Task not found")

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel task", task_id=task_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Messaging endpoints
@app.post("/api/v1/messages", response_model=APIResponse)
async def send_message(message: MessageSend):
    """Send a message to an agent."""
    try:
        message_id = await message_broker.send_message(
            from_agent="api-server",
            to_agent=message.to_agent,
            topic=message.topic,
            payload=message.content,
            message_type=message.message_type,
        )

        return APIResponse(
            success=True,
            data={
                "message_id": message_id,
                "to_agent": message.to_agent,
                "topic": message.topic,
            },
        )

    except Exception as e:
        logger.error("Failed to send message", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/broadcast", response_model=APIResponse)
async def broadcast_message(topic: str, content: dict[str, Any]):
    """Broadcast a message to all agents."""
    try:
        message_id = await message_broker.broadcast_message(
            from_agent="api-server", topic=topic, payload=content
        )

        return APIResponse(
            success=True,
            data={"message_id": message_id, "topic": topic, "type": "broadcast"},
        )

    except Exception as e:
        logger.error("Failed to broadcast message", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# System control endpoints
@app.post("/api/v1/system/analyze", response_model=APIResponse)
async def trigger_system_analysis():
    """Trigger a system analysis by the meta-agent."""
    try:
        task = Task(
            id=str(uuid.uuid4()),
            title="System Analysis",
            description="Perform comprehensive system analysis",
            type="system_analysis",
            priority=TaskPriority.HIGH,
            assigned_to="meta-agent",
            status="pending",
            created_at=time.time(),
        )

        task_id = await task_queue.add_task(task)

        return APIResponse(
            success=True,
            data={"task_id": task_id, "description": "System analysis triggered"},
        )

    except Exception as e:
        logger.error("Failed to trigger system analysis", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/system/shutdown", response_model=APIResponse)
async def shutdown_system():
    """Gracefully shutdown the entire system."""
    try:
        # Send shutdown message to all agents
        await message_broker.broadcast_message(
            from_agent="api-server",
            topic="shutdown",
            payload={"reason": "API shutdown request", "timestamp": time.time()},
        )

        return APIResponse(success=True, data={"status": "shutdown_initiated"})

    except Exception as e:
        logger.error("Failed to shutdown system", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Metrics endpoints
@app.get("/api/v1/metrics", response_model=APIResponse)
async def get_metrics():
    """Get system metrics."""
    try:
        db_manager = get_database_manager(settings.database_url)
        db_session = db_manager.get_session()

        try:
            # Get recent metrics
            metrics = (
                db_session.query(SystemMetric)
                .order_by(SystemMetric.timestamp.desc())
                .limit(100)
                .all()
            )

            metric_data = []
            for metric in metrics:
                metric_data.append(
                    {
                        "name": metric.metric_name,
                        "type": metric.metric_type,
                        "value": metric.value,
                        "unit": metric.unit,
                        "agent_id": metric.agent_id,
                        "timestamp": metric.timestamp,
                        "labels": metric.labels,
                    }
                )

            return APIResponse(success=True, data=metric_data)

        finally:
            db_session.close()

    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Performance and optimization endpoints
@app.get("/api/v1/performance", response_model=APIResponse)
async def get_performance_stats():
    """Get performance optimization statistics."""
    try:
        from ..core.performance_optimizer import get_performance_optimizer

        optimizer = get_performance_optimizer()

        # Get current performance metrics
        current_metrics = await optimizer.analyze_performance()

        # Get optimization statistics
        optimization_stats = optimizer.get_optimization_stats()

        # Get workload prediction
        workload_prediction = await optimizer.resource_predictor.predict_workload()

        return APIResponse(
            success=True,
            data={
                "current_metrics": {
                    metric.value: value for metric, value in current_metrics.items()
                },
                "optimization_stats": optimization_stats,
                "workload_prediction": {
                    "predicted_task_count": workload_prediction.predicted_task_count,
                    "predicted_queue_depth": workload_prediction.predicted_queue_depth,
                    "predicted_resource_usage": workload_prediction.predicted_resource_usage,
                    "confidence_interval": workload_prediction.confidence_interval,
                    "timestamp": workload_prediction.timestamp,
                },
            },
        )

    except Exception as e:
        logger.error("Failed to get performance stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/performance/optimize", response_model=APIResponse)
async def trigger_optimization():
    """Trigger performance optimization."""
    try:
        from ..core.performance_optimizer import get_performance_optimizer

        optimizer = get_performance_optimizer()

        # Identify optimization opportunities
        opportunities = await optimizer.identify_optimization_opportunities()

        # Apply the top opportunity if available
        if opportunities:
            top_opportunity = opportunities[0]
            success = await optimizer.apply_optimization(top_opportunity)

            return APIResponse(
                success=True,
                data={
                    "optimization_applied": success,
                    "strategy": top_opportunity.strategy.value,
                    "description": top_opportunity.description,
                    "expected_impact": {
                        metric.value: impact
                        for metric, impact in top_opportunity.expected_impact.items()
                    },
                },
            )
        else:
            return APIResponse(
                success=True,
                data={"message": "No optimization opportunities identified"},
            )

    except Exception as e:
        logger.error("Failed to trigger optimization", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Context and memory endpoints
@app.get("/api/v1/context/{agent_id}", response_model=APIResponse)
async def get_agent_context(agent_id: str, limit: int = 10):
    """Get context/memory for a specific agent."""
    try:
        from ..core.context_engine import get_context_engine

        context_engine = await get_context_engine(settings.database_url)
        memory_stats = await context_engine.get_memory_stats(agent_id)

        return APIResponse(
            success=True,
            data={
                "agent_id": agent_id,
                "total_contexts": memory_stats.total_contexts,
                "contexts_by_importance": memory_stats.contexts_by_importance,
                "contexts_by_category": memory_stats.contexts_by_category,
                "storage_size_mb": memory_stats.storage_size_mb,
                "oldest_context_age_days": memory_stats.oldest_context_age_days,
                "most_accessed_context_id": memory_stats.most_accessed_context_id,
            },
        )

    except Exception as e:
        logger.error("Failed to get agent context", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/context/{agent_id}/consolidate", response_model=APIResponse)
async def consolidate_agent_memory(agent_id: str):
    """Trigger memory consolidation for an agent."""
    try:
        from ..core.context_engine import get_context_engine

        context_engine = await get_context_engine(settings.database_url)
        consolidation_stats = await context_engine.consolidate_memory(agent_id)

        return APIResponse(
            success=True,
            data={"agent_id": agent_id, "consolidation_stats": consolidation_stats},
        )

    except Exception as e:
        logger.error("Failed to consolidate memory", agent_id=agent_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Self-modification endpoints
@app.get("/api/v1/modifications", response_model=APIResponse)
async def get_modification_stats():
    """Get self-modification statistics."""
    try:
        from ..core.self_modifier import get_self_modifier

        modifier = get_self_modifier()
        stats = modifier.get_modification_stats()

        return APIResponse(success=True, data=stats)

    except Exception as e:
        logger.error("Failed to get modification stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/modifications/proposal", response_model=APIResponse)
async def create_modification_proposal(
    title: str,
    description: str,
    modification_type: str,
    file_paths: list[str],
    agent_id: str = "api-user",
):
    """Create a new modification proposal."""
    try:
        from ..core.self_modifier import get_self_modifier, ModificationType, CodeChange

        modifier = get_self_modifier()

        # Create code changes (simplified - in practice would read actual files)
        changes = []
        for file_path in file_paths:
            changes.append(
                CodeChange(
                    file_path=file_path,
                    original_content="",  # Would read actual content
                    modified_content="",  # Would get from user input
                    change_description=f"Modification to {file_path}",
                    line_numbers=[],
                )
            )

        proposal_id = await modifier.propose_modification(
            title=title,
            description=description,
            modification_type=ModificationType(modification_type),
            changes=changes,
            created_by=agent_id,
        )

        return APIResponse(
            success=True,
            data={"proposal_id": proposal_id, "title": title, "status": "proposed"},
        )

    except Exception as e:
        logger.error("Failed to create modification proposal", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/modifications/{proposal_id}/validate", response_model=APIResponse)
async def validate_modification(proposal_id: str):
    """Validate a modification proposal."""
    try:
        from ..core.self_modifier import get_self_modifier

        modifier = get_self_modifier()
        validation_result = await modifier.validate_modification(proposal_id)

        return APIResponse(
            success=True,
            data={
                "proposal_id": proposal_id,
                "overall_success": validation_result.overall_success,
                "tests_passed": validation_result.tests_passed,
                "code_quality_score": validation_result.code_quality_score,
                "security_issues": validation_result.security_issues,
                "issues_found": validation_result.issues_found,
                "recommendations": validation_result.recommendations,
            },
        )

    except Exception as e:
        logger.error(
            "Failed to validate modification", proposal_id=proposal_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


# Git workflow endpoints
@app.get("/api/v1/workflows", response_model=APIResponse)
async def list_workflows():
    """List active development workflows."""
    try:
        from ..core.cli_git_integration import get_enhanced_cli_executor

        cli_executor = get_enhanced_cli_executor()
        workflows = await cli_executor.git_manager.list_active_workflows()

        workflow_data = []
        for workflow in workflows:
            workflow_data.append(
                {
                    "id": workflow.id,
                    "type": workflow.workflow_type.value,
                    "description": workflow.description,
                    "branch_name": workflow.branch_name,
                    "status": workflow.status.value,
                    "created_at": workflow.created_at,
                    "updated_at": workflow.updated_at,
                    "agent_id": workflow.agent_id,
                    "files_modified": workflow.files_modified,
                    "commits": len(workflow.commits),
                }
            )

        return APIResponse(success=True, data=workflow_data)

    except Exception as e:
        logger.error("Failed to list workflows", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/workflows", response_model=APIResponse)
async def create_workflow(
    workflow_type: str, description: str, agent_id: str, target_branch: str = "main"
):
    """Create a new development workflow."""
    try:
        from ..core.cli_git_integration import get_enhanced_cli_executor, WorkflowType

        cli_executor = get_enhanced_cli_executor()
        workflow = await cli_executor.git_manager.create_workflow(
            workflow_type=WorkflowType(workflow_type),
            description=description,
            agent_id=agent_id,
            target_branch=target_branch,
        )

        return APIResponse(
            success=True,
            data={
                "workflow_id": workflow.id,
                "branch_name": workflow.branch_name,
                "status": workflow.status.value,
            },
        )

    except Exception as e:
        logger.error("Failed to create workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/workflows/{workflow_id}/complete", response_model=APIResponse)
async def complete_workflow(workflow_id: str, merge_to_main: bool = False):
    """Complete a development workflow."""
    try:
        from ..core.cli_git_integration import get_enhanced_cli_executor

        cli_executor = get_enhanced_cli_executor()
        success = await cli_executor.git_manager.complete_workflow(
            workflow_id=workflow_id, merge_to_main=merge_to_main
        )

        return APIResponse(
            success=success,
            data={
                "workflow_id": workflow_id,
                "completed": success,
                "merged": merge_to_main,
            },
        )

    except Exception as e:
        logger.error(
            "Failed to complete workflow", workflow_id=workflow_id, error=str(e)
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


# System diagnostics endpoints
@app.get("/api/v1/diagnostics", response_model=APIResponse)
async def get_system_diagnostics():
    """Get comprehensive system diagnostics."""
    try:
        # Collect various diagnostic information
        diagnostics = {
            "timestamp": time.time(),
            "system_uptime": time.time() - startup_time,
            "config": {
                "database_url": settings.database_url.replace(
                    settings.database_url.split("@")[0].split("//")[1], "***"
                )
                if "@" in settings.database_url
                else settings.database_url,
                "redis_url": settings.redis_url,
                "max_agents": settings.max_concurrent_agents,
                "preferred_cli_tool": settings.preferred_cli_tool.value
                if settings.preferred_cli_tool
                else None,
            },
        }

        # Add queue statistics
        queue_stats = await task_queue.get_queue_stats()
        diagnostics["queue_stats"] = {
            "pending_tasks": queue_stats.pending_tasks,
            "assigned_tasks": queue_stats.assigned_tasks,
            "in_progress_tasks": queue_stats.in_progress_tasks,
            "completed_tasks": queue_stats.completed_tasks,
            "failed_tasks": queue_stats.failed_tasks,
            "total_tasks": queue_stats.total_tasks,
            "average_completion_time": queue_stats.average_completion_time,
            "queue_size_by_priority": queue_stats.queue_size_by_priority,
        }

        # Add orchestrator health
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        system_health = await orchestrator.get_system_health()
        diagnostics["system_health"] = {
            "total_agents": system_health.total_agents,
            "active_agents": system_health.active_agents,
            "idle_agents": system_health.idle_agents,
            "busy_agents": system_health.busy_agents,
            "error_agents": system_health.error_agents,
            "avg_load_factor": system_health.avg_load_factor,
            "queue_size": system_health.queue_size,
            "tasks_per_minute": system_health.tasks_per_minute,
        }

        return APIResponse(success=True, data=diagnostics)

    except Exception as e:
        logger.error("Failed to get diagnostics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


def start_server():
    """Start the API server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    start_server()
