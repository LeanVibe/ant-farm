"""FastAPI server for LeanVibe Agent Hive 2.0 - Agent management and coordination."""

import asyncio
import json
import logging
import sys
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    from ..core.analytics import (
        PerformanceMiddleware,
        get_performance_collector,
        start_performance_monitoring,
        stop_performance_monitoring,
    )
    from ..core.auth import (
        AuthenticationError,
        Permissions,
        SecurityMiddleware,
        get_current_user,
        get_cli_user,
        rate_limit,
        require_admin,
    )
    from ..core.config import settings
    from ..core.message_broker import MessageType, message_broker
    from ..core.orchestrator import get_orchestrator
    from ..core.security import User, create_default_admin, security_manager
    from ..core.task_queue import Task, TaskPriority, task_queue
except ImportError:  # Direct execution - add src to path
    src_path = Path(__file__).parent.parent
    sys.path.insert(0, str(src_path))
    from core.analytics import (
        PerformanceMiddleware,
        get_performance_collector,
        start_performance_monitoring,
        stop_performance_monitoring,
    )
    from core.auth import (
        AuthenticationError,
        Permissions,
        SecurityMiddleware,
        get_current_user,
        get_cli_user,
        rate_limit,
        require_admin,
    )
    from core.config import settings
    from core.message_broker import MessageType, message_broker
    from core.orchestrator import get_orchestrator
    from core.security import User, create_default_admin, security_manager
    from core.task_queue import Task, TaskPriority, task_queue

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.connection_metadata: dict[WebSocket, dict] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_metadata[websocket] = {
            "client_id": client_id or str(uuid.uuid4()),
            "connected_at": time.time(),
            "subscriptions": set(),
        }
        logger.info(
            "WebSocket client connected",
            client_id=self.connection_metadata[websocket]["client_id"],
            total_connections=len(self.active_connections),
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            client_info = self.connection_metadata.get(websocket, {})
            self.active_connections.remove(websocket)
            self.connection_metadata.pop(websocket, None)
            logger.info(
                "WebSocket client disconnected",
                client_id=client_info.get("client_id"),
                total_connections=len(self.active_connections),
            )

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error("Failed to send personal WebSocket message", error=str(e))
            self.disconnect(websocket)

    async def broadcast(self, message: dict, event_type: str = None):
        """Broadcast message to all connected clients or filtered by subscriptions."""
        if not self.active_connections:
            return

        disconnected = []
        for connection in self.active_connections:
            try:
                # Check if client is subscribed to this event type
                metadata = self.connection_metadata.get(connection, {})
                subscriptions = metadata.get("subscriptions", set())

                if event_type and subscriptions and event_type not in subscriptions:
                    continue

                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error("Failed to broadcast WebSocket message", error=str(e))
                disconnected.append(connection)

        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    def subscribe_client(self, websocket: WebSocket, event_types: list[str]):
        """Subscribe client to specific event types."""
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].update(event_types)

    def unsubscribe_client(self, websocket: WebSocket, event_types: list[str]):
        """Unsubscribe client from specific event types."""
        if websocket in self.connection_metadata:
            self.connection_metadata[websocket]["subscriptions"].difference_update(
                event_types
            )


manager = ConnectionManager()


# Background task for broadcasting system events
async def system_event_broadcaster():
    """Background task that sends periodic system updates via WebSocket."""
    while True:
        try:
            # Broadcast system status every 30 seconds
            await asyncio.sleep(30)

            if manager.active_connections:
                # Get system status
                total_tasks = await task_queue.get_total_tasks()
                completed_tasks = await task_queue.get_completed_tasks()
                queue_depth = await task_queue.get_queue_depth()

                from pathlib import Path

                orchestrator = await get_orchestrator(settings.database_url, Path("."))
                active_agents = await orchestrator.get_active_agent_count()

                status_message = {
                    "type": "system-status",
                    "payload": {
                        "timestamp": time.time(),
                        "active_agents": active_agents,
                        "total_tasks": total_tasks,
                        "completed_tasks": completed_tasks,
                        "queue_depth": queue_depth,
                        "health_score": completed_tasks / max(total_tasks, 1),
                    },
                }

                await manager.broadcast(status_message, "system-status")

        except Exception as e:
            logger.error("Error in system event broadcaster", error=str(e))
            await asyncio.sleep(5)  # Short delay before retrying


# FastAPI app
app = FastAPI(
    title="LeanVibe Agent Hive 2.0",
    description="""
    **LeanVibe Agent Hive 2.0** is an autonomous multi-agent development system that uses AI agents to build and enhance software continuously.

    ## Features

    * **Multi-Agent Orchestration**: Spawn, monitor, and coordinate multiple AI agents
    * **Real-time Task Management**: Create, assign, and track development tasks
    * **Self-Improvement Engine**: Agents can modify their own code safely
    * **Advanced Context Engine**: Semantic memory and knowledge management
    * **WebSocket Events**: Real-time dashboard updates and notifications
    * **Progressive Web App**: Installable with offline capabilities
    * **Security**: JWT authentication with role-based permissions

    ## Authentication

    Most endpoints require authentication using JWT Bearer tokens. Get your token by calling the `/api/v1/auth/login` endpoint.

    ## Rate Limiting

    API calls are rate-limited per IP address:
    - Authentication endpoints: 10 requests/minute
    - General endpoints: 100 requests/minute
    - WebSocket connections: 50 connections/minute

    ## WebSocket Events

    Connect to `/api/v1/ws/events` for real-time updates:
    - `system-status`: System health and metrics
    - `agent-update`: Agent lifecycle events
    - `task-update`: Task status changes
    - `message`: Inter-agent communication
    - `metrics-update`: Performance metrics

    ## Error Handling

    All endpoints return standardized error responses:
    - 400: Bad Request - Invalid input data
    - 401: Unauthorized - Authentication required
    - 403: Forbidden - Insufficient permissions
    - 404: Not Found - Resource not found
    - 429: Too Many Requests - Rate limit exceeded
    - 500: Internal Server Error - System error
    """,
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_tags=[
        {
            "name": "Authentication",
            "description": "User authentication and authorization endpoints",
        },
        {
            "name": "System",
            "description": "System status, health checks, and control operations",
        },
        {"name": "Agents", "description": "Agent lifecycle management and monitoring"},
        {"name": "Tasks", "description": "Task creation, assignment, and tracking"},
        {
            "name": "Messages",
            "description": "Inter-agent communication and broadcasting",
        },
        {"name": "Metrics", "description": "System performance and monitoring data"},
        {"name": "Context", "description": "Memory and context management for agents"},
        {
            "name": "Modifications",
            "description": "Self-improvement and code modification system",
        },
        {
            "name": "Workflows",
            "description": "Development workflow and Git integration",
        },
        {
            "name": "Diagnostics",
            "description": "System diagnostics and troubleshooting",
        },
    ],
    contact={
        "name": "LeanVibe Team",
        "url": "https://leanvibe.ai",
        "email": "support@leanvibe.ai",
    },
    license_info={"name": "MIT", "url": "https://opensource.org/licenses/MIT"},
    servers=[
        {"url": "http://localhost:9001", "description": "Development server"},
        {"url": "https://hive.leanvibe.ai", "description": "Production server"},
    ],
)

# Import enhanced error handling
from .error_handlers import (
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)
from .middleware import (
    RequestTimeoutMiddleware,
    RequestValidationMiddleware,
    SecurityHeadersMiddleware,
)

# Add comprehensive error handlers
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Add security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# Add request validation middleware
app.add_middleware(
    RequestValidationMiddleware,
    max_request_size_mb=10,
    max_json_depth=10,
    rate_limit_requests=100,
    rate_limit_window=60,
)

# Add request timeout middleware
app.add_middleware(RequestTimeoutMiddleware, timeout_seconds=30)

# Add security middleware
app.add_middleware(SecurityMiddleware)

# Add performance monitoring middleware
app.add_middleware(PerformanceMiddleware)

# CORS middleware (keep last in chain)
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
    short_id: str | None = None


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


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    permissions: list[str] = Field(default_factory=list)


class UserInfo(BaseModel):
    id: str
    username: str
    email: str
    is_active: bool
    is_admin: bool
    permissions: list[str]
    created_at: str
    last_login: str | None = None


# Startup and shutdown events
startup_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global startup_time
    startup_time = time.time()
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

        # Start background event broadcaster
        asyncio.create_task(system_event_broadcaster())

        # Start performance monitoring
        await start_performance_monitoring()

        # Create default admin user if none exists
        if not security_manager.users:
            create_default_admin()

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

        # Stop performance monitoring
        await stop_performance_monitoring()

        logger.info("API server shut down successfully")

    except Exception as e:
        logger.error(f"Error during API server shutdown: {str(e)}")


# WebSocket endpoint for real-time events
@app.websocket("/api/v1/ws/events")
async def websocket_endpoint(websocket: WebSocket, client_id: str = None):
    """WebSocket endpoint for real-time dashboard updates."""
    await manager.connect(websocket, client_id)

    try:
        # Send initial connection confirmation
        await manager.send_personal_message(
            {
                "type": "connection-status",
                "payload": {
                    "status": "connected",
                    "client_id": manager.connection_metadata[websocket]["client_id"],
                    "timestamp": time.time(),
                },
            },
            websocket,
        )

        # Send initial system status
        total_tasks = await task_queue.get_total_tasks()
        completed_tasks = await task_queue.get_completed_tasks()
        queue_depth = await task_queue.get_queue_depth()

        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        active_agents = await orchestrator.get_active_agent_count()

        initial_status = {
            "type": "system-status",
            "payload": {
                "timestamp": time.time(),
                "active_agents": active_agents,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "queue_depth": queue_depth,
                "health_score": completed_tasks / max(total_tasks, 1),
                "uptime": time.time() - startup_time,
            },
        }

        await manager.send_personal_message(initial_status, websocket)

        # Listen for client messages
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                await handle_websocket_message(websocket, message)
            except json.JSONDecodeError:
                await manager.send_personal_message(
                    {"type": "error", "payload": {"message": "Invalid JSON format"}},
                    websocket,
                )
            except Exception as e:
                logger.error("Error handling WebSocket message", error=str(e))
                await manager.send_personal_message(
                    {
                        "type": "error",
                        "payload": {"message": f"Error processing message: {str(e)}"},
                    },
                    websocket,
                )

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
        manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, message: dict):
    """Handle incoming WebSocket messages from clients."""
    msg_type = message.get("type")
    payload = message.get("payload", {})

    if msg_type == "subscribe":
        # Subscribe to specific event types
        event_types = payload.get("event_types", [])
        manager.subscribe_client(websocket, event_types)
        await manager.send_personal_message(
            {
                "type": "subscription-confirmed",
                "payload": {"subscribed_to": event_types},
            },
            websocket,
        )

    elif msg_type == "unsubscribe":
        # Unsubscribe from event types
        event_types = payload.get("event_types", [])
        manager.unsubscribe_client(websocket, event_types)
        await manager.send_personal_message(
            {
                "type": "unsubscription-confirmed",
                "payload": {"unsubscribed_from": event_types},
            },
            websocket,
        )

    elif msg_type == "request-status":
        # Send current status on demand
        total_tasks = await task_queue.get_total_tasks()
        completed_tasks = await task_queue.get_completed_tasks()
        queue_depth = await task_queue.get_queue_depth()

        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        active_agents = await orchestrator.get_active_agent_count()

        status = {
            "type": "system-status",
            "payload": {
                "timestamp": time.time(),
                "active_agents": active_agents,
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "queue_depth": queue_depth,
                "health_score": completed_tasks / max(total_tasks, 1),
                "uptime": time.time() - startup_time,
            },
        }
        await manager.send_personal_message(status, websocket)

    else:
        await manager.send_personal_message(
            {
                "type": "error",
                "payload": {"message": f"Unknown message type: {msg_type}"},
            },
            websocket,
        )


# Authentication endpoints
@app.post(
    "/api/v1/auth/login",
    response_model=APIResponse,
    tags=["Authentication"],
    summary="User Login",
    description="""
          Authenticate a user and receive JWT access and refresh tokens.

          The access token expires in 30 minutes and should be included in the
          Authorization header as 'Bearer <token>' for subsequent requests.

          The refresh token expires in 7 days and can be used to obtain new
          access tokens without re-entering credentials.
          """,
    responses={
        200: {
            "description": "Login successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {
                            "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                            "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
                            "token_type": "bearer",
                            "expires_in": 1800,
                            "user": {
                                "id": "123e4567-e89b-12d3-a456-426614174000",
                                "username": "admin",
                                "email": "admin@leanvibe.ai",
                                "is_admin": True,
                                "permissions": ["system:admin", "agent:write"],
                            },
                        },
                        "timestamp": 1640995200.0,
                        "request_id": "req_123456",
                    }
                }
            },
        },
        401: {
            "description": "Invalid credentials or account locked",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid username or password"}
                }
            },
        },
        429: {
            "description": "Too many login attempts",
            "content": {
                "application/json": {
                    "example": {"detail": "Rate limit exceeded: 10 requests per minute"}
                }
            },
        },
    },
)
@rate_limit(10)  # Limit login attempts
async def login(request: Request, login_request: LoginRequest):
    """Authenticate user and return JWT tokens."""
    try:
        # Authenticate user
        user = security_manager.authenticate_user(
            login_request.username, login_request.password
        )

        if not user:
            raise AuthenticationError("Invalid username or password")

        # Generate tokens
        access_token = security_manager.create_access_token(user)
        refresh_token = security_manager.create_refresh_token(user)

        # Create response
        user_data = {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "is_admin": user.is_admin,
            "permissions": user.permissions,
        }

        login_response = LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=security_manager.config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user=user_data,
        )

        return APIResponse(success=True, data=login_response.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(status_code=500, detail="Login failed") from e


@app.post("/api/v1/auth/refresh", response_model=APIResponse)
@rate_limit(20)
async def refresh_token(request: Request, refresh_token: str):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        payload = security_manager.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise AuthenticationError("Invalid refresh token")

        # Get user
        user_id = payload.get("sub")
        if not user_id or user_id not in security_manager.users:
            raise AuthenticationError("User not found")

        user = security_manager.users[user_id]
        if not user.is_active:
            raise AuthenticationError("User account is disabled")

        # Generate new access token
        access_token = security_manager.create_access_token(user)

        return APIResponse(
            success=True,
            data={
                "access_token": access_token,
                "token_type": "bearer",
                "expires_in": security_manager.config.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(status_code=500, detail="Token refresh failed") from e


@app.get("/api/v1/auth/me", response_model=APIResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information."""
    user_info = UserInfo(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        is_active=current_user.is_active,
        is_admin=current_user.is_admin,
        permissions=current_user.permissions,
        created_at=current_user.created_at.isoformat(),
        last_login=current_user.last_login.isoformat()
        if current_user.last_login
        else None,
    )

    return APIResponse(success=True, data=user_info.dict())


@app.post("/api/v1/auth/logout", response_model=APIResponse)
async def logout(current_user: User = Depends(get_current_user)):
    """Logout user (invalidate token on client side)."""
    # In a production system, you would maintain a token blacklist
    # For now, we just log the logout event
    logger.info(
        "User logged out", user_id=current_user.id, username=current_user.username
    )

    return APIResponse(success=True, data={"message": "Logged out successfully"})


@app.post("/api/v1/auth/users", response_model=APIResponse)
@Permissions.system_write()
async def create_user(
    user_create: UserCreate, current_user: User = Depends(get_current_user)
):
    """Create a new user (admin only)."""
    try:
        # Check if username already exists
        if security_manager.get_user_by_username(user_create.username):
            raise HTTPException(status_code=400, detail="Username already exists")

        # Create user
        new_user = security_manager.create_user(
            username=user_create.username,
            email=user_create.email,
            password=user_create.password,
            permissions=user_create.permissions,
        )

        user_info = UserInfo(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            is_active=new_user.is_active,
            is_admin=new_user.is_admin,
            permissions=new_user.permissions,
            created_at=new_user.created_at.isoformat(),
            last_login=None,
        )

        return APIResponse(success=True, data=user_info.dict())

    except HTTPException:
        raise
    except Exception as e:
        logger.error("User creation failed", error=str(e))
        raise HTTPException(status_code=500, detail="User creation failed") from e


# Health check endpoints
async def broadcast_event(event_type: str, payload: dict):
    """Broadcast an event to all WebSocket clients."""
    message = {"type": event_type, "payload": payload, "timestamp": time.time()}
    await manager.broadcast(message, event_type)


# Health check endpoints
@app.get(
    "/health",
    response_model=APIResponse,
    tags=["System"],
    summary="Basic Health Check",
    description="""
          Simple health check endpoint that returns system status.

          This endpoint does not require authentication and can be used for:
          - Load balancer health checks         - Monitoring system availability
         - Service discovery health probes
         - Basic connectivity testing
         """,
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "data": {"status": "healthy", "service": "agent-hive-api"},
                        "timestamp": 1640995200.0,
                        "request_id": "req_123456",
                    }
                }
            },
        }
    },
)
async def health_check():
    """Basic health check."""
    return APIResponse(
        success=True, data={"status": "healthy", "service": "agent-hive-api"}
    )


@app.get(
    "/api/v1/health",
    response_model=APIResponse,
    tags=["System"],
    summary="Detailed Health Check",
    description="Detailed health check for CLI tools and monitoring.",
)
async def detailed_health_check():
    """Detailed health check for CLI tools."""
    try:
        # Test database connection
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))

        # Test Redis connection
        await message_broker.redis_client.ping()

        return APIResponse(
            success=True,
            data={
                "status": "healthy",
                "service": "agent-hive-api",
                "database": "connected",
                "redis": "connected",
                "uptime": time.time() - startup_time,
            },
        )
    except Exception as e:
        return APIResponse(
            success=False,
            data={"status": "unhealthy", "service": "agent-hive-api", "error": str(e)},
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
        logger.error(f"Failed to get system status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


# Agent management endpoints
@app.get("/api/v1/agents", response_model=APIResponse)
async def list_agents(current_user: User = Depends(get_cli_user)):
    """List all agents."""
    try:
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        agents = (
            await orchestrator.registry.list_agents()
        )  # Fixed: use registry.list_agents
        agent_info = []

        for agent in agents:
            # Convert orchestrator AgentStatus to API AgentStatus
            api_status = "active"  # default
            if agent.status.value == "starting":
                api_status = "starting"
            elif agent.status.value == "active":
                api_status = "active"
            elif agent.status.value == "idle":
                api_status = "active"  # Map idle to active for API
            elif agent.status.value == "busy":
                api_status = "active"  # Map busy to active for API
            elif agent.status.value == "stopping":
                api_status = "stopping"
            elif agent.status.value == "stopped":
                api_status = "inactive"
            elif agent.status.value == "error":
                api_status = "error"

            # Convert capabilities list to dict format for API compatibility
            capabilities_dict = {}
            if agent.capabilities:
                if isinstance(agent.capabilities, list):
                    capabilities_dict = {cap: True for cap in agent.capabilities}
                elif isinstance(agent.capabilities, dict):
                    capabilities_dict = agent.capabilities
                else:
                    capabilities_dict = {}

            info = AgentInfo(
                name=agent.name,
                type=agent.type,
                role=agent.role,
                status=AgentStatus(api_status),
                capabilities=capabilities_dict,
                last_heartbeat=agent.last_heartbeat,
                uptime=time.time() - agent.created_at if agent.created_at else 0.0,
                short_id=agent.short_id,
            )
            agent_info.append(info.dict())

        return APIResponse(success=True, data=agent_info)

    except Exception as e:
        logger.error(f"Failed to list agents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/agents/{agent_name}", response_model=APIResponse)
# @Permissions.agent_read()  # Temporarily disabled for CLI testing
async def get_agent(agent_name: str, current_user: User = Depends(get_cli_user)):
    """Get specific agent information."""
    try:
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        agent = await orchestrator.registry.get_agent(
            agent_name
        )  # Fixed: use registry.get_agent
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
# @Permissions.agent_spawn()  # Temporarily disabled for CLI testing
async def spawn_agent(
    agent_type: str,
    agent_name: str | None = None,
    background_tasks: BackgroundTasks = None,
    current_user: User = Depends(get_cli_user),
):
    """Spawn a new agent."""
    try:
        if not agent_name:
            agent_name = f"{agent_type}-{int(time.time())}"

        # Spawn agent asynchronously
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        session_name = await orchestrator.spawn_agent(agent_type, agent_name)

        # Broadcast agent creation event
        await broadcast_event(
            "agent-update",
            {
                "action": "spawned",
                "agent": {
                    "name": agent_name,
                    "type": agent_type,
                    "status": "spawning",
                    "session_name": session_name,
                    "timestamp": time.time(),
                },
            },
        )

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
        logger.error(f"Failed to spawn agent type '{agent_type}': {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/agents/{agent_name}/stop", response_model=APIResponse)
@Permissions.agent_terminate()
async def stop_agent(agent_name: str, current_user: User = Depends(get_current_user)):
    """Stop a specific agent."""
    try:
        from pathlib import Path

        orchestrator = await get_orchestrator(settings.database_url, Path("."))
        success = await orchestrator.stop_agent(agent_name)

        if success:
            # Broadcast agent stop event
            await broadcast_event(
                "agent-update",
                {
                    "action": "stopped",
                    "agent": {
                        "name": agent_name,
                        "status": "stopping",
                        "timestamp": time.time(),
                    },
                },
            )

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


@app.get("/api/v1/test")
async def test_endpoint():
    """Simple test endpoint without authentication."""
    return {"message": "Hello from API server", "timestamp": time.time()}


# Task management endpoints
@app.get("/api/v1/tasks", response_model=APIResponse)
# @Permissions.task_read()  # Temporarily disabled for CLI testing
async def list_tasks(
    status: str | None = None,
    assigned_to: str | None = None,
    current_user: User = Depends(get_cli_user),
):
    """List tasks with optional filtering."""
    try:
        tasks = await task_queue.list_tasks(status=status, assigned_to=assigned_to)
        task_info = []

        for task in tasks:
            info = TaskInfo(
                id=task.id,
                title=task.title,
                description=task.description,
                type=task.task_type,  # This is correct
                status=task.status,
                priority=task.priority,
                assigned_to=task.agent_id,  # Fixed: use agent_id
                created_at=task.created_at,
                started_at=task.started_at,
                completed_at=task.completed_at,
                result=task.result,
                error=task.error_message,  # Fixed: use error_message
            )
            task_info.append(info.dict())

        return APIResponse(success=True, data=task_info)

    except Exception as e:
        logger.error("Failed to list tasks", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/tasks", response_model=APIResponse)
# @Permissions.task_create()  # Temporarily disabled for CLI testing
async def create_task(
    task_create: TaskCreate, current_user: User = Depends(get_cli_user)
):
    """Create a new task and submit it to the task queue for MetaAgent processing."""
    try:
        logger.info(
            f"Creating task: title={task_create.title}, type={task_create.type}, priority={task_create.priority}"
        )

        task = Task(
            id=str(uuid.uuid4()),
            title=task_create.title,
            description=task_create.description,
            task_type=task_create.type,  # Fixed: task_type not type
            priority=task_create.priority,
            agent_id=task_create.assigned_to
            or "meta-agent",  # Fixed: agent_id not assigned_to
            dependencies=task_create.dependencies,
            # Removed metadata as it's not in Task model
            status="pending",
            created_at=time.time(),
        )

        logger.info(
            f"Task object created: task_id={task.id}, task_type={task.task_type}"
        )
        task_id = await task_queue.add_task(task)
        logger.info(f"Task added to queue: task_id={task_id}")

        # Broadcast task creation event
        await broadcast_event(
            "task-update",
            {
                "action": "created",
                "task": {
                    "id": task_id,
                    "title": task.title,
                    "type": task.task_type,  # Fixed: use task_type
                    "status": task.status,  # Fixed: status is a string not enum
                    "priority": task.priority.value,
                    "assigned_to": task.agent_id,  # Fixed: use agent_id
                    "created_at": task.created_at,
                },
            },
        )

        return APIResponse(
            success=True,
            data={"task_id": task_id, "title": task.title, "status": task.status},
        )

    except Exception as e:
        logger.error(f"Failed to create task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/tasks/self-improvement", response_model=APIResponse)
# @Permissions.task_create()  # Temporarily disabled for CLI testing
async def create_self_improvement_task(
    title: str, description: str, current_user: User = Depends(get_cli_user)
):
    """Create a self-improvement task for the MetaAgent as specified in PLAN.md.

    This is the main entry point for giving the MetaAgent its first self-improvement task.
    The task will be processed using the ContextEngine → SelfModifier workflow.
    """
    try:
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            description=description,
            task_type="self_improvement",  # Fixed: task_type not type
            priority=TaskPriority.HIGH,  # High priority for self-improvement
            agent_id="meta-agent",  # Fixed: agent_id not assigned_to
            dependencies=[],
            # Removed metadata as it's not in Task model
            status="pending",
            created_at=time.time(),
        )

        task_id = await task_queue.add_task(task)

        # Broadcast self-improvement task creation event
        await broadcast_event(
            "task-update",
            {
                "action": "created",
                "task": {
                    "id": task_id,
                    "title": task.title,
                    "type": "self_improvement",
                    "status": "pending",
                    "priority": "high",
                    "assigned_to": "meta-agent",
                    "created_at": task.created_at,
                },
            },
        )

        logger.info("Self-improvement task created", task_id=task_id, title=title)

        return APIResponse(
            success=True,
            data={
                "task_id": task_id,
                "title": title,
                "type": "self_improvement",
                "status": "pending",
                "message": "Self-improvement task submitted to MetaAgent",
            },
        )

    except Exception as e:
        logger.error("Failed to create self-improvement task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/tasks/{task_id}", response_model=APIResponse)
async def get_task(task_id: str):
    """Get specific task information."""
    try:
        task = await task_queue.get_task_by_id(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        info = TaskInfo(
            id=task.id,
            title=task.title,
            description=task.description,
            type=task.task_type,  # This is correct
            status=task.status,
            priority=task.priority,
            assigned_to=task.agent_id,  # Fixed: use agent_id
            created_at=task.created_at,
            started_at=task.started_at,
            completed_at=task.completed_at,
            result=task.result,
            error=task.error_message,  # Fixed: use error_message
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
@Permissions.system_write()
async def trigger_system_analysis(current_user: User = Depends(get_current_user)):
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
@require_admin()
async def shutdown_system(current_user: User = Depends(get_current_user)):
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
@app.get(
    "/api/v1/metrics",
    response_model=APIResponse,
    tags=["Metrics"],
    summary="Get System Metrics",
    description="""
         Get comprehensive system performance metrics including:
         - CPU, memory, and disk utilization
         - Network statistics
         - Process and thread counts
         - API response times and error rates
         - Agent performance metrics
         """,
)
@Permissions.metrics_read()
async def get_metrics(
    time_window_minutes: int = 60, current_user: User = Depends(get_current_user)
):
    """Get system metrics."""
    try:
        collector = get_performance_collector()
        metrics_summary = collector.get_metrics_summary(time_window_minutes)

        return APIResponse(success=True, data=metrics_summary)

    except Exception as e:
        logger.error("Failed to get metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/api/v1/metrics/alerts",
    response_model=APIResponse,
    tags=["Metrics"],
    summary="Get Performance Alerts",
    description="Get current performance alerts based on system thresholds.",
)
@Permissions.metrics_read()
async def get_performance_alerts(current_user: User = Depends(get_current_user)):
    """Get current performance alerts."""
    try:
        collector = get_performance_collector()
        alerts = collector.get_performance_alerts()

        return APIResponse(success=True, data=alerts)

    except Exception as e:
        logger.error("Failed to get performance alerts", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/api/v1/metrics/export",
    tags=["Metrics"],
    summary="Export Metrics Data",
    description="""
         Export historical metrics data in JSON or CSV format.
         Useful for external analysis, reporting, or backup purposes.
         """,
)
@Permissions.metrics_read()
async def export_metrics(
    format: str = "json",
    time_window_hours: int = 24,
    current_user: User = Depends(get_current_user),
):
    """Export metrics data."""
    try:
        collector = get_performance_collector()

        if format.lower() not in ["json", "csv"]:
            raise HTTPException(
                status_code=400, detail="Format must be 'json' or 'csv'"
            )

        exported_data = collector.export_metrics(format, time_window_hours)

        # Set appropriate content type and filename
        content_type = "application/json" if format.lower() == "json" else "text/csv"
        filename = f"hive_metrics_{int(time.time())}.{format.lower()}"

        from fastapi.responses import Response

        return Response(
            content=exported_data,
            media_type=content_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Performance and optimization endpoints
@app.get(
    "/api/v1/performance",
    response_model=APIResponse,
    tags=["Metrics"],
    summary="Get Performance Statistics",
    description="""
         Get detailed performance optimization statistics including:
         - Current system resource utilization
         - API endpoint performance metrics
         - Optimization recommendations
         - Historical performance trends
         """,
)
@Permissions.metrics_read()
async def get_performance_stats(
    time_window_minutes: int = 60, current_user: User = Depends(get_current_user)
):
    """Get performance optimization statistics."""
    try:
        collector = get_performance_collector()

        # Get comprehensive performance data
        metrics_summary = collector.get_metrics_summary(time_window_minutes)
        api_performance = collector.get_api_performance_summary(time_window_minutes)
        system_health = collector.get_system_health_score()
        alerts = collector.get_performance_alerts()

        # Try to get optimization stats from the existing optimizer
        optimization_stats = {}
        try:
            from ..core.performance_optimizer import get_performance_optimizer

            optimizer = get_performance_optimizer()
            optimization_stats = optimizer.get_optimization_stats()
        except Exception:
            # Fallback if optimizer not available
            optimization_stats = {
                "optimizations_applied": 0,
                "performance_improvements": {},
                "last_optimization": None,
            }

        return APIResponse(
            success=True,
            data={
                "system_health_score": system_health,
                "metrics_summary": metrics_summary,
                "api_performance": {k: v.to_dict() for k, v in api_performance.items()},
                "optimization_stats": optimization_stats,
                "alerts": alerts,
                "time_window_minutes": time_window_minutes,
                "recommendations": _generate_performance_recommendations(collector),
            },
        )

    except Exception as e:
        logger.error("Failed to get performance stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


def _generate_performance_recommendations(collector) -> list[dict[str, Any]]:
    """Generate performance recommendations based on current metrics."""
    recommendations = []
    alerts = collector.get_performance_alerts()

    # Generate recommendations based on alerts
    for alert in alerts:
        if alert["metric"] == "cpu_usage" and alert["type"] == "critical":
            recommendations.append(
                {
                    "type": "optimization",
                    "priority": "high",
                    "category": "resource",
                    "title": "High CPU Usage Detected",
                    "description": "Consider scaling horizontally or optimizing CPU-intensive operations",
                    "actions": [
                        "Review CPU-intensive agents and tasks",
                        "Consider adding more worker processes",
                        "Optimize algorithms and reduce computational complexity",
                    ],
                }
            )

        elif alert["metric"] == "memory_usage" and alert["type"] == "critical":
            recommendations.append(
                {
                    "type": "optimization",
                    "priority": "high",
                    "category": "resource",
                    "title": "High Memory Usage Detected",
                    "description": "Memory usage is approaching critical levels",
                    "actions": [
                        "Review memory-intensive operations",
                        "Implement memory cleanup routines",
                        "Consider increasing available memory",
                    ],
                }
            )

        elif alert["metric"] == "api_response_time" and alert["type"] == "warning":
            recommendations.append(
                {
                    "type": "optimization",
                    "priority": "medium",
                    "category": "performance",
                    "title": "Slow API Response Times",
                    "description": "Some API endpoints are responding slowly",
                    "actions": [
                        "Review slow endpoints and optimize queries",
                        "Implement caching for frequently accessed data",
                        "Consider database indexing improvements",
                    ],
                }
            )

    # Add general recommendations if no alerts
    if not recommendations:
        recommendations.append(
            {
                "type": "maintenance",
                "priority": "low",
                "category": "general",
                "title": "System Running Optimally",
                "description": "No performance issues detected at this time",
                "actions": [
                    "Continue monitoring system metrics",
                    "Regular maintenance and updates recommended",
                    "Consider performance testing under higher loads",
                ],
            }
        )

    return recommendations


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


# Context helper function
async def resolve_agent_uuid(agent_identifier: str) -> str:
    """Resolve agent identifier to UUID for context operations."""
    from core.async_db import get_async_database_manager
    from core.models import Agent as AgentModel
    from sqlalchemy import select
    import uuid

    # First try to resolve using the database
    try:
        db_manager = await get_async_database_manager()
        async with db_manager.async_session_maker() as session:
            # Try exact UUID match first
            try:
                uuid_obj = uuid.UUID(agent_identifier)
                return str(uuid_obj)
            except ValueError:
                pass

            # Try to find by name or short_id
            stmt = select(AgentModel).where(
                (AgentModel.name == agent_identifier)
                | (AgentModel.short_id == agent_identifier)
            )
            result = await session.execute(stmt)
            agent = result.scalar_one_or_none()

            if agent:
                return str(agent.id)
    except Exception as e:
        logger.warning(f"Database agent lookup failed: {e}")

    # Fallback to uuid5 for backward compatibility
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"agent.{agent_identifier}"))


@app.get("/api/v1/context/{agent_id}", response_model=APIResponse)
async def get_agent_context(agent_id: str, limit: int = 10):
    """Get context/memory for a specific agent."""
    try:
        from core.context_engine import get_context_engine

        # Resolve agent identifier to proper UUID
        agent_uuid = await resolve_agent_uuid(agent_id)

        context_engine = await get_context_engine(settings.database_url)
        memory_stats = await context_engine.get_memory_stats(agent_uuid)

        return APIResponse(
            success=True,
            data={
                "agent_id": agent_id,
                "resolved_uuid": agent_uuid,
                "total_contexts": memory_stats.total_contexts,
                "contexts_by_importance": memory_stats.contexts_by_importance,
                "contexts_by_category": memory_stats.contexts_by_category,
                "storage_size_mb": memory_stats.storage_size_mb,
                "oldest_context_age_days": memory_stats.oldest_context_age_days,
                "most_accessed_context_id": memory_stats.most_accessed_context_id,
            },
        )

    except Exception as e:
        logger.error(f"Failed to add context for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/context/{agent_id}/search", response_model=APIResponse)
async def search_context(agent_id: str, query: str, limit: int = 10):
    """Perform semantic search on agent context."""
    try:
        from core.context_engine import get_context_engine

        # Resolve agent identifier to proper UUID
        agent_uuid = await resolve_agent_uuid(agent_id)

        context_engine = await get_context_engine(settings.database_url)

        # Search contexts and filter by agent_id
        all_results = await context_engine.search_context(
            query=query, limit=limit * 3
        )  # Get more to filter

        # Filter results by agent_uuid
        agent_results = [
            result
            for result in all_results
            if str(result.context.agent_id) == agent_uuid
        ]

        # Take only the requested limit
        results = agent_results[:limit]

        return APIResponse(
            success=True,
            data={
                "agent_id": agent_id,
                "resolved_uuid": agent_uuid,
                "query": query,
                "total_results": len(results),
                "results": [
                    {
                        "context_id": str(result.context.id),
                        "content": result.context.content,
                        "category": result.context.category,
                        "importance_score": result.context.importance_score,
                        "similarity_score": result.similarity_score,
                        "created_at": result.context.created_at.isoformat()
                        if result.context.created_at
                        else None,
                    }
                    for result in results
                ],
            },
        )

    except Exception as e:
        logger.error(
            f"Failed to search context for agent {agent_id} with query '{query}': {str(e)}"
        )
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/context/{agent_id}/add", response_model=APIResponse)
async def add_context(
    agent_id: str,
    content: str,
    content_type: str = "text",
    category: str = "general",
    importance_score: float = 0.5,
    topic: str = None,
    metadata: dict = None,
):
    """Add a document to the context engine."""
    try:
        from core.context_engine import get_context_engine

        # Resolve agent identifier to proper UUID
        agent_uuid = await resolve_agent_uuid(agent_id)

        context_engine = await get_context_engine(settings.database_url)

        context_id = await context_engine.store_context(
            content=content,
            agent_id=agent_uuid,
            content_type=content_type,
            category=category,
            importance_score=importance_score,
            topic=topic,
            metadata=metadata or {},
        )

        return APIResponse(
            success=True,
            data={
                "context_id": str(context_id),
                "agent_id": agent_id,
                "resolved_uuid": agent_uuid,
                "content_length": len(content),
                "category": category,
                "importance_score": importance_score,
            },
        )

        return APIResponse(
            success=True,
            data={
                "agent_id": agent_id,
                "agent_uuid": agent_uuid,
                "context_id": str(context_id),
                "content_length": len(content),
                "category": category,
            },
        )

    except Exception as e:
        logger.error(f"Failed to add context for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/context/{agent_id}/consolidate", response_model=APIResponse)
async def consolidate_agent_memory(agent_id: str):
    """Trigger memory consolidation for an agent."""
    try:
        from core.context_engine import get_context_engine

        context_engine = await get_context_engine(settings.database_url)
        consolidation_stats = await context_engine.consolidate_memory(agent_id)

        return APIResponse(
            success=True,
            data={"agent_id": agent_id, "consolidation_stats": consolidation_stats},
        )

    except Exception as e:
        logger.error(f"Failed to add context for agent {agent_id}: {str(e)}")
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
        from ..core.self_modifier import CodeChange, ModificationType, get_self_modifier

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
        from ..core.cli_git_integration import WorkflowType, get_enhanced_cli_executor

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


# ADW Monitoring endpoints
@app.get("/api/v1/adw/metrics/current", response_model=APIResponse)
async def get_current_adw_metrics():
    """Get current ADW monitoring metrics."""
    try:
        # This would integrate with the actual monitoring dashboard
        metrics = {
            "autonomy_score": 85.5,
            "memory_percent": 45.2,
            "cpu_percent": 32.1,
            "uptime": time.time() - startup_time,
            "session_hours": 2.5,
            "velocity_commits": 3.2,
            "active_session": True,
            "timestamp": time.time(),
        }

        return APIResponse(success=True, data=metrics)
    except Exception as e:
        logger.error("Failed to get ADW metrics", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/adw/cognitive/state", response_model=APIResponse)
async def get_cognitive_state():
    """Get current cognitive load state."""
    try:
        # This would integrate with the cognitive load manager
        cognitive_state = {
            "fatigue_level": 0.35,
            "focus_efficiency": 0.82,
            "mode": "focus",
            "session_duration": 2.5,
            "mode_transitions": 1,
            "last_transition": time.time() - 1800,  # 30 minutes ago
            "recommended_action": "continue",
            "timestamp": time.time(),
        }

        return APIResponse(success=True, data=cognitive_state)
    except Exception as e:
        logger.error("Failed to get cognitive state", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/adw/predictions/current", response_model=APIResponse)
async def get_failure_predictions():
    """Get current failure predictions."""
    try:
        # This would integrate with the failure prediction system
        predictions = {
            "risk_score": 0.15,
            "confidence": 0.87,
            "next_check": "5 minutes",
            "risk_factors": ["complexity_increase", "time_pressure"],
            "mitigation_suggestions": [
                "Consider reducing iteration scope",
                "Take a 5-minute break to maintain focus",
            ],
            "timestamp": time.time(),
        }

        return APIResponse(success=True, data=predictions)
    except Exception as e:
        logger.error("Failed to get failure predictions", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/adw/monitoring/start", response_model=APIResponse)
async def start_adw_monitoring():
    """Start ADW monitoring session."""
    try:
        # This would start the actual monitoring
        session_id = f"adw-monitor-{int(time.time())}"

        # In real implementation, this would:
        # - Initialize AutonomousDashboard
        # - Start cognitive load monitoring
        # - Begin failure prediction

        return APIResponse(
            success=True,
            data={
                "session_id": session_id,
                "monitoring_started": True,
                "start_time": time.time(),
            },
        )
    except Exception as e:
        logger.error("Failed to start ADW monitoring", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/adw/monitoring/stop", response_model=APIResponse)
async def stop_adw_monitoring():
    """Stop ADW monitoring session."""
    try:
        # This would stop the actual monitoring

        return APIResponse(
            success=True,
            data={
                "monitoring_stopped": True,
                "stop_time": time.time(),
            },
        )
    except Exception as e:
        logger.error("Failed to stop ADW monitoring", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/adw/sessions/history", response_model=APIResponse)
async def get_adw_session_history():
    """Get ADW session history."""
    try:
        # This would retrieve actual session history
        history = [
            {
                "session_id": "adw-session-1",
                "start_time": time.time() - 7200,  # 2 hours ago
                "duration": 3600,  # 1 hour
                "status": "completed",
                "commits": 8,
                "tests_passed": 15,
                "cognitive_score": 0.85,
                "autonomy_score": 82.5,
            },
            {
                "session_id": "adw-session-2",
                "start_time": time.time() - 14400,  # 4 hours ago
                "duration": 7200,  # 2 hours
                "status": "completed",
                "commits": 12,
                "tests_passed": 22,
                "cognitive_score": 0.78,
                "autonomy_score": 88.2,
            },
        ]

        return APIResponse(success=True, data=history)
    except Exception as e:
        logger.error("Failed to get ADW session history", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/adw/emergency/status", response_model=APIResponse)
async def get_emergency_status():
    """Get current emergency intervention status."""
    try:
        # Mock emergency status data
        # In a real implementation, this would connect to the active ADW session
        emergency_status = {
            "timestamp": time.time(),
            "active": True,
            "current_intervention_level": "warning",
            "total_events": 2,
            "unresolved_events": 0,
            "human_intervention_requested": False,
            "last_check_time": time.time() - 10,
            "recent_events": [
                {
                    "timestamp": time.time() - 300,
                    "failure_type": "resource_exhaustion",
                    "intervention_level": "pause",
                    "description": "Memory usage exceeded 90%",
                    "resolved": True,
                },
                {
                    "timestamp": time.time() - 600,
                    "failure_type": "repeated_failures",
                    "intervention_level": "warning",
                    "description": "3 consecutive test failures detected",
                    "resolved": True,
                },
            ],
            "system_health": {
                "memory_usage": 72.5,
                "cpu_usage": 45.2,
                "disk_usage": 68.9,
                "active_monitors": 5,
            },
        }

        return APIResponse(success=True, data=emergency_status)
    except Exception as e:
        logger.error("Failed to get emergency status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/adw/emergency/resolve/{event_index}", response_model=APIResponse)
async def resolve_emergency_event(event_index: int, resolution_notes: str = ""):
    """Resolve a specific emergency event."""
    try:
        # Mock emergency event resolution
        # In a real implementation, this would connect to the active ADW session
        result = {
            "event_index": event_index,
            "resolved": True,
            "resolution_timestamp": time.time(),
            "resolution_notes": resolution_notes,
        }

        return APIResponse(success=True, data=result)
    except Exception as e:
        logger.error("Failed to resolve emergency event", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/adw/emergency/request-intervention", response_model=APIResponse)
async def request_human_intervention(reason: str):
    """Request human intervention for the current ADW session."""
    try:
        # Mock human intervention request
        # In a real implementation, this would connect to the active ADW session
        result = {
            "intervention_requested": True,
            "timestamp": time.time(),
            "reason": reason,
            "session_id": f"adw-session-{int(time.time())}",
            "escalation_level": "human_required",
        }

        return APIResponse(success=True, data=result)
    except Exception as e:
        logger.error("Failed to request human intervention", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


# Large Project Coordination endpoints
@app.post("/api/v1/projects", response_model=APIResponse)
async def create_project_workspace(
    project_data: dict = Body(...), current_user: dict = Depends(get_current_user)
):
    """Create a new large project workspace."""
    try:
        from ..core.collaboration import get_large_project_coordinator, ProjectScale

        coordinator = await get_large_project_coordinator()

        # Validate input data
        required_fields = ["name", "description", "scale", "lead_agent"]
        for field in required_fields:
            if field not in project_data:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}"
                )

        # Convert scale string to enum
        try:
            scale = ProjectScale(project_data["scale"])
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid project scale: {project_data['scale']}",
            )

        project_id = await coordinator.create_project_workspace(
            name=project_data["name"],
            description=project_data["description"],
            scale=scale,
            lead_agent=project_data["lead_agent"],
            root_path=project_data.get("root_path"),
        )

        return APIResponse(
            success=True,
            data={
                "project_id": project_id,
                "message": "Project workspace created successfully",
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create project workspace", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/projects/{project_id}/join", response_model=APIResponse)
async def join_project(
    project_id: str,
    join_data: dict = Body(...),
    current_user: dict = Depends(get_current_user),
):
    """Add an agent to a project workspace."""
    try:
        from ..core.collaboration import get_large_project_coordinator

        coordinator = await get_large_project_coordinator()

        agent_id = join_data.get("agent_id")
        roles = join_data.get("roles", ["contributor"])

        if not agent_id:
            raise HTTPException(status_code=400, detail="Missing agent_id")

        success = await coordinator.join_project(project_id, agent_id, roles)

        if not success:
            raise HTTPException(status_code=404, detail="Project not found")

        return APIResponse(
            success=True,
            data={"message": f"Agent {agent_id} joined project successfully"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to join project", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/projects/{project_id}/tasks/decompose", response_model=APIResponse)
async def decompose_large_task(
    project_id: str,
    task_data: dict = Body(...),
    current_user: dict = Depends(get_current_user),
):
    """Decompose a large task into coordinated sub-tasks."""
    try:
        from ..core.collaboration import get_large_project_coordinator

        coordinator = await get_large_project_coordinator()

        required_fields = ["description", "estimated_complexity"]
        for field in required_fields:
            if field not in task_data:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}"
                )

        decomposition_result = await coordinator.decompose_large_task(
            project_id=project_id,
            task_description=task_data["description"],
            estimated_complexity=task_data["estimated_complexity"],
            target_agents=task_data.get("target_agents"),
        )

        return APIResponse(success=True, data=decomposition_result)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to decompose task", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/projects/{project_id}/status", response_model=APIResponse)
async def get_project_status(
    project_id: str, current_user: dict = Depends(get_current_user)
):
    """Get comprehensive status of a large project."""
    try:
        from ..core.collaboration import get_large_project_coordinator

        coordinator = await get_large_project_coordinator()
        status = await coordinator.get_project_status(project_id)

        return APIResponse(success=True, data=status)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to get project status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/api/v1/projects/{project_id}/progress", response_model=APIResponse)
async def monitor_project_progress(
    project_id: str, current_user: dict = Depends(get_current_user)
):
    """Monitor and report on project progress."""
    try:
        from ..core.collaboration import get_large_project_coordinator

        coordinator = await get_large_project_coordinator()
        progress = await coordinator.monitor_project_progress(project_id)

        return APIResponse(success=True, data=progress)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to monitor project progress", error=str(e))
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/api/v1/projects/{project_id}/conflicts/resolve", response_model=APIResponse)
async def resolve_project_conflict(
    project_id: str,
    conflict_data: dict = Body(...),
    current_user: dict = Depends(get_current_user),
):
    """Handle conflicts in large project coordination."""
    try:
        from ..core.collaboration import get_large_project_coordinator

        coordinator = await get_large_project_coordinator()

        required_fields = ["conflict_type", "involved_agents"]
        for field in required_fields:
            if field not in conflict_data:
                raise HTTPException(
                    status_code=400, detail=f"Missing required field: {field}"
                )

        resolution_result = await coordinator.handle_conflict_resolution(
            project_id=project_id,
            conflict_type=conflict_data["conflict_type"],
            involved_agents=conflict_data["involved_agents"],
            context=conflict_data.get("context", {}),
        )

        return APIResponse(success=True, data=resolution_result)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Failed to resolve project conflict", error=str(e))
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
    from ..core.config import get_settings

    settings = get_settings()
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)


if __name__ == "__main__":
    start_server()
