"""Authentication and authorization middleware for FastAPI."""

import structlog
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .security import Permission, User, security_manager

logger = structlog.get_logger()

# HTTP Bearer token scheme
security_scheme = HTTPBearer(auto_error=False)


class AuthenticationError(HTTPException):
    """Authentication failed."""

    def __init__(self, detail: str = "Authentication required"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class AuthorizationError(HTTPException):
    """Authorization failed."""

    def __init__(self, detail: str = "Insufficient permissions"):
        super().__init__(status_code=status.HTTP_403_FORBIDDEN, detail=detail)


class RateLimitError(HTTPException):
    """Rate limit exceeded."""

    def __init__(self, detail: str = "Rate limit exceeded"):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": "60"},
        )


def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip

    return request.client.host if request.client else "unknown"


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> User:
    """Get current authenticated user from JWT token."""

    if not credentials:
        raise AuthenticationError("No authentication token provided")

    # Verify token
    payload = security_manager.verify_token(credentials.credentials)
    if not payload:
        raise AuthenticationError("Invalid or expired token")

    # Get user
    user_id = payload.get("sub")
    if not user_id or user_id not in security_manager.users:
        raise AuthenticationError("User not found")

    user = security_manager.users[user_id]
    if not user.is_active:
        raise AuthenticationError("User account is disabled")

    # Log access
    client_ip = get_client_ip(request)
    logger.info(
        "User authenticated",
        user_id=user.id,
        username=user.username,
        client_ip=client_ip,
        endpoint=str(request.url),
    )

    return user


async def get_optional_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> User | None:
    """Get current user if authenticated, otherwise None."""
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def require_permissions(*permissions: str):
    """Decorator to require specific permissions for an endpoint."""

    def decorator(func):
        async def wrapper(
            *args, current_user: User = Depends(get_current_user), **kwargs
        ):
            # Check permissions
            if not security_manager.require_permissions(
                current_user, list(permissions)
            ):
                missing_perms = [
                    p for p in permissions if p not in current_user.permissions
                ]
                logger.warning(
                    "Permission denied",
                    user_id=current_user.id,
                    required_permissions=list(permissions),
                    missing_permissions=missing_perms,
                )
                raise AuthorizationError(
                    f"Missing required permissions: {', '.join(missing_perms)}"
                )

            return await func(*args, current_user=current_user, **kwargs)

        return wrapper

    return decorator


def require_admin():
    """Decorator to require admin privileges."""

    def decorator(func):
        async def wrapper(
            *args, current_user: User = Depends(get_current_user), **kwargs
        ):
            if not current_user.is_admin:
                logger.warning("Admin access denied", user_id=current_user.id)
                raise AuthorizationError("Admin privileges required")

            return await func(*args, current_user=current_user, **kwargs)

        return wrapper

    return decorator


def rate_limit(limit: int = 100):
    """Rate limiting decorator."""

    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = get_client_ip(request)

            if not security_manager.check_rate_limit(client_ip, limit):
                raise RateLimitError(
                    f"Rate limit exceeded: {limit} requests per minute"
                )

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator


class SecurityMiddleware:
    """Security middleware for FastAPI application."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            # Add security headers to response
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = security_manager.get_security_headers()
                    message["headers"] = message.get("headers", [])

                    for name, value in headers.items():
                        message["headers"].append([name.encode(), value.encode()])

                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Permission shortcuts for common operations
class Permissions:
    """Convenience class for permission checks."""

    @staticmethod
    def system_read():
        return require_permissions(Permission.SYSTEM_READ)

    @staticmethod
    def system_write():
        return require_permissions(Permission.SYSTEM_WRITE)

    @staticmethod
    def agent_read():
        return require_permissions(Permission.AGENT_READ)

    @staticmethod
    def agent_write():
        return require_permissions(Permission.AGENT_WRITE, Permission.AGENT_READ)

    @staticmethod
    def agent_spawn():
        return require_permissions(Permission.AGENT_SPAWN, Permission.AGENT_WRITE)

    @staticmethod
    def agent_terminate():
        return require_permissions(Permission.AGENT_TERMINATE, Permission.AGENT_WRITE)

    @staticmethod
    def task_read():
        return require_permissions(Permission.TASK_READ)

    @staticmethod
    def task_write():
        return require_permissions(Permission.TASK_WRITE, Permission.TASK_READ)

    @staticmethod
    def task_create():
        return require_permissions(Permission.TASK_CREATE, Permission.TASK_WRITE)

    @staticmethod
    def message_read():
        return require_permissions(Permission.MESSAGE_READ)

    @staticmethod
    def message_send():
        return require_permissions(Permission.MESSAGE_SEND, Permission.MESSAGE_READ)

    @staticmethod
    def metrics_read():
        return require_permissions(Permission.METRICS_READ)

    @staticmethod
    def modification_read():
        return require_permissions(Permission.MODIFICATION_READ)

    @staticmethod
    def modification_propose():
        return require_permissions(
            Permission.MODIFICATION_PROPOSE, Permission.MODIFICATION_READ
        )
