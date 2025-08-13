"""Security module for authentication, authorization, and security hardening."""

import secrets
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
import structlog
from passlib.context import CryptContext
from pydantic import BaseModel, Field

logger = structlog.get_logger()

# Password context for secure hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(BaseModel):
    """User model for authentication."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    hashed_password: str
    is_active: bool = True
    is_admin: bool = False
    permissions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: datetime | None = None
    failed_login_attempts: int = 0
    locked_until: datetime | None = None


class Permission:
    """Permission constants for role-based access control."""

    # System permissions
    SYSTEM_READ = "system:read"
    SYSTEM_WRITE = "system:write"
    SYSTEM_ADMIN = "system:admin"

    # Agent permissions
    AGENT_READ = "agent:read"
    AGENT_WRITE = "agent:write"
    AGENT_SPAWN = "agent:spawn"
    AGENT_TERMINATE = "agent:terminate"

    # Task permissions
    TASK_READ = "task:read"
    TASK_WRITE = "task:write"
    TASK_CREATE = "task:create"
    TASK_CANCEL = "task:cancel"

    # Message permissions
    MESSAGE_READ = "message:read"
    MESSAGE_SEND = "message:send"
    MESSAGE_BROADCAST = "message:broadcast"

    # Metrics permissions
    METRICS_READ = "metrics:read"
    METRICS_WRITE = "metrics:write"

    # Context permissions
    CONTEXT_READ = "context:read"
    CONTEXT_WRITE = "context:write"

    # Modification permissions
    MODIFICATION_READ = "modification:read"
    MODIFICATION_PROPOSE = "modification:propose"
    MODIFICATION_APPROVE = "modification:approve"
    MODIFICATION_APPLY = "modification:apply"


class SecurityConfig:
    """Security configuration settings."""

    # JWT settings
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Rate limiting
    MAX_LOGIN_ATTEMPTS: int = 5
    LOCKOUT_DURATION_MINUTES: int = 15

    # API rate limits (requests per minute)
    API_RATE_LIMIT_GENERAL: int = 100
    API_RATE_LIMIT_AUTH: int = 10
    API_RATE_LIMIT_WEBSOCKET: int = 50

    # Security headers
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:9001"]
    ALLOWED_HOSTS: list[str] = ["localhost", "127.0.0.1"]

    # Content Security Policy
    CSP_DIRECTIVES = {
        "default-src": ["'self'"],
        "script-src": ["'self'", "'unsafe-inline'"],
        "style-src": ["'self'", "'unsafe-inline'"],
        "img-src": ["'self'", "data:", "blob:"],
        "connect-src": ["'self'", "ws:", "wss:"],
        "frame-ancestors": ["'none'"],
    }


class SecurityManager:
    """Central security manager for authentication and authorization."""

    def __init__(self, config: SecurityConfig = None):
        self.config = config or SecurityConfig()
        self.users: dict[str, User] = {}
        self.sessions: dict[str, dict] = {}
        self.rate_limits: dict[str, list[float]] = {}

    # Password management
    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return pwd_context.verify(plain_password, hashed_password)

    # User management
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        is_admin: bool = False,
        permissions: list[str] = None,
    ) -> User:
        """Create a new user with hashed password."""
        user = User(
            username=username,
            email=email,
            hashed_password=self.hash_password(password),
            is_admin=is_admin,
            permissions=permissions or [],
        )
        self.users[user.id] = user
        logger.info("User created", username=username, user_id=user.id)
        return user

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        for user in self.users.values():
            if user.username == username:
                return user
        return None

    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate user credentials."""
        user = self.get_user_by_username(username)
        if not user:
            logger.warning("Login attempt with invalid username", username=username)
            return None

        # Check if account is locked
        if user.locked_until and datetime.now(UTC) < user.locked_until:
            logger.warning("Login attempt on locked account", username=username)
            return None

        # Verify password
        if not self.verify_password(password, user.hashed_password):
            user.failed_login_attempts += 1

            # Lock account after max attempts
            if user.failed_login_attempts >= self.config.MAX_LOGIN_ATTEMPTS:
                user.locked_until = datetime.now(UTC) + timedelta(
                    minutes=self.config.LOCKOUT_DURATION_MINUTES
                )
                logger.warning(
                    "Account locked due to failed attempts",
                    username=username,
                    attempts=user.failed_login_attempts,
                )

            logger.warning(
                "Failed login attempt",
                username=username,
                attempts=user.failed_login_attempts,
            )
            return None

        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(UTC)

        logger.info("User authenticated successfully", username=username)
        return user

    # JWT token management
    def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        expire = datetime.now(UTC) + timedelta(
            minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES
        )

        payload = {
            "sub": user.id,
            "username": user.username,
            "permissions": user.permissions,
            "is_admin": user.is_admin,
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "access",
        }

        token = jwt.encode(
            payload, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM
        )
        logger.info("Access token created", user_id=user.id, expires=expire)
        return token

    def create_refresh_token(self, user: User) -> str:
        """Create JWT refresh token."""
        expire = datetime.now(UTC) + timedelta(
            days=self.config.REFRESH_TOKEN_EXPIRE_DAYS
        )

        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh",
        }

        return jwt.encode(
            payload, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM
        )

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM]
            )
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except Exception as e:
            logger.warning("Invalid token", error=str(e))
            return None

    # Authorization
    def check_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission."""
        if user.is_admin:
            return True
        return permission in user.permissions

    def require_permissions(self, user: User, permissions: list[str]) -> bool:
        """Check if user has all required permissions."""
        if user.is_admin:
            return True
        return all(perm in user.permissions for perm in permissions)

    # Rate limiting
    def check_rate_limit(self, client_id: str, limit: int) -> bool:
        """Check if client exceeds rate limit."""
        now = time.time()
        window_start = now - 60  # 1-minute window

        if client_id not in self.rate_limits:
            self.rate_limits[client_id] = []

        # Remove old requests outside the window
        self.rate_limits[client_id] = [
            req_time
            for req_time in self.rate_limits[client_id]
            if req_time > window_start
        ]

        # Check if under limit
        if len(self.rate_limits[client_id]) >= limit:
            logger.warning(
                "Rate limit exceeded",
                client_id=client_id,
                requests=len(self.rate_limits[client_id]),
            )
            return False

        # Add current request
        self.rate_limits[client_id].append(now)
        return True

    # API key management
    def generate_api_key(self, user: User, name: str = None) -> str:
        """Generate API key for a user."""
        key_id = str(uuid.uuid4())
        key_secret = secrets.token_urlsafe(32)
        api_key = f"hive_{key_id}_{key_secret}"

        # Store API key metadata (in production, store in database)

        logger.info("API key generated", user_id=user.id, key_id=key_id)
        return api_key

    # Security headers
    def get_security_headers(self) -> dict[str, str]:
        """Get security headers for HTTP responses."""
        csp = "; ".join(
            [
                f"{directive} {' '.join(sources)}"
                for directive, sources in self.config.CSP_DIRECTIVES.items()
            ]
        )

        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": csp,
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }

    # Input validation and sanitization
    def validate_input(self, data: Any, max_length: int = 1000) -> bool:
        """Basic input validation."""
        if isinstance(data, str):
            if len(data) > max_length:
                return False
            # Check for common injection patterns
            dangerous_patterns = [
                "<script",
                "javascript:",
                "data:text/html",
                "vbscript:",
            ]
            data_lower = data.lower()
            if any(pattern in data_lower for pattern in dangerous_patterns):
                return False
        return True

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal."""
        # Remove path separators and dangerous characters
        dangerous_chars = ["../", "..\\", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]
        sanitized = filename
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, "_")
        return sanitized[:255]  # Limit filename length


# Global security manager instance
security_manager = SecurityManager()


def create_default_admin():
    """Create default admin user for initial setup."""
    admin_permissions = [
        Permission.SYSTEM_ADMIN,
        Permission.AGENT_READ,
        Permission.AGENT_WRITE,
        Permission.AGENT_SPAWN,
        Permission.AGENT_TERMINATE,
        Permission.TASK_READ,
        Permission.TASK_WRITE,
        Permission.TASK_CREATE,
        Permission.TASK_CANCEL,
        Permission.MESSAGE_READ,
        Permission.MESSAGE_SEND,
        Permission.MESSAGE_BROADCAST,
        Permission.METRICS_READ,
        Permission.METRICS_WRITE,
        Permission.CONTEXT_READ,
        Permission.CONTEXT_WRITE,
        Permission.MODIFICATION_READ,
        Permission.MODIFICATION_PROPOSE,
        Permission.MODIFICATION_APPROVE,
        Permission.MODIFICATION_APPLY,
    ]

    admin_user = security_manager.create_user(
        username="admin",
        email="admin@leanvibe.ai",
        password="change_me_now_123!",  # Force password change on first login
        is_admin=True,
        permissions=admin_permissions,
    )

    logger.info("Default admin user created", user_id=admin_user.id)
    return admin_user


if __name__ == "__main__":
    # Create default admin user for development
    create_default_admin()
