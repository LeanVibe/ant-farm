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

    def __init__(self, config: SecurityConfig | None = None):
        self.config = config or SecurityConfig()
        self.users: dict[str, User] = {}
        self.sessions: dict[str, dict] = {}
        self.rate_limits: dict[str, list[float]] = {}
        self.refresh_tokens: dict[str, dict] = {}  # Store refresh token metadata
        self.token_blacklist: set[str] = set()  # Blacklist for invalidated tokens
        self.audit_log: list[dict] = []  # Security audit log

    def _audit_log(self, event: str, details: dict):
        """Log security event for audit purposes."""
        audit_entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            "details": details,
        }
        self.audit_log.append(audit_entry)

        # Keep only last 1000 entries to prevent memory issues
        if len(self.audit_log) > 1000:
            self.audit_log.pop(0)

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
        permissions: list[str] | None = None,
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
        """Create JWT refresh token with rotation support."""
        expire = datetime.now(UTC) + timedelta(
            days=self.config.REFRESH_TOKEN_EXPIRE_DAYS
        )

        # Generate unique token ID for tracking
        token_id = str(uuid.uuid4())

        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.now(UTC),
            "type": "refresh",
            "jti": token_id,  # JWT ID for token tracking
        }

        token = jwt.encode(
            payload, self.config.SECRET_KEY, algorithm=self.config.ALGORITHM
        )

        # Store refresh token metadata for rotation
        self.refresh_tokens[token_id] = {
            "user_id": user.id,
            "created_at": datetime.now(UTC),
            "expires_at": expire,
            "used": False,
        }

        # Audit log token creation
        self._audit_log("refresh_token_created", {
            "user_id": user.id,
            "token_id": token_id,
            "expires_at": expire.isoformat(),
        })

        logger.info("Refresh token created", user_id=user.id, token_id=token_id)
        return token

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify and decode JWT token with blacklist checking."""
        try:
            payload = jwt.decode(
                token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM]
            )

            # Check if token is blacklisted
            token_id = payload.get("jti")
            if token_id and token_id in self.token_blacklist:
                logger.warning("Token has been blacklisted", token_id=token_id)
                return None

            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except Exception as e:
            logger.warning("Invalid token", error=str(e))
            return None

    def rotate_refresh_token(self, old_token: str) -> str | None:
        """Rotate refresh token - invalidate old one and create new one."""
        try:
            payload = jwt.decode(
                old_token, self.config.SECRET_KEY, algorithms=[self.config.ALGORITHM]
            )

            if payload.get("type") != "refresh":
                logger.warning("Attempted to rotate non-refresh token")
                return None

            token_id = payload.get("jti")
            user_id = payload.get("sub")

            if not token_id or not user_id:
                logger.warning("Invalid token payload for rotation")
                return None

            # Check if token was already used
            if token_id in self.refresh_tokens:
                token_data = self.refresh_tokens[token_id]
                if token_data.get("used", False):
                    logger.warning("Refresh token already used", token_id=token_id)
                    return None

                # Mark old token as used
                token_data["used"] = True
                token_data["used_at"] = datetime.now(UTC)

            # Blacklist the old token
            self.token_blacklist.add(token_id)

            # Get user
            if user_id not in self.users:
                logger.warning("User not found for token rotation", user_id=user_id)
                return None

            user = self.users[user_id]

            # Create new refresh token
            new_token = self.create_refresh_token(user)

            logger.info("Refresh token rotated", user_id=user_id, old_token_id=token_id)
            return new_token

        except Exception as e:
            logger.error("Token rotation failed", error=str(e))
            return None

    def invalidate_user_tokens(self, user_id: str):
        """Invalidate all tokens for a user."""
        # Blacklist all refresh tokens for this user
        tokens_to_blacklist = []
        for token_id, token_data in self.refresh_tokens.items():
            if token_data.get("user_id") == user_id:
                tokens_to_blacklist.append(token_id)

        for token_id in tokens_to_blacklist:
            self.token_blacklist.add(token_id)
            if token_id in self.refresh_tokens:
                self.refresh_tokens[token_id]["used"] = True
                self.refresh_tokens[token_id]["invalidated_at"] = datetime.now(UTC)

        logger.info("All tokens invalidated for user", user_id=user_id, tokens_blacklisted=len(tokens_to_blacklist))

    def cleanup_expired_tokens(self):
        """Clean up expired refresh tokens and old blacklist entries."""
        now = datetime.now(UTC)

        # Remove expired refresh tokens
        expired_tokens = []
        for token_id, token_data in self.refresh_tokens.items():
            if token_data.get("expires_at", now) < now:
                expired_tokens.append(token_id)

        for token_id in expired_tokens:
            del self.refresh_tokens[token_id]

        # Clean up old blacklist entries (older than 30 days)
        old_blacklist_cutoff = now - timedelta(days=30)
        old_blacklist = []
        for token_id, token_data in self.refresh_tokens.items():
            invalidated_at = token_data.get("invalidated_at")
            if invalidated_at and invalidated_at < old_blacklist_cutoff:
                old_blacklist.append(token_id)

        for token_id in old_blacklist:
            self.token_blacklist.discard(token_id)

        if expired_tokens or old_blacklist:
            logger.info("Cleaned up expired tokens",
                       expired_refresh_tokens=len(expired_tokens),
                       cleaned_blacklist_entries=len(old_blacklist))

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
    def generate_api_key(self, user: User, name: str | None = None) -> str:
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
