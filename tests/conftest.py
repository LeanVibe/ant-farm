"""Test fixtures and configuration for LeanVibe Agent Hive tests."""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any

import pytest
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.models import User
from src.core.security import security_manager

# Test configuration - use environment variables with fallbacks
TEST_DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://test_user:test_pass@localhost:5432/leanvibe_hive_test",
)
TEST_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Create a Redis client for testing."""
    client = redis.Redis.from_url(TEST_REDIS_URL, decode_responses=True)
    try:
        yield client
    finally:
        await client.flushdb()  # Clean up after test
        await client.aclose()


@pytest.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create a database session for testing."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with async_session() as session:
        yield session


@pytest.fixture
def mock_cli_tools() -> dict:
    """Mock CLI tools for testing."""
    return {
        "opencode": {
            "name": "OpenCode",
            "command": "opencode",
            "execute_pattern": 'opencode "{prompt}"',
            "available": True,
        },
        "claude": {
            "name": "Claude Code CLI",
            "command": "claude",
            "execute_pattern": 'claude --no-interactive "{prompt}"',
            "available": True,
        },
    }


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.name = "test-agent"
    agent.agent_type = "test"
    agent.status = "active"
    agent.execute_with_cli_tool = AsyncMock(return_value="Mock response")
    return agent


@pytest.fixture
def sample_task():
    """Create a sample task for testing."""
    from src.core.task_queue import Task, TaskPriority

    return Task(
        id="test-task-123",
        title="Test Task",
        description="A test task for unit testing",
        task_type="test",
        payload={"test_data": "value"},
        priority=TaskPriority.MEDIUM,
    )


class AsyncMockRedis:
    """Mock Redis client for testing."""

    def __init__(self):
        self.data = {}
        self.lists = {}
        self.pubsub_messages = []

    async def hset(self, key: str, mapping: dict) -> None:
        if key not in self.data:
            self.data[key] = {}
        self.data[key].update(mapping)

    async def hgetall(self, key: str) -> dict:
        return self.data.get(key, {})

    async def lpush(self, key: str, value: str) -> None:
        if key not in self.lists:
            self.lists[key] = []
        self.lists[key].insert(0, value)

    async def brpop(self, key: str, timeout: int = 1) -> tuple:
        if key in self.lists and self.lists[key]:
            return (key, self.lists[key].pop())
        return None

    async def publish(self, channel: str, message: str) -> None:
        self.pubsub_messages.append((channel, message))

    async def ping(self) -> bool:
        return True


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    return AsyncMockRedis()


# User Profile Testing Fixtures

@pytest.fixture
def test_client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Create mock user for testing."""
    user = MagicMock(spec=User)
    user.id = "test-user-id"
    user.username = "testuser"
    user.email = "test@example.com"
    user.is_active = True
    user.is_admin = False
    user.permissions = ["read"]
    user.first_name = "Test"
    user.last_name = "User"
    user.bio = "Test bio"
    user.preferences = {"theme": "dark"}
    return user


@pytest.fixture
def mock_admin_user():
    """Create mock admin user for testing."""
    user = MagicMock(spec=User)
    user.id = "admin-user-id"
    user.username = "admin"
    user.email = "admin@example.com"
    user.is_active = True
    user.is_admin = True
    user.permissions = ["system:admin"]
    user.first_name = "Admin"
    user.last_name = "User"
    user.bio = "Admin bio"
    user.preferences = {"theme": "light"}
    return user


@pytest.fixture
def auth_headers(mock_user):
    """Create authentication headers for test user."""
    # Mock the token creation for testing
    from unittest.mock import patch
    with patch.object(security_manager, 'create_access_token', return_value='mock.jwt.token'):
        token = security_manager.create_access_token(mock_user)
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def admin_auth_headers(mock_admin_user):
    """Create authentication headers for admin user."""
    from unittest.mock import patch
    with patch.object(security_manager, 'create_access_token', return_value='mock.admin.jwt.token'):
        token = security_manager.create_access_token(mock_admin_user)
        return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "SecurePass123!",
        "first_name": "Test",
        "last_name": "User",
        "bio": "Test bio",
        "preferences": {"theme": "dark", "notifications": True},
        "timezone": "UTC",
        "language": "en",
    }


@pytest.fixture
def sample_update_data():
    """Sample update data for testing."""
    return {
        "first_name": "Updated",
        "last_name": "Name",
        "bio": "Updated bio",
        "preferences": {"theme": "light"},
        "timezone": "America/New_York",
        "language": "es",
    }


@pytest.fixture
def large_user_data():
    """Large user data for performance testing."""
    return {
        "username": "largetestuser",
        "email": "large@example.com",
        "password": "SecurePass123!",
        "first_name": "Large",
        "last_name": "Test",
        "bio": "A" * 4000,  # 4KB bio
        "preferences": {f"key_{i}": f"value_{i}" * 10 for i in range(100)},  # Large preferences
        "timezone": "UTC",
        "language": "en",
    }


@pytest.fixture
def edge_case_user_data():
    """Edge case user data for testing."""
    return {
        "max_username": "a" * 100,
        "max_email": "a" * 243 + "@example.com",
        "unicode_name": "José María ñoño 😀",
        "special_chars": "user_ñ_ü_ç",
        "whitespace_name": "  John  ",
        "empty_optional": "",
        "none_optional": None,
    }


@pytest.fixture
def malicious_user_data():
    """Malicious user data for security testing."""
    return {
        "sql_injection": {
            "username": "'; DROP TABLE users; --",
            "bio": "' OR '1'='1",
            "first_name": "'; UPDATE users SET is_admin=true; --",
        },
        "xss_payloads": {
            "bio": "<script>alert('xss')</script>",
            "first_name": "<img src=x onerror=alert(1)>",
            "last_name": "javascript:alert('xss')",
        },
        "invalid_types": {
            "preferences": "not_a_dict",
            "is_admin": "not_a_bool",
            "permissions": "not_a_list",
        }
    }


@pytest.fixture
def performance_test_data():
    """Performance test data sets."""
    return {
        "small_batch": [
            {
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "password": "SecurePass123!",
                "first_name": f"User{i}",
                "last_name": "Test",
            }
            for i in range(10)
        ],
        "medium_batch": [
            {
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "password": "SecurePass123!",
                "first_name": f"User{i}",
                "last_name": "Test",
                "bio": f"Bio for user {i}",
            }
            for i in range(100)
        ],
        "large_batch": [
            {
                "username": f"user_{i}",
                "email": f"user_{i}@example.com",
                "password": "SecurePass123!",
                "first_name": f"User{i}",
                "last_name": "Test",
                "bio": f"Bio for user {i}" * 10,  # Larger bio
                "preferences": {"theme": "dark", "index": i},
            }
            for i in range(1000)
        ],
    }


@pytest.fixture
def concurrent_test_setup():
    """Setup for concurrent operation testing."""
    return {
        "user_count": 50,
        "operations_per_user": 10,
        "max_concurrent": 20,
        "timeout_seconds": 30,
    }


@pytest.fixture
def load_test_configuration():
    """Load testing configuration."""
    return {
        "target_rps": {
            "read": 100,
            "update": 50,
            "create": 20,
            "delete": 10,
        },
        "response_time_limits": {
            "read_p95": 200,  # ms
            "update_p95": 500,  # ms
            "create_p95": 1000,  # ms
            "delete_p95": 2000,  # ms
        },
        "concurrent_users": {
            "read": 500,
            "update": 250,
            "create": 100,
            "delete": 50,
        },
        "test_duration_seconds": 60,
    }


@pytest.fixture
def security_test_scenarios():
    """Security test scenarios configuration."""
    return {
        "authentication_tests": [
            "expired_tokens",
            "invalid_signatures",
            "tampered_payloads",
            "missing_tokens",
        ],
        "authorization_tests": [
            "user_isolation",
            "admin_privileges",
            "permission_elevation",
            "cross_user_access",
        ],
        "input_validation_tests": [
            "sql_injection",
            "xss_prevention",
            "command_injection",
            "path_traversal",
        ],
        "data_protection_tests": [
            "sensitive_data_exposure",
            "audit_logging",
            "encryption_verification",
        ],
    }


@pytest.fixture
def compliance_test_requirements():
    """Compliance testing requirements."""
    return {
        "gdpr": {
            "data_portability": True,
            "right_to_erasure": True,
            "consent_management": True,
            "data_minimization": True,
        },
        "security_standards": {
            "owasp_top_10": True,
            "password_policy": True,
            "session_management": True,
            "error_handling": True,
        },
        "accessibility": {
            "wcag_compliance": False,  # Not applicable for API
            "internationalization": True,
        },
    }
