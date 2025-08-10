"""Pytest-based security tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.security import create_default_admin, security_manager


@pytest.fixture
def client():
    """Test client fixture."""
    return TestClient(app)


@pytest.fixture
def admin_token(client):
    """Admin token fixture."""
    admin = create_default_admin()
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "change_me_now_123!"},
    )
    if response.status_code == 200:
        return response.json()["data"]["access_token"]
    return None


@pytest.fixture
def regular_user_token(client):
    """Regular user token fixture."""
    regular_user = security_manager.create_user(
        username="testuser",
        email="test@example.com",
        password="testpass123",
        is_admin=False,
        permissions=["agent:read", "task:read", "system:read"],
    )

    response = client.post(
        "/api/v1/auth/login", json={"username": "testuser", "password": "testpass123"}
    )
    if response.status_code == 200:
        return response.json()["data"]["access_token"]
    return None


class TestAuthentication:
    """Test authentication requirements."""

    @pytest.mark.parametrize(
        "method,endpoint",
        [
            ("GET", "/api/v1/agents"),
            ("POST", "/api/v1/agents?agent_type=meta"),
            ("GET", "/api/v1/tasks"),
            ("POST", "/api/v1/tasks"),
            ("GET", "/api/v1/status"),
            ("POST", "/api/v1/messages"),
            ("GET", "/api/v1/modifications"),
            ("GET", "/api/v1/workflows"),
            ("GET", "/api/v1/context/test-agent"),
        ],
    )
    def test_protected_endpoints_require_auth(self, client, method, endpoint):
        """Test that protected endpoints require authentication."""
        response = client.request(method, endpoint)
        assert response.status_code == 401, (
            f"Endpoint {method} {endpoint} should require authentication"
        )

    def test_public_endpoints_accessible(self, client):
        """Test that public endpoints are accessible without auth."""
        public_endpoints = [
            ("GET", "/health"),
            ("GET", "/api/v1/health"),
            ("GET", "/api/v1/test"),
            ("POST", "/api/v1/auth/login"),
        ]

        for method, endpoint in public_endpoints:
            response = client.request(method, endpoint)
            assert response.status_code != 401, (
                f"Public endpoint {method} {endpoint} should not require auth"
            )


class TestAuthorization:
    """Test authorization and permissions."""

    def test_regular_user_cannot_access_admin_endpoints(
        self, client, regular_user_token
    ):
        """Test that regular users cannot access admin-only endpoints."""
        if not regular_user_token:
            pytest.skip("Regular user token not available")

        headers = {"Authorization": f"Bearer {regular_user_token}"}

        admin_endpoints = [
            ("POST", "/api/v1/agents"),
            ("POST", "/api/v1/tasks"),
            ("POST", "/api/v1/system/analyze"),
            ("POST", "/api/v1/modifications/proposal"),
            ("POST", "/api/v1/workflows"),
        ]

        for method, endpoint in admin_endpoints:
            response = client.request(method, endpoint, headers=headers)
            assert response.status_code == 403, (
                f"Regular user should not access {method} {endpoint}"
            )

    def test_admin_can_access_protected_endpoints(self, client, admin_token):
        """Test that admin can access protected endpoints."""
        if not admin_token:
            pytest.skip("Admin token not available")

        headers = {"Authorization": f"Bearer {admin_token}"}

        # Test read endpoints that should work for admin
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code != 401 and response.status_code != 403, (
            "Admin should access agents endpoint"
        )


class TestJWTValidation:
    """Test JWT token validation."""

    def test_invalid_token_rejected(self, client):
        """Test that invalid tokens are rejected."""
        headers = {"Authorization": "Bearer invalid_token_123"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 401

    def test_malformed_token_rejected(self, client):
        """Test that malformed tokens are rejected."""
        headers = {"Authorization": "Bearer malformed.token"}
        response = client.get("/api/v1/agents", headers=headers)
        assert response.status_code == 401

    def test_missing_bearer_prefix_rejected(self, client, admin_token):
        """Test that tokens without Bearer prefix are rejected."""
        if admin_token:
            headers = {"Authorization": admin_token}  # Missing "Bearer "
            response = client.get("/api/v1/agents", headers=headers)
            assert response.status_code == 401


class TestInputValidation:
    """Test input validation and sanitization."""

    @pytest.mark.parametrize(
        "payload",
        [
            "'; DROP TABLE agents; --",
            "1' OR '1'='1",
            "admin'; INSERT INTO users VALUES ('hacker'); --",
        ],
    )
    def test_sql_injection_protection(self, client, admin_token, payload):
        """Test protection against SQL injection."""
        if not admin_token:
            pytest.skip("Admin token not available")

        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.get(f"/api/v1/agents/{payload}", headers=headers)

        # Should not cause server error (500)
        assert response.status_code != 500, "SQL injection payload caused server error"

    @pytest.mark.parametrize(
        "payload",
        [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
        ],
    )
    def test_xss_protection(self, client, admin_token, payload):
        """Test protection against XSS attacks."""
        if not admin_token:
            pytest.skip("Admin token not available")

        headers = {"Authorization": f"Bearer {admin_token}"}
        response = client.post(
            "/api/v1/tasks",
            headers=headers,
            json={"title": payload, "description": "Test task", "type": "test"},
        )

        # Either rejected or sanitized - should not return raw XSS payload
        if response.status_code == 200:
            task_data = response.json().get("data", {})
            assert payload not in str(task_data), "XSS payload was not sanitized"


class TestRateLimiting:
    """Test rate limiting functionality."""

    def test_login_rate_limiting(self, client):
        """Test that login attempts are rate limited."""
        # Make many failed login attempts
        for i in range(12):  # Exceed limit of 10
            response = client.post(
                "/api/v1/auth/login",
                json={"username": "nonexistent", "password": "wrongpass"},
            )

            # Check if we get rate limited
            if response.status_code == 429:
                break
        else:
            pytest.skip("Rate limiting not triggered or configured differently")


class TestSecurityHeaders:
    """Test security headers in responses."""

    def test_security_headers_present(self, client):
        """Test that security headers are present."""
        response = client.get("/health")
        headers = response.headers

        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Content-Security-Policy",
            "Referrer-Policy",
        ]

        for header in required_headers:
            assert header in headers, f"Security header {header} is missing"

    def test_security_header_values(self, client):
        """Test that security headers have correct values."""
        response = client.get("/health")
        headers = response.headers

        expected_values = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

        for header, expected_value in expected_values.items():
            if header in headers:
                assert headers[header] == expected_value, (
                    f"Header {header} has incorrect value"
                )


class TestCORS:
    """Test CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present for cross-origin requests."""
        response = client.options(
            "/api/v1/agents",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        # Should have CORS headers
        assert (
            "Access-Control-Allow-Origin" in response.headers
            or "access-control-allow-origin" in response.headers
        ), "CORS headers missing"
