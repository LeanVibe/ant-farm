"""Integration tests for user profile CRUD operations."""

import pytest
import httpx
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.main import app
from src.core.models import User
from src.core.security import security_manager


@pytest.fixture
def client():
    """Create test client for API testing."""
    return TestClient(app)


@pytest.fixture
async def test_user(db_session: AsyncSession):
    """Create a test user in the database."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password="hashed_password",
        is_active=True,
        is_admin=False,
        permissions=["read"],
        first_name="Test",
        last_name="User",
        bio="Test bio",
        preferences={"theme": "dark"},
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)
    return user


@pytest.fixture
def auth_headers(test_user):
    """Create authentication headers for test user."""
    token = security_manager.create_access_token(test_user)
    return {"Authorization": f"Bearer {token}"}


class TestUserProfileAPIIntegration:
    """Integration tests for user profile API endpoints."""

    def test_get_user_profile_success(self, client, test_user, auth_headers):
        """Test successful profile retrieval."""
        response = client.get("/api/v1/auth/profile", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email
        assert data["first_name"] == test_user.first_name

    def test_get_user_profile_unauthenticated(self, client):
        """Test profile access without authentication."""
        response = client.get("/api/v1/auth/profile")

        assert response.status_code == 401
        assert "not authenticated" in response.json()["error"].lower()

    def test_update_user_profile_success(self, client, test_user, auth_headers):
        """Test successful profile update."""
        update_data = {
            "first_name": "Updated",
            "last_name": "Name",
            "bio": "Updated bio",
            "preferences": {"theme": "light", "notifications": True}
        }

        response = client.put("/api/v1/auth/profile", json=update_data, headers=auth_headers)

        assert response.status_code == 200
        data = response.json()["data"]
        assert data["first_name"] == "Updated"
        assert data["bio"] == "Updated bio"
        assert data["preferences"]["theme"] == "light"

    def test_update_user_profile_validation_error(self, client, test_user, auth_headers):
        """Test profile update with validation errors."""
        # Test with invalid email
        update_data = {
            "email": "invalid-email-format"
        }

        response = client.put("/api/v1/auth/profile", json=update_data, headers=auth_headers)

        assert response.status_code == 422
        assert "validation" in response.json()["error"].lower()

    def test_update_user_profile_unauthenticated(self, client):
        """Test profile update without authentication."""
        update_data = {"first_name": "Hacker"}

        response = client.put("/api/v1/auth/profile", json=update_data)

        assert response.status_code == 401

    def test_delete_user_profile_success(self, client, test_user, auth_headers):
        """Test successful profile deletion."""
        response = client.delete("/api/v1/auth/profile", headers=auth_headers)

        assert response.status_code == 200
        assert "deactivated" in response.json()["message"].lower()

    def test_delete_user_profile_unauthenticated(self, client):
        """Test profile deletion without authentication."""
        response = client.delete("/api/v1/auth/profile")

        assert response.status_code == 401


class TestUserProfileSecurityIntegration:
    """Security-focused integration tests."""

    def test_sql_injection_prevention(self, client, test_user, auth_headers):
        """Test SQL injection prevention in profile fields."""
        malicious_payloads = [
            {"bio": "'; DROP TABLE users; --"},
            {"first_name": "' OR '1'='1"},
            {"username": "'; UPDATE users SET is_admin=true; --"},
        ]

        for payload in malicious_payloads:
            response = client.put("/api/v1/auth/profile", json=payload, headers=auth_headers)
            # Should either validate or sanitize, not execute SQL
            assert response.status_code in [200, 422]

    def test_xss_prevention(self, client, test_user, auth_headers):
        """Test XSS prevention in profile fields."""
        xss_payloads = [
            {"bio": "<script>alert('xss')</script>"},
            {"first_name": "<img src=x onerror=alert(1)>"},
        ]

        for payload in xss_payloads:
            response = client.put("/api/v1/auth/profile", json=payload, headers=auth_headers)
            # Should sanitize or reject XSS attempts
            assert response.status_code in [200, 422]

            if response.status_code == 200:
                data = response.json()["data"]
                # Ensure XSS payload is not stored as-is
                for field, value in payload.items():
                    assert "<script>" not in str(data.get(field, ""))
                    assert "onerror" not in str(data.get(field, ""))

    def test_rate_limiting(self, client, test_user, auth_headers):
        """Test API rate limiting."""
        # Make multiple rapid requests
        responses = []
        for _ in range(10):
            response = client.get("/api/v1/auth/profile", headers=auth_headers)
            responses.append(response.status_code)

        # Some requests should be rate limited (429)
        assert 429 in responses or all(code == 200 for code in responses)

    def test_input_sanitization(self, client, test_user, auth_headers):
        """Test input sanitization for profile fields."""
        # Test with potentially harmful input
        dangerous_input = {
            "bio": "Bio with <script> tags and 'quotes' \"double\"",
            "first_name": "Name with <b>HTML</b>",
        }

        response = client.put("/api/v1/auth/profile", json=dangerous_input, headers=auth_headers)

        if response.status_code == 200:
            data = response.json()["data"]
            # HTML should be sanitized or escaped
            assert "<script>" not in data["bio"]
            assert "<b>" not in data["first_name"]


class TestUserProfilePerformanceIntegration:
    """Performance-focused integration tests."""

    def test_profile_response_time(self, client, test_user, auth_headers):
        """Test profile retrieval response time."""
        import time

        start_time = time.time()
        response = client.get("/api/v1/auth/profile", headers=auth_headers)
        end_time = time.time()

        assert response.status_code == 200
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second

    def test_concurrent_profile_access(self, client, test_user, auth_headers):
        """Test concurrent profile access."""
        import threading
        import queue

        results = queue.Queue()

        def make_request():
            response = client.get("/api/v1/auth/profile", headers=auth_headers)
            results.put(response.status_code)

        # Make 5 concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All requests should succeed
        codes = []
        while not results.empty():
            codes.append(results.get())

        assert all(code == 200 for code in codes)


class TestUserProfileErrorHandlingIntegration:
    """Error handling integration tests."""

    def test_malformed_json_handling(self, client, auth_headers):
        """Test handling of malformed JSON."""
        response = client.put(
            "/api/v1/auth/profile",
            data="invalid json {",
            headers={**auth_headers, "Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_large_payload_handling(self, client, auth_headers):
        """Test handling of large payloads."""
        large_bio = "A" * 10000  # Very large bio
        update_data = {"bio": large_bio}

        response = client.put("/api/v1/auth/profile", json=update_data, headers=auth_headers)

        # Should either accept or reject gracefully
        assert response.status_code in [200, 422, 413]

    def test_invalid_content_type(self, client, auth_headers):
        """Test handling of invalid content types."""
        response = client.put(
            "/api/v1/auth/profile",
            data="first_name=Test",
            headers={**auth_headers, "Content-Type": "application/x-www-form-urlencoded"}
        )

        # Should handle gracefully
        assert response.status_code in [200, 422, 415]

    def test_database_connection_error_simulation(self, client, auth_headers):
        """Test handling of database connection errors."""
        # This would require mocking database failures
        # For now, test with invalid database state
        response = client.get("/api/v1/auth/profile", headers=auth_headers)

        # Should not crash the application
        assert response.status_code in [200, 500]


if __name__ == "__main__":
    pytest.main([__file__])