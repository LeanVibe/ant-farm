"""OAuth2 password flow and scope enforcement tests."""

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.security import security_manager, create_default_admin


def get_client():
    return TestClient(app)


def test_oauth2_token_success():
    client = get_client()
    # Ensure admin exists
    create_default_admin()

    # Request token via OAuth2 password flow
    resp = client.post(
        "/api/v1/auth/token",
        data={
            "username": "admin",
            "password": "change_me_now_123!",
            "grant_type": "password",
            "scope": "system:read agent:read",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert resp.status_code == 200
    data = resp.json()["data"]
    assert "access_token" in data and data["token_type"] == "bearer"
    assert isinstance(data.get("expires_in"), int)


def test_oauth2_scope_enforcement_forbidden_on_insufficient_scope():
    client = get_client()

    # Create a limited-permission user
    user = security_manager.create_user(
        username="limited",
        email="limited@example.com",
        password="testpass123",
        is_admin=False,
        permissions=["agent:read"],
    )

    # Obtain token requesting higher scope than user has
    resp = client.post(
        "/api/v1/auth/token",
        data={
            "username": "limited",
            "password": "testpass123",
            "grant_type": "password",
            "scope": "system:write",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert resp.status_code == 200
    token = resp.json()["data"]["access_token"]

    # Try to access an endpoint requiring system:write
    headers = {"Authorization": f"Bearer {token}"}
    r2 = client.post("/api/v1/system/analyze", headers=headers)

    assert r2.status_code == 403


def test_oauth2_token_invalid_credentials_unauthorized():
    client = get_client()

    resp = client.post(
        "/api/v1/auth/token",
        data={
            "username": "ghost",
            "password": "nope",
            "grant_type": "password",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    assert resp.status_code == 401
