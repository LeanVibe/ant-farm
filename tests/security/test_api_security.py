"""
Comprehensive security test suite for API endpoints.

Tests authentication, authorization, rate limiting, input validation,
and security headers across all API endpoints.
"""

import asyncio
import json
import time
from typing import Any, Dict, List

import httpx
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.security import create_default_admin, security_manager


class SecurityTestSuite:
    """Comprehensive security testing for API endpoints."""

    def __init__(self):
        self.client = TestClient(app)
        self.base_url = "http://testserver"
        self.admin_token = None
        self.regular_user_token = None

    def setup_test_users(self):
        """Create test users and get tokens."""
        # Create admin user
        admin = create_default_admin()

        # Create regular user
        regular_user = security_manager.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            is_admin=False,
            permissions=["agent:read", "task:read", "system:read"],
        )

        # Get tokens
        admin_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "change_me_now_123!"},
        )
        if admin_response.status_code == 200:
            self.admin_token = admin_response.json()["data"]["access_token"]

        user_response = self.client.post(
            "/api/v1/auth/login",
            json={"username": "testuser", "password": "testpass123"},
        )
        if user_response.status_code == 200:
            self.regular_user_token = user_response.json()["data"]["access_token"]

    def get_auth_header(self, token: str) -> dict[str, str]:
        """Get authorization header for requests."""
        return {"Authorization": f"Bearer {token}"}

    def test_authentication_required(self):
        """Test that protected endpoints require authentication."""
        protected_endpoints = [
            ("GET", "/api/v1/agents"),
            ("POST", "/api/v1/agents?agent_type=meta"),
            ("GET", "/api/v1/tasks"),
            ("POST", "/api/v1/tasks"),
            ("GET", "/api/v1/status"),
            ("POST", "/api/v1/messages"),
            ("GET", "/api/v1/modifications"),
            ("POST", "/api/v1/modifications/proposal"),
            ("GET", "/api/v1/workflows"),
            ("POST", "/api/v1/workflows"),
            ("GET", "/api/v1/adw/metrics/current"),
            ("POST", "/api/v1/adw/monitoring/start"),
            ("GET", "/api/v1/context/test-agent"),
            ("POST", "/api/v1/context/test-agent/search?query=test"),
        ]

        results = []
        for method, endpoint in protected_endpoints:
            response = self.client.request(method, endpoint)

            # Should return 401 Unauthorized
            if response.status_code != 401:
                results.append(
                    {
                        "endpoint": f"{method} {endpoint}",
                        "expected": 401,
                        "actual": response.status_code,
                        "status": "FAIL",
                    }
                )
            else:
                results.append(
                    {
                        "endpoint": f"{method} {endpoint}",
                        "expected": 401,
                        "actual": response.status_code,
                        "status": "PASS",
                    }
                )

        return results

    def test_authorization_enforcement(self):
        """Test that users can only access endpoints they have permissions for."""
        if not self.regular_user_token:
            return [{"error": "No regular user token available"}]

        # Test endpoints that require higher privileges
        admin_only_endpoints = [
            ("POST", "/api/v1/agents"),  # requires agent:spawn
            ("POST", "/api/v1/tasks"),  # requires task:create
            ("POST", "/api/v1/system/analyze"),  # requires system:write
            ("POST", "/api/v1/modifications/proposal"),  # requires modification:propose
            ("POST", "/api/v1/workflows"),  # requires system:write
        ]

        results = []
        for method, endpoint in admin_only_endpoints:
            headers = self.get_auth_header(self.regular_user_token)
            response = self.client.request(method, endpoint, headers=headers)

            # Should return 403 Forbidden (insufficient permissions)
            if response.status_code != 403:
                results.append(
                    {
                        "endpoint": f"{method} {endpoint}",
                        "expected": 403,
                        "actual": response.status_code,
                        "status": "FAIL",
                        "message": "Regular user should not have access",
                    }
                )
            else:
                results.append(
                    {
                        "endpoint": f"{method} {endpoint}",
                        "expected": 403,
                        "actual": response.status_code,
                        "status": "PASS",
                    }
                )

        return results

    def test_jwt_token_validation(self):
        """Test JWT token validation and expiration."""
        results = []

        # Test invalid token
        invalid_headers = {"Authorization": "Bearer invalid_token_123"}
        response = self.client.get("/api/v1/agents", headers=invalid_headers)

        if response.status_code == 401:
            results.append(
                {
                    "test": "Invalid JWT token",
                    "status": "PASS",
                    "expected": 401,
                    "actual": response.status_code,
                }
            )
        else:
            results.append(
                {
                    "test": "Invalid JWT token",
                    "status": "FAIL",
                    "expected": 401,
                    "actual": response.status_code,
                }
            )

        # Test malformed token
        malformed_headers = {"Authorization": "Bearer malformed.token"}
        response = self.client.get("/api/v1/agents", headers=malformed_headers)

        if response.status_code == 401:
            results.append(
                {
                    "test": "Malformed JWT token",
                    "status": "PASS",
                    "expected": 401,
                    "actual": response.status_code,
                }
            )
        else:
            results.append(
                {
                    "test": "Malformed JWT token",
                    "status": "FAIL",
                    "expected": 401,
                    "actual": response.status_code,
                }
            )

        # Test missing Bearer prefix
        no_bearer_headers = {"Authorization": self.admin_token or "token123"}
        response = self.client.get("/api/v1/agents", headers=no_bearer_headers)

        if response.status_code == 401:
            results.append(
                {
                    "test": "Missing Bearer prefix",
                    "status": "PASS",
                    "expected": 401,
                    "actual": response.status_code,
                }
            )
        else:
            results.append(
                {
                    "test": "Missing Bearer prefix",
                    "status": "FAIL",
                    "expected": 401,
                    "actual": response.status_code,
                }
            )

        return results

    def test_input_validation(self):
        """Test input validation and sanitization."""
        if not self.admin_token:
            return [{"error": "No admin token available"}]

        results = []
        headers = self.get_auth_header(self.admin_token)

        # Test SQL injection attempts
        sql_injection_payloads = [
            "'; DROP TABLE agents; --",
            "1' OR '1'='1",
            "admin'; INSERT INTO users VALUES ('hacker'); --",
        ]

        for payload in sql_injection_payloads:
            response = self.client.get(f"/api/v1/agents/{payload}", headers=headers)

            # Should not cause server error (500) - should be handled gracefully
            if response.status_code == 500:
                results.append(
                    {
                        "test": f"SQL injection protection ({payload[:20]}...)",
                        "status": "FAIL",
                        "message": "Server error indicates possible SQL injection vulnerability",
                    }
                )
            else:
                results.append(
                    {
                        "test": f"SQL injection protection ({payload[:20]}...)",
                        "status": "PASS",
                        "message": f"Handled gracefully with status {response.status_code}",
                    }
                )

        # Test XSS attempts
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
        ]

        for payload in xss_payloads:
            # Test in task creation
            response = self.client.post(
                "/api/v1/tasks",
                headers=headers,
                json={"title": payload, "description": "Test task", "type": "test"},
            )

            # Check if XSS payload is sanitized or rejected
            if response.status_code == 200:
                task_data = response.json().get("data", {})
                if payload in str(task_data):
                    results.append(
                        {
                            "test": f"XSS protection ({payload[:20]}...)",
                            "status": "FAIL",
                            "message": "XSS payload not sanitized",
                        }
                    )
                else:
                    results.append(
                        {
                            "test": f"XSS protection ({payload[:20]}...)",
                            "status": "PASS",
                            "message": "XSS payload handled safely",
                        }
                    )
            else:
                results.append(
                    {
                        "test": f"XSS protection ({payload[:20]}...)",
                        "status": "PASS",
                        "message": f"XSS payload rejected with status {response.status_code}",
                    }
                )

        return results

    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        results = []

        # Test login rate limiting (10 requests/minute)
        login_attempts = []
        for i in range(12):  # Exceed limit of 10
            response = self.client.post(
                "/api/v1/auth/login",
                json={"username": "nonexistent", "password": "wrongpass"},
            )
            login_attempts.append(response.status_code)

        # Check if rate limiting kicked in
        rate_limited = any(
            status == 429 for status in login_attempts[-3:]
        )  # Check last 3 attempts

        if rate_limited:
            results.append(
                {
                    "test": "Login rate limiting",
                    "status": "PASS",
                    "message": "Rate limiting active after excessive requests",
                }
            )
        else:
            results.append(
                {
                    "test": "Login rate limiting",
                    "status": "FAIL",
                    "message": "No rate limiting detected",
                }
            )

        return results

    def test_security_headers(self):
        """Test security headers in responses."""
        response = self.client.get("/health")
        headers = response.headers

        required_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Content-Security-Policy": True,  # Just check presence
            "Referrer-Policy": "strict-origin-when-cross-origin",
        }

        results = []
        for header, expected_value in required_headers.items():
            if header in headers:
                if expected_value is True:  # Just check presence
                    results.append(
                        {
                            "test": f"Security header: {header}",
                            "status": "PASS",
                            "value": headers[header],
                        }
                    )
                elif headers[header] == expected_value:
                    results.append(
                        {
                            "test": f"Security header: {header}",
                            "status": "PASS",
                            "value": headers[header],
                        }
                    )
                else:
                    results.append(
                        {
                            "test": f"Security header: {header}",
                            "status": "FAIL",
                            "expected": expected_value,
                            "actual": headers[header],
                        }
                    )
            else:
                results.append(
                    {
                        "test": f"Security header: {header}",
                        "status": "FAIL",
                        "message": "Header missing",
                    }
                )

        return results

    def test_cors_configuration(self):
        """Test CORS configuration."""
        results = []

        # Test preflight request
        response = self.client.options(
            "/api/v1/agents",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Authorization",
            },
        )

        if "Access-Control-Allow-Origin" in response.headers:
            results.append(
                {
                    "test": "CORS headers present",
                    "status": "PASS",
                    "headers": dict(response.headers),
                }
            )
        else:
            results.append(
                {
                    "test": "CORS headers present",
                    "status": "FAIL",
                    "message": "CORS headers missing",
                }
            )

        return results

    def run_comprehensive_security_audit(self) -> dict[str, Any]:
        """Run all security tests and return comprehensive report."""
        print("🔒 Starting Comprehensive Security Audit...")

        self.setup_test_users()

        audit_results = {"timestamp": time.time(), "summary": {}, "tests": {}}

        # Run all test categories
        test_categories = [
            ("authentication", self.test_authentication_required),
            ("authorization", self.test_authorization_enforcement),
            ("jwt_validation", self.test_jwt_token_validation),
            ("input_validation", self.test_input_validation),
            ("rate_limiting", self.test_rate_limiting),
            ("security_headers", self.test_security_headers),
            ("cors", self.test_cors_configuration),
        ]

        total_tests = 0
        passed_tests = 0

        for category, test_function in test_categories:
            print(f"  Running {category} tests...")
            try:
                results = test_function()
                audit_results["tests"][category] = results

                # Count pass/fail
                for result in results:
                    if isinstance(result, dict) and "status" in result:
                        total_tests += 1
                        if result["status"] == "PASS":
                            passed_tests += 1

            except Exception as e:
                audit_results["tests"][category] = [
                    {"error": str(e), "status": "ERROR"}
                ]

        # Calculate summary
        pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        audit_results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": round(pass_rate, 1),
            "security_score": self._calculate_security_score(audit_results["tests"]),
        }

        return audit_results

    def _calculate_security_score(self, test_results: dict) -> float:
        """Calculate overall security score based on test results."""
        category_weights = {
            "authentication": 30,  # Most critical
            "authorization": 25,  # Very critical
            "jwt_validation": 20,  # Critical
            "input_validation": 15,  # Important
            "rate_limiting": 5,  # Moderate
            "security_headers": 3,  # Nice to have
            "cors": 2,  # Configuration
        }

        total_weight = sum(category_weights.values())
        weighted_score = 0

        for category, weight in category_weights.items():
            if category in test_results:
                results = test_results[category]
                if results:
                    passed = sum(
                        1
                        for r in results
                        if isinstance(r, dict) and r.get("status") == "PASS"
                    )
                    total = len(
                        [r for r in results if isinstance(r, dict) and "status" in r]
                    )

                    if total > 0:
                        category_score = (passed / total) * weight
                        weighted_score += category_score

        return round((weighted_score / total_weight) * 100, 1)


def main():
    """Run security audit and generate report."""
    suite = SecurityTestSuite()
    results = suite.run_comprehensive_security_audit()

    # Print summary
    summary = results["summary"]
    print("\n🔒 Security Audit Complete")
    print("=" * 50)
    print(f"📊 Total Tests: {summary['total_tests']}")
    print(f"✅ Passed: {summary['passed_tests']}")
    print(f"❌ Failed: {summary['failed_tests']}")
    print(f"🎯 Pass Rate: {summary['pass_rate']}%")
    print(f"🛡️  Security Score: {summary['security_score']}/100")

    # Detailed results
    print("\n📋 Detailed Results:")
    print("-" * 30)

    for category, tests in results["tests"].items():
        print(f"\n{category.upper()}:")
        for test in tests:
            if isinstance(test, dict):
                status_icon = (
                    "✅"
                    if test.get("status") == "PASS"
                    else "❌"
                    if test.get("status") == "FAIL"
                    else "⚠️"
                )
                test_name = test.get("test", test.get("endpoint", "Unknown"))
                print(f"  {status_icon} {test_name}")

                if test.get("status") == "FAIL":
                    message = test.get(
                        "message",
                        f"Expected {test.get('expected')}, got {test.get('actual')}",
                    )
                    print(f"      ↳ {message}")

    # Save detailed report
    with open("security_test_report.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n📝 Detailed report saved to: security_test_report.json")

    # Return appropriate exit code
    return 0 if summary["security_score"] >= 80 else 1


if __name__ == "__main__":
    exit(main())
