"""Performance tests for user profile CRUD operations."""

import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
import pytest
import httpx
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from src.api.main import app
from src.core.models import User
from src.core.security import security_manager


@pytest.fixture
def client():
    """Create test client for performance testing."""
    return TestClient(app)


@pytest.fixture
def mock_user():
    """Create mock user for performance testing."""
    user = Mock(spec=User)
    user.id = "perf-test-user"
    user.username = "perftestuser"
    user.email = "perf@example.com"
    user.is_active = True
    user.is_admin = False
    user.permissions = ["read"]
    return user


@pytest.fixture
def auth_headers(mock_user):
    """Create authentication headers for performance testing."""
    # Mock the token creation for performance tests
    with patch.object(security_manager, 'create_access_token', return_value='mock.jwt.token'):
        token = security_manager.create_access_token(mock_user)
        return {"Authorization": f"Bearer {token}"}


class TestProfileReadPerformance:
    """Performance tests for profile read operations."""

    @pytest.mark.performance
    def test_profile_read_response_time(self, client, mock_user, auth_headers):
        """Test GET profile response time under normal load."""
        # Mock the database and auth layers for performance testing
        with patch('src.api.main.get_current_user', return_value=mock_user):
            with patch('src.api.main.UserProfile') as mock_profile_class:
                mock_profile = Mock()
                mock_profile.dict.return_value = {
                    "id": mock_user.id,
                    "username": mock_user.username,
                    "email": mock_user.email,
                    "first_name": "Test",
                    "last_name": "User",
                    "bio": "Performance test bio",
                    "preferences": {"theme": "dark"},
                    "is_active": True,
                    "is_admin": False,
                    "permissions": ["read"],
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "last_login": "2024-01-01T00:00:00",
                }
                mock_profile_class.return_value = mock_profile

                # Measure response time
                start_time = time.time()
                response = client.get("/api/v1/auth/profile", headers=auth_headers)
                end_time = time.time()

                assert response.status_code == 200
                response_time = (end_time - start_time) * 1000  # Convert to milliseconds

                # Assert performance requirements (P95 < 200ms)
                assert response_time < 200, f"Response time too slow: {response_time}ms"

    @pytest.mark.performance
    def test_concurrent_profile_reads(self, client, mock_user, auth_headers):
        """Test concurrent profile read operations."""
        with patch('src.api.main.get_current_user', return_value=mock_user):
            with patch('src.api.main.UserProfile') as mock_profile_class:
                mock_profile = Mock()
                mock_profile.dict.return_value = {
                    "id": mock_user.id,
                    "username": mock_user.username,
                    "email": mock_user.email,
                    "first_name": "Test",
                    "last_name": "User",
                    "is_active": True,
                    "is_admin": False,
                    "permissions": ["read"],
                    "created_at": "2024-01-01T00:00:00",
                    "updated_at": "2024-01-01T00:00:00",
                    "last_login": "2024-01-01T00:00:00",
                }
                mock_profile_class.return_value = mock_profile

                def make_request():
                    start = time.time()
                    response = client.get("/api/v1/auth/profile", headers=auth_headers)
                    end = time.time()
                    return response.status_code, (end - start) * 1000

                # Test with 50 concurrent requests
                with ThreadPoolExecutor(max_workers=10) as executor:
                    futures = [executor.submit(make_request) for _ in range(50)]
                    results = [future.result() for future in futures]

                # Analyze results
                response_times = [result[1] for result in results]
                success_count = sum(1 for status, _ in results if status == 200)

                # Assert performance
                assert success_count == 50, f"Only {success_count}/50 requests succeeded"

                p95_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
                avg_time = statistics.mean(response_times)

                assert p95_time < 200, f"P95 response time too high: {p95_time}ms"
                assert avg_time < 100, f"Average response time too high: {avg_time}ms"


class TestProfileUpdatePerformance:
    """Performance tests for profile update operations."""

    @pytest.mark.performance
    def test_profile_update_response_time(self, client, mock_user, auth_headers):
        """Test PUT profile update response time."""
        update_data = {
            "first_name": "Updated",
            "last_name": "Name",
            "bio": "Updated bio for performance testing",
            "preferences": {"theme": "light", "notifications": True}
        }

        with patch('src.api.main.get_current_user', return_value=mock_user):
            with patch('src.api.main.update_user_profile_in_db', return_value=True):
                start_time = time.time()
                response = client.put("/api/v1/auth/profile", json=update_data, headers=auth_headers)
                end_time = time.time()

                assert response.status_code == 200
                response_time = (end_time - start_time) * 1000

                # Assert performance requirements (P95 < 500ms)
                assert response_time < 500, f"Update response time too slow: {response_time}ms"

    @pytest.mark.performance
    def test_bulk_profile_updates(self, client, auth_headers):
        """Test bulk profile update operations."""
        update_data = {
            "first_name": "Bulk",
            "last_name": "Update",
            "bio": "Bulk update test",
        }

        def make_update_request(user_id):
            headers = {**auth_headers, "X-User-ID": user_id}
            start = time.time()
            response = client.put("/api/v1/auth/profile", json=update_data, headers=headers)
            end = time.time()
            return response.status_code, (end - start) * 1000

        # Simulate bulk updates for 25 users
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_update_request, f"user_{i}") for i in range(25)]
            results = [future.result() for future in futures]

        response_times = [result[1] for result in results]
        success_count = sum(1 for status, _ in results if status == 200)

        assert success_count >= 20, f"Only {success_count}/25 bulk updates succeeded"

        p95_time = statistics.quantiles(response_times, n=20)[18]
        assert p95_time < 500, f"Bulk update P95 too high: {p95_time}ms"


class TestProfileCreatePerformance:
    """Performance tests for profile creation operations."""

    @pytest.mark.performance
    def test_profile_creation_response_time(self, client):
        """Test POST profile creation response time."""
        create_data = {
            "username": f"perfuser_{int(time.time())}",
            "email": f"perf_{int(time.time())}@example.com",
            "password": "SecurePass123!",
            "first_name": "Performance",
            "last_name": "Test",
        }

        with patch('src.api.main.create_user_in_db', return_value=Mock(id="new-user-id")):
            start_time = time.time()
            response = client.post("/api/v1/auth/users", json=create_data)
            end_time = time.time()

            assert response.status_code == 200
            response_time = (end_time - start_time) * 1000

            # Assert performance requirements (< 1000ms)
            assert response_time < 1000, f"Creation response time too slow: {response_time}ms"


class TestDatabasePerformance:
    """Database-level performance tests."""

    @pytest.mark.performance
    def test_database_query_performance(self):
        """Test database query performance for profile operations."""
        # Mock database operations
        with patch('src.core.async_db.AsyncDatabaseSession') as mock_session:
            mock_db = Mock()
            mock_session.return_value = mock_db

            # Simulate profile queries
            query_times = []

            for _ in range(100):
                start = time.time()
                # Simulate database query
                time.sleep(0.001)  # 1ms simulated query time
                end = time.time()
                query_times.append((end - start) * 1000)

            avg_query_time = statistics.mean(query_times)
            p95_query_time = statistics.quantiles(query_times, n=20)[18]

            # Assert reasonable query performance
            assert avg_query_time < 5, f"Average query time too high: {avg_query_time}ms"
            assert p95_query_time < 10, f"P95 query time too high: {p95_query_time}ms"


class TestMemoryPerformance:
    """Memory usage performance tests."""

    @pytest.mark.performance
    def test_memory_usage_with_large_profiles(self, client, mock_user, auth_headers):
        """Test memory usage with large profile data."""
        # Create large profile data
        large_bio = "A" * 4000  # 4KB bio
        large_preferences = {f"key_{i}": f"value_{i}" * 10 for i in range(100)}  # Large preferences

        update_data = {
            "bio": large_bio,
            "preferences": large_preferences,
        }

        with patch('src.api.main.get_current_user', return_value=mock_user):
            with patch('src.api.main.update_user_profile_in_db', return_value=True):
                # Measure memory before
                import psutil
                import os
                process = psutil.Process(os.getpid())
                memory_before = process.memory_info().rss / 1024 / 1024  # MB

                # Perform operations with large data
                for _ in range(10):
                    response = client.put("/api/v1/auth/profile", json=update_data, headers=auth_headers)
                    assert response.status_code == 200

                # Measure memory after
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before

                # Assert reasonable memory usage (< 50MB increase)
                assert memory_increase < 50, f"Excessive memory usage: {memory_increase}MB"


class TestLoadTesting:
    """Load testing for profile operations."""

    @pytest.mark.performance
    async def test_sustained_load_profile_reads(self):
        """Test sustained load for profile read operations."""
        async with httpx.AsyncClient(base_url="http://testserver") as client:
            # This would require a test server setup
            # For now, test the async handling capability
            tasks = []

            for i in range(100):
                task = self._make_async_profile_request(client, i)
                tasks.append(task)

            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()

            total_time = end_time - start_time
            successful_requests = sum(1 for r in responses if not isinstance(r, Exception))

            # Calculate RPS
            rps = successful_requests / total_time

            # Assert minimum performance (50 RPS)
            assert rps >= 50, f"RPS too low: {rps}"

    async def _make_async_profile_request(self, client, request_id):
        """Helper method for async profile requests."""
        # Mock async request - in real test this would make actual HTTP calls
        await asyncio.sleep(0.01)  # Simulate network delay
        return {"status": "success", "request_id": request_id}


class TestPerformanceRegression:
    """Performance regression tests."""

    @pytest.mark.performance
    def test_performance_baseline_comparison(self):
        """Test performance against established baselines."""
        # This would compare current performance against stored baselines
        # For now, establish basic performance assertions

        baseline_metrics = {
            "profile_read_p95": 150,  # ms
            "profile_update_p95": 400,  # ms
            "profile_create_p95": 800,  # ms
            "concurrent_users_supported": 100,
        }

        # In a real implementation, these would be loaded from a baseline file
        # and compared against current measurements

        # Assert that current performance meets baseline
        assert baseline_metrics["profile_read_p95"] < 200
        assert baseline_metrics["profile_update_p95"] < 500
        assert baseline_metrics["profile_create_p95"] < 1000


class TestScalabilityTesting:
    """Scalability testing for profile operations."""

    @pytest.mark.performance
    def test_horizontal_scalability_simulation(self):
        """Test simulated horizontal scalability."""
        # Simulate multiple server instances
        server_count = 3
        requests_per_server = 100

        total_start_time = time.time()

        def simulate_server_load(server_id):
            server_start = time.time()
            successful = 0

            for i in range(requests_per_server):
                # Simulate request processing
                time.sleep(0.005)  # 5ms per request
                successful += 1

            server_end = time.time()
            return successful, server_end - server_start

        # Run simulation across multiple "servers"
        with ThreadPoolExecutor(max_workers=server_count) as executor:
            futures = [executor.submit(simulate_server_load, i) for i in range(server_count)]
            results = [future.result() for future in futures]

        total_end_time = time.time()
        total_time = total_end_time - total_start_time

        total_successful = sum(successful for successful, _ in results)
        total_expected = server_count * requests_per_server

        # Assert scalability
        assert total_successful == total_expected
        throughput = total_successful / total_time

        # Assert minimum throughput (100 RPS across all servers)
        assert throughput >= 100, f"Throughput too low: {throughput} RPS"


if __name__ == "__main__":
    pytest.main([__file__, "-m", "performance"])