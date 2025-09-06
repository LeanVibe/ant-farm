# Performance and Edge Case Testing - User Profile CRUD Operations

## Performance Testing Framework

### Load Testing Specifications

#### Target Performance Metrics
| Operation | Target RPS | P95 Response Time | P99 Response Time | Concurrent Users |
|-----------|------------|-------------------|-------------------|------------------|
| **GET Profile** | 100 RPS | <200ms | <300ms | 500 |
| **POST Create User** | 20 RPS | <1000ms | <1500ms | 100 |
| **PUT Update Profile** | 50 RPS | <500ms | <750ms | 250 |
| **DELETE Profile** | 10 RPS | <2000ms | <3000ms | 50 |

### Performance Test Implementation

```python
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import httpx
import pytest

class TestUserProfilePerformance:
    """Performance testing for user profile CRUD operations."""
    
    @pytest.mark.performance
    async def test_profile_read_performance(self):
        """Test GET profile performance under load"""
        async with httpx.AsyncClient() as client:
            # Create test user first
            user_response = await self.create_test_user(client)
            user_id = user_response["data"]["id"]
            
            # Setup authentication
            auth_headers = await self.get_auth_headers(client)
            
            # Performance test configuration
            target_rps = 100
            test_duration = 30  # seconds
            total_requests = target_rps * test_duration
            
            start_time = time.time()
            tasks = []
            
            # Generate concurrent requests
            for _ in range(total_requests):
                task = self.make_profile_request(client, user_id, auth_headers)
                tasks.append(task)
            
            # Execute all requests
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Analyze results
            successful_requests = [r for r in responses if not isinstance(r, Exception)]
            actual_rps = len(successful_requests) / (end_time - start_time)
            
            # Calculate response time percentiles
            response_times = [r.elapsed.total_seconds() * 1000 for r in successful_requests]
            p95_time = self.calculate_percentile(response_times, 95)
            p99_time = self.calculate_percentile(response_times, 99)
            
            # Assertions
            assert actual_rps >= target_rps * 0.9, f"RPS too low: {actual_rps} < {target_rps}"
            assert p95_time < 200, f"P95 response time too high: {p95_time}ms"
            assert p99_time < 300, f"P99 response time too high: {p99_time}ms"
    
    @pytest.mark.performance
    async def test_profile_update_performance(self):
        """Test PUT profile update performance"""
        async with httpx.AsyncClient() as client:
            # Create test users
            users = await self.create_multiple_users(client, count=250)
            auth_headers = await self.get_auth_headers(client)
            
            target_rps = 50
            test_duration = 30
            
            start_time = time.time()
            tasks = []
            
            for i in range(target_rps * test_duration):
                user = users[i % len(users)]
                update_data = {
                    "first_name": f"Updated{i}",
                    "bio": f"Performance test bio {i}"
                }
                task = self.update_profile_request(client, user["id"], update_data, auth_headers)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            # Performance validation
            successful_requests = [r for r in responses if not isinstance(r, Exception)]
            actual_rps = len(successful_requests) / (end_time - start_time)
            
            response_times = [r.elapsed.total_seconds() * 1000 for r in successful_requests]
            p95_time = self.calculate_percentile(response_times, 95)
            
            assert actual_rps >= target_rps * 0.9
            assert p95_time < 500
    
    @pytest.mark.performance
    async def test_concurrent_user_creation_performance(self):
        """Test concurrent user creation performance"""
        async with httpx.AsyncClient() as client:
            target_rps = 20
            test_duration = 10  # Shorter for creation tests
            
            tasks = []
            start_time = time.time()
            
            for i in range(target_rps * test_duration):
                user_data = {
                    "username": f"perftest_user_{i}_{int(time.time())}",
                    "email": f"perftest_{i}@example.com",
                    "password": "TestPassword123!",
                    "first_name": f"Test{i}",
                    "last_name": "User"
                }
                task = client.post("/api/v1/auth/users", json=user_data)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.time()
            
            successful_requests = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
            actual_rps = len(successful_requests) / (end_time - start_time)
            
            response_times = [r.elapsed.total_seconds() * 1000 for r in successful_requests]
            p95_time = self.calculate_percentile(response_times, 95)
            
            assert actual_rps >= target_rps * 0.8  # More lenient for creation
            assert p95_time < 1000
```

### Database Performance Testing

```python
class TestDatabasePerformance:
    """Database-level performance testing."""
    
    @pytest.mark.performance
    async def test_profile_query_optimization(self):
        """Test database query performance with proper indexing"""
        # Create large dataset
        await self.create_test_users(count=10000)
        
        # Test various query patterns
        query_tests = [
            ("SELECT by ID", "SELECT * FROM users WHERE id = %s"),
            ("SELECT by username", "SELECT * FROM users WHERE username = %s"),
            ("SELECT by email", "SELECT * FROM users WHERE email = %s"),
            ("Complex profile query", """
                SELECT u.*, COUNT(t.id) as task_count 
                FROM users u 
                LEFT JOIN tasks t ON u.id = t.assigned_to 
                WHERE u.is_active = true 
                GROUP BY u.id
            """)
        ]
        
        for test_name, query in query_tests:
            start_time = time.time()
            
            # Execute query multiple times
            for _ in range(100):
                await self.execute_query(query, test_params)
            
            end_time = time.time()
            avg_query_time = (end_time - start_time) / 100
            
            # Query should complete in reasonable time
            assert avg_query_time < 0.050, f"{test_name} too slow: {avg_query_time}s"
    
    @pytest.mark.performance
    async def test_bulk_operations_performance(self):
        """Test bulk profile operations performance"""
        batch_sizes = [100, 500, 1000, 2000]
        
        for batch_size in batch_sizes:
            # Bulk insert test
            user_data = self.generate_user_data_batch(batch_size)
            
            start_time = time.time()
            await self.bulk_insert_users(user_data)
            insert_time = time.time() - start_time
            
            # Bulk update test
            update_data = [{"id": user["id"], "bio": f"Updated {i}"} 
                          for i, user in enumerate(user_data)]
            
            start_time = time.time()
            await self.bulk_update_profiles(update_data)
            update_time = time.time() - start_time
            
            # Performance assertions
            insert_rate = batch_size / insert_time
            update_rate = batch_size / update_time
            
            assert insert_rate > 50, f"Insert rate too low for batch {batch_size}: {insert_rate}/s"
            assert update_rate > 100, f"Update rate too low for batch {batch_size}: {update_rate}/s"
```

## Edge Case and Error Handling Test Matrix

### Data Boundary Edge Cases

```python
class TestDataBoundaryEdgeCases:
    """Test edge cases for data boundaries and limits."""
    
    @pytest.mark.edge_case
    async def test_maximum_field_lengths(self):
        """Test behavior at maximum field length limits"""
        edge_cases = [
            ("username", "a" * 100, True),      # Max allowed
            ("username", "a" * 101, False),     # Over limit
            ("email", "a" * 243 + "@example.com", True),   # Max email (255 total)
            ("email", "a" * 244 + "@example.com", False),  # Over limit
            ("first_name", "a" * 100, True),
            ("first_name", "a" * 101, False),
            ("bio", "a" * 5000, True),          # Assume reasonable bio limit
            ("bio", "a" * 5001, False),
        ]
        
        for field, value, should_succeed in edge_cases:
            user_data = self.create_base_user_data()
            user_data[field] = value
            
            response = await self.client.post("/api/v1/auth/users", json=user_data)
            
            if should_succeed:
                assert response.status_code == 200, f"Failed for valid {field} length"
            else:
                assert response.status_code == 422, f"Accepted invalid {field} length"
    
    @pytest.mark.edge_case
    async def test_unicode_character_handling(self):
        """Test Unicode character handling in profile fields"""
        unicode_tests = [
            ("Emoji in name", {"first_name": "John 😀", "last_name": "Doe 🎉"}),
            ("Multi-language", {"bio": "Hello 世界 مرحبا Привет"}),
            ("Special characters", {"username": "user_ñ_ü_ç"}),
            ("RTL text", {"bio": "English text مع العربية"}),
            ("Zero-width characters", {"first_name": "John\u200b\u200cDoe"}),
        ]
        
        for test_name, data in unicode_tests:
            user_data = self.create_base_user_data()
            user_data.update(data)
            
            # Create user
            response = await self.client.post("/api/v1/auth/users", json=user_data)
            assert response.status_code == 200, f"Unicode test failed: {test_name}"
            
            # Verify data integrity
            user_id = response.json()["data"]["id"]
            profile_response = await self.get_profile(user_id)
            
            for key, expected_value in data.items():
                actual_value = profile_response["data"][key]
                assert actual_value == expected_value, f"Unicode data corruption in {key}"
    
    @pytest.mark.edge_case
    async def test_null_and_empty_value_handling(self):
        """Test null and empty value handling"""
        null_tests = [
            # Optional fields - should accept null/empty
            ("first_name", None, True),
            ("first_name", "", True),
            ("last_name", None, True),
            ("bio", None, True),
            ("avatar_url", None, True),
            
            # Required fields - should reject null/empty
            ("username", None, False),
            ("username", "", False),
            ("email", None, False),
            ("email", "", False),
            ("password", None, False),
            ("password", "", False),
        ]
        
        for field, value, should_succeed in null_tests:
            user_data = self.create_base_user_data()
            if value is None:
                user_data[field] = None
            else:
                user_data[field] = value
            
            response = await self.client.post("/api/v1/auth/users", json=user_data)
            
            if should_succeed:
                assert response.status_code == 200, f"Rejected valid null/empty {field}"
            else:
                assert response.status_code == 422, f"Accepted invalid null/empty {field}"
```

### Concurrent Operations Edge Cases

```python
class TestConcurrentOperationsEdgeCases:
    """Test edge cases for concurrent operations."""
    
    @pytest.mark.edge_case
    async def test_concurrent_profile_updates(self):
        """Test concurrent updates to the same profile"""
        user = await self.create_test_user()
        user_id = user["id"]
        auth_headers = await self.get_auth_headers()
        
        # Define conflicting updates
        update_tasks = [
            self.update_profile(user_id, {"first_name": "Alice"}, auth_headers),
            self.update_profile(user_id, {"first_name": "Bob"}, auth_headers),
            self.update_profile(user_id, {"bio": "Bio A"}, auth_headers),
            self.update_profile(user_id, {"bio": "Bio B"}, auth_headers),
        ]
        
        # Execute concurrent updates
        responses = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        # At least one should succeed
        successful_updates = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
        assert len(successful_updates) > 0, "All concurrent updates failed"
        
        # Verify final state is consistent
        final_profile = await self.get_profile(user_id)
        assert final_profile["data"]["updated_at"] is not None
        
        # Check for any data corruption
        self.validate_profile_data_integrity(final_profile["data"])
    
    @pytest.mark.edge_case
    async def test_concurrent_user_creation_with_duplicates(self):
        """Test concurrent creation of users with potential duplicates"""
        base_username = f"testuser_{int(time.time())}"
        base_email = f"test_{int(time.time())}@example.com"
        
        # Attempt to create users with same username/email concurrently
        creation_tasks = []
        for i in range(10):
            user_data = {
                "username": base_username,  # Same username
                "email": f"{base_email}_{i}",  # Different emails
                "password": "TestPassword123!",
            }
            task = self.client.post("/api/v1/auth/users", json=user_data)
            creation_tasks.append(task)
        
        responses = await asyncio.gather(*creation_tasks, return_exceptions=True)
        
        # Only one should succeed due to unique constraint
        successful_creations = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
        assert len(successful_creations) == 1, "Unique constraint not enforced properly"
        
        # Others should fail with appropriate error
        failed_responses = [r for r in responses if not isinstance(r, Exception) and r.status_code != 200]
        assert all(r.status_code in [400, 409, 422] for r in failed_responses)
```

### System Resource Edge Cases

```python
class TestSystemResourceEdgeCases:
    """Test edge cases related to system resources."""
    
    @pytest.mark.edge_case
    async def test_database_connection_exhaustion(self):
        """Test behavior when database connections are exhausted"""
        # Simulate high connection usage
        connection_tasks = []
        
        for i in range(100):  # Attempt many concurrent operations
            task = self.create_test_user_with_db_intensive_operation()
            connection_tasks.append(task)
        
        responses = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        # System should handle gracefully, not crash
        exceptions = [r for r in responses if isinstance(r, Exception)]
        successful_responses = [r for r in responses if not isinstance(r, Exception)]
        
        # At least some operations should succeed
        assert len(successful_responses) > 0, "All operations failed under connection pressure"
        
        # Failures should be graceful (503 Service Unavailable, not 500 Internal Error)
        if exceptions:
            # Check that exceptions are handled gracefully
            assert all("timeout" in str(e).lower() or "connection" in str(e).lower() for e in exceptions[:5])
    
    @pytest.mark.edge_case
    async def test_memory_usage_with_large_profiles(self):
        """Test memory usage with large profile data"""
        large_profile_data = {
            "username": "large_profile_user",
            "email": "large@example.com",
            "password": "TestPassword123!",
            "bio": "A" * 4000,  # Large bio
            "preferences": {f"key_{i}": f"value_{i}" for i in range(1000)},  # Large preferences
        }
        
        # Monitor memory usage
        initial_memory = self.get_memory_usage()
        
        # Create multiple large profiles
        tasks = []
        for i in range(50):
            data = large_profile_data.copy()
            data["username"] = f"large_user_{i}"
            data["email"] = f"large_{i}@example.com"
            task = self.client.post("/api/v1/auth/users", json=data)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable (not excessive)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase}MB"
        
        # Operations should still succeed
        successful_responses = [r for r in responses if not isinstance(r, Exception) and r.status_code == 200]
        assert len(successful_responses) > 40, "Too many failures with large profiles"
```

### Network and Timeout Edge Cases

```python
class TestNetworkTimeoutEdgeCases:
    """Test edge cases related to network timeouts and interruptions."""
    
    @pytest.mark.edge_case
    async def test_request_timeout_handling(self):
        """Test handling of request timeouts"""
        # Simulate slow network conditions
        async with httpx.AsyncClient(timeout=1.0) as slow_client:  # 1 second timeout
            
            # Create operation that might take longer than timeout
            large_user_data = {
                "username": "timeout_test_user",
                "email": "timeout@example.com", 
                "password": "TestPassword123!",
                "bio": "A" * 10000,  # Very large bio
                "preferences": {f"key_{i}": "value_" * 100 for i in range(500)},
            }
            
            try:
                response = await slow_client.post("/api/v1/auth/users", json=large_user_data)
                # If successful, verify data integrity
                if response.status_code == 200:
                    self.validate_profile_data_integrity(response.json()["data"])
                    
            except httpx.TimeoutException:
                # Timeout is acceptable behavior
                pass
            
            # Verify system remains stable after timeout
            health_response = await self.client.get("/api/v1/health")
            assert health_response.status_code == 200
    
    @pytest.mark.edge_case
    async def test_partial_request_handling(self):
        """Test handling of incomplete/partial requests"""
        partial_requests = [
            {},  # Empty request
            {"username": "partial_user"},  # Missing required fields
            {"email": "incomplete@example.com"},  # Missing required fields
            {
                "username": "test_user",
                "email": "test@example.com"
                # Missing password
            },
        ]
        
        for partial_data in partial_requests:
            response = await self.client.post("/api/v1/auth/users", json=partial_data)
            
            # Should return validation error, not crash
            assert response.status_code == 422, f"Improper handling of partial request: {partial_data}"
            assert "validation" in response.json().get("error", "").lower()
```

### Data Consistency Edge Cases

```python
class TestDataConsistencyEdgeCases:
    """Test edge cases for data consistency and integrity."""
    
    @pytest.mark.edge_case
    async def test_transaction_rollback_scenarios(self):
        """Test transaction rollback in failure scenarios"""
        # Simulate failure during user creation process
        user_data = {
            "username": "rollback_test_user",
            "email": "rollback@example.com",
            "password": "TestPassword123!",
        }
        
        # Mock database failure during user creation
        with self.simulate_database_error_after_user_insert():
            response = await self.client.post("/api/v1/auth/users", json=user_data)
            
            # Should fail gracefully
            assert response.status_code == 500
        
        # Verify partial data was not committed
        # User should not exist in database
        user_check = await self.find_user_by_username("rollback_test_user")
        assert user_check is None, "Partial user data was committed despite failure"
    
    @pytest.mark.edge_case
    async def test_data_corruption_detection(self):
        """Test detection and handling of data corruption"""
        # Create user with valid data
        user = await self.create_test_user()
        user_id = user["id"]
        
        # Simulate data corruption scenarios
        corruption_tests = [
            ("Invalid JSON in preferences", {"preferences": "not_json_string"}),
            ("Malformed email", {"email": "corrupted_email_format"}),
            ("Invalid timezone", {"timezone": "Not/A/Valid/Timezone"}),
        ]
        
        for test_name, corrupted_data in corruption_tests:
            auth_headers = await self.get_auth_headers()
            
            response = await self.client.put(
                f"/api/v1/auth/profile", 
                json=corrupted_data,
                headers=auth_headers
            )
            
            # Should reject corrupted data
            assert response.status_code in [400, 422], f"Accepted corrupted data: {test_name}"
            
            # Verify original data remains intact
            profile_response = await self.get_profile(user_id)
            self.validate_profile_data_integrity(profile_response["data"])
```

## Implementation Notes

### Test Environment Setup
- Use isolated test database for each test run
- Implement proper test data cleanup
- Mock external dependencies (email services, file storage)
- Configure realistic test data volumes

### Performance Monitoring
- Collect detailed timing metrics
- Monitor resource usage (CPU, memory, database connections)
- Generate performance trend reports
- Set up automated performance regression detection

### Edge Case Coverage
- Systematically test boundary conditions
- Include both positive and negative test cases  
- Test error recovery and system stability
- Validate data integrity after failures

This comprehensive test matrix ensures robust handling of edge cases and maintains system performance under various load conditions.