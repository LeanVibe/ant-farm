# Security Test Scenarios - User Profile CRUD Operations

## Overview
This document provides detailed security test scenarios for user profile CRUD operations, designed to validate authentication, authorization, input validation, and data protection mechanisms.

## Authentication Security Tests

### Test Suite: JWT Token Security
```python
class TestJWTTokenSecurity:
    """Test JWT token-based authentication security."""
    
    def test_access_token_expiration(self):
        """Verify access tokens expire correctly"""
        # Create user and get token
        # Wait for token expiration
        # Attempt API call with expired token
        # Expect 401 Unauthorized
        
    def test_refresh_token_rotation(self):
        """Test refresh token rotation security"""
        # Get refresh token
        # Use refresh token to get new access token
        # Attempt to reuse old refresh token
        # Expect 401 Unauthorized
        
    def test_token_tampering_detection(self):
        """Verify tampered tokens are rejected"""
        # Get valid token
        # Modify token payload
        # Attempt API call with tampered token
        # Expect 401 Unauthorized with appropriate error
        
    def test_invalid_signature_rejection(self):
        """Test rejection of tokens with invalid signatures"""
        # Create token with wrong secret
        # Attempt API call
        # Expect 401 Unauthorized
```

### Test Suite: Password Security Validation
```python
class TestPasswordSecurity:
    """Test password security requirements."""
    
    def test_weak_password_rejection(self):
        """Verify weak passwords are rejected"""
        weak_passwords = [
            "123456", "password", "abc123", "qwerty",
            "admin", "test", "12345678"  # Common weak passwords
        ]
        for password in weak_passwords:
            # Attempt user creation with weak password
            # Expect validation error
            
    def test_password_complexity_requirements(self):
        """Test password complexity validation"""
        test_cases = [
            ("short1", False),  # Too short
            ("nouppercase123!", False),  # No uppercase
            ("NOLOWERCASE123!", False),  # No lowercase  
            ("NoDigits!!", False),  # No digits
            ("NoSpecialChars123", False),  # No special chars
            ("ValidPass123!", True),  # Valid password
        ]
        
    def test_password_brute_force_protection(self):
        """Test account lockout after failed attempts"""
        # Attempt 5 failed logins (MAX_LOGIN_ATTEMPTS)
        # Verify account is locked
        # Attempt login with correct password
        # Expect 401 due to lockout
        # Wait for lockout duration
        # Verify account unlocks
```

## Authorization Security Tests

### Test Suite: Permission-Based Access Control
```python
class TestPermissionSecurity:
    """Test role-based permission system."""
    
    def test_user_profile_access_isolation(self):
        """Verify users can only access their own profiles"""
        # Create two users
        # User A attempts to read User B's profile
        # Expect 403 Forbidden
        # User A attempts to update User B's profile
        # Expect 403 Forbidden
        
    def test_admin_privilege_validation(self):
        """Test admin user access to other profiles"""
        # Create admin user and regular user
        # Admin accesses regular user profile
        # Expect 200 OK with profile data
        # Regular user attempts admin functions
        # Expect 403 Forbidden
        
    def test_permission_elevation_prevention(self):
        """Prevent users from granting themselves admin privileges"""
        # Create regular user
        # Attempt to update profile with is_admin: true
        # Expect validation error or 403 Forbidden
        # Verify user remains non-admin
        
    def test_cross_user_profile_modification(self):
        """Prevent unauthorized profile modifications"""
        # Create User A and User B
        # User A attempts to modify User B's profile
        # Include various fields (name, email, preferences)
        # Expect 403 Forbidden for all attempts
```

### Test Suite: Endpoint-Level Security
```python
class TestEndpointSecurity:
    """Test security at API endpoint level."""
    
    def test_unauthenticated_access_denial(self):
        """Verify all profile endpoints require authentication"""
        endpoints = [
            ("GET", "/api/v1/auth/profile"),
            ("PUT", "/api/v1/auth/profile"), 
            ("DELETE", "/api/v1/auth/profile"),
            ("GET", "/api/v1/auth/users/{user_id}/profile")
        ]
        for method, url in endpoints:
            # Make request without auth token
            # Expect 401 Unauthorized
            
    def test_insufficient_permissions_denial(self):
        """Test denial with insufficient permissions"""
        # Create user with limited permissions
        # Attempt operations requiring higher permissions
        # Expect 403 Forbidden
```

## Input Validation Security Tests

### Test Suite: SQL Injection Prevention
```python
class TestSQLInjectionSecurity:
    """Test protection against SQL injection attacks."""
    
    def test_profile_field_sql_injection(self):
        """Test SQL injection in profile fields"""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "'; UPDATE users SET is_admin=true; --",
            "' UNION SELECT * FROM users; --"
        ]
        
        fields_to_test = [
            "first_name", "last_name", "bio", 
            "avatar_url", "username", "email"
        ]
        
        for field in fields_to_test:
            for payload in malicious_inputs:
                # Attempt profile update with malicious input
                # Verify no SQL injection occurs
                # Check database integrity
                
    def test_search_parameter_injection(self):
        """Test SQL injection in search parameters"""
        # Test profile search/filter endpoints
        # Include malicious SQL in search terms
        # Verify parameterized queries prevent injection
```

### Test Suite: XSS Prevention
```python
class TestXSSPrevention:
    """Test protection against XSS attacks."""
    
    def test_script_tag_sanitization(self):
        """Test script tag removal/sanitization"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "<img src='x' onerror='alert(1)'>",
            "javascript:alert('xss')",
            "<svg onload='alert(1)'>",
            "data:text/html,<script>alert('xss')</script>"
        ]
        
        for payload in xss_payloads:
            # Update profile with XSS payload in bio/name fields
            # Verify payload is sanitized
            # Check stored data doesn't contain executable scripts
            
    def test_html_entity_encoding(self):
        """Test proper HTML entity encoding"""
        # Include HTML entities in profile data
        # Verify proper encoding/decoding
        # Ensure no executable content passes through
```

### Test Suite: File Upload Security
```python
class TestFileUploadSecurity:
    """Test avatar upload security."""
    
    def test_malicious_file_upload_prevention(self):
        """Prevent upload of malicious files"""
        malicious_files = [
            ("malware.exe", "application/executable"),
            ("script.php", "application/x-php"),
            ("payload.js", "text/javascript"),
            ("virus.bat", "application/batch")
        ]
        
        for filename, content_type in malicious_files:
            # Attempt upload of malicious file as avatar
            # Expect rejection due to file type validation
            
    def test_file_size_limits(self):
        """Test file size restrictions"""
        # Attempt upload of oversized file
        # Expect rejection due to size limits
        # Verify system stability after large upload attempt
        
    def test_image_format_validation(self):
        """Validate only allowed image formats"""
        valid_formats = ["image/jpeg", "image/png", "image/gif", "image/webp"]
        invalid_formats = ["application/pdf", "text/plain", "video/mp4"]
        
        # Test valid formats - expect success
        # Test invalid formats - expect rejection
```

## Data Protection Security Tests

### Test Suite: Sensitive Data Handling
```python
class TestSensitiveDataSecurity:
    """Test protection of sensitive user data."""
    
    def test_password_hash_exposure_prevention(self):
        """Ensure password hashes are never exposed"""
        # Create user account
        # Fetch profile via API
        # Verify response doesn't contain hashed_password
        # Test all profile endpoints for data leakage
        
    def test_personal_data_encryption(self):
        """Verify sensitive data is properly protected"""
        # Check database storage of sensitive fields
        # Verify proper encryption/hashing where required
        # Test data at rest protection
        
    def test_audit_trail_security(self):
        """Test security of audit logs"""
        # Perform profile operations
        # Check audit logs don't expose sensitive data
        # Verify log access is restricted
```

### Test Suite: Data Validation
```python
class TestDataValidationSecurity:
    """Test data validation security."""
    
    def test_email_format_validation(self):
        """Test email validation security"""
        invalid_emails = [
            "plaintext",
            "@domain.com",
            "user@",
            "user..double.dot@domain.com",
            "user@domain..com",
            " user@domain.com ",  # Leading/trailing spaces
        ]
        
        for email in invalid_emails:
            # Attempt user creation/update with invalid email
            # Expect validation error
            
    def test_timezone_validation(self):
        """Test timezone string validation"""
        invalid_timezones = [
            "NotATimeZone",
            "../../etc/passwd",  # Path traversal attempt
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --"
        ]
        
        for tz in invalid_timezones:
            # Attempt profile update with invalid timezone
            # Expect validation error
```

## API Security Tests

### Test Suite: Rate Limiting
```python
class TestRateLimitingSecurity:
    """Test API rate limiting security."""
    
    def test_authentication_endpoint_rate_limiting(self):
        """Test rate limiting on auth endpoints"""
        # Make rapid successive login attempts
        # Verify rate limiting kicks in
        # Test both valid and invalid credentials
        
    def test_profile_operation_rate_limiting(self):
        """Test rate limiting on profile operations"""
        # Make rapid profile update requests
        # Verify rate limiting prevents abuse
        # Test across different users/sessions
```

### Test Suite: CORS and Headers
```python
class TestSecurityHeaders:
    """Test security headers and CORS configuration."""
    
    def test_security_headers_present(self):
        """Verify all required security headers"""
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options", 
            "X-XSS-Protection",
            "Strict-Transport-Security",
            "Content-Security-Policy"
        ]
        
        # Make API request
        # Verify all security headers are present
        # Validate header values
        
    def test_cors_configuration(self):
        """Test CORS policy enforcement"""
        # Test allowed origins
        # Test forbidden origins
        # Verify preflight requests work correctly
```

## Advanced Security Test Scenarios

### Test Suite: Session Security
```python
class TestSessionSecurity:
    """Test session management security."""
    
    def test_concurrent_session_handling(self):
        """Test multiple concurrent sessions"""
        # Login from multiple locations/devices
        # Verify session isolation
        # Test session invalidation
        
    def test_session_fixation_prevention(self):
        """Prevent session fixation attacks"""
        # Test session ID regeneration on login
        # Verify old session IDs become invalid
```

### Test Suite: Business Logic Security
```python
class TestBusinessLogicSecurity:
    """Test business logic security vulnerabilities."""
    
    def test_profile_deletion_security(self):
        """Test secure profile deletion"""
        # Delete user profile
        # Verify all related data is cleaned up
        # Ensure no data resurrection possible
        # Test soft delete vs hard delete behavior
        
    def test_concurrent_modification_handling(self):
        """Test handling of concurrent profile modifications"""
        # Two users modify same profile simultaneously
        # Verify proper conflict resolution
        # Test optimistic locking mechanisms
```

## Implementation Guidelines

### Test Data Security
- Use randomized test data to prevent data leakage
- Clean up test data after each test
- Use separate test databases
- Mock external services (email, file storage)

### Security Test Environment
- Isolated test environment
- Network segmentation
- Monitoring and logging enabled
- Regular security test data refresh

### Continuous Security Testing
- Integrate security tests in CI/CD pipeline
- Automated vulnerability scanning
- Regular penetration testing
- Security regression testing

### Test Metrics and Reporting
- Security test coverage metrics
- Vulnerability detection rates
- False positive tracking
- Security test execution time

This comprehensive security test suite ensures robust protection of user profile data and prevents common security vulnerabilities in CRUD operations.