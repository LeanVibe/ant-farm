"""Security testing configuration and compliance checks for LeanVibe Agent Hive.

This module defines security testing requirements, vulnerability thresholds,
and compliance validation for the user profile CRUD operations.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class VulnerabilitySeverity(Enum):
    """Vulnerability severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class SecurityScanType(Enum):
    """Types of security scans."""
    SAST = "SAST"  # Static Application Security Testing
    DAST = "DAST"  # Dynamic Application Security Testing
    SCA = "SCA"   # Software Composition Analysis
    CONTAINER = "CONTAINER"  # Container security scanning
    INFRASTRUCTURE = "INFRASTRUCTURE"  # Infrastructure as Code security


@dataclass
class SecurityThreshold:
    """Security vulnerability thresholds."""
    severity: VulnerabilitySeverity
    max_allowed: int
    blocking: bool
    scan_types: List[SecurityScanType]


@dataclass
class SecurityTestCase:
    """Security test case definition."""
    name: str
    category: str
    severity: VulnerabilitySeverity
    description: str
    test_file: str
    expected_result: str


@dataclass
class ComplianceRequirement:
    """Compliance requirement definition."""
    standard: str
    requirement: str
    description: str
    test_cases: List[str]
    automated_check: bool


class SecurityConfiguration:
    """Security testing configuration for the system."""

    # Vulnerability thresholds by severity
    VULNERABILITY_THRESHOLDS = [
        SecurityThreshold(
            severity=VulnerabilitySeverity.CRITICAL,
            max_allowed=0,
            blocking=True,
            scan_types=[SecurityScanType.SAST, SecurityScanType.DAST, SecurityScanType.SCA],
        ),
        SecurityThreshold(
            severity=VulnerabilitySeverity.HIGH,
            max_allowed=5,
            blocking=True,
            scan_types=[SecurityScanType.SAST, SecurityScanType.DAST, SecurityScanType.SCA],
        ),
        SecurityThreshold(
            severity=VulnerabilitySeverity.MEDIUM,
            max_allowed=20,
            blocking=False,  # Allow but monitor
            scan_types=[SecurityScanType.SAST, SecurityScanType.SCA],
        ),
        SecurityThreshold(
            severity=VulnerabilitySeverity.LOW,
            max_allowed=50,
            blocking=False,
            scan_types=[SecurityScanType.SAST],
        ),
    ]

    # Security test cases for user profile operations
    SECURITY_TEST_CASES = [
        SecurityTestCase(
            name="sql_injection_prevention",
            category="Input Validation",
            severity=VulnerabilitySeverity.CRITICAL,
            description="Verify SQL injection attacks are prevented in profile fields",
            test_file="tests/security/test_user_profile_security.py::TestInputValidationSecurity::test_sql_injection_prevention",
            expected_result="All SQL injection attempts should be rejected or sanitized",
        ),
        SecurityTestCase(
            name="xss_prevention",
            category="Input Validation",
            severity=VulnerabilitySeverity.HIGH,
            description="Verify XSS attacks are prevented in profile fields",
            test_file="tests/security/test_user_profile_security.py::TestInputValidationSecurity::test_xss_prevention",
            expected_result="All XSS payloads should be rejected or sanitized",
        ),
        SecurityTestCase(
            name="authentication_bypass",
            category="Authentication",
            severity=VulnerabilitySeverity.CRITICAL,
            description="Verify JWT token security and prevent authentication bypass",
            test_file="tests/security/test_user_profile_security.py::TestAuthenticationSecurity::test_jwt_token_expiration_handling",
            expected_result="Invalid and expired tokens should be rejected",
        ),
        SecurityTestCase(
            name="authorization_enforcement",
            category="Authorization",
            severity=VulnerabilitySeverity.HIGH,
            description="Verify users can only access their own profile data",
            test_file="tests/security/test_user_profile_security.py::TestAuthorizationSecurity::test_user_profile_isolation",
            expected_result="Users should only access their own data",
        ),
        SecurityTestCase(
            name="password_security",
            category="Authentication",
            severity=VulnerabilitySeverity.HIGH,
            description="Verify password complexity and storage security",
            test_file="tests/security/test_user_profile_security.py::TestPasswordSecurity::test_weak_password_rejection",
            expected_result="Weak passwords should be rejected, strong passwords accepted",
        ),
        SecurityTestCase(
            name="rate_limiting",
            category="DoS Protection",
            severity=VulnerabilitySeverity.MEDIUM,
            description="Verify API rate limiting prevents abuse",
            test_file="tests/security/test_user_profile_security.py::TestRateLimitingSecurity::test_authentication_rate_limiting",
            expected_result="Excessive requests should be rate limited",
        ),
        SecurityTestCase(
            name="data_exposure_prevention",
            category="Data Protection",
            severity=VulnerabilitySeverity.HIGH,
            description="Verify sensitive data is not exposed in API responses",
            test_file="tests/security/test_user_profile_security.py::TestDataProtectionSecurity::test_sensitive_data_exposure_prevention",
            expected_result="Sensitive fields should not appear in responses",
        ),
    ]

    # Compliance requirements
    COMPLIANCE_REQUIREMENTS = [
        ComplianceRequirement(
            standard="OWASP Top 10",
            requirement="A01:2021 - Broken Access Control",
            description="Verify proper access controls for user profile operations",
            test_cases=[
                "authorization_enforcement",
                "authentication_bypass",
            ],
            automated_check=True,
        ),
        ComplianceRequirement(
            standard="OWASP Top 10",
            requirement="A02:2021 - Cryptographic Failures",
            description="Verify proper password hashing and data protection",
            test_cases=[
                "password_security",
                "data_exposure_prevention",
            ],
            automated_check=True,
        ),
        ComplianceRequirement(
            standard="OWASP Top 10",
            requirement="A03:2021 - Injection",
            description="Verify prevention of SQL injection and other injection attacks",
            test_cases=[
                "sql_injection_prevention",
                "xss_prevention",
            ],
            automated_check=True,
        ),
        ComplianceRequirement(
            standard="GDPR",
            requirement="Article 25 - Data Protection by Design",
            description="Verify data protection measures are implemented by design",
            test_cases=[
                "data_exposure_prevention",
                "authorization_enforcement",
            ],
            automated_check=True,
        ),
        ComplianceRequirement(
            standard="GDPR",
            requirement="Article 32 - Security of Processing",
            description="Verify appropriate security measures for personal data",
            test_cases=[
                "password_security",
                "rate_limiting",
            ],
            automated_check=True,
        ),
    ]

    @classmethod
    def get_vulnerability_threshold(cls, severity: VulnerabilitySeverity) -> SecurityThreshold:
        """Get vulnerability threshold for a severity level."""
        for threshold in cls.VULNERABILITY_THRESHOLDS:
            if threshold.severity == severity:
                return threshold
        raise ValueError(f"No threshold defined for severity: {severity}")

    @classmethod
    def validate_vulnerability_count(
        cls,
        severity: VulnerabilitySeverity,
        count: int
    ) -> Dict[str, Any]:
        """Validate if vulnerability count is within acceptable limits."""
        threshold = cls.get_vulnerability_threshold(severity)

        within_limit = count <= threshold.max_allowed

        return {
            "severity": severity.value,
            "count": count,
            "max_allowed": threshold.max_allowed,
            "within_limit": within_limit,
            "blocking": threshold.blocking,
            "action_required": not within_limit and threshold.blocking,
        }

    @classmethod
    def get_security_test_cases(cls, category: Optional[str] = None) -> List[SecurityTestCase]:
        """Get security test cases, optionally filtered by category."""
        if category:
            return [tc for tc in cls.SECURITY_TEST_CASES if tc.category == category]
        return cls.SECURITY_TEST_CASES

    @classmethod
    def get_compliance_requirements(cls, standard: Optional[str] = None) -> List[ComplianceRequirement]:
        """Get compliance requirements, optionally filtered by standard."""
        if standard:
            return [req for req in cls.COMPLIANCE_REQUIREMENTS if req.standard == standard]
        return cls.COMPLIANCE_REQUIREMENTS

    @classmethod
    def generate_security_report(cls, scan_results: Dict[str, Any]) -> str:
        """Generate a comprehensive security report."""
        report_lines = [
            "# Security Assessment Report - LeanVibe Agent Hive",
            "",
            f"Generated: {os.environ.get('GITHUB_RUN_ID', 'Local')} - {os.environ.get('GITHUB_SHA', 'HEAD')[:8]}",
            "",
            "## Vulnerability Summary",
            "",
        ]

        # Add vulnerability counts
        for severity in VulnerabilitySeverity:
            count = scan_results.get(f"vulnerabilities_{severity.value.lower()}", 0)
            validation = cls.validate_vulnerability_count(severity, count)

            status = "✅" if validation["within_limit"] else "❌"
            blocking = " (BLOCKING)" if validation["blocking"] and not validation["within_limit"] else ""

            report_lines.extend([
                f"- **{severity.value}**: {status} {count}/{validation['max_allowed']} {blocking}",
            ])

        report_lines.extend([
            "",
            "## Security Test Results",
            "",
        ])

        # Add test results
        for test_case in cls.SECURITY_TEST_CASES:
            test_result = scan_results.get(f"test_{test_case.name}", {})
            status = test_result.get("status", "NOT_RUN")

            if status == "PASSED":
                icon = "✅"
            elif status == "FAILED":
                icon = "❌"
            else:
                icon = "⏭️"

            report_lines.extend([
                f"- **{test_case.name}**: {icon} {status}",
                f"  - Category: {test_case.category}",
                f"  - Severity: {test_case.severity.value}",
                "",
            ])

        # Add compliance status
        report_lines.extend([
            "## Compliance Status",
            "",
        ])

        for requirement in cls.COMPLIANCE_REQUIREMENTS:
            compliant_tests = 0
            total_tests = len(requirement.test_cases)

            for test_name in requirement.test_cases:
                test_result = scan_results.get(f"test_{test_name}", {})
                if test_result.get("status") == "PASSED":
                    compliant_tests += 1

            compliance_percentage = (compliant_tests / total_tests) * 100 if total_tests > 0 else 0
            status = "✅ COMPLIANT" if compliance_percentage == 100 else f"⚠️ {compliance_percentage:.0f}% COMPLIANT"

            report_lines.extend([
                f"- **{requirement.standard} - {requirement.requirement}**: {status}",
                f"  - {requirement.description}",
                "",
            ])

        # Add recommendations
        report_lines.extend([
            "## Security Recommendations",
            "",
        ])

        # Check for critical issues
        critical_count = scan_results.get("vulnerabilities_critical", 0)
        if critical_count > 0:
            report_lines.extend([
                "- **CRITICAL**: Address all critical vulnerabilities immediately",
                "- Review authentication and authorization mechanisms",
                "- Implement additional input validation",
            ])

        high_count = scan_results.get("vulnerabilities_high", 0)
        if high_count > cls.get_vulnerability_threshold(VulnerabilitySeverity.HIGH).max_allowed:
            report_lines.extend([
                "- **HIGH**: Review high-severity vulnerabilities",
                "- Consider security hardening measures",
                "- Implement additional monitoring",
            ])

        # Check test failures
        failed_tests = [
            tc for tc in cls.SECURITY_TEST_CASES
            if scan_results.get(f"test_{tc.name}", {}).get("status") == "FAILED"
        ]

        if failed_tests:
            report_lines.extend([
                "- **FAILED TESTS**: Review and fix failed security tests",
                "- Ensure security controls are properly implemented",
            ])

        if not failed_tests and critical_count == 0:
            report_lines.extend([
                "- **EXCELLENT**: All security tests passed",
                "- System meets security requirements",
                "- Continue monitoring for new vulnerabilities",
            ])

        return "\n".join(report_lines)


# Security scanning configuration for CI/CD
def get_security_scan_config() -> Dict[str, Any]:
    """Get security scanning configuration for CI/CD."""
    return {
        "bandit": {
            "enabled": True,
            "config_file": "pyproject.toml",
            "severity_threshold": "medium",
            "confidence_threshold": "medium",
        },
        "trivy": {
            "enabled": True,
            "scan_types": ["fs", "config"],
            "severity_threshold": "medium",
            "ignore_unfixed": True,
        },
        "owasp_zap": {
            "enabled": True,
            "target_url": "http://localhost:8000",
            "scan_type": "baseline",
            "risk_threshold": "medium",
        },
        "safety": {
            "enabled": True,
            "severity_threshold": "medium",
        },
    }


def should_run_security_scan() -> bool:
    """Determine if security scans should run."""
    # Run on all PRs, main branch, and scheduled runs
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    ref = os.environ.get("GITHUB_REF", "")

    return (
        event in ["pull_request", "push", "schedule"] or
        "main" in ref or
        "master" in ref
    )


# Export key functions and classes
__all__ = [
    "SecurityConfiguration",
    "VulnerabilitySeverity",
    "SecurityScanType",
    "SecurityThreshold",
    "SecurityTestCase",
    "ComplianceRequirement",
    "get_security_scan_config",
    "should_run_security_scan",
]