"""Comprehensive Test Strategy Configuration for LeanVibe Agent Hive.

This module defines the testing strategy, quality gates, and test execution
configuration for the user profile CRUD operations and overall system.
"""

import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum


class TestCategory(Enum):
    """Test categories for comprehensive testing strategy."""
    UNIT = "unit"
    INTEGRATION = "integration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EDGE_CASE = "edge_case"
    E2E = "e2e"


class QualityGate(Enum):
    """Quality gates that must pass for deployment."""
    UNIT_TEST_COVERAGE = "unit_test_coverage"
    SECURITY_SCAN = "security_scan"
    PERFORMANCE_SLA = "performance_sla"
    INTEGRATION_TESTS = "integration_tests"
    LINTING = "linting"


@dataclass
class TestConfiguration:
    """Configuration for test execution."""
    category: TestCategory
    test_files: List[str]
    coverage_target: float
    timeout_seconds: int
    required_for_ci: bool
    quality_gate: bool


@dataclass
class PerformanceSLA:
    """Performance Service Level Agreement configuration."""
    operation: str
    target_rps: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    concurrent_users: int


@dataclass
class SecurityRequirement:
    """Security testing requirements."""
    vulnerability_severity: str
    max_allowed: int
    scan_types: List[str]


class TestStrategy:
    """Comprehensive test strategy for the system."""

    # Test configurations by category
    TEST_CONFIGURATIONS = {
        TestCategory.UNIT: TestConfiguration(
            category=TestCategory.UNIT,
            test_files=[
                "tests/unit/test_user_profile_crud.py",
                "tests/unit/test_user_profile_edge_cases.py",
                "tests/unit/test_models.py",
                "tests/unit/test_security.py",
            ],
            coverage_target=95.0,
            timeout_seconds=300,
            required_for_ci=True,
            quality_gate=True,
        ),
        TestCategory.INTEGRATION: TestConfiguration(
            category=TestCategory.INTEGRATION,
            test_files=[
                "tests/integration/test_user_profile_crud_integration.py",
                "tests/integration/test_api.py",
                "tests/integration/test_core_system_integration.py",
            ],
            coverage_target=85.0,
            timeout_seconds=600,
            required_for_ci=True,
            quality_gate=True,
        ),
        TestCategory.SECURITY: TestConfiguration(
            category=TestCategory.SECURITY,
            test_files=[
                "tests/security/test_user_profile_security.py",
                "tests/security/test_api_security.py",
                "tests/security/test_oauth2_scopes.py",
            ],
            coverage_target=90.0,
            timeout_seconds=900,
            required_for_ci=True,
            quality_gate=True,
        ),
        TestCategory.PERFORMANCE: TestConfiguration(
            category=TestCategory.PERFORMANCE,
            test_files=[
                "tests/performance/test_user_profile_performance.py",
            ],
            coverage_target=80.0,
            timeout_seconds=1800,  # 30 minutes
            required_for_ci=False,  # Run on schedule/manual
            quality_gate=False,
        ),
        TestCategory.EDGE_CASE: TestConfiguration(
            category=TestCategory.EDGE_CASE,
            test_files=[
                "tests/unit/test_user_profile_edge_cases.py",
            ],
            coverage_target=90.0,
            timeout_seconds=300,
            required_for_ci=True,
            quality_gate=True,
        ),
    }

    # Performance SLAs for user profile operations
    PERFORMANCE_SLAS = [
        PerformanceSLA(
            operation="GET /api/v1/auth/profile",
            target_rps=100.0,
            p95_response_time_ms=200.0,
            p99_response_time_ms=300.0,
            concurrent_users=500,
        ),
        PerformanceSLA(
            operation="PUT /api/v1/auth/profile",
            target_rps=50.0,
            p95_response_time_ms=500.0,
            p99_response_time_ms=750.0,
            concurrent_users=250,
        ),
        PerformanceSLA(
            operation="POST /api/v1/auth/users",
            target_rps=20.0,
            p95_response_time_ms=1000.0,
            p99_response_time_ms=1500.0,
            concurrent_users=100,
        ),
        PerformanceSLA(
            operation="DELETE /api/v1/auth/profile",
            target_rps=10.0,
            p95_response_time_ms=2000.0,
            p99_response_time_ms=3000.0,
            concurrent_users=50,
        ),
    ]

    # Security requirements
    SECURITY_REQUIREMENTS = [
        SecurityRequirement(
            vulnerability_severity="CRITICAL",
            max_allowed=0,
            scan_types=["SAST", "DAST", "dependency_scan"],
        ),
        SecurityRequirement(
            vulnerability_severity="HIGH",
            max_allowed=5,
            scan_types=["SAST", "DAST", "dependency_scan"],
        ),
        SecurityRequirement(
            vulnerability_severity="MEDIUM",
            max_allowed=20,
            scan_types=["SAST", "dependency_scan"],
        ),
    ]

    # Quality gates configuration
    QUALITY_GATES = {
        QualityGate.UNIT_TEST_COVERAGE: {
            "target": 90.0,
            "blocking": True,
            "categories": [TestCategory.UNIT],
        },
        QualityGate.SECURITY_SCAN: {
            "target": 0,  # Zero critical vulnerabilities
            "blocking": True,
            "categories": [TestCategory.SECURITY],
        },
        QualityGate.PERFORMANCE_SLA: {
            "target": 95.0,  # 95% of SLAs must pass
            "blocking": False,  # Don't block deployment, but alert
            "categories": [TestCategory.PERFORMANCE],
        },
        QualityGate.INTEGRATION_TESTS: {
            "target": 100.0,  # All integration tests must pass
            "blocking": True,
            "categories": [TestCategory.INTEGRATION],
        },
        QualityGate.LINTING: {
            "target": 100.0,  # No linting errors
            "blocking": True,
            "categories": [],  # Applied to all code
        },
    }

    @classmethod
    def get_test_configuration(cls, category: TestCategory) -> TestConfiguration:
        """Get test configuration for a specific category."""
        return cls.TEST_CONFIGURATIONS[category]

    @classmethod
    def get_required_tests_for_ci(cls) -> List[TestConfiguration]:
        """Get all tests required for CI execution."""
        return [
            config for config in cls.TEST_CONFIGURATIONS.values()
            if config.required_for_ci
        ]

    @classmethod
    def get_quality_gate_tests(cls) -> List[TestConfiguration]:
        """Get all tests that are quality gates."""
        return [
            config for config in cls.TEST_CONFIGURATIONS.values()
            if config.quality_gate
        ]

    @classmethod
    def get_performance_slas(cls) -> List[PerformanceSLA]:
        """Get all performance SLAs."""
        return cls.PERFORMANCE_SLAS

    @classmethod
    def validate_quality_gate(
        cls,
        gate: QualityGate,
        actual_value: float,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate if a quality gate passes."""
        gate_config = cls.QUALITY_GATES[gate]
        target = gate_config["target"]
        blocking = gate_config["blocking"]

        passed = actual_value >= target

        return {
            "gate": gate.value,
            "target": target,
            "actual": actual_value,
            "passed": passed,
            "blocking": blocking,
            "context": context or {},
        }

    @classmethod
    def generate_test_report(cls, results: Dict[str, Any]) -> str:
        """Generate a comprehensive test report."""
        report_lines = [
            "# Comprehensive Test Report - LeanVibe Agent Hive",
            "",
            f"Generated: {os.environ.get('GITHUB_RUN_ID', 'Local')} - {os.environ.get('GITHUB_SHA', 'HEAD')[:8]}",
            "",
            "## Test Results Summary",
            "",
        ]

        # Add test category results
        for category in TestCategory:
            if category.value in results:
                category_results = results[category.value]
                report_lines.extend([
                    f"### {category.value.upper()} Tests",
                    f"- **Status**: {'✅ PASSED' if category_results.get('passed', False) else '❌ FAILED'}",
                    f"- **Coverage**: {category_results.get('coverage', 0):.1f}%",
                    f"- **Duration**: {category_results.get('duration', 0):.1f}s",
                    f"- **Tests Run**: {category_results.get('tests_run', 0)}",
                    "",
                ])

        # Add quality gate results
        report_lines.extend([
            "## Quality Gates",
            "",
        ])

        for gate in QualityGate:
            if f"gate_{gate.value}" in results:
                gate_result = results[f"gate_{gate.value}"]
                status = "✅ PASSED" if gate_result["passed"] else "❌ FAILED"
                blocking = " (BLOCKING)" if gate_result["blocking"] else ""
                report_lines.extend([
                    f"- **{gate.value.upper()}**: {status}{blocking}",
                    f"  - Target: {gate_result['target']}",
                    f"  - Actual: {gate_result['actual']}",
                    "",
                ])

        # Add performance SLAs
        report_lines.extend([
            "## Performance SLAs",
            "",
        ])

        for sla in cls.PERFORMANCE_SLAS:
            sla_status = results.get(f"sla_{sla.operation}", {}).get("passed", False)
            status = "✅ MET" if sla_status else "❌ NOT MET"
            report_lines.extend([
                f"- **{sla.operation}**: {status}",
                f"  - Target RPS: {sla.target_rps}",
                f"  - P95 Response Time: {sla.p95_response_time_ms}ms",
                "",
            ])

        # Add security summary
        report_lines.extend([
            "## Security Summary",
            "",
            "- **Critical Vulnerabilities**: 0 allowed",
            "- **High Vulnerabilities**: ≤5 allowed",
            "- **Medium Vulnerabilities**: ≤20 allowed",
            "- **Security Tests**: Authentication, Authorization, Input Validation",
            "",
            "## Recommendations",
            "",
        ])

        # Add recommendations based on results
        if not all(cat_results.get("passed", False) for cat_results in results.values() if isinstance(cat_results, dict)):
            report_lines.extend([
                "- Review failed tests and fix issues",
                "- Address security vulnerabilities",
                "- Improve test coverage in failing areas",
            ])
        else:
            report_lines.extend([
                "- All quality gates passed ✅",
                "- System ready for deployment",
                "- Monitor performance in production",
            ])

        return "\n".join(report_lines)


# Environment-specific test configuration
def get_test_environment_config() -> Dict[str, Any]:
    """Get test configuration based on environment."""
    env = os.environ.get("ENVIRONMENT", "development")

    configs = {
        "development": {
            "database_url": "postgresql://test_user:test_pass@localhost:5432/leanvibe_hive_test",
            "redis_url": "redis://localhost:6379",
            "test_timeout": 300,
            "performance_enabled": False,
        },
        "ci": {
            "database_url": "postgresql://test_user:test_pass@localhost:5432/leanvibe_hive_test",
            "redis_url": "redis://localhost:6379",
            "test_timeout": 600,
            "performance_enabled": True,
        },
        "staging": {
            "database_url": os.environ.get("DATABASE_URL", ""),
            "redis_url": os.environ.get("REDIS_URL", ""),
            "test_timeout": 900,
            "performance_enabled": True,
        },
        "production": {
            "database_url": os.environ.get("DATABASE_URL", ""),
            "redis_url": os.environ.get("REDIS_URL", ""),
            "test_timeout": 1800,
            "performance_enabled": True,
        },
    }

    return configs.get(env, configs["development"])


# Export key functions and classes
__all__ = [
    "TestStrategy",
    "TestCategory",
    "QualityGate",
    "TestConfiguration",
    "PerformanceSLA",
    "SecurityRequirement",
    "get_test_environment_config",
]