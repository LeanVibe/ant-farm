"""
Quality Gate Agent - Validates subagent outputs and ensures consistency.

This agent implements quality gates for complex workflows, ensuring that
subagent outputs meet quality standards, are consistent with requirements,
and pass validation checks before being considered complete.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import structlog

from ..constants import Intervals
from ..message_broker import MessageType, message_broker

logger = structlog.get_logger()


class QualityGateType(Enum):
    """Types of quality gates."""
    CODE_REVIEW = "code_review"
    TEST_VALIDATION = "test_validation"
    PERFORMANCE_CHECK = "performance_check"
    SECURITY_SCAN = "security_scan"
    DOCUMENTATION_REVIEW = "documentation_review"
    DEPLOYMENT_VALIDATION = "deployment_validation"


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue found during quality gate checks."""
    gate_type: QualityGateType
    severity: ValidationSeverity
    description: str
    location: Optional[str] = None
    suggestion: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class QualityGateResult:
    """Result of a quality gate validation."""
    gate_type: QualityGateType
    passed: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    score: float = 0.0
    execution_time: float = 0.0
    validated_by: str = "quality_gate_agent"


@dataclass
class ConsolidatedValidation:
    """Consolidated validation results for a complete workflow."""
    workflow_id: str
    overall_passed: bool
    gate_results: List[QualityGateResult] = field(default_factory=list)
    total_issues: int = 0
    critical_issues: int = 0
    validation_score: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class QualityGateAgent:
    """Agent responsible for implementing quality gates and validation."""

    def __init__(self):
        self.quality_gates: Dict[QualityGateType, callable] = {
            QualityGateType.CODE_REVIEW: self._run_code_review,
            QualityGateType.TEST_VALIDATION: self._run_test_validation,
            QualityGateType.PERFORMANCE_CHECK: self._run_performance_check,
            QualityGateType.SECURITY_SCAN: self._run_security_scan,
            QualityGateType.DOCUMENTATION_REVIEW: self._run_documentation_review,
            QualityGateType.DEPLOYMENT_VALIDATION: self._run_deployment_validation,
        }

        self.validation_history: Dict[str, ConsolidatedValidation] = {}
        self.quality_metrics = {
            "validations_performed": 0,
            "gates_passed": 0,
            "gates_failed": 0,
            "average_quality_score": 0.0,
            "critical_issues_found": 0
        }

    async def initialize(self):
        """Initialize the quality gate agent."""
        logger.info("Quality gate agent initialized")

    async def validate_workflow_output(self, workflow_id: str, workflow_output: Dict[str, Any]) -> ConsolidatedValidation:
        """Validate the complete output of a workflow through all quality gates."""
        logger.info("Starting workflow validation", workflow_id=workflow_id)

        validation = ConsolidatedValidation(workflow_id=workflow_id)
        gate_results = []

        # Run all applicable quality gates
        for gate_type, gate_function in self.quality_gates.items():
            if self._should_run_gate(gate_type, workflow_output):
                gate_result = await gate_function(workflow_output)
                gate_results.append(gate_result)

                # Update metrics
                if gate_result.passed:
                    self.quality_metrics["gates_passed"] += 1
                else:
                    self.quality_metrics["gates_failed"] += 1

                # Count issues
                validation.total_issues += len(gate_result.issues)
                validation.critical_issues += len([
                    issue for issue in gate_result.issues
                    if issue.severity == ValidationSeverity.CRITICAL
                ])

        validation.gate_results = gate_results
        validation.overall_passed = all(result.passed for result in gate_results)
        validation.end_time = time.time()

        # Calculate overall validation score
        if gate_results:
            total_score = sum(result.score for result in gate_results)
            validation.validation_score = total_score / len(gate_results)

        # Store validation result
        self.validation_history[workflow_id] = validation
        self.quality_metrics["validations_performed"] += 1

        # Update average quality score
        total_validations = self.quality_metrics["validations_performed"]
        if total_validations > 0:
            self.quality_metrics["average_quality_score"] = (
                (self.quality_metrics["average_quality_score"] * (total_validations - 1)) +
                validation.validation_score
            ) / total_validations

        logger.info(
            "Workflow validation completed",
            workflow_id=workflow_id,
            passed=validation.overall_passed,
            total_issues=validation.total_issues,
            critical_issues=validation.critical_issues,
            quality_score=validation.validation_score
        )

        return validation

    def _should_run_gate(self, gate_type: QualityGateType, workflow_output: Dict[str, Any]) -> bool:
        """Determine if a quality gate should be run based on the workflow output."""
        # Check if the workflow output contains relevant content for this gate
        output_content = json.dumps(workflow_output).lower()

        gate_indicators = {
            QualityGateType.CODE_REVIEW: ["def ", "class ", "import ", "function", "method"],
            QualityGateType.TEST_VALIDATION: ["test", "assert", "pytest", "unittest"],
            QualityGateType.PERFORMANCE_CHECK: ["performance", "optimization", "speed", "efficiency"],
            QualityGateType.SECURITY_SCAN: ["security", "auth", "password", "token", "encrypt"],
            QualityGateType.DOCUMENTATION_REVIEW: ["doc", "readme", "comment", "documentation"],
            QualityGateType.DEPLOYMENT_VALIDATION: ["deploy", "docker", "kubernetes", "production"]
        }

        indicators = gate_indicators.get(gate_type, [])
        return any(indicator in output_content for indicator in indicators)

    async def _run_code_review(self, workflow_output: Dict[str, Any]) -> QualityGateResult:
        """Run code review quality gate."""
        start_time = time.time()
        issues = []

        # Extract code from workflow output
        code_content = self._extract_code_content(workflow_output)

        if code_content:
            # Check for common code quality issues
            issues.extend(self._check_code_style(code_content))
            issues.extend(self._check_code_complexity(code_content))
            issues.extend(self._check_error_handling(code_content))
            issues.extend(self._check_security_issues(code_content))

        # Calculate score based on issues
        score = self._calculate_gate_score(issues)

        execution_time = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)

        return QualityGateResult(
            gate_type=QualityGateType.CODE_REVIEW,
            passed=passed,
            issues=issues,
            score=score,
            execution_time=execution_time
        )

    async def _run_test_validation(self, workflow_output: Dict[str, Any]) -> QualityGateResult:
        """Run test validation quality gate."""
        start_time = time.time()
        issues = []

        # Extract test content from workflow output
        test_content = self._extract_test_content(workflow_output)

        if test_content:
            # Check test quality
            issues.extend(self._check_test_coverage(test_content))
            issues.extend(self._check_test_structure(test_content))
            issues.extend(self._check_test_best_practices(test_content))

        # Calculate score
        score = self._calculate_gate_score(issues)
        execution_time = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)

        return QualityGateResult(
            gate_type=QualityGateType.TEST_VALIDATION,
            passed=passed,
            issues=issues,
            score=score,
            execution_time=execution_time
        )

    async def _run_performance_check(self, workflow_output: Dict[str, Any]) -> QualityGateResult:
        """Run performance check quality gate."""
        start_time = time.time()
        issues = []

        # Check for performance considerations
        content = json.dumps(workflow_output).lower()

        # Check for potential performance issues
        if "select * from" in content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.PERFORMANCE_CHECK,
                severity=ValidationSeverity.WARNING,
                description="Potential N+1 query or inefficient database access",
                suggestion="Use specific column selection and consider query optimization"
            ))

        if "for " in content and " in " in content and "len(" not in content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.PERFORMANCE_CHECK,
                severity=ValidationSeverity.INFO,
                description="Loop detected - consider vectorization or optimization",
                suggestion="Review loop efficiency and consider alternative approaches"
            ))

        # Calculate score
        score = self._calculate_gate_score(issues)
        execution_time = time.time() - start_time
        passed = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

        return QualityGateResult(
            gate_type=QualityGateType.PERFORMANCE_CHECK,
            passed=passed,
            issues=issues,
            score=score,
            execution_time=execution_time
        )

    async def _run_security_scan(self, workflow_output: Dict[str, Any]) -> QualityGateResult:
        """Run security scan quality gate."""
        start_time = time.time()
        issues = []

        content = json.dumps(workflow_output).lower()

        # Check for common security issues
        security_patterns = {
            "password": "Potential hardcoded password detected",
            "secret": "Potential secret exposure",
            "token": "Token handling detected - ensure secure storage",
            "sql": "SQL query detected - check for injection vulnerabilities",
            "eval(": "Use of eval() detected - security risk",
            "exec(": "Use of exec() detected - security risk"
        }

        for pattern, description in security_patterns.items():
            if pattern in content:
                severity = ValidationSeverity.CRITICAL if pattern in ["eval(", "exec("] else ValidationSeverity.WARNING
                issues.append(ValidationIssue(
                    gate_type=QualityGateType.SECURITY_SCAN,
                    severity=severity,
                    description=description,
                    suggestion="Review and implement secure alternatives"
                ))

        # Calculate score
        score = self._calculate_gate_score(issues)
        execution_time = time.time() - start_time
        passed = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

        return QualityGateResult(
            gate_type=QualityGateType.SECURITY_SCAN,
            passed=passed,
            issues=issues,
            score=score,
            execution_time=execution_time
        )

    async def _run_documentation_review(self, workflow_output: Dict[str, Any]) -> QualityGateResult:
        """Run documentation review quality gate."""
        start_time = time.time()
        issues = []

        content = json.dumps(workflow_output).lower()

        # Check documentation quality
        if "todo" in content or "fixme" in content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.DOCUMENTATION_REVIEW,
                severity=ValidationSeverity.WARNING,
                description="TODO/FIXME comments found in code",
                suggestion="Resolve or document technical debt"
            ))

        if len(content) < 100:  # Very short documentation
            issues.append(ValidationIssue(
                gate_type=QualityGateType.DOCUMENTATION_REVIEW,
                severity=ValidationSeverity.INFO,
                description="Documentation appears minimal",
                suggestion="Consider adding more detailed documentation"
            ))

        # Calculate score
        score = self._calculate_gate_score(issues)
        execution_time = time.time() - start_time
        passed = not any(issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL] for issue in issues)

        return QualityGateResult(
            gate_type=QualityGateType.DOCUMENTATION_REVIEW,
            passed=passed,
            issues=issues,
            score=score,
            execution_time=execution_time
        )

    async def _run_deployment_validation(self, workflow_output: Dict[str, Any]) -> QualityGateResult:
        """Run deployment validation quality gate."""
        start_time = time.time()
        issues = []

        content = json.dumps(workflow_output).lower()

        # Check deployment readiness
        if "dockerfile" in content or "docker" in content:
            if "healthcheck" not in content:
                issues.append(ValidationIssue(
                    gate_type=QualityGateType.DEPLOYMENT_VALIDATION,
                    severity=ValidationSeverity.WARNING,
                    description="Docker container missing health check",
                    suggestion="Add HEALTHCHECK instruction to Dockerfile"
                ))

        if "kubernetes" in content or "k8s" in content:
            if "resource" not in content:
                issues.append(ValidationIssue(
                    gate_type=QualityGateType.DEPLOYMENT_VALIDATION,
                    severity=ValidationSeverity.INFO,
                    description="Kubernetes deployment without resource limits",
                    suggestion="Add resource requests and limits"
                ))

        # Calculate score
        score = self._calculate_gate_score(issues)
        execution_time = time.time() - start_time
        passed = not any(issue.severity == ValidationSeverity.CRITICAL for issue in issues)

        return QualityGateResult(
            gate_type=QualityGateType.DEPLOYMENT_VALIDATION,
            passed=passed,
            issues=issues,
            score=score,
            execution_time=execution_time
        )

    def _extract_code_content(self, workflow_output: Dict[str, Any]) -> str:
        """Extract code content from workflow output."""
        # Look for code in various possible locations
        code_locations = ["code", "implementation", "result", "output", "content"]

        for location in code_locations:
            if location in workflow_output:
                content = workflow_output[location]
                if isinstance(content, str):
                    return content
                elif isinstance(content, dict) and "code" in content:
                    return content["code"]

        # Convert entire output to string as fallback
        return json.dumps(workflow_output)

    def _extract_test_content(self, workflow_output: Dict[str, Any]) -> str:
        """Extract test content from workflow output."""
        # Look for test-related content
        test_locations = ["tests", "test_code", "test_content", "test_output"]

        for location in test_locations:
            if location in workflow_output:
                content = workflow_output[location]
                if isinstance(content, str):
                    return content

        return ""

    def _check_code_style(self, code_content: str) -> List[ValidationIssue]:
        """Check code style and formatting."""
        issues = []

        # Check line length (basic check)
        long_lines = [line for line in code_content.split('\n') if len(line) > 88]
        if long_lines:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.CODE_REVIEW,
                severity=ValidationSeverity.INFO,
                description=f"Found {len(long_lines)} lines exceeding 88 characters",
                suggestion="Consider breaking long lines for better readability"
            ))

        # Check for common style issues
        if '\t' in code_content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.CODE_REVIEW,
                severity=ValidationSeverity.WARNING,
                description="Tabs detected in code",
                suggestion="Use spaces for consistent indentation"
            ))

        return issues

    def _check_code_complexity(self, code_content: str) -> List[ValidationIssue]:
        """Check code complexity metrics."""
        issues = []

        # Count functions and classes
        function_count = len(re.findall(r'def \w+', code_content))
        class_count = len(re.findall(r'class \w+', code_content))

        if function_count > 20:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.CODE_REVIEW,
                severity=ValidationSeverity.WARNING,
                description=f"High function count ({function_count}) - consider refactoring",
                suggestion="Break down into smaller modules or classes"
            ))

        if class_count > 10:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.CODE_REVIEW,
                severity=ValidationSeverity.INFO,
                description=f"High class count ({class_count}) - review architecture",
                suggestion="Consider if all classes are necessary"
            ))

        return issues

    def _check_error_handling(self, code_content: str) -> List[ValidationIssue]:
        """Check error handling patterns."""
        issues = []

        # Check for bare except clauses
        if "except:" in code_content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.CODE_REVIEW,
                severity=ValidationSeverity.WARNING,
                description="Bare except clause found",
                suggestion="Specify specific exception types to catch"
            ))

        # Check for missing try-except blocks around risky operations
        risky_operations = ["open(", "requests.", "subprocess.", "os.system"]
        for operation in risky_operations:
            if operation in code_content and "try:" not in code_content:
                issues.append(ValidationIssue(
                    gate_type=QualityGateType.CODE_REVIEW,
                    severity=ValidationSeverity.INFO,
                    description=f"Risky operation {operation} without try-except",
                    suggestion="Add proper error handling around risky operations"
                ))
                break

        return issues

    def _check_security_issues(self, code_content: str) -> List[ValidationIssue]:
        """Check for security vulnerabilities."""
        issues = []

        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']'
        ]

        for pattern in secret_patterns:
            if re.search(pattern, code_content, re.IGNORECASE):
                issues.append(ValidationIssue(
                    gate_type=QualityGateType.CODE_REVIEW,
                    severity=ValidationSeverity.CRITICAL,
                    description="Potential hardcoded secret detected",
                    suggestion="Use environment variables or secure secret management"
                ))
                break

        return issues

    def _check_test_coverage(self, test_content: str) -> List[ValidationIssue]:
        """Check test coverage and quality."""
        issues = []

        # Check for basic test structure
        if "def test_" not in test_content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.TEST_VALIDATION,
                severity=ValidationSeverity.ERROR,
                description="No test functions found",
                suggestion="Add proper test functions following naming conventions"
            ))

        # Check for assertions
        if "assert" not in test_content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.TEST_VALIDATION,
                severity=ValidationSeverity.ERROR,
                description="No assertions found in tests",
                suggestion="Add assertions to validate expected behavior"
            ))

        return issues

    def _check_test_structure(self, test_content: str) -> List[ValidationIssue]:
        """Check test structure and organization."""
        issues = []

        # Check for test class structure
        if "class Test" not in test_content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.TEST_VALIDATION,
                severity=ValidationSeverity.INFO,
                description="No test classes found",
                suggestion="Consider organizing tests into classes for better structure"
            ))

        return issues

    def _check_test_best_practices(self, test_content: str) -> List[ValidationIssue]:
        """Check for test best practices."""
        issues = []

        # Check for mocking without assertions
        if "mock" in test_content.lower() and "assert" not in test_content:
            issues.append(ValidationIssue(
                gate_type=QualityGateType.TEST_VALIDATION,
                severity=ValidationSeverity.WARNING,
                description="Mock usage without assertions",
                suggestion="Add assertions to verify mock behavior"
            ))

        return issues

    def _calculate_gate_score(self, issues: List[ValidationIssue]) -> float:
        """Calculate a quality score for the gate based on issues found."""
        if not issues:
            return 1.0  # Perfect score

        # Weight issues by severity
        severity_weights = {
            ValidationSeverity.INFO: 0.1,
            ValidationSeverity.WARNING: 0.3,
            ValidationSeverity.ERROR: 0.7,
            ValidationSeverity.CRITICAL: 1.0
        }

        total_penalty = sum(severity_weights[issue.severity] for issue in issues)
        max_penalty = len(issues) * 1.0  # Maximum possible penalty

        # Calculate score (1.0 - normalized penalty)
        score = 1.0 - (total_penalty / max_penalty)
        return max(0.0, score)  # Ensure non-negative score

    async def get_validation_history(self, workflow_id: str) -> Optional[ConsolidatedValidation]:
        """Get validation history for a workflow."""
        return self.validation_history.get(workflow_id)

    async def get_quality_metrics(self) -> Dict[str, Any]:
        """Get quality gate performance metrics."""
        return {
            **self.quality_metrics,
            "validation_history_size": len(self.validation_history),
            "active_gates": len(self.quality_gates)
        }

    async def add_custom_gate(self, gate_type: str, gate_function: callable):
        """Add a custom quality gate."""
        custom_gate_type = QualityGateType(gate_type)
        self.quality_gates[custom_gate_type] = gate_function

        logger.info("Custom quality gate added", gate_type=gate_type)

    async def remove_gate(self, gate_type: QualityGateType):
        """Remove a quality gate."""
        if gate_type in self.quality_gates:
            del self.quality_gates[gate_type]
            logger.info("Quality gate removed", gate_type=gate_type.value)

    async def run_single_gate(self, gate_type: QualityGateType, content: Dict[str, Any]) -> QualityGateResult:
        """Run a single quality gate on provided content."""
        if gate_type not in self.quality_gates:
            raise ValueError(f"Quality gate {gate_type} not found")

        gate_function = self.quality_gates[gate_type]
        return await gate_function(content)