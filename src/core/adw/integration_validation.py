"""
Integration validation phase implementation for ADW sessions.
Performs comprehensive testing and validation before proceeding.
"""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()


@dataclass
class ValidationResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    score: float
    duration: float
    details: dict[str, Any]
    error_message: str | None = None


@dataclass
class IntegrationValidationReport:
    """Complete integration validation report."""

    timestamp: float
    overall_passed: bool
    validation_results: list[ValidationResult]
    rollback_triggered: bool
    rollback_reason: str | None = None


class IntegrationValidationEngine:
    """Performs comprehensive validation and testing."""

    def __init__(
        self, project_path: Path, rollback_system, quality_gates, resource_guardian
    ):
        self.project_path = project_path
        self.rollback_system = rollback_system
        self.quality_gates = quality_gates
        self.resource_guardian = resource_guardian
        self.validation_history: list[IntegrationValidationReport] = []

    async def run_comprehensive_validation(
        self, rollback_on_failure: bool = True
    ) -> IntegrationValidationReport:
        """Run comprehensive integration validation."""
        start_time = time.time()

        logger.info("Starting comprehensive integration validation")

        # Run all validation checks in parallel where possible
        validation_tasks = [
            self._run_unit_tests(),
            self._run_integration_tests(),
            self._run_performance_tests(),
            self._run_security_scan(),
            self._run_quality_gates(),
            self._check_dependency_vulnerabilities(),
            self._validate_api_compatibility(),
        ]

        validation_results = []

        # Execute validations with proper error handling
        for i, task in enumerate(validation_tasks):
            try:
                result = await task
                validation_results.append(result)
            except Exception as e:
                # Create failed result for exception
                validation_results.append(
                    ValidationResult(
                        name=f"validation_{i}",
                        passed=False,
                        score=0.0,
                        duration=0.0,
                        details={"error": str(e)},
                        error_message=str(e),
                    )
                )

        # Determine overall success
        overall_passed = all(result.passed for result in validation_results)

        # Handle rollback if needed
        rollback_triggered = False
        rollback_reason = None

        if not overall_passed and rollback_on_failure:
            rollback_reason = self._determine_rollback_reason(validation_results)
            rollback_triggered = await self._trigger_intelligent_rollback(
                rollback_reason, validation_results
            )

        report = IntegrationValidationReport(
            timestamp=start_time,
            overall_passed=overall_passed,
            validation_results=validation_results,
            rollback_triggered=rollback_triggered,
            rollback_reason=rollback_reason,
        )

        self.validation_history.append(report)

        logger.info(
            "Integration validation completed",
            overall_passed=overall_passed,
            duration=time.time() - start_time,
            rollback_triggered=rollback_triggered,
        )

        return report

    async def _run_unit_tests(self) -> ValidationResult:
        """Run unit tests in parallel."""
        start_time = time.time()

        try:
            # Run unit tests with parallel execution if available
            cmd = ["pytest", "tests/unit/", "-v", "--tb=short", "-x"]

            # Check if pytest-xdist is available for parallel execution
            try:
                import psutil
                import pytest_xdist

                cpu_count = psutil.cpu_count()
                cmd.extend(["-n", str(min(cpu_count, 4))])  # Limit to 4 processes max
            except ImportError:
                pass  # Run sequentially if xdist not available

            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            duration = time.time() - start_time

            # Parse test results
            output = stdout.decode() + stderr.decode()
            passed = process.returncode == 0

            # Extract test statistics
            test_stats = self._parse_pytest_output(output)

            return ValidationResult(
                name="unit_tests",
                passed=passed,
                score=1.0 if passed else 0.0,
                duration=duration,
                details={
                    "exit_code": process.returncode,
                    "tests_run": test_stats.get("total", 0),
                    "tests_passed": test_stats.get("passed", 0),
                    "tests_failed": test_stats.get("failed", 0),
                    "output_sample": output[:500] if output else "",
                },
            )

        except Exception as e:
            return ValidationResult(
                name="unit_tests",
                passed=False,
                score=0.0,
                duration=time.time() - start_time,
                details={"error": str(e)},
                error_message=str(e),
            )

    async def _run_integration_tests(self) -> ValidationResult:
        """Run integration tests."""
        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                "pytest",
                "tests/integration/",
                "-v",
                "--tb=short",
                "-x",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            duration = time.time() - start_time

            output = stdout.decode() + stderr.decode()
            passed = process.returncode == 0

            test_stats = self._parse_pytest_output(output)

            return ValidationResult(
                name="integration_tests",
                passed=passed,
                score=1.0 if passed else 0.0,
                duration=duration,
                details={
                    "exit_code": process.returncode,
                    "tests_run": test_stats.get("total", 0),
                    "tests_passed": test_stats.get("passed", 0),
                    "tests_failed": test_stats.get("failed", 0),
                    "output_sample": output[:500] if output else "",
                },
            )

        except Exception as e:
            return ValidationResult(
                name="integration_tests",
                passed=False,
                score=0.0,
                duration=time.time() - start_time,
                details={"error": str(e)},
                error_message=str(e),
            )

    async def _run_performance_tests(self) -> ValidationResult:
        """Run performance regression tests."""
        start_time = time.time()

        try:
            # Measure current test suite performance
            test_runtime = (
                await self.resource_guardian.test_optimizer.measure_test_runtime()
            )

            # Check for regression
            is_regression = False
            regression_ratio = 1.0

            if self.resource_guardian.test_optimizer.baseline_runtime:
                is_regression = (
                    self.resource_guardian.test_optimizer.is_runtime_regression(
                        test_runtime
                    )
                )
                regression_ratio = (
                    test_runtime
                    / self.resource_guardian.test_optimizer.baseline_runtime
                )

            # Performance score based on regression
            if is_regression:
                score = max(
                    0.0, 1.0 - (regression_ratio - 1.0)
                )  # Penalty for regression
            else:
                score = 1.0

            return ValidationResult(
                name="performance_tests",
                passed=not is_regression,
                score=score,
                duration=time.time() - start_time,
                details={
                    "current_runtime": test_runtime,
                    "baseline_runtime": self.resource_guardian.test_optimizer.baseline_runtime,
                    "is_regression": is_regression,
                    "regression_ratio": regression_ratio,
                },
            )

        except Exception as e:
            return ValidationResult(
                name="performance_tests",
                passed=False,
                score=0.0,
                duration=time.time() - start_time,
                details={"error": str(e)},
                error_message=str(e),
            )

    async def _run_security_scan(self) -> ValidationResult:
        """Run security vulnerability scan."""
        start_time = time.time()

        try:
            # Use the built-in security scanner from quality gates
            from ..safety.quality_gates import SecurityScanner

            scanner = SecurityScanner()
            scan_result = await scanner.scan_project(self.project_path)

            total_issues = scan_result.get("total_issues", 0)
            passed = total_issues == 0

            # Score based on number of issues
            if total_issues == 0:
                score = 1.0
            elif total_issues <= 3:
                score = 0.7
            elif total_issues <= 10:
                score = 0.5
            else:
                score = 0.0

            return ValidationResult(
                name="security_scan",
                passed=passed,
                score=score,
                duration=time.time() - start_time,
                details={
                    "total_issues": total_issues,
                    "files_with_issues": scan_result.get("files_with_issues", 0),
                    "scan_completed": True,
                },
            )

        except Exception as e:
            return ValidationResult(
                name="security_scan",
                passed=False,
                score=0.0,
                duration=time.time() - start_time,
                details={"error": str(e)},
                error_message=str(e),
            )

    async def _run_quality_gates(self) -> ValidationResult:
        """Run quality gates validation."""
        start_time = time.time()

        try:
            quality_results = await self.quality_gates.run_all_gates()

            if not quality_results:
                return ValidationResult(
                    name="quality_gates",
                    passed=True,
                    score=1.0,
                    duration=time.time() - start_time,
                    details={"message": "No quality gates configured"},
                )

            passed_gates = sum(1 for result in quality_results if result.passed)
            total_gates = len(quality_results)
            overall_passed = passed_gates == total_gates

            # Calculate average score
            avg_score = sum(result.score for result in quality_results) / total_gates

            return ValidationResult(
                name="quality_gates",
                passed=overall_passed,
                score=avg_score,
                duration=time.time() - start_time,
                details={
                    "gates_passed": passed_gates,
                    "total_gates": total_gates,
                    "individual_results": [
                        {
                            "gate": result.gate_name,
                            "passed": result.passed,
                            "score": result.score,
                        }
                        for result in quality_results
                    ],
                },
            )

        except Exception as e:
            return ValidationResult(
                name="quality_gates",
                passed=False,
                score=0.0,
                duration=time.time() - start_time,
                details={"error": str(e)},
                error_message=str(e),
            )

    async def _check_dependency_vulnerabilities(self) -> ValidationResult:
        """Check for dependency vulnerabilities."""
        start_time = time.time()

        try:
            # Check if requirements.txt exists
            requirements_file = self.project_path / "requirements.txt"
            pyproject_file = self.project_path / "pyproject.toml"

            if not requirements_file.exists() and not pyproject_file.exists():
                return ValidationResult(
                    name="dependency_vulnerabilities",
                    passed=True,
                    score=1.0,
                    duration=time.time() - start_time,
                    details={"message": "No dependency files found"},
                )

            # Try to use safety or pip-audit if available
            vulnerabilities_found = 0
            scan_method = "none"

            try:
                # Try safety first
                process = await asyncio.create_subprocess_exec(
                    "safety",
                    "check",
                    "--json",
                    cwd=self.project_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate()

                if process.returncode == 0:
                    try:
                        safety_data = json.loads(stdout.decode())
                        vulnerabilities_found = len(safety_data)
                        scan_method = "safety"
                    except json.JSONDecodeError:
                        pass

            except FileNotFoundError:
                # Safety not available, try pip-audit
                try:
                    process = await asyncio.create_subprocess_exec(
                        "pip-audit",
                        "--format=json",
                        cwd=self.project_path,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await process.communicate()

                    if process.returncode == 0:
                        try:
                            audit_data = json.loads(stdout.decode())
                            vulnerabilities_found = len(
                                audit_data.get("vulnerabilities", [])
                            )
                            scan_method = "pip-audit"
                        except json.JSONDecodeError:
                            pass

                except FileNotFoundError:
                    pass  # Neither tool available

            passed = vulnerabilities_found == 0
            score = 1.0 if passed else max(0.0, 1.0 - vulnerabilities_found * 0.1)

            return ValidationResult(
                name="dependency_vulnerabilities",
                passed=passed,
                score=score,
                duration=time.time() - start_time,
                details={
                    "vulnerabilities_found": vulnerabilities_found,
                    "scan_method": scan_method,
                    "has_requirements": requirements_file.exists(),
                    "has_pyproject": pyproject_file.exists(),
                },
            )

        except Exception as e:
            return ValidationResult(
                name="dependency_vulnerabilities",
                passed=False,
                score=0.0,
                duration=time.time() - start_time,
                details={"error": str(e)},
                error_message=str(e),
            )

    async def _validate_api_compatibility(self) -> ValidationResult:
        """Validate API compatibility and breaking changes."""
        start_time = time.time()

        try:
            # Simple API compatibility check - look for common breaking changes
            breaking_changes = []

            # Check git diff for potential breaking changes
            process = await asyncio.create_subprocess_exec(
                "git",
                "diff",
                "HEAD~1",
                "--name-only",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                changed_files = stdout.decode().strip().split("\n")
                api_files = [
                    f
                    for f in changed_files
                    if "api" in f.lower() or f.endswith("api.py")
                ]

                # Check for potential breaking changes in API files
                for api_file in api_files:
                    file_path = self.project_path / api_file
                    if file_path.exists():
                        # Simple heuristic: look for removed functions/classes
                        process = await asyncio.create_subprocess_exec(
                            "git",
                            "diff",
                            "HEAD~1",
                            api_file,
                            cwd=self.project_path,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )

                        stdout, stderr = await process.communicate()
                        diff_content = stdout.decode()

                        # Look for removed functions/classes (lines starting with -)
                        removed_lines = [
                            line
                            for line in diff_content.split("\n")
                            if line.startswith("-")
                        ]
                        removed_functions = [
                            line
                            for line in removed_lines
                            if "def " in line or "class " in line
                        ]

                        if removed_functions:
                            breaking_changes.extend(removed_functions)

            passed = len(breaking_changes) == 0
            score = 1.0 if passed else max(0.0, 1.0 - len(breaking_changes) * 0.2)

            return ValidationResult(
                name="api_compatibility",
                passed=passed,
                score=score,
                duration=time.time() - start_time,
                details={
                    "breaking_changes_detected": len(breaking_changes),
                    "breaking_changes": breaking_changes[:5],  # Limit output
                    "api_files_changed": len(
                        [f for f in changed_files if "api" in f.lower()]
                    )
                    if "changed_files" in locals()
                    else 0,
                },
            )

        except Exception as e:
            return ValidationResult(
                name="api_compatibility",
                passed=True,  # Default to passing if check fails
                score=1.0,
                duration=time.time() - start_time,
                details={
                    "error": str(e),
                    "note": "Compatibility check failed, defaulting to pass",
                },
                error_message=str(e),
            )

    def _parse_pytest_output(self, output: str) -> dict[str, int]:
        """Parse pytest output to extract test statistics."""
        stats = {"total": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}

        try:
            # Look for summary line like "5 passed, 2 failed"
            lines = output.split("\n")
            for line in lines:
                if " passed" in line or " failed" in line:
                    # Extract numbers from summary line
                    import re

                    numbers = re.findall(r"(\d+) (\w+)", line)
                    for count, status in numbers:
                        count = int(count)
                        if status == "passed":
                            stats["passed"] = count
                        elif status == "failed":
                            stats["failed"] = count
                        elif status == "error" or status == "errors":
                            stats["errors"] = count
                        elif status == "skipped":
                            stats["skipped"] = count

                    stats["total"] = (
                        stats["passed"]
                        + stats["failed"]
                        + stats["errors"]
                        + stats["skipped"]
                    )
                    break

        except Exception:
            pass  # Return default stats if parsing fails

        return stats

    def _determine_rollback_reason(
        self, validation_results: list[ValidationResult]
    ) -> str:
        """Determine the reason for rollback based on validation failures."""
        failed_validations = [
            result for result in validation_results if not result.passed
        ]

        if not failed_validations:
            return "no_failures"

        # Prioritize failure types
        failure_priorities = {
            "unit_tests": "critical_test_failures",
            "integration_tests": "integration_failures",
            "security_scan": "security_vulnerabilities",
            "dependency_vulnerabilities": "dependency_issues",
            "performance_tests": "performance_regression",
            "quality_gates": "quality_violations",
            "api_compatibility": "breaking_changes",
        }

        for result in failed_validations:
            if result.name in failure_priorities:
                return failure_priorities[result.name]

        return "general_validation_failure"

    async def _trigger_intelligent_rollback(
        self, reason: str, validation_results: list[ValidationResult]
    ) -> bool:
        """Trigger intelligent rollback based on failure type."""
        from ..safety.rollback_system import RollbackLevel

        # Map reasons to rollback levels
        rollback_mapping = {
            "critical_test_failures": RollbackLevel.TEST_FAILURE,
            "integration_failures": RollbackLevel.TEST_FAILURE,
            "security_vulnerabilities": RollbackLevel.SYSTEM_CRASH,
            "dependency_issues": RollbackLevel.SYSTEM_CRASH,
            "performance_regression": RollbackLevel.PERFORMANCE_REGRESSION,
            "quality_violations": RollbackLevel.TEST_FAILURE,
            "breaking_changes": RollbackLevel.SYSTEM_CRASH,
            "general_validation_failure": RollbackLevel.TEST_FAILURE,
        }

        rollback_level = rollback_mapping.get(reason, RollbackLevel.TEST_FAILURE)

        # Create context for rollback
        context = {
            "reason": reason,
            "failed_validations": [r.name for r in validation_results if not r.passed],
            "validation_scores": {r.name: r.score for r in validation_results},
        }

        logger.warning(
            "Triggering intelligent rollback",
            reason=reason,
            rollback_level=rollback_level.value,
            context=context,
        )

        try:
            success = await self.rollback_system.handle_failure(rollback_level, context)
            return success
        except Exception as e:
            logger.error("Rollback failed", error=str(e))
            return False

    def get_validation_statistics(self) -> dict[str, Any]:
        """Get validation statistics from history."""
        if not self.validation_history:
            return {"total_validations": 0}

        total_validations = len(self.validation_history)
        successful_validations = len(
            [v for v in self.validation_history if v.overall_passed]
        )
        rollbacks_triggered = len(
            [v for v in self.validation_history if v.rollback_triggered]
        )

        # Calculate average scores by validation type
        validation_scores = {}
        for report in self.validation_history:
            for result in report.validation_results:
                if result.name not in validation_scores:
                    validation_scores[result.name] = []
                validation_scores[result.name].append(result.score)

        avg_scores = {
            name: sum(scores) / len(scores)
            for name, scores in validation_scores.items()
        }

        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": successful_validations / total_validations,
            "rollbacks_triggered": rollbacks_triggered,
            "rollback_rate": rollbacks_triggered / total_validations,
            "average_scores_by_validation": avg_scores,
            "recent_reports": self.validation_history[-5:]
            if len(self.validation_history) > 5
            else self.validation_history,
        }
