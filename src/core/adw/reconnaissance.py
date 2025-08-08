"""
Reconnaissance phase implementation for ADW sessions.
Performs comprehensive system assessment before development begins.
"""

import asyncio
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class ReconnaissanceReport:
    """Report generated during reconnaissance phase."""

    timestamp: float
    system_health: Dict[str, Any]
    test_coverage: Dict[str, Any]
    performance_baseline: Dict[str, Any]
    error_patterns: List[Dict[str, Any]]
    repository_status: Dict[str, Any]
    recommendations: List[str]


class ReconnaissanceEngine:
    """Performs comprehensive system assessment for ADW sessions."""

    def __init__(self, project_path: Path):
        self.project_path = project_path

    async def run_comprehensive_assessment(self) -> ReconnaissanceReport:
        """Run comprehensive system assessment."""
        logger.info("Starting comprehensive reconnaissance assessment")

        start_time = time.time()

        # Run all assessments in parallel for efficiency
        tasks = [
            self._assess_system_health(),
            self._analyze_test_coverage(),
            self._measure_performance_baseline(),
            self._analyze_error_patterns(),
            self._check_repository_status(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        system_health = (
            results[0]
            if not isinstance(results[0], Exception)
            else {"error": str(results[0])}
        )
        test_coverage = (
            results[1]
            if not isinstance(results[1], Exception)
            else {"error": str(results[1])}
        )
        performance_baseline = (
            results[2]
            if not isinstance(results[2], Exception)
            else {"error": str(results[2])}
        )
        error_patterns = results[3] if not isinstance(results[3], Exception) else []
        repository_status = (
            results[4]
            if not isinstance(results[4], Exception)
            else {"error": str(results[4])}
        )

        # Generate recommendations based on findings
        recommendations = self._generate_recommendations(
            system_health,
            test_coverage,
            performance_baseline,
            error_patterns,
            repository_status,
        )

        report = ReconnaissanceReport(
            timestamp=start_time,
            system_health=system_health,
            test_coverage=test_coverage,
            performance_baseline=performance_baseline,
            error_patterns=error_patterns,
            repository_status=repository_status,
            recommendations=recommendations,
        )

        logger.info(
            "Reconnaissance assessment completed",
            duration=time.time() - start_time,
            recommendations_count=len(recommendations),
        )

        return report

    async def _assess_system_health(self) -> Dict[str, Any]:
        """Assess overall system health."""
        import psutil
        import shutil

        health_data = {}

        # Resource usage
        memory = psutil.virtual_memory()
        disk = shutil.disk_usage(self.project_path)

        health_data["resources"] = {
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "disk_free_gb": disk.free / (1024**3),
            "disk_total_gb": disk.total / (1024**3),
            "cpu_count": psutil.cpu_count(),
        }

        # Process health
        health_data["processes"] = {
            "total_processes": len(psutil.pids()),
            "python_processes": len(
                [p for p in psutil.process_iter() if "python" in p.name().lower()]
            ),
        }

        # Development environment check
        health_data["environment"] = await self._check_development_environment()

        return health_data

    async def _check_development_environment(self) -> Dict[str, Any]:
        """Check development environment setup."""
        env_status = {}

        # Check Python version
        try:
            process = await asyncio.create_subprocess_exec(
                "python",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                env_status["python_version"] = stdout.decode().strip()
            else:
                env_status["python_version"] = {"error": stderr.decode().strip()}
        except Exception as e:
            env_status["python_version"] = {"error": str(e)}

        # Check git status
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "--version",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                env_status["git_version"] = stdout.decode().strip()
            else:
                env_status["git_version"] = {"error": stderr.decode().strip()}
        except Exception as e:
            env_status["git_version"] = {"error": str(e)}

        # Check pytest availability
        try:
            process = await asyncio.create_subprocess_exec(
                "pytest",
                "--version",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                env_status["pytest_version"] = stdout.decode().strip()
            else:
                env_status["pytest_version"] = {"error": stderr.decode().strip()}
        except Exception as e:
            env_status["pytest_version"] = {"error": str(e)}

        return env_status

    async def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze current test coverage."""
        coverage_data = {}

        try:
            # Run pytest with coverage
            process = await asyncio.create_subprocess_exec(
                "pytest",
                "--cov=src",
                "--cov-report=json",
                "--cov-report=term-missing",
                "--tb=no",
                "-q",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            # Try to read coverage report
            coverage_file = self.project_path / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, "r") as f:
                    coverage_json = json.load(f)

                coverage_data["overall_percentage"] = coverage_json.get(
                    "totals", {}
                ).get("percent_covered", 0)
                coverage_data["lines_covered"] = coverage_json.get("totals", {}).get(
                    "covered_lines", 0
                )
                coverage_data["lines_total"] = coverage_json.get("totals", {}).get(
                    "num_statements", 0
                )
                coverage_data["missing_lines"] = coverage_json.get("totals", {}).get(
                    "missing_lines", 0
                )

                # Identify files with low coverage
                files_coverage = coverage_json.get("files", {})
                low_coverage_files = [
                    {
                        "file": file,
                        "coverage": data.get("summary", {}).get("percent_covered", 0),
                    }
                    for file, data in files_coverage.items()
                    if data.get("summary", {}).get("percent_covered", 0) < 80
                ]
                coverage_data["low_coverage_files"] = low_coverage_files

            else:
                coverage_data["error"] = "Coverage report not generated"

        except Exception as e:
            coverage_data["error"] = str(e)

        return coverage_data

    async def _measure_performance_baseline(self) -> Dict[str, Any]:
        """Measure performance baseline."""
        performance_data = {}

        try:
            # Measure test suite runtime
            test_start = time.time()
            process = await asyncio.create_subprocess_exec(
                "pytest",
                "tests/",
                "--tb=no",
                "-q",
                "--maxfail=1",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await process.communicate()
            test_duration = time.time() - test_start

            performance_data["test_suite_runtime"] = test_duration
            performance_data["test_suite_status"] = (
                "passed" if process.returncode == 0 else "failed"
            )

            # Measure import time for main modules
            import_times = await self._measure_import_times()
            performance_data["import_times"] = import_times

            # Memory usage during tests
            import psutil

            performance_data["memory_during_tests"] = psutil.virtual_memory().percent

        except Exception as e:
            performance_data["error"] = str(e)

        return performance_data

    async def _measure_import_times(self) -> Dict[str, float]:
        """Measure import times for key modules."""
        import_times = {}

        key_modules = [
            "src.core.orchestrator",
            "src.core.safety",
            "src.core.adw",
            "src.agents",
        ]

        for module in key_modules:
            try:
                start_time = time.time()
                process = await asyncio.create_subprocess_exec(
                    "python",
                    "-c",
                    f"import {module}",
                    cwd=self.project_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                await process.communicate()
                import_time = time.time() - start_time

                if process.returncode == 0:
                    import_times[module] = import_time
                else:
                    import_times[module] = {"error": "import_failed"}

            except Exception as e:
                import_times[module] = {"error": str(e)}

        return import_times

    async def _analyze_error_patterns(self) -> List[Dict[str, Any]]:
        """Analyze error patterns from logs and git history."""
        error_patterns = []

        try:
            # Analyze recent git commits for error-related patterns
            process = await asyncio.create_subprocess_exec(
                "git",
                "log",
                "--oneline",
                "-20",
                "--grep=fix",
                "--grep=error",
                "--grep=bug",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                commits = stdout.decode().strip().split("\n")
                for commit in commits:
                    if commit.strip():
                        parts = commit.split(" ", 1)
                        if len(parts) == 2:
                            error_patterns.append(
                                {
                                    "type": "git_commit",
                                    "hash": parts[0],
                                    "message": parts[1],
                                    "category": self._categorize_error(parts[1]),
                                }
                            )

            # Look for common error patterns in test failures
            try:
                process = await asyncio.create_subprocess_exec(
                    "pytest",
                    "tests/",
                    "--tb=line",
                    "-v",
                    cwd=self.project_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    # Parse test failures
                    error_lines = [
                        line
                        for line in stdout.decode().split("\n")
                        if "FAILED" in line or "ERROR" in line
                    ]

                    for error_line in error_lines[:5]:  # Limit to first 5 errors
                        error_patterns.append(
                            {
                                "type": "test_failure",
                                "description": error_line.strip(),
                                "category": self._categorize_error(error_line),
                            }
                        )

            except Exception:
                pass  # Test failure analysis is optional

        except Exception as e:
            error_patterns.append(
                {
                    "type": "analysis_error",
                    "description": str(e),
                    "category": "system",
                }
            )

        return error_patterns

    def _categorize_error(self, message: str) -> str:
        """Categorize error based on message content."""
        message_lower = message.lower()

        if any(keyword in message_lower for keyword in ["syntax", "import", "name"]):
            return "syntax"
        elif any(keyword in message_lower for keyword in ["test", "assert", "expect"]):
            return "test"
        elif any(
            keyword in message_lower for keyword in ["performance", "slow", "timeout"]
        ):
            return "performance"
        elif any(
            keyword in message_lower for keyword in ["security", "auth", "permission"]
        ):
            return "security"
        else:
            return "other"

    async def _check_repository_status(self) -> Dict[str, Any]:
        """Check repository status and health."""
        repo_status = {}

        try:
            # Check git status
            process = await asyncio.create_subprocess_exec(
                "git",
                "status",
                "--porcelain",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                status_lines = stdout.decode().strip().split("\n")
                repo_status["uncommitted_changes"] = len(
                    [l for l in status_lines if l.strip()]
                )
                repo_status["is_clean"] = not any(l.strip() for l in status_lines)

                # Categorize changes
                changes = {"modified": 0, "added": 0, "deleted": 0, "untracked": 0}
                for line in status_lines:
                    if line.strip():
                        status_code = line[:2]
                        if "M" in status_code:
                            changes["modified"] += 1
                        elif "A" in status_code:
                            changes["added"] += 1
                        elif "D" in status_code:
                            changes["deleted"] += 1
                        elif "??" in status_code:
                            changes["untracked"] += 1

                repo_status["changes_breakdown"] = changes

            # Check branch information
            process = await asyncio.create_subprocess_exec(
                "git",
                "branch",
                "--show-current",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                repo_status["current_branch"] = stdout.decode().strip()

            # Check for recent commits
            process = await asyncio.create_subprocess_exec(
                "git",
                "log",
                "--oneline",
                "-5",
                cwd=self.project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                commits = stdout.decode().strip().split("\n")
                repo_status["recent_commits"] = [
                    c.strip() for c in commits if c.strip()
                ]

        except Exception as e:
            repo_status["error"] = str(e)

        return repo_status

    def _generate_recommendations(
        self,
        system_health: Dict[str, Any],
        test_coverage: Dict[str, Any],
        performance_baseline: Dict[str, Any],
        error_patterns: List[Dict[str, Any]],
        repository_status: Dict[str, Any],
    ) -> List[str]:
        """Generate actionable recommendations based on assessment."""
        recommendations = []

        # Memory recommendations
        memory_percent = system_health.get("resources", {}).get("memory_percent", 0)
        if memory_percent > 80:
            recommendations.append(
                "High memory usage detected - consider memory optimization"
            )

        # Test coverage recommendations
        coverage_percent = test_coverage.get("overall_percentage", 0)
        if coverage_percent < 80:
            recommendations.append(
                f"Test coverage is {coverage_percent:.1f}% - aim for >80%"
            )

        low_coverage_files = test_coverage.get("low_coverage_files", [])
        if low_coverage_files:
            recommendations.append(
                f"Focus testing on {len(low_coverage_files)} files with low coverage"
            )

        # Performance recommendations
        test_runtime = performance_baseline.get("test_suite_runtime", 0)
        if test_runtime > 60:  # More than 1 minute
            recommendations.append("Test suite runtime is slow - consider optimization")

        # Error pattern recommendations
        syntax_errors = [p for p in error_patterns if p.get("category") == "syntax"]
        if len(syntax_errors) > 2:
            recommendations.append(
                "Multiple syntax errors detected - review code quality practices"
            )

        test_errors = [p for p in error_patterns if p.get("category") == "test"]
        if len(test_errors) > 3:
            recommendations.append(
                "High number of test failures - review test reliability"
            )

        # Repository recommendations
        uncommitted_changes = repository_status.get("uncommitted_changes", 0)
        if uncommitted_changes > 5:
            recommendations.append(
                "Many uncommitted changes - consider committing work in progress"
            )

        # Add default recommendations if none generated
        if not recommendations:
            recommendations.append(
                "System health looks good - proceed with development"
            )

        return recommendations
