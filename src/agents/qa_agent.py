"""Quality Assurance Agent for LeanVibe Agent Hive 2.0."""

import asyncio
import json
import time
from typing import Any

import structlog

from ..core.config import settings
from ..core.task_queue import Task
from .base_agent import BaseAgent, HealthStatus, TaskResult

logger = structlog.get_logger()


class QAAgent(BaseAgent):
    """Quality Assurance Agent specializing in testing and code quality."""

    def __init__(self, name: str = "qa-agent"):
        super().__init__(
            name=name,
            agent_type="qa",
            role="Quality Assurance Engineer - Testing, code review, and quality validation",
        )

        # QA-specific capabilities
        self.test_frameworks = ["pytest", "unittest", "mypy", "ruff", "bandit"]
        self.code_quality_tools = ["mypy", "ruff", "bandit", "safety", "pylint"]
        self.coverage_tools = ["pytest-cov", "coverage"]

        # Performance tracking
        self.tests_run = 0
        self.tests_passed = 0
        self.tests_failed = 0
        self.code_issues_found = 0

        logger.info("QA Agent initialized", agent=self.name)

    async def run(self) -> None:
        """Main QA agent execution loop."""
        logger.info("QA Agent starting main execution loop", agent=self.name)

        while self.status == "active":
            try:
                # Check for QA tasks
                task = await self._get_next_qa_task()

                if task:
                    logger.info(
                        "Processing QA task",
                        agent=self.name,
                        task_id=task.id,
                        task_type=task.task_type,
                    )
                    await self.process_task(task)
                else:
                    # No tasks available, wait
                    await asyncio.sleep(5)

                # Periodic health monitoring
                await self._monitor_codebase_health()

            except Exception as e:
                logger.error("QA Agent execution error", agent=self.name, error=str(e))
                await asyncio.sleep(10)

    async def _get_next_qa_task(self) -> Task | None:
        """Get next QA-related task from queue."""
        from ..core.task_queue import task_queue

        # Look for QA-specific task types
        qa_task_types = [
            "test_code",
            "review_code",
            "check_quality",
            "run_tests",
            "security_scan",
            "performance_test",
            "validate_changes",
        ]

        return await task_queue.get_next_task(
            agent_capabilities=qa_task_types, agent_id=self.name
        )

    async def _process_task_implementation(self, task: Task) -> TaskResult:
        """QA-specific task processing."""
        start_time = time.time()

        try:
            # Route to specific QA method based on task type
            if task.task_type == "test_code":
                result = await self._test_code(task)
            elif task.task_type == "review_code":
                result = await self._review_code(task)
            elif task.task_type == "check_quality":
                result = await self._check_code_quality(task)
            elif task.task_type == "run_tests":
                result = await self._run_tests(task)
            elif task.task_type == "security_scan":
                result = await self._security_scan(task)
            elif task.task_type == "performance_test":
                result = await self._performance_test(task)
            elif task.task_type == "validate_changes":
                result = await self._validate_changes(task)
            else:
                # Fallback to base implementation
                result = await super()._process_task_implementation(task)

            execution_time = time.time() - start_time

            # Store QA context
            await self.store_context(
                content=f"QA Task: {task.task_type}\nResult: {'Success' if result.success else 'Failed'}\nDetails: {str(result.data)[:500]}",
                importance_score=0.8,
                category="qa_execution",
                metadata={
                    "task_type": task.task_type,
                    "execution_time": execution_time,
                    "success": result.success,
                },
            )

            return result

        except Exception as e:
            logger.error(
                "QA task processing failed",
                agent=self.name,
                task_id=task.id,
                error=str(e),
            )
            return TaskResult(success=False, error=str(e))

    async def _test_code(self, task: Task) -> TaskResult:
        """Generate and run tests for code."""
        code_path = task.metadata.get("code_path", "src/")
        test_type = task.metadata.get("test_type", "unit")

        # Get context about the code
        context_results = await self.retrieve_context(
            f"testing {code_path} {test_type}", limit=5
        )

        context_text = "\n".join([r.context.content for r in context_results])

        prompt = f"""
        As a QA Engineer, analyze the code at {code_path} and create comprehensive {test_type} tests.
        
        Task: {task.description}
        Code Path: {code_path}
        Test Type: {test_type}
        
        Context:
        {context_text}
        
        Please:
        1. Analyze the code structure and functionality
        2. Generate appropriate test cases
        3. Write tests using pytest framework
        4. Include edge cases and error conditions
        5. Ensure proper test organization and naming
        6. Run the tests and report results
        
        Focus on testing coverage, edge cases, and maintainability.
        """
        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            # Try to run the generated tests
            test_results = await self._run_pytest(code_path)

            self.tests_run += test_results.get("total", 0)
            self.tests_passed += test_results.get("passed", 0)
            self.tests_failed += test_results.get("failed", 0)

            return TaskResult(
                success=True,
                data={
                    "test_generation": result.output,
                    "test_results": test_results,
                    "coverage_report": test_results.get("coverage"),
                    "tool_used": result.tool_used,
                },
                metrics={
                    "tests_run": test_results.get("total", 0),
                    "tests_passed": test_results.get("passed", 0),
                    "tests_failed": test_results.get("failed", 0),
                    "coverage_percentage": test_results.get("coverage_percent", 0),
                },
            )
        else:
            return TaskResult(success=False, error=result.error)

    async def _review_code(self, task: Task) -> TaskResult:
        """Perform code review."""
        code_path = task.metadata.get("code_path", "src/")
        review_focus = task.metadata.get("focus", "all")

        # Get recent changes context
        context_results = await self.retrieve_context(
            f"code review {code_path}", limit=10
        )

        context_text = "\n".join([r.context.content for r in context_results])

        prompt = f"""
        As a Senior QA Engineer, perform a comprehensive code review of {code_path}.
        
        Task: {task.description}
        Code Path: {code_path}
        Review Focus: {review_focus}
        
        Previous Context:
        {context_text}
        
        Please review the code for:
        1. Code quality and maintainability
        2. Security vulnerabilities
        3. Performance implications
        4. Testing coverage
        5. Documentation quality
        6. Error handling
        7. Code consistency and standards
        8. Potential bugs or issues
        
        Provide specific recommendations and severity levels for each issue found.
        Format the output as a structured review report.
        """
        result = await self.execute_with_cli_tool(prompt)

        if result.success:
            # Run additional quality checks
            quality_results = await self._run_quality_checks(code_path)

            self.code_issues_found += len(quality_results.get("issues", []))

            return TaskResult(
                success=True,
                data={
                    "review_report": result.output,
                    "quality_checks": quality_results,
                    "tool_used": result.tool_used,
                    "recommendations": self._extract_recommendations(result.output),
                },
                metrics={
                    "issues_found": len(quality_results.get("issues", [])),
                    "security_issues": len(quality_results.get("security_issues", [])),
                    "quality_score": quality_results.get("quality_score", 0),
                },
            )
        else:
            return TaskResult(success=False, error=result.error)

    async def _check_code_quality(self, task: Task) -> TaskResult:
        """Run comprehensive code quality checks."""
        code_path = task.metadata.get("code_path", "src/")

        quality_results = {}

        # Run multiple quality tools
        for tool in self.code_quality_tools:
            try:
                tool_result = await self._run_quality_tool(tool, code_path)
                quality_results[tool] = tool_result
            except Exception as e:
                logger.warning(f"Quality tool {tool} failed", error=str(e))
                quality_results[tool] = {"error": str(e)}

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(quality_results)

        return TaskResult(
            success=True,
            data={
                "quality_results": quality_results,
                "quality_score": quality_score,
                "recommendations": self._generate_quality_recommendations(
                    quality_results
                ),
            },
            metrics={
                "quality_score": quality_score,
                "tools_run": len(quality_results),
                "issues_total": sum(
                    len(r.get("issues", []))
                    for r in quality_results.values()
                    if isinstance(r, dict)
                ),
            },
        )

    async def _run_tests(self, task: Task) -> TaskResult:
        """Run existing test suite."""
        test_path = task.metadata.get("test_path", "tests/")
        test_pattern = task.metadata.get("pattern", "test_*.py")

        test_results = await self._run_pytest(test_path, test_pattern)

        self.tests_run += test_results.get("total", 0)
        self.tests_passed += test_results.get("passed", 0)
        self.tests_failed += test_results.get("failed", 0)

        return TaskResult(
            success=test_results.get("failed", 0) == 0,
            data={
                "test_results": test_results,
                "coverage_report": test_results.get("coverage"),
                "test_summary": f"{test_results.get('passed', 0)}/{test_results.get('total', 0)} tests passed",
            },
            metrics={
                "tests_run": test_results.get("total", 0),
                "tests_passed": test_results.get("passed", 0),
                "tests_failed": test_results.get("failed", 0),
                "coverage_percentage": test_results.get("coverage_percent", 0),
            },
        )

    async def _security_scan(self, task: Task) -> TaskResult:
        """Run security vulnerability scan."""
        code_path = task.metadata.get("code_path", "src/")

        security_results = {}

        # Run bandit for security issues
        try:
            bandit_result = await self._run_bandit(code_path)
            security_results["bandit"] = bandit_result
        except Exception as e:
            security_results["bandit"] = {"error": str(e)}

        # Run safety for dependency vulnerabilities
        try:
            safety_result = await self._run_safety()
            security_results["safety"] = safety_result
        except Exception as e:
            security_results["safety"] = {"error": str(e)}

        # Count security issues
        total_issues = 0
        for tool_result in security_results.values():
            if isinstance(tool_result, dict) and "issues" in tool_result:
                total_issues += len(tool_result["issues"])

        self.code_issues_found += total_issues

        return TaskResult(
            success=total_issues == 0,
            data={
                "security_results": security_results,
                "total_vulnerabilities": total_issues,
                "recommendations": self._generate_security_recommendations(
                    security_results
                ),
            },
            metrics={
                "vulnerabilities_found": total_issues,
                "security_score": max(0, 100 - total_issues * 10),  # Simple scoring
            },
        )

    async def _performance_test(self, task: Task) -> TaskResult:
        """Run performance tests."""
        # This would integrate with performance testing tools
        # For now, provide a framework for performance testing

        prompt = f"""
        As a QA Engineer, design and execute performance tests for the system.
        
        Task: {task.description}
        Target: {task.metadata.get("target", "API endpoints")}
        
        Please:
        1. Identify performance test scenarios
        2. Set up performance test framework
        3. Execute load tests
        4. Analyze performance metrics
        5. Provide recommendations for optimization
        
        Focus on response times, throughput, and resource utilization.
        """
        result = await self.execute_with_cli_tool(prompt)

        return TaskResult(
            success=result.success,
            data={"performance_analysis": result.output, "tool_used": result.tool_used},
            metrics={"execution_time": result.execution_time},
        )

    async def _validate_changes(self, task: Task) -> TaskResult:
        """Validate code changes before deployment."""
        changes_path = task.metadata.get("changes_path", ".")

        validation_results = {}

        # Run comprehensive validation
        validation_results["tests"] = await self._run_pytest(changes_path)
        validation_results["quality"] = await self._run_quality_checks(changes_path)
        validation_results["security"] = await self._run_bandit(changes_path)

        # Check if all validations pass
        all_passed = (
            validation_results["tests"].get("failed", 0) == 0
            and len(validation_results["quality"].get("critical_issues", [])) == 0
            and len(validation_results["security"].get("high_severity", [])) == 0
        )

        return TaskResult(
            success=all_passed,
            data={
                "validation_results": validation_results,
                "deployment_ready": all_passed,
                "blockers": self._identify_deployment_blockers(validation_results),
            },
            metrics={
                "tests_passed": validation_results["tests"].get("passed", 0),
                "quality_issues": len(validation_results["quality"].get("issues", [])),
                "security_issues": len(
                    validation_results["security"].get("issues", [])
                ),
            },
        )

    async def _run_pytest(
        self, path: str, pattern: str = "test_*.py"
    ) -> dict[str, Any]:
        """Run pytest and return results."""
        try:
            cmd = [
                "python",
                "-m",
                "pytest",
                path,
                "--tb=short",
                "--quiet",
                "--cov=src",
                "--cov-report=term-missing",
                "--json-report",
                "--json-report-file=/tmp/pytest_report.json",
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=settings.project_root,
            )

            stdout, stderr = await process.communicate()

            # Parse JSON report if available
            try:
                with open("/tmp/pytest_report.json") as f:
                    report = json.load(f)

                return {
                    "total": report["summary"]["total"],
                    "passed": report["summary"]["passed"],
                    "failed": report["summary"]["failed"],
                    "skipped": report["summary"]["skipped"],
                    "coverage_percent": self._extract_coverage_from_output(
                        stdout.decode()
                    ),
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                }
            except (FileNotFoundError, json.JSONDecodeError):
                # Fallback to parsing stdout
                output = stdout.decode()
                return {
                    "total": output.count("PASSED") + output.count("FAILED"),
                    "passed": output.count("PASSED"),
                    "failed": output.count("FAILED"),
                    "output": output,
                    "errors": stderr.decode(),
                }

        except Exception as e:
            return {"error": str(e), "total": 0, "passed": 0, "failed": 1}

    async def _run_quality_tool(self, tool: str, path: str) -> dict[str, Any]:
        """Run a specific quality tool."""
        try:
            if tool == "mypy":
                cmd = ["python", "-m", "mypy", path, "--ignore-missing-imports"]
            elif tool == "ruff":
                cmd = ["python", "-m", "ruff", "check", path, "--output-format=json"]
            elif tool == "bandit":
                cmd = ["python", "-m", "bandit", "-r", path, "-f", "json"]
            else:
                return {"error": f"Unknown tool: {tool}"}

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=settings.project_root,
            )

            stdout, stderr = await process.communicate()

            if tool == "ruff" and process.returncode == 0:
                # Parse ruff JSON output
                try:
                    issues = json.loads(stdout.decode())
                    return {"issues": issues, "tool": tool}
                except json.JSONDecodeError:
                    return {"issues": [], "tool": tool}
            elif tool == "bandit":
                try:
                    report = json.loads(stdout.decode())
                    return {"issues": report.get("results", []), "tool": tool}
                except json.JSONDecodeError:
                    return {"issues": [], "tool": tool}
            else:
                return {
                    "output": stdout.decode(),
                    "errors": stderr.decode(),
                    "return_code": process.returncode,
                    "tool": tool,
                }

        except Exception as e:
            return {"error": str(e), "tool": tool}

    async def _run_quality_checks(self, path: str) -> dict[str, Any]:
        """Run comprehensive quality checks."""
        results = {}

        for tool in self.code_quality_tools:
            results[tool] = await self._run_quality_tool(tool, path)

        # Aggregate issues
        all_issues = []
        for tool_result in results.values():
            if "issues" in tool_result:
                all_issues.extend(tool_result["issues"])

        return {
            "tools": results,
            "issues": all_issues,
            "total_issues": len(all_issues),
            "quality_score": self._calculate_quality_score(results),
        }

    async def _run_bandit(self, path: str) -> dict[str, Any]:
        """Run bandit security scanner."""
        return await self._run_quality_tool("bandit", path)

    async def _run_safety(self) -> dict[str, Any]:
        """Run safety dependency scanner."""
        try:
            cmd = ["python", "-m", "safety", "check", "--json"]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=settings.project_root,
            )

            stdout, stderr = await process.communicate()

            try:
                vulnerabilities = json.loads(stdout.decode())
                return {"vulnerabilities": vulnerabilities, "tool": "safety"}
            except json.JSONDecodeError:
                return {"vulnerabilities": [], "tool": "safety"}

        except Exception as e:
            return {"error": str(e), "tool": "safety"}

    def _calculate_quality_score(self, results: dict[str, Any]) -> float:
        """Calculate overall quality score from tool results."""
        total_score = 100.0

        for tool, result in results.items():
            if isinstance(result, dict) and "issues" in result:
                # Deduct points for issues
                issues = result["issues"]
                total_score -= len(issues) * 2  # 2 points per issue

        return max(0.0, total_score)

    def _extract_coverage_from_output(self, output: str) -> float:
        """Extract coverage percentage from pytest output."""
        # Look for coverage percentage in output
        import re

        match = re.search(r"TOTAL.*?(\d+)%", output)
        if match:
            return float(match.group(1))
        return 0.0

    def _extract_recommendations(self, review_output: str) -> list[str]:
        """Extract actionable recommendations from review output."""
        # Simple extraction - in practice this would be more sophisticated
        recommendations = []
        lines = review_output.split("\n")

        for line in lines:
            if "recommend" in line.lower() or "should" in line.lower():
                recommendations.append(line.strip())

        return recommendations[:10]  # Limit to top 10 recommendations

    def _generate_quality_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate recommendations based on quality check results."""
        recommendations = []

        for tool, result in results.items():
            if isinstance(result, dict) and "issues" in result:
                issue_count = len(result["issues"])
                if issue_count > 0:
                    recommendations.append(
                        f"Fix {issue_count} {tool} issues for better code quality"
                    )

        return recommendations

    def _generate_security_recommendations(self, results: dict[str, Any]) -> list[str]:
        """Generate security recommendations."""
        recommendations = []

        for tool, result in results.items():
            if isinstance(result, dict) and "issues" in result:
                high_severity = [
                    i for i in result["issues"] if i.get("severity") == "HIGH"
                ]
                if high_severity:
                    recommendations.append(
                        f"Address {len(high_severity)} high-severity {tool} issues immediately"
                    )

        return recommendations

    def _identify_deployment_blockers(
        self, validation_results: dict[str, Any]
    ) -> list[str]:
        """Identify issues that block deployment."""
        blockers = []

        if validation_results["tests"].get("failed", 0) > 0:
            blockers.append("Failed test cases must be fixed")

        if validation_results["security"].get("high_severity"):
            blockers.append("High-severity security issues must be resolved")

        return blockers

    async def _monitor_codebase_health(self) -> None:
        """Periodic monitoring of codebase health."""
        # This could run quality checks on a schedule
        # For now, just update health metrics
        health = await self.health_check()

        if health == HealthStatus.DEGRADED:
            logger.warning("QA Agent health degraded", agent=self.name)
        elif health == HealthStatus.UNHEALTHY:
            logger.error("QA Agent unhealthy", agent=self.name)

    async def health_check(self) -> HealthStatus:
        """QA-specific health check."""
        base_health = await super().health_check()

        if base_health != HealthStatus.HEALTHY:
            return base_health

        # Check QA-specific health
        try:
            # Check if pytest is available
            result = await asyncio.create_subprocess_exec(
                "python",
                "-m",
                "pytest",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()

            if result.returncode != 0:
                return HealthStatus.DEGRADED

            # Check test failure rate
            if self.tests_run > 0:
                failure_rate = self.tests_failed / self.tests_run
                if failure_rate > 0.3:  # More than 30% failure rate
                    return HealthStatus.DEGRADED

            return HealthStatus.HEALTHY

        except Exception:
            return HealthStatus.DEGRADED
