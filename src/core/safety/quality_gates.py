"""Autonomous quality gates for code validation during extended development sessions.

This module provides automated code quality checks that prevent bad code from
entering the system during autonomous development workflows.
"""

import ast
import asyncio
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

logger = structlog.get_logger()


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""

    gate_name: str
    passed: bool
    score: float
    threshold: float
    details: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None


@dataclass
class QualityGateConfig:
    """Configuration for quality gates."""

    max_complexity: int = 10
    min_test_coverage: float = 0.80
    max_duplicate_lines: int = 10
    max_function_length: int = 50
    max_class_length: int = 200
    security_scan_enabled: bool = True
    performance_check_enabled: bool = True
    documentation_check_enabled: bool = True


class CodeComplexityAnalyzer:
    """Analyzes code complexity metrics."""

    def __init__(self):
        self.complexity_cache: Dict[str, int] = {}

    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze complexity metrics for a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            complexity_visitor = ComplexityVisitor()
            complexity_visitor.visit(tree)

            return {
                "cyclomatic_complexity": complexity_visitor.complexity,
                "function_count": len(complexity_visitor.functions),
                "class_count": len(complexity_visitor.classes),
                "max_function_complexity": max(
                    complexity_visitor.function_complexities.values()
                )
                if complexity_visitor.function_complexities
                else 0,
                "avg_function_complexity": sum(
                    complexity_visitor.function_complexities.values()
                )
                / len(complexity_visitor.function_complexities)
                if complexity_visitor.function_complexities
                else 0,
                "function_complexities": complexity_visitor.function_complexities,
                "long_functions": [
                    name
                    for name, lines in complexity_visitor.function_lengths.items()
                    if lines > 50
                ],
                "total_lines": content.count("\n") + 1,
            }

        except Exception as e:
            logger.error(
                "Failed to analyze file complexity", file=str(file_path), error=str(e)
            )
            return {"error": str(e)}

    def analyze_project(self, project_path: Path) -> Dict[str, Any]:
        """Analyze complexity metrics for entire project."""
        python_files = list(project_path.rglob("*.py"))

        total_complexity = 0
        total_functions = 0
        total_classes = 0
        max_complexity = 0
        problematic_files = []

        for file_path in python_files:
            if any(
                exclude in str(file_path)
                for exclude in [".venv", "__pycache__", ".git"]
            ):
                continue

            analysis = self.analyze_file(file_path)
            if "error" in analysis:
                continue

            file_complexity = analysis.get("cyclomatic_complexity", 0)
            total_complexity += file_complexity
            total_functions += analysis.get("function_count", 0)
            total_classes += analysis.get("class_count", 0)

            max_func_complexity = analysis.get("max_function_complexity", 0)
            if max_func_complexity > max_complexity:
                max_complexity = max_func_complexity

            # Flag files with high complexity
            if file_complexity > 100 or max_func_complexity > 20:
                problematic_files.append(
                    {
                        "file": str(file_path.relative_to(project_path)),
                        "total_complexity": file_complexity,
                        "max_function_complexity": max_func_complexity,
                        "long_functions": analysis.get("long_functions", []),
                    }
                )

        return {
            "total_files_analyzed": len(python_files),
            "total_complexity": total_complexity,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "max_function_complexity": max_complexity,
            "avg_complexity_per_file": total_complexity / len(python_files)
            if python_files
            else 0,
            "problematic_files": problematic_files,
        }


class ComplexityVisitor(ast.NodeVisitor):
    """AST visitor for calculating cyclomatic complexity."""

    def __init__(self):
        self.complexity = 1  # Start with 1 for the main path
        self.functions = []
        self.classes = []
        self.function_complexities = {}
        self.function_lengths = {}
        self.current_function = None

    def visit_FunctionDef(self, node):
        """Visit function definition."""
        self.functions.append(node.name)

        # Calculate function complexity
        old_complexity = self.complexity
        self.complexity = 1  # Reset for function
        self.current_function = node.name

        self.generic_visit(node)

        # Store function complexity and length
        func_complexity = self.complexity
        func_length = (
            node.end_lineno - node.lineno if hasattr(node, "end_lineno") else 0
        )

        self.function_complexities[node.name] = func_complexity
        self.function_lengths[node.name] = func_length

        # Add function complexity to total
        self.complexity = old_complexity + func_complexity
        self.current_function = None

    def visit_AsyncFunctionDef(self, node):
        """Visit async function definition."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node):
        """Visit class definition."""
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_If(self, node):
        """Visit if statement."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        """Visit for loop."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_AsyncFor(self, node):
        """Visit async for loop."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        """Visit while loop."""
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        """Visit exception handler."""
        self.complexity += 1
        self.generic_visit(node)


class SecurityScanner:
    """Basic security vulnerability scanner."""

    def __init__(self):
        self.security_patterns = [
            # SQL injection patterns
            r"execute\s*\(\s*['\"].*%.*['\"]",
            r"cursor\.execute\s*\(\s*['\"].*\+.*['\"]",
            # Command injection patterns
            r"os\.system\s*\(\s*.*\+",
            r"subprocess\.(call|run|Popen)\s*\(\s*.*\+",
            # Hardcoded secrets (simplified)
            r"password\s*=\s*['\"][^'\"]{3,}['\"]",
            r"api_key\s*=\s*['\"][^'\"]{10,}['\"]",
            r"secret\s*=\s*['\"][^'\"]{5,}['\"]",
            # Dangerous functions
            r"eval\s*\(",
            r"exec\s*\(",
            r"pickle\.loads\s*\(",
        ]

    async def scan_file(self, file_path: Path) -> Dict[str, Any]:
        """Scan a single file for security issues."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            issues = []

            for i, line in enumerate(content.split("\n"), 1):
                for pattern in self.security_patterns:
                    import re

                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(
                            {"line": i, "pattern": pattern, "content": line.strip()}
                        )

            return {
                "file": str(file_path),
                "issues_found": len(issues),
                "issues": issues,
            }

        except Exception as e:
            logger.error(
                "Failed to scan file for security issues",
                file=str(file_path),
                error=str(e),
            )
            return {"error": str(e)}

    async def scan_project(self, project_path: Path) -> Dict[str, Any]:
        """Scan entire project for security issues."""
        python_files = list(project_path.rglob("*.py"))

        total_issues = 0
        files_with_issues = []

        for file_path in python_files:
            if any(
                exclude in str(file_path)
                for exclude in [".venv", "__pycache__", ".git"]
            ):
                continue

            scan_result = await self.scan_file(file_path)
            if "error" in scan_result:
                continue

            issues_count = scan_result.get("issues_found", 0)
            total_issues += issues_count

            if issues_count > 0:
                files_with_issues.append(scan_result)

        return {
            "total_files_scanned": len(python_files),
            "total_issues": total_issues,
            "files_with_issues": len(files_with_issues),
            "details": files_with_issues,
        }


class PerformanceAnalyzer:
    """Analyzes performance impact of code changes."""

    async def analyze_import_complexity(self, file_path: Path) -> Dict[str, Any]:
        """Analyze import complexity and potential performance issues."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")

            # Check for potentially expensive imports
            expensive_imports = [
                imp
                for imp in imports
                if any(
                    pattern in imp
                    for pattern in [
                        "pandas",
                        "numpy",
                        "tensorflow",
                        "torch",
                        "matplotlib",
                    ]
                )
            ]

            return {
                "total_imports": len(imports),
                "expensive_imports": expensive_imports,
                "import_score": len(expensive_imports) / len(imports) if imports else 0,
            }

        except Exception as e:
            logger.error(
                "Failed to analyze import complexity", file=str(file_path), error=str(e)
            )
            return {"error": str(e)}


class AutonomousQualityGates:
    """Main quality gate system for autonomous development."""

    def __init__(self, project_path: Path, config: Optional[QualityGateConfig] = None):
        self.project_path = project_path
        self.config = config or QualityGateConfig()

        self.complexity_analyzer = CodeComplexityAnalyzer()
        self.security_scanner = SecurityScanner()
        self.performance_analyzer = PerformanceAnalyzer()

        self.gate_history: List[Dict[str, Any]] = []

    async def run_all_gates(
        self, files: Optional[List[Path]] = None
    ) -> List[QualityGateResult]:
        """Run all quality gates on specified files or entire project."""
        if files is None:
            files = list(self.project_path.rglob("*.py"))
            files = [
                f
                for f in files
                if not any(
                    exclude in str(f) for exclude in [".venv", "__pycache__", ".git"]
                )
            ]

        results = []

        # Complexity gate
        results.append(await self._run_complexity_gate(files))

        # Security gate
        if self.config.security_scan_enabled:
            results.append(await self._run_security_gate(files))

        # Performance gate
        if self.config.performance_check_enabled:
            results.append(await self._run_performance_gate(files))

        # Test coverage gate
        results.append(await self._run_test_coverage_gate())

        # Documentation gate
        if self.config.documentation_check_enabled:
            results.append(await self._run_documentation_gate(files))

        # Record results
        self._record_gate_run(results)

        return results

    async def _run_complexity_gate(self, files: List[Path]) -> QualityGateResult:
        """Run complexity analysis gate."""
        start_time = time.time()

        try:
            analysis = self.complexity_analyzer.analyze_project(self.project_path)

            max_complexity = analysis.get("max_function_complexity", 0)
            passed = max_complexity <= self.config.max_complexity

            return QualityGateResult(
                gate_name="complexity",
                passed=passed,
                score=min(1.0, self.config.max_complexity / max(max_complexity, 1)),
                threshold=self.config.max_complexity,
                details=analysis,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="complexity",
                passed=False,
                score=0.0,
                threshold=self.config.max_complexity,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def _run_security_gate(self, files: List[Path]) -> QualityGateResult:
        """Run security scanning gate."""
        start_time = time.time()

        try:
            scan_result = await self.security_scanner.scan_project(self.project_path)

            total_issues = scan_result.get("total_issues", 0)
            passed = total_issues == 0

            return QualityGateResult(
                gate_name="security",
                passed=passed,
                score=1.0 if passed else 0.0,
                threshold=0,
                details=scan_result,
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="security",
                passed=False,
                score=0.0,
                threshold=0,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def _run_performance_gate(self, files: List[Path]) -> QualityGateResult:
        """Run performance analysis gate."""
        start_time = time.time()

        try:
            # Simple performance check - analyze imports
            total_expensive_imports = 0
            total_imports = 0

            for file_path in files:
                analysis = await self.performance_analyzer.analyze_import_complexity(
                    file_path
                )
                if "error" not in analysis:
                    total_expensive_imports += len(
                        analysis.get("expensive_imports", [])
                    )
                    total_imports += analysis.get("total_imports", 0)

            performance_score = 1.0 - (total_expensive_imports / max(total_imports, 1))
            passed = performance_score > 0.8  # 80% threshold

            return QualityGateResult(
                gate_name="performance",
                passed=passed,
                score=performance_score,
                threshold=0.8,
                details={
                    "total_imports": total_imports,
                    "expensive_imports": total_expensive_imports,
                    "performance_score": performance_score,
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="performance",
                passed=False,
                score=0.0,
                threshold=0.8,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def _run_test_coverage_gate(self) -> QualityGateResult:
        """Run test coverage gate."""
        start_time = time.time()

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
            coverage_percentage = 0.0

            if coverage_file.exists():
                import json

                with open(coverage_file, "r") as f:
                    coverage_data = json.load(f)
                    coverage_percentage = (
                        coverage_data.get("totals", {}).get("percent_covered", 0) / 100
                    )

            passed = coverage_percentage >= self.config.min_test_coverage

            return QualityGateResult(
                gate_name="test_coverage",
                passed=passed,
                score=coverage_percentage,
                threshold=self.config.min_test_coverage,
                details={
                    "coverage_percentage": coverage_percentage * 100,
                    "test_exit_code": process.returncode,
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="test_coverage",
                passed=False,
                score=0.0,
                threshold=self.config.min_test_coverage,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
            )

    async def _run_documentation_gate(self, files: List[Path]) -> QualityGateResult:
        """Run documentation completeness gate."""
        start_time = time.time()

        try:
            total_functions = 0
            documented_functions = 0

            for file_path in files:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        if not node.name.startswith("_"):  # Skip private functions
                            total_functions += 1

                            # Check if function has docstring
                            if (
                                node.body
                                and isinstance(node.body[0], ast.Expr)
                                and isinstance(node.body[0].value, ast.Constant)
                                and isinstance(node.body[0].value.value, str)
                            ):
                                documented_functions += 1

            documentation_score = documented_functions / max(total_functions, 1)
            passed = documentation_score >= 0.8  # 80% documentation threshold

            return QualityGateResult(
                gate_name="documentation",
                passed=passed,
                score=documentation_score,
                threshold=0.8,
                details={
                    "total_functions": total_functions,
                    "documented_functions": documented_functions,
                    "documentation_percentage": documentation_score * 100,
                },
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return QualityGateResult(
                gate_name="documentation",
                passed=False,
                score=0.0,
                threshold=0.8,
                details={},
                execution_time=time.time() - start_time,
                error=str(e),
            )

    def _record_gate_run(self, results: List[QualityGateResult]) -> None:
        """Record quality gate run for analysis."""
        record = {
            "timestamp": time.time(),
            "total_gates": len(results),
            "passed_gates": sum(1 for r in results if r.passed),
            "overall_passed": all(r.passed for r in results),
            "results": [
                {
                    "gate": r.gate_name,
                    "passed": r.passed,
                    "score": r.score,
                    "execution_time": r.execution_time,
                    "error": r.error,
                }
                for r in results
            ],
        }

        self.gate_history.append(record)

        # Keep only last 50 runs
        if len(self.gate_history) > 50:
            self.gate_history = self.gate_history[-50:]

        logger.info(
            "Quality gate run completed",
            **{k: v for k, v in record.items() if k != "results"},
        )

    def get_gate_statistics(self) -> Dict[str, Any]:
        """Get quality gate statistics."""
        if not self.gate_history:
            return {"total_runs": 0}

        total_runs = len(self.gate_history)
        successful_runs = sum(1 for run in self.gate_history if run["overall_passed"])

        gate_success_rates = {}
        for run in self.gate_history:
            for result in run["results"]:
                gate_name = result["gate"]
                if gate_name not in gate_success_rates:
                    gate_success_rates[gate_name] = {"passed": 0, "total": 0}
                gate_success_rates[gate_name]["total"] += 1
                if result["passed"]:
                    gate_success_rates[gate_name]["passed"] += 1

        # Calculate success rates
        for gate_name in gate_success_rates:
            stats = gate_success_rates[gate_name]
            stats["success_rate"] = stats["passed"] / stats["total"]

        return {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "overall_success_rate": successful_runs / total_runs,
            "gate_success_rates": gate_success_rates,
            "recent_runs": self.gate_history[-10:]
            if len(self.gate_history) > 10
            else self.gate_history,
        }
