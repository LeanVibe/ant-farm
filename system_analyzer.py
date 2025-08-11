#!/usr/bin/env python3
"""
LeanVibe Agent Hive - Comprehensive System Analysis
Analyzes architecture, dependencies, and component health
"""

import ast
import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
import json


class SystemAnalyzer:
    """Analyzes the entire LeanVibe system architecture."""

    def __init__(self, root_path: str = "src"):
        self.root_path = Path(root_path)
        self.components = {}
        self.dependencies = {}
        self.contracts = {}
        self.test_coverage = {}

    def analyze_complete_system(self) -> Dict[str, Any]:
        """Perform comprehensive system analysis."""
        print("🔍 LeanVibe Agent Hive - System Architecture Analysis")
        print("=" * 60)

        # 1. Component Discovery
        components = self._discover_components()

        # 2. Dependency Analysis
        dependencies = self._analyze_dependencies()

        # 3. Contract Analysis
        contracts = self._analyze_contracts()

        # 4. CLI Interface Analysis
        cli_analysis = self._analyze_cli_interface()

        # 5. Test Coverage Analysis
        test_analysis = self._analyze_test_coverage()

        # 6. Critical Path Analysis
        critical_paths = self._analyze_critical_paths()

        return {
            "components": components,
            "dependencies": dependencies,
            "contracts": contracts,
            "cli_interface": cli_analysis,
            "test_coverage": test_analysis,
            "critical_paths": critical_paths,
            "recommendations": self._generate_recommendations(),
        }

    def _discover_components(self) -> Dict[str, Any]:
        """Discover all system components and their types."""
        print("\n📊 Component Discovery")
        print("-" * 30)

        components = {
            "agents": [],
            "core_services": [],
            "api_endpoints": [],
            "cli_commands": [],
            "models": [],
            "utilities": [],
        }

        for py_file in self.root_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    tree = ast.parse(content)

                component_info = self._analyze_file_component(py_file, tree)

                # Categorize component
                if "agents/" in str(py_file):
                    components["agents"].append(component_info)
                elif "api/" in str(py_file):
                    components["api_endpoints"].append(component_info)
                elif "cli/" in str(py_file):
                    components["cli_commands"].append(component_info)
                elif "models.py" in str(py_file):
                    components["models"].append(component_info)
                elif "core/" in str(py_file):
                    components["core_services"].append(component_info)
                else:
                    components["utilities"].append(component_info)

            except Exception as e:
                print(f"⚠️  Error analyzing {py_file}: {e}")

        # Print summary
        for category, items in components.items():
            print(f"  {category.replace('_', ' ').title()}: {len(items)}")

        return components

    def _analyze_file_component(self, file_path: Path, tree: ast.AST) -> Dict[str, Any]:
        """Analyze a single Python file component."""
        classes = []
        functions = []
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract class info
                base_classes = [
                    base.id if isinstance(base, ast.Name) else str(base)
                    for base in node.bases
                ]
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]

                classes.append(
                    {
                        "name": node.name,
                        "bases": base_classes,
                        "methods": methods,
                        "is_async": any("async" in method for method in methods),
                        "line": node.lineno,
                    }
                )

            elif isinstance(node, ast.FunctionDef):
                if node.col_offset == 0:  # Top-level function
                    functions.append(
                        {
                            "name": node.name,
                            "is_async": isinstance(node, ast.AsyncFunctionDef),
                            "args": [arg.arg for arg in node.args.args],
                            "line": node.lineno,
                        }
                    )

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    imports.append({"type": "from", "module": module, "names": names})
                else:
                    names = [alias.name for alias in node.names]
                    imports.append({"type": "import", "names": names})

        return {
            "file": str(file_path),
            "classes": classes,
            "functions": functions,
            "imports": imports,
            "size_lines": len(open(file_path).readlines()) if file_path.exists() else 0,
        }

    def _analyze_dependencies(self) -> Dict[str, Any]:
        """Analyze component dependencies and identify cycles."""
        print("\n🔗 Dependency Analysis")
        print("-" * 30)

        # Map internal dependencies
        internal_deps = {}
        external_deps = set()

        for py_file in self.root_path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, "r") as f:
                    content = f.read()
                    tree = ast.parse(content)

                file_deps = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith(("src.", "..")):
                            # Internal dependency
                            file_deps.append(node.module)
                        elif node.module:
                            # External dependency
                            external_deps.add(node.module.split(".")[0])

                internal_deps[str(py_file)] = file_deps

            except Exception as e:
                print(f"⚠️  Error analyzing dependencies for {py_file}: {e}")

        # Find dependency cycles
        cycles = self._find_dependency_cycles(internal_deps)

        print(f"  Internal dependencies mapped: {len(internal_deps)}")
        print(f"  External dependencies: {len(external_deps)}")
        print(f"  Dependency cycles found: {len(cycles)}")

        return {
            "internal": internal_deps,
            "external": list(external_deps),
            "cycles": cycles,
            "analysis": {
                "most_dependent": self._find_most_dependent_files(internal_deps),
                "least_dependent": self._find_least_dependent_files(internal_deps),
            },
        }

    def _find_dependency_cycles(self, deps: Dict[str, List[str]]) -> List[List[str]]:
        """Find circular dependencies using DFS."""
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for dep in deps.get(node, []):
                dfs(dep, path.copy())

            rec_stack.remove(node)

        for file in deps:
            if file not in visited:
                dfs(file, [])

        return cycles

    def _find_most_dependent_files(
        self, deps: Dict[str, List[str]]
    ) -> List[Tuple[str, int]]:
        """Find files with most dependencies."""
        return sorted(
            [(f, len(d)) for f, d in deps.items()], key=lambda x: x[1], reverse=True
        )[:10]

    def _find_least_dependent_files(
        self, deps: Dict[str, List[str]]
    ) -> List[Tuple[str, int]]:
        """Find files with least dependencies."""
        return sorted([(f, len(d)) for f, d in deps.items()], key=lambda x: x[1])[:10]

    def _analyze_contracts(self) -> Dict[str, Any]:
        """Analyze system contracts (interfaces, APIs, protocols)."""
        print("\n📋 Contract Analysis")
        print("-" * 30)

        contracts = {
            "agent_contracts": self._analyze_agent_contracts(),
            "api_contracts": self._analyze_api_contracts(),
            "message_contracts": self._analyze_message_contracts(),
            "database_contracts": self._analyze_database_contracts(),
        }

        return contracts

    def _analyze_agent_contracts(self) -> Dict[str, Any]:
        """Analyze agent interface contracts."""
        agent_contracts = {}

        # Find BaseAgent contract
        base_agent_file = self.root_path / "agents" / "base_agent.py"
        if base_agent_file.exists():
            with open(base_agent_file, "r") as f:
                content = f.read()
                tree = ast.parse(content)

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == "BaseAgent":
                    methods = []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            is_abstract = any(
                                isinstance(decorator, ast.Name)
                                and decorator.id == "abstractmethod"
                                for decorator in item.decorator_list
                            )
                            methods.append(
                                {
                                    "name": item.name,
                                    "is_abstract": is_abstract,
                                    "is_async": isinstance(item, ast.AsyncFunctionDef),
                                    "args": [arg.arg for arg in item.args.args],
                                }
                            )

                    agent_contracts["BaseAgent"] = {
                        "methods": methods,
                        "abstract_methods": [m for m in methods if m["is_abstract"]],
                    }

        return agent_contracts

    def _analyze_api_contracts(self) -> Dict[str, Any]:
        """Analyze API endpoint contracts."""
        api_contracts = {}

        main_api_file = self.root_path / "api" / "main.py"
        if main_api_file.exists():
            with open(main_api_file, "r") as f:
                content = f.read()

            # Extract FastAPI routes (simplified)
            endpoints = []
            lines = content.split("\n")
            current_endpoint = None

            for i, line in enumerate(lines):
                if "@app." in line and any(
                    method in line for method in ["get", "post", "put", "delete"]
                ):
                    # Found endpoint decorator
                    method = None
                    for m in ["get", "post", "put", "delete"]:
                        if f"@app.{m}" in line:
                            method = m.upper()
                            break

                    # Extract path
                    path_start = line.find('"')
                    path_end = line.find('"', path_start + 1)
                    path = (
                        line[path_start + 1 : path_end]
                        if path_start != -1 and path_end != -1
                        else "unknown"
                    )

                    # Look for function definition on next lines
                    func_name = None
                    for j in range(i + 1, min(i + 5, len(lines))):
                        if "async def " in lines[j] or "def " in lines[j]:
                            func_name = lines[j].split("def ")[1].split("(")[0].strip()
                            break

                    endpoints.append(
                        {"method": method, "path": path, "function": func_name}
                    )

            api_contracts["endpoints"] = endpoints

        return api_contracts

    def _analyze_message_contracts(self) -> Dict[str, Any]:
        """Analyze message passing contracts."""
        message_contracts = {}

        # Look for message broker and models
        message_files = [
            "core/message_broker.py",
            "core/enhanced_message_broker.py",
            "core/models.py",
        ]

        for file_path in message_files:
            full_path = self.root_path / file_path
            if full_path.exists():
                # Analyze message types, structures
                message_contracts[file_path] = {"analyzed": True}

        return message_contracts

    def _analyze_database_contracts(self) -> Dict[str, Any]:
        """Analyze database schema contracts."""
        db_contracts = {}

        models_file = self.root_path / "core" / "models.py"
        if models_file.exists():
            with open(models_file, "r") as f:
                content = f.read()
                tree = ast.parse(content)

            models = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    # Check if it's a SQLAlchemy model
                    is_model = any(
                        isinstance(base, ast.Name) and base.id in ["Base", "Model"]
                        for base in node.bases
                    )
                    if is_model or "Model" in node.name:
                        models.append(node.name)

            db_contracts["models"] = models

        return db_contracts

    def _analyze_cli_interface(self) -> Dict[str, Any]:
        """Analyze CLI interface completeness for power users."""
        print("\n💻 CLI Interface Analysis")
        print("-" * 30)

        cli_commands = {}

        cli_dir = self.root_path / "cli" / "commands"
        if cli_dir.exists():
            for py_file in cli_dir.glob("*.py"):
                if py_file.name == "__init__.py":
                    continue

                with open(py_file, "r") as f:
                    content = f.read()

                # Extract Typer commands
                commands = []
                lines = content.split("\n")

                for i, line in enumerate(lines):
                    if "@app.command()" in line:
                        # Look for function definition
                        for j in range(i + 1, min(i + 3, len(lines))):
                            if "def " in lines[j]:
                                func_name = (
                                    lines[j].split("def ")[1].split("(")[0].strip()
                                )

                                # Extract docstring
                                docstring = ""
                                if j + 1 < len(lines) and '"""' in lines[j + 1]:
                                    docstring = lines[j + 1].strip().replace('"""', "")

                                commands.append(
                                    {"name": func_name, "description": docstring}
                                )
                                break

                cli_commands[py_file.name] = commands

        print(f"  CLI command modules: {len(cli_commands)}")
        total_commands = sum(len(cmds) for cmds in cli_commands.values())
        print(f"  Total CLI commands: {total_commands}")

        return cli_commands

    def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage and quality."""
        print("\n🧪 Test Coverage Analysis")
        print("-" * 30)

        test_analysis = {
            "unit_tests": [],
            "integration_tests": [],
            "e2e_tests": [],
            "coverage_gaps": [],
        }

        tests_dir = Path("tests")
        if tests_dir.exists():
            for test_file in tests_dir.rglob("test_*.py"):
                category = "unit_tests"
                if "integration" in str(test_file):
                    category = "integration_tests"
                elif "e2e" in str(test_file):
                    category = "e2e_tests"

                test_analysis[category].append(str(test_file))

        print(f"  Unit tests: {len(test_analysis['unit_tests'])}")
        print(f"  Integration tests: {len(test_analysis['integration_tests'])}")
        print(f"  E2E tests: {len(test_analysis['e2e_tests'])}")

        return test_analysis

    def _analyze_critical_paths(self) -> Dict[str, Any]:
        """Analyze critical system execution paths."""
        print("\n🎯 Critical Path Analysis")
        print("-" * 30)

        critical_paths = {
            "agent_lifecycle": [
                "Agent Creation",
                "Agent Registration",
                "Task Assignment",
                "Task Execution",
                "Result Reporting",
                "Agent Cleanup",
            ],
            "collaboration_flow": [
                "Session Creation",
                "Agent Joining",
                "Context Sharing",
                "Message Exchange",
                "Conflict Resolution",
                "Session Completion",
            ],
            "api_request_flow": [
                "Authentication",
                "Request Validation",
                "Business Logic",
                "Database Operation",
                "Response Formation",
            ],
        }

        print(f"  Critical paths identified: {len(critical_paths)}")

        return critical_paths

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for system improvement."""
        recommendations = [
            "1. Implement comprehensive unit tests for each component",
            "2. Add contract testing between components",
            "3. Create component isolation test framework",
            "4. Implement proper integration test suite",
            "5. Add performance benchmarking tests",
            "6. Create CLI command completeness audit",
            "7. Implement health check endpoints for all services",
            "8. Add monitoring and observability",
            "9. Create deployment verification tests",
            "10. Implement automated dependency vulnerability scanning",
        ]

        return recommendations


def main():
    """Run comprehensive system analysis."""
    analyzer = SystemAnalyzer()
    analysis = analyzer.analyze_complete_system()

    # Save analysis to file
    with open("system_analysis_report.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    print(f"\n📄 Analysis saved to: system_analysis_report.json")
    print(
        f"📊 Total components analyzed: {sum(len(cat) for cat in analysis['components'].values())}"
    )

    print("\n🎯 Key Recommendations:")
    for rec in analysis["recommendations"][:5]:
        print(f"  {rec}")

    return analysis


if __name__ == "__main__":
    main()
