"""Unit tests for DeveloperAgent with TDD approach."""

from unittest.mock import AsyncMock, patch

import pytest

from src.agents.base_agent import HealthStatus, TaskResult
from src.agents.developer_agent import DeveloperAgent
from src.core.task_queue import Task


@pytest.fixture
def developer_agent():
    """Create a DeveloperAgent instance for testing."""
    return DeveloperAgent()


@pytest.fixture
def mock_task():
    """Create a mock task for testing."""
    return Task(
        id="test-task-123",
        title="Authentication System",
        description="Implement a user authentication system",
        priority=1,
        task_type="implementation",
        payload={"framework": "FastAPI", "database": "PostgreSQL"},
    )


@pytest.fixture
def mock_code_generation_task():
    """Create a mock code generation task."""
    return Task(
        id="code-gen-456",
        title="User Model",
        description="Create a User model with SQLAlchemy",
        priority=1,
        task_type="code_generation",
        payload={
            "class_name": "User",
            "fields": ["id", "username", "email", "password_hash"],
            "framework": "SQLAlchemy",
        },
    )


@pytest.fixture
def mock_code_generation_task():
    """Create a mock code generation task."""
    return Task(
        id="code-gen-456",
        description="Create a User model with SQLAlchemy",
        priority=1,
        task_type="code_generation",
        metadata={
            "class_name": "User",
            "fields": ["id", "username", "email", "password_hash"],
            "framework": "SQLAlchemy",
        },
    )


@pytest.mark.asyncio
class TestDeveloperAgent:
    """Test cases for DeveloperAgent functionality."""

    async def test_developer_agent_initialization(self, developer_agent):
        """Test that DeveloperAgent initializes correctly."""
        assert developer_agent.name == "developer-agent"
        assert developer_agent.agent_type == "developer"
        assert "Software Engineer" in developer_agent.role
        # Check initial health through health_check method
        health = await developer_agent.health_check()
        assert health in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]

    async def test_developer_agent_has_programming_languages(self, developer_agent):
        """Test that DeveloperAgent has programming language capabilities."""
        assert hasattr(developer_agent, "programming_languages")
        assert "python" in developer_agent.programming_languages
        assert "javascript" in developer_agent.programming_languages
        assert "typescript" in developer_agent.programming_languages

    async def test_developer_agent_has_frameworks(self, developer_agent):
        """Test that DeveloperAgent knows about common frameworks."""
        assert hasattr(developer_agent, "frameworks")
        assert "fastapi" in developer_agent.frameworks
        assert "react" in developer_agent.frameworks
        assert "sqlalchemy" in developer_agent.frameworks

    async def test_developer_agent_has_development_tools(self, developer_agent):
        """Test that DeveloperAgent has development tools."""
        assert hasattr(developer_agent, "development_tools")
        assert "git" in developer_agent.development_tools
        assert "pytest" in developer_agent.development_tools
        assert "ruff" in developer_agent.development_tools

    @patch("src.agents.developer_agent.DeveloperAgent.execute_with_cli_tool")
    async def test_process_implementation_task(
        self, mock_cli, developer_agent, mock_task
    ):
        """Test processing an implementation task."""
        # Mock CLI tool response
        mock_cli.return_value = AsyncMock()
        mock_cli.return_value.success = True
        mock_cli.return_value.output = "Implementation completed successfully"

        result = await developer_agent.process_task(mock_task)

        assert isinstance(result, TaskResult)
        assert result.success is True
        assert "implementation" in result.data["message"].lower()
        mock_cli.assert_called_once()

    @patch("src.agents.developer_agent.DeveloperAgent.execute_with_cli_tool")
    async def test_generate_code_with_tdd(self, mock_cli, developer_agent):
        """Test code generation following TDD principles."""
        # Mock CLI tool responses for TDD workflow
        mock_cli.return_value = AsyncMock()
        mock_cli.return_value.success = True
        mock_cli.return_value.output = "Tests and code generated successfully"

        # Create a simple task without using the problematic fixture
        task = Task(
            id="code-gen-456",
            title="User Model",
            description="Create a User model with SQLAlchemy",
            priority=1,
            task_type="code_generation",
            payload={
                "class_name": "User",
                "fields": ["id", "username", "email", "password_hash"],
                "framework": "SQLAlchemy",
            },
        )

        result = await developer_agent.generate_code_with_tdd(task)

        assert isinstance(result, TaskResult)
        assert result.success is True
        assert (
            "tdd" in result.data["message"].lower()
            or "test" in result.data["message"].lower()
        )
        mock_cli.assert_called_once()

    @patch("src.agents.developer_agent.DeveloperAgent.execute_with_cli_tool")
    async def test_implement_feature(self, mock_cli, developer_agent, mock_task):
        """Test feature implementation workflow."""
        # Mock CLI tool response
        mock_cli.return_value = AsyncMock()
        mock_cli.return_value.success = True
        mock_cli.return_value.output = "Feature implemented with tests"

        result = await developer_agent.implement_feature(mock_task)

        assert isinstance(result, TaskResult)
        assert result.success is True
        assert "feature" in result.data["message"].lower()

    @patch("src.agents.developer_agent.DeveloperAgent.execute_with_cli_tool")
    async def test_refactor_code(self, mock_cli, developer_agent):
        """Test code refactoring capabilities."""
        # Mock CLI tool response
        mock_cli.return_value = AsyncMock()
        mock_cli.return_value.success = True
        mock_cli.return_value.output = "Code refactored successfully"

        file_path = "/path/to/code.py"
        refactor_type = "extract_method"

        result = await developer_agent.refactor_code(file_path, refactor_type)

        assert isinstance(result, TaskResult)
        assert result.success is True
        assert "refactor" in result.data["message"].lower()

    async def test_analyze_code_complexity(self, developer_agent):
        """Test code complexity analysis."""
        code = """
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            return x
    else:
        return 0
"""

        complexity = await developer_agent.analyze_code_complexity(code)

        assert isinstance(complexity, dict)
        assert "cyclomatic_complexity" in complexity
        assert complexity["cyclomatic_complexity"] > 1

    async def test_suggest_improvements(self, developer_agent):
        """Test code improvement suggestions."""
        code = """
def bad_function():
    x = 1
    y = 2
    z = x + y
    print(z)
    return z
"""

        suggestions = await developer_agent.suggest_improvements(code)

        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        assert all(isinstance(suggestion, str) for suggestion in suggestions)

    @patch("src.agents.developer_agent.DeveloperAgent.execute_with_cli_tool")
    async def test_task_processing_failure_handling(
        self, mock_cli, developer_agent, mock_task
    ):
        """Test handling of task processing failures."""
        # Mock CLI tool failure
        mock_cli.return_value = AsyncMock()
        mock_cli.return_value.success = False
        mock_cli.return_value.error = "Compilation error"

        result = await developer_agent.process_task(mock_task)

        assert isinstance(result, TaskResult)
        assert result.success is False
        assert "error" in result.error.lower() or "fail" in result.error.lower()

    async def test_health_check_success(self, developer_agent):
        """Test successful health check."""
        with patch.object(
            developer_agent, "check_development_tools", return_value=True
        ):
            health = await developer_agent.health_check()
            assert health == HealthStatus.HEALTHY

    async def test_health_check_degraded(self, developer_agent):
        """Test degraded health status."""
        with patch.object(
            developer_agent, "check_development_tools", return_value=False
        ):
            health = await developer_agent.health_check()
            assert health == HealthStatus.DEGRADED

    async def test_get_capabilities(self, developer_agent):
        """Test capabilities reporting."""
        capabilities = developer_agent.get_capabilities()

        assert isinstance(capabilities, dict)
        assert "programming_languages" in capabilities
        assert "frameworks" in capabilities
        assert "development_tools" in capabilities
        assert "specializations" in capabilities

    @patch("src.agents.developer_agent.DeveloperAgent.execute_with_cli_tool")
    async def test_collaborative_development(self, mock_cli, developer_agent):
        """Test collaborative development features."""
        # Mock CLI tool response
        mock_cli.return_value = AsyncMock()
        mock_cli.return_value.success = True
        mock_cli.return_value.output = "Collaboration request processed"

        collaboration_request = {
            "partner_agent": "qa-agent",
            "task": "code_review",
            "code": "def example(): pass",
        }

        result = await developer_agent.handle_collaboration(collaboration_request)

        assert isinstance(result, dict)
        assert "success" in result
        assert result["success"] is True

    async def test_performance_tracking(self, developer_agent):
        """Test performance metrics tracking."""
        initial_tasks = developer_agent.tasks_completed
        initial_lines = developer_agent.lines_of_code_written

        # Simulate task completion
        developer_agent.track_task_completion(100)  # 100 lines of code

        assert developer_agent.tasks_completed == initial_tasks + 1
        assert developer_agent.lines_of_code_written == initial_lines + 100

    async def test_specialization_detection(self, developer_agent):
        """Test automatic specialization detection based on task patterns."""
        # Simulate processing multiple web development tasks
        web_tasks = [
            Task(
                id=f"web-{i}",
                title=f"Web Component {i}",
                description=f"Implement React component {i}",
                priority=1,
                task_type="web_development",
            )
            for i in range(3)
        ]

        for task in web_tasks:
            developer_agent.track_specialization(task)

        specializations = developer_agent.get_specializations()
        assert "web_development" in specializations
        assert specializations["web_development"] >= 3
