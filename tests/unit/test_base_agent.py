"""Unit tests for base agent functionality."""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.base_agent import BaseAgent, CLIToolManager, CLIToolType

import pytest


class TestBaseAgent:
    """Test cases for BaseAgent class."""

    @pytest.fixture
    def mock_async_db_manager(self):
        """Mock async database manager."""
        return AsyncMock()

    @pytest.fixture
    def mock_cli_tools(self):
        """Mock CLI tool manager."""
        return MagicMock(spec=CLIToolManager)

    @pytest.fixture
    def base_agent(self, mock_async_db_manager, mock_cli_tools):
        """Create a test BaseAgent instance."""
        with patch('src.agents.base_agent.get_persistent_cli_manager'), \
             patch('src.agents.base_agent.get_async_database_manager', return_value=mock_async_db_manager), \
             patch('src.agents.base_agent.message_broker'), \
             patch('asyncio.create_task', return_value=MagicMock()):

            # Create a concrete implementation for testing
            class TestAgent(BaseAgent):
                async def run(self):
                    pass

            agent = TestAgent(
                name="test_agent",
                agent_type="test_type",
                role="test_role"
            )
            agent.cli_tools = mock_cli_tools
            agent.async_db_manager = mock_async_db_manager
            agent.agent_uuid = "test-uuid-123"
            return agent

    @pytest.mark.asyncio
    async def test_flush_metrics_if_enabled_disabled(self, base_agent, mock_cli_tools):
        """Test that metrics are not flushed when feature flag is disabled."""
        # Ensure feature flag is not set
        with patch.dict(os.environ, {}, clear=True):
            await base_agent.flush_metrics_if_enabled()

        # flush_metrics should not be called
        mock_cli_tools.flush_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_metrics_if_enabled_enabled(self, base_agent, mock_cli_tools, mock_async_db_manager):
        """Test that metrics are flushed when feature flag is enabled."""
        # Set feature flag
        with patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"}):
            await base_agent.flush_metrics_if_enabled()

        # flush_metrics should be called with correct parameters
        mock_cli_tools.flush_metrics.assert_called_once_with(
            mock_async_db_manager,
            "test-uuid-123",
            "test_agent",
            "test_type"
        )

    @pytest.mark.asyncio
    async def test_flush_metrics_if_enabled_no_db_manager(self, base_agent, mock_cli_tools):
        """Test that metrics are not flushed when database manager is not available."""
        base_agent.async_db_manager = None

        with patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"}):
            await base_agent.flush_metrics_if_enabled()

        mock_cli_tools.flush_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_metrics_if_enabled_no_agent_uuid(self, base_agent, mock_cli_tools):
        """Test that metrics are not flushed when agent UUID is not available."""
        base_agent.agent_uuid = None

        with patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"}):
            await base_agent.flush_metrics_if_enabled()

        mock_cli_tools.flush_metrics.assert_not_called()

    @pytest.mark.asyncio
    async def test_flush_metrics_if_enabled_exception_handling(self, base_agent, mock_cli_tools):
        """Test that exceptions during metrics flush are handled gracefully."""
        mock_cli_tools.flush_metrics.side_effect = Exception("Test error")

        with patch.dict(os.environ, {"HIVE_METRICS_FLUSH": "1"}):
            # Should not raise exception
            await base_agent.flush_metrics_if_enabled()

        mock_cli_tools.flush_metrics.assert_called_once()

    @pytest.mark.asyncio
    async def test_base_agent_initialization(self, mock_cli_tools):
        """Test BaseAgent initialization."""
        # Test that BaseAgent initializes with correct properties
        pass

    def test_cli_tool_detection(self):
        """Test detection of available CLI tools."""
        # Test that available CLI tools are detected correctly
        pass

    def test_cli_tool_priority_selection(self, mock_cli_tools):
        """Test CLI tool selection by priority."""
        # Test that opencode > claude > gemini priority is respected
        pass

    @pytest.mark.asyncio
    async def test_execute_with_cli_tool_success(self, mock_agent):
        """Test successful CLI tool execution."""
        # Test that CLI tools are executed correctly
        pass

    @pytest.mark.asyncio
    async def test_execute_with_cli_tool_fallback(self, mock_agent):
        """Test CLI tool fallback mechanism."""
        # Test that fallback to other tools works when primary fails
        pass

    @pytest.mark.asyncio
    async def test_api_fallback(self, mock_agent):
        """Test fallback to API when CLI tools fail."""
        # Test that API is used when no CLI tools work
        pass

    @pytest.mark.asyncio
    async def test_agent_lifecycle(self, mock_agent):
        """Test agent start/stop lifecycle."""
        # Test agent start(), run(), cleanup() lifecycle
        pass

    @pytest.mark.asyncio
    async def test_context_storage(self, mock_agent):
        """Test context storage functionality."""
        # Test store_context() method
        pass

    @pytest.mark.asyncio
    async def test_message_sending(self, mock_agent):
        """Test agent-to-agent messaging."""
        # Test send_message() functionality
        pass


class TestCLIToolDetection:
    """Test cases for CLI tool detection logic."""

    @patch("subprocess.run")
    def test_opencode_detection(self, mock_subprocess):
        """Test opencode CLI tool detection."""
        # Test that opencode is detected when available
        mock_subprocess.return_value.returncode = 0
        # Test detection logic
        pass

    @patch("subprocess.run")
    def test_claude_detection(self, mock_subprocess):
        """Test Claude CLI tool detection."""
        # Test that Claude CLI is detected when available
        mock_subprocess.return_value.returncode = 0
        # Test detection logic
        pass

    @patch("subprocess.run")
    def test_gemini_detection(self, mock_subprocess):
        """Test Gemini CLI tool detection."""
        # Test that Gemini CLI is detected when available
        mock_subprocess.return_value.returncode = 0
        # Test detection logic
        pass

    @patch("subprocess.run")
    def test_no_tools_detected(self, mock_subprocess):
        """Test behavior when no CLI tools are detected."""
        # Test that system handles absence of CLI tools gracefully
        mock_subprocess.side_effect = FileNotFoundError()
        # Test fallback behavior
        pass


class TestErrorClassification:
    def test_classify_timeout(self):
        assert CLIToolManager.classify_error_text("Request timed out after 30s") == "timeout"

    def test_classify_rate_limit(self):
        assert CLIToolManager.classify_error_text("429 Too Many Requests: rate limit exceeded") == "rate_limit"

    def test_classify_auth(self):
        assert CLIToolManager.classify_error_text("401 Unauthorized: invalid API key") == "auth"

    def test_classify_oom(self):
        assert CLIToolManager.classify_error_text("Process killed: OOM") == "oom"

    def test_classify_other(self):
        assert CLIToolManager.classify_error_text("unexpected error") == "other"


class TestCLIBudgets:
    @pytest.mark.asyncio
    async def test_budget_enforcement(self, monkeypatch):
        m = CLIToolManager()
        # Force a single preferred tool
        m.available_tools = {
            CLIToolType.CLAUDE: {
                "name": "Claude",
                "command": "claude",
                "execute_pattern": lambda prompt: _immediate_fail(),
                "supports_interactive": False,
            }
        }
        m.preferred_tool = CLIToolType.CLAUDE
        m.per_tool_budget_per_minute = 1

        async def _immediate_fail():
            from src.agents.base_agent import ToolResult, ErrorCategory

            return ToolResult(success=False, output="", error="boom", error_category=ErrorCategory.OTHER.value)

        # First call uses the budget
        r1 = await m.execute_prompt("a")
        # Second should be blocked by budget
        r2 = await m.execute_prompt("b")
        assert r2.success is False and r2.error_category == "rate_limit"


class TestAgentExecution:
    """Test cases for agent execution patterns."""

    @pytest.mark.asyncio
    async def test_task_processing_loop(self, mock_agent):
        """Test agent task processing loop."""
        # Test that agents can process tasks in a loop
        pass

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_agent):
        """Test agent error handling."""
        # Test that agents handle errors gracefully
        pass

    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_agent):
        """Test agent graceful shutdown."""
        # Test that agents can be stopped cleanly
        pass


# Integration test placeholder
class TestBaseAgentIntegration:
    """Integration tests for BaseAgent."""

    @pytest.mark.asyncio
    async def test_real_cli_tool_execution(self):
        """Test execution with real CLI tools (if available)."""
        # Test actual CLI tool execution (only if tools are installed)
        pass

    @pytest.mark.asyncio
    async def test_agent_with_redis_backend(self, redis_client):
        """Test agent with real Redis backend."""
        # Test agent functionality with actual Redis
        pass
