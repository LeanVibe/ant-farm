"""
Unit tests for monitor CLI commands
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile

from src.cli.commands.monitor import (
    _monitor_agents_live,
    _list_agents_snapshot,
    _monitor_tasks_live,
    _list_tasks_snapshot,
    _monitor_system_live,
    _show_system_snapshot,
    _monitor_resources_live,
    _show_resources_snapshot,
    _show_metrics,
    _display_agent_metrics,
    _display_task_metrics,
    _display_system_metrics,
    _output_metrics_csv,
)


class TestMonitorCommands:
    """Test cases for monitor CLI commands"""

    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx AsyncClient"""
        with patch('src.cli.commands.monitor.httpx.AsyncClient') as mock_client:
            mock_instance = AsyncMock()
            mock_client.return_value.__aenter__.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def mock_console(self):
        """Mock rich console"""
        with patch('src.cli.commands.monitor.console') as mock_console:
            yield mock_console

    @pytest.fixture
    def sample_agent_data(self):
        """Sample agent data for testing"""
        return [
            {
                "short_id": "abc123",
                "name": "test-agent-1",
                "type": "meta",
                "status": "active",
                "uptime": 120.5,
                "last_heartbeat": 1234567890.0,
                "cpu_percent": 15.2
            },
            {
                "short_id": "def456",
                "name": "test-agent-2",
                "type": "developer",
                "status": "inactive",
                "uptime": 60.0,
                "last_heartbeat": "never",
                "cpu_percent": 0.0
            }
        ]

    @pytest.fixture
    def sample_task_data(self):
        """Sample task data for testing"""
        return [
            {
                "short_id": "task123",
                "title": "Test Task 1",
                "type": "general",
                "status": "completed",
                "priority": 5,
                "assigned_to": "meta-agent"
            },
            {
                "short_id": "task456",
                "title": "Test Task 2",
                "type": "code_review",
                "status": "pending",
                "priority": 3,
                "assigned_to": "qa-agent"
            }
        ]

    @pytest.fixture
    def sample_system_data(self):
        """Sample system data for testing"""
        return {
            "status": "healthy",
            "database": {"status": "healthy"},
            "redis": {"status": "healthy"}
        }

    @pytest.fixture
    def sample_metrics_data(self):
        """Sample metrics data for testing"""
        return {
            "cpu_percent": 45.2,
            "memory_percent": 67.8,
            "disk_percent": 23.1,
            "memory_used": 8589934592,  # 8GB
            "memory_total": 17179869184,  # 16GB
            "disk_used": 107374182400,  # 100GB
            "disk_total": 500000000000,  # 500GB
        }

    @pytest.mark.asyncio
    async def test_list_agents_snapshot_success(self, mock_httpx_client, mock_console, sample_agent_data):
        """Test successful agent snapshot listing"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": sample_agent_data}
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock create_status_table
            with patch('src.cli.commands.monitor.create_status_table') as mock_table:
                mock_table.return_value = "Mocked Table"

                await _list_agents_snapshot()

                # Verify API call
                mock_httpx_client.get.assert_called_once()
                call_args = mock_httpx_client.get.call_args
                assert "api/v1/cli/agents" in call_args[0][0]

                # Verify table creation and display
                mock_table.assert_called_once()
                mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_list_agents_snapshot_with_filters(self, mock_httpx_client, mock_console, sample_agent_data):
        """Test agent snapshot listing with filters"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": sample_agent_data}
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock create_status_table
            with patch('src.cli.commands.monitor.create_status_table') as mock_table:
                mock_table.return_value = "Mocked Table"

                await _list_agents_snapshot("meta", "active")

                # Verify table was called (filtered data should still be displayed)
                mock_table.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_agents_snapshot_api_error(self, mock_httpx_client, mock_console):
        """Test agent snapshot listing with API error"""
        # Mock API error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock error_handler
            with patch('src.cli.commands.monitor.error_handler') as mock_error:
                await _list_agents_snapshot()

                # Verify error handler was called
                mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_tasks_snapshot_success(self, mock_httpx_client, mock_console, sample_task_data):
        """Test successful task snapshot listing"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": sample_task_data}
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock create_status_table
            with patch('src.cli.commands.monitor.create_status_table') as mock_table:
                mock_table.return_value = "Mocked Table"

                await _list_tasks_snapshot()

                # Verify API call
                mock_httpx_client.get.assert_called_once()
                call_args = mock_httpx_client.get.call_args
                assert "api/v1/cli/tasks" in call_args[0][0]

                # Verify table creation and display
                mock_table.assert_called_once()
                mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_show_system_snapshot_success(self, mock_httpx_client, mock_console, sample_system_data):
        """Test successful system snapshot display"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_system_data
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock create_system_status_table
            with patch('src.cli.commands.monitor.create_system_status_table') as mock_table:
                mock_table.return_value = "Mocked System Table"

                await _show_system_snapshot()

                # Verify API call
                mock_httpx_client.get.assert_called_once()
                call_args = mock_httpx_client.get.call_args
                assert "api/v1/health" in call_args[0][0]

                # Verify table creation and display
                mock_table.assert_called_once()
                mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_show_resources_snapshot_success(self, mock_httpx_client, mock_console, sample_metrics_data):
        """Test successful resource snapshot display"""
        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": sample_metrics_data}
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock create_status_table
            with patch('src.cli.commands.monitor.create_status_table') as mock_table:
                mock_table.return_value = "Mocked Resources Table"

                await _show_resources_snapshot()

                # Verify API call
                mock_httpx_client.get.assert_called_once()
                call_args = mock_httpx_client.get.call_args
                assert "api/v1/metrics/system" in call_args[0][0]

                # Verify table creation and display
                mock_table.assert_called_once()
                mock_console.print.assert_called()

    @pytest.mark.asyncio
    async def test_show_metrics_agents_success(self, mock_httpx_client, mock_console):
        """Test successful agent metrics display"""
        sample_metrics = {
            "performance": {
                "total_agents": 5,
                "avg_tasks_per_agent": 2.3,
                "success_rate": 85.5
            },
            "by_type": {
                "meta": 2,
                "developer": 2,
                "qa": 1
            }
        }

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": sample_metrics}
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock create_status_table
            with patch('src.cli.commands.monitor.create_status_table') as mock_table:
                mock_table.return_value = "Mocked Metrics Table"

                await _show_metrics("agents", "1h", "table")

                # Verify API call
                mock_httpx_client.get.assert_called_once()
                call_args = mock_httpx_client.get.call_args
                assert "api/v1/metrics/agents" in call_args[0][0]
                assert call_args[1]["params"]["period"] == "1h"

                # Verify table creation
                assert mock_table.call_count >= 1

    @pytest.mark.asyncio
    async def test_show_metrics_json_format(self, mock_httpx_client, mock_console):
        """Test metrics display in JSON format"""
        sample_metrics = {"performance": {"total_agents": 5}}

        # Mock API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": sample_metrics}
        mock_httpx_client.get.return_value = mock_response

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            await _show_metrics("agents", "1h", "json")

            # Verify JSON output
            mock_console.print.assert_called()
            printed_content = mock_console.print.call_args[0][0]
            assert isinstance(printed_content, str)
            # Should be valid JSON
            parsed = json.loads(printed_content)
            assert "performance" in parsed

    def test_display_agent_metrics(self, mock_console):
        """Test agent metrics display formatting"""
        metrics = {
            "performance": {
                "total_agents": 5,
                "avg_tasks_per_agent": 2.3,
                "success_rate": 85.5
            },
            "by_type": {
                "meta": 2,
                "developer": 2,
                "qa": 1
            }
        }

        with patch('src.cli.commands.monitor.create_status_table') as mock_table:
            mock_table.return_value = "Mocked Table"

            _display_agent_metrics(metrics)

            # Verify table creation for performance metrics
            assert mock_table.call_count >= 1
            mock_console.print.assert_called()

    def test_display_task_metrics(self, mock_console):
        """Test task metrics display formatting"""
        metrics = {
            "performance": {
                "total_tasks": 10,
                "avg_completion_time": 45.2,
                "success_rate": 90.0
            },
            "by_status": {
                "pending": 3,
                "completed": 7
            }
        }

        with patch('src.cli.commands.monitor.create_status_table') as mock_table:
            mock_table.return_value = "Mocked Table"

            _display_task_metrics(metrics)

            # Verify table creation for performance metrics
            assert mock_table.call_count >= 1
            mock_console.print.assert_called()

    def test_display_system_metrics(self, mock_console):
        """Test system metrics display formatting"""
        metrics = {
            "performance": {
                "avg_cpu_percent": 45.2,
                "avg_memory_percent": 67.8,
                "avg_response_time": 150.0
            },
            "health": {
                "uptime_hours": 24.5,
                "error_rate": 0.5
            }
        }

        with patch('src.cli.commands.monitor.create_status_table') as mock_table:
            mock_table.return_value = "Mocked Table"

            _display_system_metrics(metrics)

            # Verify table creation for performance metrics
            assert mock_table.call_count >= 1
            mock_console.print.assert_called()

    def test_output_metrics_csv(self):
        """Test CSV output formatting"""
        metrics = {
            "performance": {
                "total_agents": 5,
                "avg_tasks_per_agent": 2.3,
                "success_rate": 85.5
            }
        }

        # Capture stdout
        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            _output_metrics_csv("agents", metrics)

        csv_output = f.getvalue()
        assert "Metric,Value,Description" in csv_output
        assert "Total Agents,5" in csv_output
        assert "Avg Tasks/Agent,2.3" in csv_output

    @pytest.mark.asyncio
    async def test_show_metrics_invalid_target(self, mock_console):
        """Test metrics display with invalid target"""
        with patch('src.cli.commands.monitor.error_handler') as mock_error:
            await _show_metrics("invalid", "1h", "table")

            # Verify error handler was called
            mock_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, mock_httpx_client, mock_console):
        """Test connection error handling"""
        # Mock connection error
        mock_httpx_client.get.side_effect = Exception("Connection refused")

        # Mock get_api_headers
        with patch('src.cli.commands.monitor.get_api_headers', return_value={"Authorization": "Bearer test"}):
            # Mock error_handler
            with patch('src.cli.commands.monitor.error_handler') as mock_error:
                await _list_agents_snapshot()

                # Verify error handler was called
                mock_error.assert_called_once()

    def test_metrics_display_with_empty_data(self, mock_console):
        """Test metrics display with empty data"""
        empty_metrics = {}

        with patch('src.cli.commands.monitor.create_status_table') as mock_table:
            mock_table.return_value = "Mocked Table"

            _display_agent_metrics(empty_metrics)

            # Should handle empty data gracefully
            mock_console.print.assert_called()

    def test_csv_output_with_missing_data(self):
        """Test CSV output with missing data"""
        incomplete_metrics = {
            "performance": {
                "total_agents": 3
                # Missing other fields
            }
        }

        import io
        from contextlib import redirect_stdout

        f = io.StringIO()
        with redirect_stdout(f):
            _output_metrics_csv("agents", incomplete_metrics)

        csv_output = f.getvalue()
        assert "Total Agents,3" in csv_output
        # Should not crash on missing fields