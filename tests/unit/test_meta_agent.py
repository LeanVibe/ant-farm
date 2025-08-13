"""Unit tests for MetaAgent."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
import time

from src.agents.meta_agent import MetaAgent
from src.core.task_queue import Task, TaskPriority, TaskStatus


@pytest.fixture
async def meta_agent():
    """Create a MetaAgent instance for testing."""
    with patch("src.agents.base_agent.get_async_database_manager"), \
         patch("src.agents.base_agent.message_broker"), \
         patch("src.agents.base_agent.task_queue"), \
         patch("src.agents.base_agent.get_context_engine"), \
         patch("src.agents.base_agent.get_persistent_cli_manager"):
        
        agent = MetaAgent("test-meta-agent")
        
        # Mock components
        agent.async_db_manager = AsyncMock()
        agent.context_engine = AsyncMock()
        agent.agent_uuid = "test-uuid-123"
        
        # Mock CLI tools
        agent.cli_tools.available_tools = {"claude": {"name": "Claude CLI"}}
        agent.cli_tools.preferred_tool = MagicMock()
        agent.cli_tools.preferred_tool.value = "claude"
        
        yield agent


class TestMetaAgentInitialization:
    """Test MetaAgent initialization."""
    
    def test_meta_agent_initialization(self):
        """Test that MetaAgent initializes correctly."""
        with patch("src.agents.base_agent.get_async_database_manager"), \
             patch("src.agents.base_agent.message_broker"), \
             patch("src.agents.base_agent.task_queue"), \
             patch("src.agents.base_agent.get_context_engine"), \
             patch("src.agents.base_agent.get_persistent_cli_manager"):
            
            agent = MetaAgent("test-meta")
            
            assert agent.name == "test-meta"
            assert agent.agent_type == "meta"
            assert agent.role == "meta_coordinator"
            assert "system_analysis" in agent.meta_capabilities
            assert "strategic_planning" in agent.meta_capabilities
            assert "agent_coordination" in agent.meta_capabilities
            assert agent.improvement_tasks_created == 0
            assert agent.coordination_sessions == 0
            assert agent.current_strategic_focus == "system_stability"
    
    def test_meta_agent_capabilities(self):
        """Test that MetaAgent has correct capabilities."""
        with patch("src.agents.base_agent.get_async_database_manager"), \
             patch("src.agents.base_agent.message_broker"), \
             patch("src.agents.base_agent.task_queue"), \
             patch("src.agents.base_agent.get_context_engine"), \
             patch("src.agents.base_agent.get_persistent_cli_manager"):
            
            agent = MetaAgent("test-meta")
            capabilities = agent.capabilities
            
            # Should have base capabilities plus meta-specific ones
            assert "system_analysis" in capabilities
            assert "strategic_planning" in capabilities
            assert "agent_coordination" in capabilities
            assert "performance_optimization" in capabilities


class TestSystemHealthAnalysis:
    """Test system health analysis functionality."""
    
    @pytest.mark.asyncio
    async def test_analyze_system_health_success(self, meta_agent):
        """Test successful system health analysis."""
        # Mock database manager
        mock_agents = [
            MagicMock(status="active", type="meta"),
            MagicMock(status="active", type="architect"),
            MagicMock(status="error", type="qa"),
        ]
        meta_agent.async_db_manager.get_active_agents.return_value = mock_agents
        meta_agent.async_db_manager.health_check.return_value = True
        
        # Mock task queue
        with patch("src.agents.meta_agent.task_queue") as mock_task_queue:
            mock_stats = MagicMock()
            mock_stats.queue_size_by_priority = {"high": 5, "medium": 10, "low": 15}
            mock_stats.completed_tasks = 100
            mock_stats.failed_tasks = 10
            mock_stats.processing_rate = 2.5
            mock_stats.average_wait_time = 30
            mock_task_queue.get_queue_stats.return_value = mock_stats
            
            # Mock performance monitoring
            meta_agent.monitor_performance = AsyncMock(return_value={
                "avg_response_time": 0.5,
                "tasks_per_minute": 10.0,
                "error_rate": 0.05
            })
            
            result = await meta_agent.analyze_system_health()
            
            assert "overall_score" in result
            assert "timestamp" in result
            assert "agents" in result
            assert "task_queue" in result
            assert "database" in result
            assert "cli_tools" in result
            assert "performance" in result
            
            # Check agent analysis
            assert result["agents"]["total_agents"] == 3
            assert result["agents"]["active_count"] == 2
            assert result["agents"]["error_count"] == 1
            
            # Check task queue analysis
            assert result["task_queue"]["total_queued"] == 30
            assert result["task_queue"]["completed_tasks"] == 100
            
            # Check CLI tools
            assert result["cli_tools"]["tool_count"] == 1
            assert "claude" in result["cli_tools"]["available_tools"]
    
    @pytest.mark.asyncio
    async def test_analyze_system_health_with_errors(self, meta_agent):
        """Test system health analysis with component errors."""
        # Mock database error
        meta_agent.async_db_manager.get_active_agents.side_effect = Exception("DB Error")
        
        result = await meta_agent.analyze_system_health()
        
        assert result["overall_score"] == 0
        assert "error" in result
    
    @pytest.mark.asyncio
    async def test_analyze_agent_ecosystem(self, meta_agent):
        """Test agent ecosystem analysis."""
        # Mock agents with different statuses and types
        mock_agents = [
            MagicMock(status="active", type="meta"),
            MagicMock(status="active", type="architect"),
            MagicMock(status="idle", type="qa"),
            MagicMock(status="error", type="developer"),
        ]
        meta_agent.async_db_manager.get_active_agents.return_value = mock_agents
        
        result = await meta_agent._analyze_agent_ecosystem()
        
        assert result["total_agents"] == 4
        assert result["active_count"] == 2
        assert result["idle_count"] == 1
        assert result["error_count"] == 1
        assert "agent_types" in result
        assert result["agent_types"]["meta"]["count"] == 1
        assert result["agent_types"]["architect"]["count"] == 1
        assert result["health_score"] > 0  # Should calculate based on ratios
    
    @pytest.mark.asyncio
    async def test_analyze_task_queue_health(self, meta_agent):
        """Test task queue health analysis."""
        with patch("src.agents.meta_agent.task_queue") as mock_task_queue:
            mock_stats = MagicMock()
            mock_stats.queue_size_by_priority = {"high": 2, "medium": 8, "low": 5}
            mock_stats.completed_tasks = 150
            mock_stats.failed_tasks = 25
            mock_stats.processing_rate = 3.0
            mock_stats.average_wait_time = 20
            mock_task_queue.get_queue_stats.return_value = mock_stats
            
            result = await meta_agent._analyze_task_queue_health()
            
            assert result["total_queued"] == 15
            assert result["completed_tasks"] == 150
            assert result["failed_tasks"] == 25
            assert result["processing_rate"] == 3.0
            assert result["health_score"] > 0  # Should calculate success rate


class TestImprovementPlanning:
    """Test improvement task planning functionality."""
    
    @pytest.mark.asyncio
    async def test_plan_improvement_tasks_critical(self, meta_agent):
        """Test planning improvement tasks for critical issues."""
        analysis_results = {
            "overall_score": 25,  # Critical score
            "database": {"health_score": 20},
            "agents": {"error_count": 5, "active_count": 2}
        }
        
        # Mock task submission
        meta_agent._submit_improvement_task = AsyncMock(return_value="task-123")
        
        tasks = await meta_agent.plan_improvement_tasks(analysis_results)
        
        assert len(tasks) > 0
        assert any("Critical Database" in task.title for task in tasks)
        assert any("Failed Agents" in task.title for task in tasks)
        assert meta_agent.improvement_tasks_created > 0
    
    @pytest.mark.asyncio
    async def test_plan_improvement_tasks_performance(self, meta_agent):
        """Test planning improvement tasks for performance issues."""
        analysis_results = {
            "overall_score": 60,  # Performance score
            "performance": {"health_score": 50},
            "task_queue": {"total_queued": 75}
        }
        
        # Mock task submission
        meta_agent._submit_improvement_task = AsyncMock(return_value="task-456")
        
        tasks = await meta_agent.plan_improvement_tasks(analysis_results)
        
        assert len(tasks) > 0
        assert any("Optimize System Performance" in task.title for task in tasks)
        assert any("Task Queue Processing" in task.title for task in tasks)
    
    @pytest.mark.asyncio
    async def test_plan_improvement_tasks_enhancement(self, meta_agent):
        """Test planning improvement tasks for system enhancements."""
        analysis_results = {
            "overall_score": 85,  # Good score - enhancement opportunities
            "agents": {"health_score": 80, "agent_types": {"meta": {"count": 1}}}
        }
        
        # Mock task submission
        meta_agent._submit_improvement_task = AsyncMock(return_value="task-789")
        
        tasks = await meta_agent.plan_improvement_tasks(analysis_results)
        
        assert len(tasks) > 0
        # Should include missing essential agents
        assert any("Implement Architect Agent" in task.title for task in tasks)
        assert any("Implement Qa Agent" in task.title for task in tasks)
    
    @pytest.mark.asyncio
    async def test_plan_agent_improvements(self, meta_agent):
        """Test planning agent-specific improvements."""
        analysis_results = {
            "agents": {
                "agent_types": {
                    "meta": {"count": 1, "active": 1}
                    # Missing architect, qa, developer agents
                }
            }
        }
        
        tasks = await meta_agent._plan_agent_improvements(analysis_results)
        
        assert len(tasks) == 3  # Should create tasks for missing essential agents
        titles = [task.title for task in tasks]
        assert "Implement Architect Agent" in titles
        assert "Implement Qa Agent" in titles
        assert "Implement Developer Agent" in titles


class TestAgentCoordination:
    """Test agent coordination functionality."""
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_success(self, meta_agent):
        """Test successful agent coordination."""
        task = Task(
            title="Complex Development Task",
            description="A complex task requiring multiple agents",
            task_type="development",
            priority=TaskPriority.HIGH
        )
        
        # Mock coordination methods
        coordination_plan = {
            "requires_coordination": True,
            "required_capabilities": ["development", "testing"],
            "min_agents": 2
        }
        meta_agent._analyze_task_coordination_needs = AsyncMock(return_value=coordination_plan)
        meta_agent._find_suitable_agents = AsyncMock(return_value=["dev-agent", "qa-agent"])
        meta_agent._create_coordination_session = AsyncMock(return_value="session-123")
        
        result = await meta_agent.coordinate_agents(task)
        
        assert result is True
        assert meta_agent.coordination_sessions == 1
        assert "session-123" in meta_agent.coordination_active
        
        # Check coordination context stored
        meta_agent.store_context.assert_called()
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_no_coordination_needed(self, meta_agent):
        """Test coordination when task doesn't require it."""
        task = Task(
            title="Simple Task",
            description="A simple task",
            task_type="simple",
            priority=TaskPriority.LOW
        )
        
        coordination_plan = {"requires_coordination": False}
        meta_agent._analyze_task_coordination_needs = AsyncMock(return_value=coordination_plan)
        
        result = await meta_agent.coordinate_agents(task)
        
        assert result is True
        assert meta_agent.coordination_sessions == 0
    
    @pytest.mark.asyncio
    async def test_coordinate_agents_insufficient_agents(self, meta_agent):
        """Test coordination failure due to insufficient agents."""
        task = Task(
            title="Complex Task",
            description="Complex task",
            task_type="complex",
            priority=TaskPriority.HIGH
        )
        
        coordination_plan = {
            "requires_coordination": True,
            "required_capabilities": ["rare_capability"],
            "min_agents": 3
        }
        meta_agent._analyze_task_coordination_needs = AsyncMock(return_value=coordination_plan)
        meta_agent._find_suitable_agents = AsyncMock(return_value=["agent-1"])  # Only 1 agent
        
        result = await meta_agent.coordinate_agents(task)
        
        assert result is False
        assert meta_agent.coordination_sessions == 0


class TestPerformanceMonitoring:
    """Test performance monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_monitor_performance_success(self, meta_agent):
        """Test successful performance monitoring."""
        # Mock performance collection methods
        meta_agent._collect_agent_performance_metrics = AsyncMock(return_value={
            "avg_agent_response_time": 1.2,
            "agent_utilization": 0.75
        })
        meta_agent._collect_task_processing_metrics = AsyncMock(return_value={
            "tasks_per_minute": 8.5,
            "avg_task_duration": 45.0
        })
        meta_agent._collect_resource_metrics = AsyncMock(return_value={
            "cpu_usage": 0.65,
            "memory_usage": 0.55
        })
        meta_agent._collect_communication_metrics = AsyncMock(return_value={
            "message_latency": 0.1,
            "message_throughput": 50.0
        })
        meta_agent._identify_performance_bottlenecks = AsyncMock(return_value=[
            {"component": "database", "severity": "medium"}
        ])
        meta_agent._analyze_performance_trends = AsyncMock(return_value={
            "trend": "improving",
            "confidence": 0.8
        })
        
        result = await meta_agent.monitor_performance()
        
        assert "avg_agent_response_time" in result
        assert "tasks_per_minute" in result
        assert "cpu_usage" in result
        assert "message_latency" in result
        assert "bottlenecks" in result
        assert "trend_analysis" in result
        assert len(result["bottlenecks"]) == 1
    
    @pytest.mark.asyncio
    async def test_monitor_performance_with_errors(self, meta_agent):
        """Test performance monitoring with errors."""
        meta_agent._collect_agent_performance_metrics = AsyncMock(
            side_effect=Exception("Metrics collection failed")
        )
        
        result = await meta_agent.monitor_performance()
        
        assert "error" in result
        assert result["error"] == "Metrics collection failed"


class TestHealthScoreCalculation:
    """Test health score calculation logic."""
    
    def test_calculate_health_score_all_healthy(self, meta_agent):
        """Test health score calculation with all components healthy."""
        health_metrics = {
            "agents": {"health_score": 90},
            "task_queue": {"health_score": 85},
            "database": {"health_score": 95},
            "cli_tools": {"health_score": 100},
            "performance": {"health_score": 80}
        }
        
        score = meta_agent._calculate_health_score(health_metrics)
        
        assert 80 <= score <= 100  # Should be weighted average
        assert "key_issues" in health_metrics
        assert len(health_metrics["key_issues"]) == 0  # No issues
    
    def test_calculate_health_score_with_issues(self, meta_agent):
        """Test health score calculation with component issues."""
        health_metrics = {
            "agents": {"health_score": 30},  # Unhealthy
            "task_queue": {"health_score": 40},  # Unhealthy
            "database": {"health_score": 90},
            "cli_tools": {"health_score": 75},
            "performance": {"health_score": 60}
        }
        
        score = meta_agent._calculate_health_score(health_metrics)
        
        assert score < 70  # Should be lower due to issues
        assert "key_issues" in health_metrics
        assert "agents_unhealthy" in health_metrics["key_issues"]
        assert "task_queue_unhealthy" in health_metrics["key_issues"]
    
    def test_calculate_health_score_missing_components(self, meta_agent):
        """Test health score calculation with missing components."""
        health_metrics = {
            "agents": {"health_score": 80},
            # Missing other components
        }
        
        score = meta_agent._calculate_health_score(health_metrics)
        
        assert score >= 0  # Should handle missing components gracefully


class TestTaskCreation:
    """Test task creation functionality."""
    
    @pytest.mark.asyncio
    async def test_create_task(self, meta_agent):
        """Test task creation with proper metadata."""
        task = await meta_agent._create_task(
            title="Test Task",
            description="Test description",
            task_type="testing",
            priority=TaskPriority.HIGH
        )
        
        assert task.title == "Test Task"
        assert task.description == "Test description"
        assert task.task_type == "testing"
        assert task.priority == TaskPriority.HIGH
        assert task.payload["created_by"] == meta_agent.name
        assert task.payload["meta_analysis"] is True
        assert "created_at" in task.payload


class TestMainExecutionLoop:
    """Test the main execution loop behavior."""
    
    @pytest.mark.asyncio
    async def test_run_execution_cycle(self, meta_agent):
        """Test a single execution cycle of the main loop."""
        # Mock all the periodic methods
        meta_agent._perform_system_analysis = AsyncMock()
        meta_agent._monitor_performance = AsyncMock()
        meta_agent._review_coordination_opportunities = AsyncMock()
        meta_agent._process_improvement_pipeline = AsyncMock()
        meta_agent._handle_pending_collaborations = AsyncMock()
        
        # Set status to inactive after first cycle to stop the loop
        async def stop_after_first_cycle():
            await asyncio.sleep(0.1)  # Brief delay
            meta_agent.status = "inactive"
        
        stop_task = asyncio.create_task(stop_after_first_cycle())
        run_task = asyncio.create_task(meta_agent.run())
        
        # Wait for both tasks
        await asyncio.gather(stop_task, run_task, return_exceptions=True)
        
        # Verify that periodic methods were called
        meta_agent._process_improvement_pipeline.assert_called()
        meta_agent._handle_pending_collaborations.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])