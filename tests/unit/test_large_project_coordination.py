"""Tests for large project coordination system."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.collaboration.large_project_coordination import (
    LargeProjectCoordinator,
    ProjectPhase,
    ProjectScale,
    ProjectWorkspace,
    ResourcePool,
    ResourceType,
    TaskDependencyGraph,
    get_large_project_coordinator,
)


class TestTaskDependencyGraph:
    """Test the task dependency graph functionality."""

    def test_add_task_without_dependencies(self):
        """Test adding a task without dependencies."""
        graph = TaskDependencyGraph()
        graph.add_task("task1", metadata={"type": "development"})

        assert "task1" in graph.dependencies
        assert graph.dependencies["task1"] == set()
        assert graph.task_metadata["task1"]["type"] == "development"

    def test_add_task_with_dependencies(self):
        """Test adding a task with dependencies."""
        graph = TaskDependencyGraph()
        graph.add_task("task1")
        graph.add_task("task2", dependencies=["task1"])

        assert graph.dependencies["task2"] == {"task1"}
        assert graph.reverse_dependencies["task1"] == {"task2"}

    def test_get_ready_tasks(self):
        """Test getting tasks ready for execution."""
        graph = TaskDependencyGraph()
        graph.add_task("task1")
        graph.add_task("task2", dependencies=["task1"])
        graph.add_task("task3", dependencies=["task1", "task2"])

        # Initially, only task1 is ready
        ready = graph.get_ready_tasks(set())
        assert ready == ["task1"]

        # After task1 is completed, task2 is ready
        ready = graph.get_ready_tasks({"task1"})
        assert ready == ["task2"]

        # After both task1 and task2 are completed, task3 is ready
        ready = graph.get_ready_tasks({"task1", "task2"})
        assert ready == ["task3"]

    def test_critical_path_calculation(self):
        """Test critical path calculation."""
        graph = TaskDependencyGraph()
        graph.add_task("task1")
        graph.add_task("task2", dependencies=["task1"])
        graph.add_task("task3", dependencies=["task2"])

        critical_path = graph.get_critical_path()
        assert len(critical_path) >= 2  # Should include multiple tasks in sequence


class TestResourcePool:
    """Test resource pool management."""

    def test_resource_allocation(self):
        """Test basic resource allocation."""
        pool = ResourcePool(ResourceType.CPU, 100.0, 100.0)

        # Successful allocation
        success = pool.allocate("agent1", 30.0)
        assert success is True
        assert pool.available_capacity == 70.0
        assert pool.allocations["agent1"] == 30.0

    def test_resource_allocation_insufficient(self):
        """Test allocation when insufficient resources."""
        pool = ResourcePool(ResourceType.CPU, 100.0, 50.0)

        # Failed allocation
        success = pool.allocate("agent1", 60.0)
        assert success is False
        assert pool.available_capacity == 50.0
        assert "agent1" not in pool.allocations

    def test_resource_deallocation(self):
        """Test resource deallocation."""
        pool = ResourcePool(ResourceType.CPU, 100.0, 100.0)

        # Allocate then deallocate
        pool.allocate("agent1", 30.0)
        freed = pool.deallocate("agent1", 20.0)

        assert freed == 20.0
        assert pool.available_capacity == 90.0
        assert pool.allocations["agent1"] == 10.0

    def test_complete_deallocation(self):
        """Test complete resource deallocation."""
        pool = ResourcePool(ResourceType.CPU, 100.0, 100.0)

        pool.allocate("agent1", 30.0)
        freed = pool.deallocate("agent1")  # Deallocate all

        assert freed == 30.0
        assert pool.available_capacity == 100.0
        assert "agent1" not in pool.allocations


class TestLargeProjectCoordinator:
    """Test the large project coordinator."""

    @pytest.fixture
    def mock_base_coordinator(self):
        """Create a mock base coordinator."""
        return MagicMock()

    @pytest.fixture
    def coordinator(self, mock_base_coordinator):
        """Create a test coordinator instance."""
        return LargeProjectCoordinator(mock_base_coordinator)

    @pytest.mark.asyncio
    async def test_create_project_workspace(self, coordinator):
        """Test creating a project workspace."""
        with patch(
            "src.core.collaboration.large_project_coordination.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            project_id = await coordinator.create_project_workspace(
                name="Test Project",
                description="A test project",
                scale=ProjectScale.MEDIUM,
                lead_agent="agent1",
            )

            assert project_id in coordinator.active_projects
            workspace = coordinator.active_projects[project_id]
            assert workspace.name == "Test Project"
            assert workspace.scale == ProjectScale.MEDIUM
            assert workspace.lead_agent == "agent1"
            assert project_id in coordinator.dependency_graphs

            # Verify message was published
            mock_broker.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_join_project(self, coordinator):
        """Test agent joining a project."""
        # Create project first
        with patch(
            "src.core.collaboration.large_project_coordination.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            project_id = await coordinator.create_project_workspace(
                name="Test Project",
                description="A test project",
                scale=ProjectScale.SMALL,
                lead_agent="lead_agent",
            )

            # Join project
            success = await coordinator.join_project(
                project_id, "agent2", ["developer"]
            )

            assert success is True
            workspace = coordinator.active_projects[project_id]
            assert "agent2" in workspace.participating_agents
            assert workspace.agent_roles["agent2"] == ["developer"]

            # Verify message was published
            assert mock_broker.publish.call_count == 2  # Create + join

    @pytest.mark.asyncio
    async def test_join_nonexistent_project(self, coordinator):
        """Test joining a non-existent project."""
        success = await coordinator.join_project("nonexistent", "agent1")
        assert success is False

    @pytest.mark.asyncio
    async def test_decompose_large_task(self, coordinator):
        """Test large task decomposition."""
        with patch(
            "src.core.collaboration.large_project_coordination.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            # Create project and add agents
            project_id = await coordinator.create_project_workspace(
                name="Test Project",
                description="A test project",
                scale=ProjectScale.LARGE,
                lead_agent="lead_agent",
            )

            await coordinator.join_project(project_id, "dev_agent", ["developer"])
            await coordinator.join_project(project_id, "qa_agent", ["qa"])

            # Decompose a complex task
            result = await coordinator.decompose_large_task(
                project_id=project_id,
                task_description="Implement user authentication system",
                estimated_complexity=8,
            )

            assert "task_id" in result
            assert "sub_tasks" in result
            assert "assignments" in result
            assert len(result["sub_tasks"]) > 0

            # Verify tasks were added to dependency graph
            dependency_graph = coordinator.dependency_graphs[project_id]
            assert len(dependency_graph.dependencies) > 0

    @pytest.mark.asyncio
    async def test_monitor_project_progress(self, coordinator):
        """Test project progress monitoring."""
        with patch(
            "src.core.collaboration.large_project_coordination.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            project_id = await coordinator.create_project_workspace(
                name="Test Project",
                description="A test project",
                scale=ProjectScale.MEDIUM,
                lead_agent="lead_agent",
            )

            # Add some tasks to the dependency graph
            dependency_graph = coordinator.dependency_graphs[project_id]
            dependency_graph.add_task("task1")
            dependency_graph.add_task("task2", dependencies=["task1"])

            # Mark one task as completed
            workspace = coordinator.active_projects[project_id]
            workspace.completed_tasks.add("task1")

            progress = await coordinator.monitor_project_progress(project_id)

            assert "completion_percentage" in progress
            assert "total_tasks" in progress
            assert progress["total_tasks"] == 2
            assert progress["completed_tasks"] == 1
            assert progress["completion_percentage"] == 50.0

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, coordinator):
        """Test conflict resolution handling."""
        with patch(
            "src.core.collaboration.large_project_coordination.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            project_id = await coordinator.create_project_workspace(
                name="Test Project",
                description="A test project",
                scale=ProjectScale.LARGE,
                lead_agent="lead_agent",
            )

            result = await coordinator.handle_conflict_resolution(
                project_id=project_id,
                conflict_type="merge_conflict",
                involved_agents=["agent1", "agent2"],
                context={"file": "test.py", "lines": [10, 15]},
            )

            assert "conflict_id" in result
            assert "resolution_strategy" in result
            assert result["involved_agents"] == ["agent1", "agent2"]
            assert coordinator.coordination_metrics["conflict_resolutions"] == 1

    @pytest.mark.asyncio
    async def test_get_project_status(self, coordinator):
        """Test getting comprehensive project status."""
        with patch(
            "src.core.collaboration.large_project_coordination.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            project_id = await coordinator.create_project_workspace(
                name="Test Project",
                description="A test project",
                scale=ProjectScale.LARGE,
                lead_agent="lead_agent",
            )

            await coordinator.join_project(project_id, "agent1", ["developer"])

            status = await coordinator.get_project_status(project_id)

            assert "workspace" in status
            assert "progress" in status
            assert "resource_pools" in status
            assert "coordination_metrics" in status
            assert status["workspace"]["name"] == "Test Project"
            assert status["workspace"]["scale"] == "large"
            assert len(status["workspace"]["participating_agents"]) == 1


@pytest.mark.asyncio
async def test_get_large_project_coordinator():
    """Test the global coordinator getter."""
    with patch(
        "src.core.collaboration.large_project_coordination.CollaborationCoordinator"
    ) as mock_collab:
        with patch("src.core.collaboration.large_project_coordination.message_broker"):
            coordinator1 = await get_large_project_coordinator()
            coordinator2 = await get_large_project_coordinator()

            # Should return the same instance (singleton pattern)
            assert coordinator1 is coordinator2
            assert isinstance(coordinator1, LargeProjectCoordinator)


class TestProjectWorkspace:
    """Test the ProjectWorkspace dataclass."""

    def test_workspace_creation(self):
        """Test basic workspace creation."""
        workspace = ProjectWorkspace(
            id="test_project",
            name="Test Project",
            description="A test workspace",
            scale=ProjectScale.MEDIUM,
            phase=ProjectPhase.PLANNING,
            root_path=Path("/tmp/test"),
            lead_agent="lead_agent",
        )

        assert workspace.id == "test_project"
        assert workspace.name == "Test Project"
        assert workspace.scale == ProjectScale.MEDIUM
        assert workspace.phase == ProjectPhase.PLANNING
        assert workspace.lead_agent == "lead_agent"
        assert len(workspace.participating_agents) == 0
        assert len(workspace.task_graph) == 0


class TestIntegration:
    """Integration tests for large project coordination."""

    @pytest.mark.asyncio
    async def test_full_project_lifecycle(self):
        """Test a complete project lifecycle."""
        with patch(
            "src.core.collaboration.large_project_coordination.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            # Create coordinator
            base_coordinator = MagicMock()
            coordinator = LargeProjectCoordinator(base_coordinator)

            # 1. Create project
            project_id = await coordinator.create_project_workspace(
                name="Integration Test Project",
                description="Full lifecycle test",
                scale=ProjectScale.LARGE,
                lead_agent="lead_agent",
            )

            # 2. Add multiple agents
            await coordinator.join_project(project_id, "architect", ["architect"])
            await coordinator.join_project(project_id, "developer1", ["developer"])
            await coordinator.join_project(project_id, "developer2", ["developer"])
            await coordinator.join_project(project_id, "qa_agent", ["qa"])

            # 3. Decompose a large task
            decomposition = await coordinator.decompose_large_task(
                project_id=project_id,
                task_description="Build complete e-commerce platform",
                estimated_complexity=9,
            )

            # 4. Monitor progress
            progress = await coordinator.monitor_project_progress(project_id)

            # 5. Handle a conflict
            conflict_resolution = await coordinator.handle_conflict_resolution(
                project_id=project_id,
                conflict_type="resource_contention",
                involved_agents=["developer1", "developer2"],
                context={"resource": "database_connection"},
            )

            # 6. Get final status
            final_status = await coordinator.get_project_status(project_id)

            # Verify the full lifecycle worked
            assert len(decomposition["sub_tasks"]) > 0
            assert progress["participating_agents"] == 4
            assert "conflict_id" in conflict_resolution
            assert final_status["workspace"]["participating_agents"] == [
                "architect",
                "developer1",
                "developer2",
                "qa_agent",
            ]

            # Verify resource allocation worked
            workspace = coordinator.active_projects[project_id]
            assert len(workspace.resource_allocations) == 4  # All agents got resources
