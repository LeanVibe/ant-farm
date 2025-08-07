"""Tests for advanced agent coordination system."""

import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from src.core.agent_coordination import (
    AgentCapabilityMap,
    CollaborationContext,
    CollaborationCoordinator,
    CollaborationType,
    TaskDecomposer,
    TaskPhase,
)
from src.core.message_broker import Message, MessageBroker, MessageType


@pytest.fixture
async def mock_message_broker():
    """Create a mock message broker."""
    broker = AsyncMock(spec=MessageBroker)
    broker.send_message = AsyncMock(return_value="test-message-id")
    broker.start_listening = AsyncMock()
    return broker


@pytest.fixture
async def coordinator(mock_message_broker):
    """Create a coordination system for testing."""
    coordinator = CollaborationCoordinator(mock_message_broker)
    return coordinator


@pytest.fixture
def sample_agents():
    """Create sample agent capability maps."""
    return [
        AgentCapabilityMap(
            agent_name="architect-01",
            capabilities=["system_design", "architecture_planning", "analysis"],
            specializations=["system_design"],
            load_factor=0.3,
            availability=True,
        ),
        AgentCapabilityMap(
            agent_name="developer-01",
            capabilities=["code_generation", "testing", "debugging"],
            specializations=["code_generation"],
            load_factor=0.5,
            availability=True,
        ),
        AgentCapabilityMap(
            agent_name="qa-01",
            capabilities=["testing", "quality_assurance", "evaluation"],
            specializations=["testing"],
            load_factor=0.2,
            availability=True,
        ),
        AgentCapabilityMap(
            agent_name="devops-01",
            capabilities=["deployment", "infrastructure", "monitoring"],
            specializations=["deployment"],
            load_factor=0.4,
            availability=True,
        ),
    ]


class TestTaskDecomposer:
    """Test task decomposition strategies."""

    def setup_method(self):
        self.decomposer = TaskDecomposer()

    @pytest.mark.asyncio
    async def test_sequential_decomposition(self, sample_agents):
        """Test sequential task decomposition."""
        context = CollaborationContext(
            id="test-seq",
            title="Develop User Authentication",
            description="Implement complete user authentication system",
            collaboration_type=CollaborationType.SEQUENTIAL,
            phase=TaskPhase.PLANNING,
            coordinator_agent="architect-01",
            metadata={"required_capabilities": ["code_generation"]},
        )

        sub_tasks = await self.decomposer.decompose_task(context, sample_agents)

        assert len(sub_tasks) > 0

        # Check that tasks have proper dependencies
        task_ids = list(sub_tasks.keys())
        for i, task_id in enumerate(task_ids[1:], 1):
            depends_on = sub_tasks[task_id]["depends_on"]
            assert len(depends_on) == 1
            assert depends_on[0] == task_ids[i - 1]

    @pytest.mark.asyncio
    async def test_parallel_decomposition(self, sample_agents):
        """Test parallel task decomposition."""
        context = CollaborationContext(
            id="test-parallel",
            title="System Optimization",
            description="Optimize system performance",
            collaboration_type=CollaborationType.PARALLEL,
            phase=TaskPhase.PLANNING,
            coordinator_agent="architect-01",
            metadata={"task_type": "system_optimization"},
        )

        sub_tasks = await self.decomposer.decompose_task(context, sample_agents)

        assert len(sub_tasks) > 0

        # Check that parallel tasks have no dependencies
        for task in sub_tasks.values():
            assert task["depends_on"] == []

    @pytest.mark.asyncio
    async def test_consensus_decomposition(self, sample_agents):
        """Test consensus-based task decomposition."""
        context = CollaborationContext(
            id="test-consensus",
            title="Architecture Decision",
            description="Decide on system architecture",
            collaboration_type=CollaborationType.CONSENSUS,
            phase=TaskPhase.PLANNING,
            coordinator_agent="architect-01",
            metadata={"required_capabilities": ["system_design"]},
        )

        sub_tasks = await self.decomposer.decompose_task(context, sample_agents)

        assert len(sub_tasks) > 0

        # Should have proposal tasks and evaluation task
        proposal_tasks = [t for t in sub_tasks.keys() if "proposal" in t]
        evaluation_tasks = [t for t in sub_tasks.keys() if "evaluation" in t]

        assert len(proposal_tasks) > 0
        assert len(evaluation_tasks) == 1

    @pytest.mark.asyncio
    async def test_competitive_decomposition(self, sample_agents):
        """Test competitive task decomposition."""
        context = CollaborationContext(
            id="test-competitive",
            title="Algorithm Optimization",
            description="Find best algorithm implementation",
            collaboration_type=CollaborationType.COMPETITIVE,
            phase=TaskPhase.PLANNING,
            coordinator_agent="architect-01",
            metadata={"required_capabilities": ["code_generation"]},
        )

        sub_tasks = await self.decomposer.decompose_task(context, sample_agents)

        assert len(sub_tasks) > 0

        # Should have competing solution tasks and evaluation
        solution_tasks = [t for t in sub_tasks.keys() if "solution" in t]
        evaluation_tasks = [t for t in sub_tasks.keys() if "evaluation" in t]

        assert len(solution_tasks) > 0
        assert len(evaluation_tasks) == 1


class TestCollaborationCoordinator:
    """Test collaboration coordination."""

    @pytest.mark.asyncio
    async def test_start_collaboration(self, coordinator, sample_agents):
        """Test starting a new collaboration."""
        with patch.object(
            coordinator, "_get_available_agents", return_value=sample_agents
        ):
            collaboration_id = await coordinator.start_collaboration(
                title="Test Project",
                description="A test collaboration project",
                collaboration_type=CollaborationType.SEQUENTIAL,
                coordinator_agent="architect-01",
                required_capabilities=["code_generation"],
                priority=5,
            )

        assert collaboration_id is not None
        assert collaboration_id in coordinator.active_collaborations

        context = coordinator.active_collaborations[collaboration_id]
        assert context.title == "Test Project"
        assert context.collaboration_type == CollaborationType.SEQUENTIAL
        assert context.phase == TaskPhase.ASSIGNMENT
        assert len(context.participating_agents) > 0

    @pytest.mark.asyncio
    async def test_sub_task_completion(self, coordinator, sample_agents):
        """Test handling sub-task completion."""
        with patch.object(
            coordinator, "_get_available_agents", return_value=sample_agents
        ):
            collaboration_id = await coordinator.start_collaboration(
                title="Test Project",
                description="A test collaboration project",
                collaboration_type=CollaborationType.PARALLEL,
                coordinator_agent="architect-01",
                required_capabilities=["code_generation"],
                priority=5,
            )

        context = coordinator.active_collaborations[collaboration_id]
        task_ids = list(context.sub_tasks.keys())

        # Simulate task completion
        message = Message(
            id="test-msg",
            from_agent="developer-01",
            to_agent="coordination_system",
            topic="sub_task_completed",
            message_type=MessageType.NOTIFICATION,
            payload={
                "collaboration_id": collaboration_id,
                "task_id": task_ids[0],
                "result": {"success": True, "output": "Task completed"},
            },
            timestamp=time.time(),
        )

        result = await coordinator._handle_sub_task_completed(message)

        assert result["status"] == "acknowledged"
        assert task_ids[0] in context.results
        assert context.results[task_ids[0]]["result"]["success"] is True

    @pytest.mark.asyncio
    async def test_collaboration_completion(self, coordinator, sample_agents):
        """Test collaboration completion."""
        with patch.object(
            coordinator, "_get_available_agents", return_value=sample_agents
        ):
            collaboration_id = await coordinator.start_collaboration(
                title="Test Project",
                description="A test collaboration project",
                collaboration_type=CollaborationType.PARALLEL,
                coordinator_agent="architect-01",
                required_capabilities=["code_generation"],
                priority=5,
            )

        context = coordinator.active_collaborations[collaboration_id]

        # Complete all tasks
        for task_id in context.sub_tasks.keys():
            message = Message(
                id=f"test-msg-{task_id}",
                from_agent="test-agent",
                to_agent="coordination_system",
                topic="sub_task_completed",
                message_type=MessageType.NOTIFICATION,
                payload={
                    "collaboration_id": collaboration_id,
                    "task_id": task_id,
                    "result": {"success": True, "output": f"Task {task_id} completed"},
                },
                timestamp=time.time(),
            )

            await coordinator._handle_sub_task_completed(message)

        # Collaboration should be completed and removed
        assert collaboration_id not in coordinator.active_collaborations

    @pytest.mark.asyncio
    async def test_agent_unavailable_handling(self, coordinator, sample_agents):
        """Test handling when an agent becomes unavailable."""
        with patch.object(
            coordinator, "_get_available_agents", return_value=sample_agents
        ):
            collaboration_id = await coordinator.start_collaboration(
                title="Test Project",
                description="A test collaboration project",
                collaboration_type=CollaborationType.SEQUENTIAL,
                coordinator_agent="architect-01",
                required_capabilities=["code_generation"],
                priority=5,
            )

        # Simulate agent becoming unavailable
        await coordinator._handle_agent_unavailable("developer-01")

        # Check if tasks were reassigned or collaboration handled gracefully
        context = coordinator.active_collaborations.get(collaboration_id)
        if context:  # If collaboration still exists
            # Check that no tasks are still assigned to the unavailable agent
            for task in context.sub_tasks.values():
                assert task.get("assigned_agent") != "developer-01"

    @pytest.mark.asyncio
    async def test_collaboration_status(self, coordinator, sample_agents):
        """Test getting collaboration status."""
        with patch.object(
            coordinator, "_get_available_agents", return_value=sample_agents
        ):
            collaboration_id = await coordinator.start_collaboration(
                title="Test Project",
                description="A test collaboration project",
                collaboration_type=CollaborationType.SEQUENTIAL,
                coordinator_agent="architect-01",
                required_capabilities=["code_generation"],
                priority=5,
            )

        status = await coordinator.get_collaboration_status(collaboration_id)

        assert status["id"] == collaboration_id
        assert status["title"] == "Test Project"
        assert status["phase"] == TaskPhase.ASSIGNMENT.value
        assert status["collaboration_type"] == CollaborationType.SEQUENTIAL.value
        assert "progress" in status
        assert "total_tasks" in status
        assert "completed_tasks" in status

    @pytest.mark.asyncio
    async def test_collaboration_not_found(self, coordinator):
        """Test handling unknown collaboration ID."""
        status = await coordinator.get_collaboration_status("unknown-id")
        assert "error" in status
        assert status["error"] == "Collaboration not found"


class TestIntegration:
    """Integration tests for the coordination system."""

    @pytest.mark.asyncio
    async def test_end_to_end_collaboration(self, sample_agents):
        """Test complete collaboration workflow."""
        # Mock the message broker
        mock_broker = AsyncMock(spec=MessageBroker)
        mock_broker.send_message = AsyncMock(return_value="test-message-id")

        coordinator = CollaborationCoordinator(mock_broker)

        with patch.object(
            coordinator, "_get_available_agents", return_value=sample_agents
        ):
            # Start collaboration
            collaboration_id = await coordinator.start_collaboration(
                title="Full Stack Development",
                description="Build a complete web application",
                collaboration_type=CollaborationType.SEQUENTIAL,
                coordinator_agent="architect-01",
                required_capabilities=["code_generation", "testing"],
                deadline=datetime.now() + timedelta(hours=2),
                priority=3,
            )

            # Verify collaboration started
            assert collaboration_id in coordinator.active_collaborations
            context = coordinator.active_collaborations[collaboration_id]
            assert len(context.sub_tasks) > 0
            assert len(context.participating_agents) > 0

            # Simulate task completions
            completed_tasks = 0
            for task_id in context.sub_tasks.keys():
                message = Message(
                    id=f"completion-{task_id}",
                    from_agent=context.sub_tasks[task_id]["assigned_agent"],
                    to_agent="coordination_system",
                    topic="sub_task_completed",
                    message_type=MessageType.NOTIFICATION,
                    payload={
                        "collaboration_id": collaboration_id,
                        "task_id": task_id,
                        "result": {
                            "success": True,
                            "output": f"Completed {task_id}",
                            "artifacts": [f"{task_id}.py"],
                        },
                    },
                    timestamp=time.time(),
                )

                await coordinator._handle_sub_task_completed(message)
                completed_tasks += 1

            # Verify collaboration completed
            assert collaboration_id not in coordinator.active_collaborations

            # Check that completion messages were sent
            assert (
                mock_broker.send_message.call_count
                >= len(context.participating_agents) + 1
            )
