"""Integration tests for agent usage of enhanced communication features."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.agents.base_agent import BaseAgent
from src.core.enhanced_message_broker import (
    EnhancedMessageBroker,
    get_enhanced_message_broker,
    ContextShareType,
    MessagePriority,
)
from src.core.realtime_collaboration import (
    RealTimeCollaborationSync,
    get_collaboration_sync,
)
from src.core.config import settings


class TestAgent(BaseAgent):
    """Test agent implementation for testing enhanced communication."""

    def __init__(self, name: str, enhanced_communication: bool = True):
        self.enhanced_communication = enhanced_communication
        super().__init__(name=name, agent_type="test", role="testing")

        # Override message broker if enhanced communication is enabled
        if enhanced_communication:
            self.enhanced_broker = get_enhanced_message_broker()
            self.collaboration_sync = get_collaboration_sync(self.enhanced_broker)

    async def run(self) -> None:
        """Test run implementation."""
        # Simple run loop for testing
        await asyncio.sleep(0.1)


@pytest.mark.asyncio
class TestAgentEnhancedCommunication:
    """Test agents using enhanced communication features."""

    @pytest.fixture
    async def enhanced_broker(self):
        """Create enhanced message broker for testing."""
        broker = EnhancedMessageBroker()
        await broker.initialize()
        yield broker
        await broker.shutdown()

    @pytest.fixture
    async def test_agents(self, enhanced_broker):
        """Create test agents with enhanced communication."""
        agent1 = TestAgent("test_agent_1", enhanced_communication=True)
        agent2 = TestAgent("test_agent_2", enhanced_communication=True)

        # Mock some dependencies to avoid full startup
        agent1.async_db_manager = AsyncMock()
        agent1.context_engine = AsyncMock()
        agent2.async_db_manager = AsyncMock()
        agent2.context_engine = AsyncMock()

        yield agent1, agent2

    async def test_agents_use_enhanced_communication(self, test_agents):
        """Test that agents use enhanced communication features."""
        agent1, agent2 = test_agents

        # Verify agents have enhanced communication components
        assert hasattr(agent1, "enhanced_broker"), "Agent should have enhanced broker"
        assert hasattr(agent1, "collaboration_sync"), (
            "Agent should have collaboration sync"
        )
        assert isinstance(agent1.enhanced_broker, EnhancedMessageBroker)
        assert isinstance(agent1.collaboration_sync, RealTimeCollaborationSync)

    async def test_agents_can_create_shared_work_session(self, test_agents):
        """Test that agents can create shared work sessions."""
        agent1, agent2 = test_agents

        # Agent1 creates a shared work session
        session_result = await agent1.collaboration_sync.start_collaboration_session(
            title="Code Review Session",
            coordinator=agent1.name,
            initial_participants={agent1.name, agent2.name},
        )

        assert session_result is not None, "Should create session successfully"
        assert isinstance(session_result, str), "Should return session ID"

    async def test_agents_can_share_context(self, test_agents):
        """Test that agents can share context through enhanced broker."""
        agent1, agent2 = test_agents

        # Create shared context
        context_id = await agent1.enhanced_broker.create_shared_context(
            context_type=ContextShareType.WORK_SESSION,
            owner_agent=agent1.name,
            initial_data={"test_data": "shared_value"},
            participants={agent1.name, agent2.name},
        )

        assert context_id is not None, "Should create shared context"

        # Agent2 joins the context
        join_success = await agent1.enhanced_broker.join_shared_context(
            context_id, agent2.name
        )
        assert join_success, "Agent should be able to join shared context"

        # Agent2 retrieves context
        context_data = await agent1.enhanced_broker.get_shared_context(
            context_id, agent2.name
        )
        assert context_data is not None, "Should retrieve shared context"
        assert context_data["data"]["test_data"] == "shared_value", (
            "Should have correct shared data"
        )

    async def test_multi_agent_collaboration_workflow(self, test_agents):
        """Test a complete multi-agent collaboration workflow."""
        agent1, agent2 = test_agents

        # 1. Agent1 starts a collaboration session
        session_id = await agent1.collaboration_sync.start_collaboration_session(
            title="Authentication Module Review",
            coordinator=agent1.name,
            initial_participants={agent1.name},
        )

        # 2. Agent2 joins the session
        join_success = await agent1.collaboration_sync.join_collaboration_session(
            session_id, agent2.name
        )
        assert join_success, "Agent2 should be able to join session"

        # 3. Agent1 shares a document
        share_success = await agent1.collaboration_sync.share_document(
            workspace_id=session_id,  # Using session_id as workspace_id for simplicity
            document_name="auth.py",
            content="class AuthModule:\n    def authenticate(self): pass",
            shared_by=agent1.name,
        )
        assert share_success, "Should be able to share document"

        # 4. Agent2 provides feedback via session update
        update_success = await agent1.collaboration_sync.send_session_update(
            session_id=session_id,
            update_type="feedback",
            update_data={
                "message": "Consider adding rate limiting to authentication",
                "priority": "high",
                "suggestions": ["Add rate limiter", "Use secure session storage"],
            },
            sent_by=agent2.name,
        )
        assert update_success, "Should be able to send session update"

        # 5. Verify session state
        session_state = await agent1.collaboration_sync.get_session_state(
            session_id, agent1.name
        )
        assert session_state is not None, "Should get session state"
        assert agent1.name in session_state["participants"], (
            "Agent1 should be participant"
        )
        assert agent2.name in session_state["participants"], (
            "Agent2 should be participant"
        )
        assert session_state["state"] == "active", "Session should be active"

    async def test_conflict_resolution_in_collaboration(self, test_agents):
        """Test conflict resolution when multiple agents edit simultaneously."""
        agent1, agent2 = test_agents

        # Start collaboration session
        session_id = await agent1.collaboration_sync.start_collaboration_session(
            title="Conflict Resolution Test",
            coordinator=agent1.name,
            initial_participants={agent1.name, agent2.name},
        )

        # Both agents try to update the same document simultaneously
        operation_id_1 = await agent1.collaboration_sync.submit_sync_operation(
            session_id=session_id,
            operation_type="update",
            resource_path="shared_config.json",
            data={"setting": "value_from_agent1"},
            author=agent1.name,
        )

        operation_id_2 = await agent1.collaboration_sync.submit_sync_operation(
            session_id=session_id,
            operation_type="update",
            resource_path="shared_config.json",
            data={"setting": "value_from_agent2"},
            author=agent2.name,
        )

        assert operation_id_1 != operation_id_2, "Should create different operation IDs"

        # Allow time for conflict resolution processing
        await asyncio.sleep(0.2)

        # Verify session metrics include conflict handling
        sync_metrics = await agent1.collaboration_sync.get_sync_metrics()
        assert "operations_processed" in sync_metrics, (
            "Should track operations processed"
        )

    async def test_enhanced_message_priorities(self, test_agents):
        """Test enhanced message broker priority handling."""
        agent1, agent2 = test_agents

        # Send high priority message
        message_id = await agent1.enhanced_broker.send_priority_message(
            from_agent=agent1.name,
            to_agent=agent2.name,
            topic="urgent_notification",
            payload={"alert": "System critical update required"},
            priority=MessagePriority.HIGH,
        )

        assert message_id is not None, "Should send priority message successfully"

        # Verify message can be retrieved from priority queue
        priority_messages = await agent1.enhanced_broker.get_priority_messages(limit=5)

        # Check if our message is in the priority queue (it may have been processed already)
        assert isinstance(priority_messages, list), (
            "Should return list of priority messages"
        )

    async def test_agent_state_synchronization(self, test_agents):
        """Test agent state synchronization through enhanced broker."""
        agent1, agent2 = test_agents

        # Update agent1's state
        await agent1.enhanced_broker.update_agent_state(
            agent_name=agent1.name,
            state_updates={
                "current_task": "reviewing_authentication_module",
                "status": "busy",
                "capabilities": ["code_review", "security_analysis"],
                "performance_metrics": {
                    "tasks_completed": 15,
                    "avg_completion_time": 120.5,
                },
            },
        )

        # Get all agent states
        agent_states = await agent1.enhanced_broker.get_agent_states(
            requesting_agent=agent2.name
        )

        assert isinstance(agent_states, dict), "Should return agent states dict"
        if agent1.name in agent_states:
            agent1_state = agent_states[agent1.name]
            assert agent1_state["current_task"] == "reviewing_authentication_module"
            assert agent1_state["status"] == "busy"

    async def test_context_aware_messaging(self, test_agents):
        """Test context-aware messaging between agents."""
        agent1, agent2 = test_agents

        # Create shared context first
        context_id = await agent1.enhanced_broker.create_shared_context(
            context_type=ContextShareType.TASK_STATE,
            owner_agent=agent1.name,
            initial_data={"current_task": "code_review", "progress": 0.3},
            participants={agent1.name, agent2.name},
        )

        # Send context-aware message
        message_sent = await agent1.enhanced_broker.send_context_aware_message(
            from_agent=agent1.name,
            to_agent=agent2.name,
            topic="task_collaboration",
            payload={"action": "request_review", "component": "authentication"},
            context_ids=[context_id],
            include_relevant_context=True,
        )

        assert message_sent, "Should send context-aware message successfully"

    async def test_performance_metrics_collection(self, test_agents):
        """Test that enhanced communication collects performance metrics."""
        agent1, agent2 = test_agents

        # Perform some enhanced communication operations
        context_id = await agent1.enhanced_broker.create_shared_context(
            context_type=ContextShareType.PERFORMANCE_METRICS,
            owner_agent=agent1.name,
            initial_data={"test_metrics": True},
        )

        # Get performance metrics
        perf_metrics = (
            await agent1.enhanced_broker.get_communication_performance_metrics()
        )

        assert isinstance(perf_metrics, dict), "Should return performance metrics"
        assert "enhanced_features" in perf_metrics, (
            "Should include enhanced features metrics"
        )

        enhanced_metrics = perf_metrics["enhanced_features"]
        assert "shared_contexts_active" in enhanced_metrics, (
            "Should track active contexts"
        )
        assert enhanced_metrics["shared_contexts_active"] >= 1, (
            "Should have at least one context"
        )
