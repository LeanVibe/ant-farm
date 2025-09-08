"""Tests for enhanced multi-agent collaboration functionality."""

import asyncio
import time
from unittest.mock import AsyncMock, patch

import pytest

from src.core.collaboration.enhanced_pair_programming import (
    CollaborationMode,
    ContextShareType,
    EnhancedAIPairProgramming,
    PairProgrammingSession,
    SessionPhase,
)


@pytest.fixture
def mock_pair_session():
    """Create a mock pair programming session."""
    from unittest.mock import MagicMock

    # Create mock agents
    developer_agent = MagicMock()
    developer_agent.name = "developer"
    qa_agent = MagicMock()
    qa_agent.name = "qa"

    return PairProgrammingSession(
        session_id="test-session",
        developer_agent=developer_agent,
        qa_agent=qa_agent,
    )


@pytest.fixture
async def enhanced_collaboration(mock_pair_session):
    """Create enhanced collaboration instance."""
    # Mock get_context_engine to avoid database dependency
    with patch('src.core.collaboration.enhanced_pair_programming.get_context_engine') as mock_get_context:
        mock_context_engine = AsyncMock()
        mock_get_context.return_value = mock_context_engine
        
        collaboration = EnhancedAIPairProgramming(mock_pair_session)
        await collaboration.initialize()
        return collaboration


class TestMultiAgentCollaboration:
    """Test multi-agent collaboration features."""

    @pytest.mark.asyncio
    async def test_start_multi_agent_session(self, enhanced_collaboration):
        """Test starting a multi-agent collaboration session."""
        participants = ["agent1", "agent2", "agent3"]
        project_context = {"project": "test", "version": "1.0"}

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.MOB_PROGRAMMING,
            project_context=project_context,
            task_description="Implement new feature",
        )

        assert session_id is not None
        assert session_id in enhanced_collaboration.active_sessions

        session = enhanced_collaboration.active_sessions[session_id]
        assert session.participants == participants
        assert session.mode == CollaborationMode.MOB_PROGRAMMING
        assert session.project_context == project_context
        assert session.current_driver == "agent1"  # First participant as initial driver

    @pytest.mark.asyncio
    async def test_multi_agent_session_requires_minimum_participants(self, enhanced_collaboration):
        """Test that multi-agent session requires at least 2 participants."""
        with pytest.raises(ValueError, match="requires at least 2 participants"):
            await enhanced_collaboration.start_multi_agent_session(
                participants=["agent1"],
                mode=CollaborationMode.PARALLEL_CODING,
                project_context={},
                task_description="Test",
            )

    @pytest.mark.asyncio
    async def test_driver_rotation_in_mob_programming(self, enhanced_collaboration):
        """Test driver rotation in mob programming mode."""
        participants = ["agent1", "agent2", "agent3"]

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.MOB_PROGRAMMING,
            project_context={},
            task_description="Test task",
        )

        session = enhanced_collaboration.active_sessions[session_id]
        initial_driver = session.current_driver

        # Manually trigger driver rotation
        await enhanced_collaboration._rotate_driver(session)

        # Driver should have changed
        assert session.current_driver != initial_driver
        assert session.current_driver in participants

    @pytest.mark.asyncio
    async def test_parallel_coding_mode_no_single_driver(self, enhanced_collaboration):
        """Test that parallel coding mode has no single driver."""
        participants = ["agent1", "agent2", "agent3"]

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.PARALLEL_CODING,
            project_context={},
            task_description="Test task",
        )

        session = enhanced_collaboration.active_sessions[session_id]
        assert session.current_driver is None  # No single driver in parallel mode

    @pytest.mark.asyncio
    async def test_context_sharing_between_agents(self, enhanced_collaboration):
        """Test context sharing between multiple agents."""
        participants = ["agent1", "agent2", "agent3"]

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.PARALLEL_CODING,
            project_context={},
            task_description="Test task",
        )

        # Share context
        test_content = {"pattern": "async def", "description": "Async function pattern"}
        await enhanced_collaboration.share_context(
            session_id=session_id,
            context_type=ContextShareType.CODE_PATTERNS,
            content=test_content,
            source_agent="agent1",
        )

        session = enhanced_collaboration.active_sessions[session_id]
        assert len(session.shared_contexts) == 2  # Initial project context + shared context
        assert session.shared_contexts[1].content == test_content
        assert session.shared_contexts[1].source_agent == "agent1"
        assert session.shared_contexts[1].context_type == ContextShareType.CODE_PATTERNS

    @pytest.mark.asyncio
    async def test_context_relevance_calculation(self, enhanced_collaboration):
        """Test context relevance score calculation."""
        participants = ["agent1", "agent2"]

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.PARALLEL_CODING,
            project_context={},
            task_description="Test task",
        )

        session = enhanced_collaboration.active_sessions[session_id]

        # Test code patterns relevance
        relevance = await enhanced_collaboration._calculate_context_relevance(
            session, ContextShareType.CODE_PATTERNS, {"pattern": "test"}
        )
        assert relevance == 0.9  # High relevance for code patterns

        # Test lower relevance context
        relevance = await enhanced_collaboration._calculate_context_relevance(
            session, ContextShareType.PROJECT_HISTORY, {"history": "old"}
        )
        assert relevance == 0.5  # Lower relevance

    @pytest.mark.asyncio
    async def test_session_status_retrieval(self, enhanced_collaboration):
        """Test retrieving session status."""
        participants = ["agent1", "agent2", "agent3"]

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.MOB_PROGRAMMING,
            project_context={"project": "test"},
            task_description="Test task",
        )

        status = await enhanced_collaboration.get_session_status(session_id)

        assert status["session_id"] == session_id
        assert status["participants"] == participants
        assert status["mode"] == "mob_programming"
        assert status["current_driver"] == "agent1"
        assert status["context_exchanges"] == 0
        assert "duration" in status

    @pytest.mark.asyncio
    async def test_session_termination(self, enhanced_collaboration):
        """Test session termination."""
        participants = ["agent1", "agent2"]

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.PARALLEL_CODING,
            project_context={},
            task_description="Test task",
        )

        # Verify session exists
        assert session_id in enhanced_collaboration.active_sessions

        # End session
        await enhanced_collaboration.end_session(session_id)

        # Verify session is removed
        assert session_id not in enhanced_collaboration.active_sessions

    @pytest.mark.asyncio
    async def test_context_expiry_handling(self, enhanced_collaboration):
        """Test that expired contexts are cleaned up."""
        participants = ["agent1", "agent2"]

        session_id = await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.PARALLEL_CODING,
            project_context={},
            task_description="Test task",
        )

        session = enhanced_collaboration.active_sessions[session_id]

        # Add context with short expiry
        expired_context = await enhanced_collaboration.share_context(
            session_id=session_id,
            context_type=ContextShareType.CODE_PATTERNS,
            content={"pattern": "test"},
            source_agent="agent1",
        )

        # Manually set expiry to past time
        session.shared_contexts[-1].expiry_time = time.time() - 1

        # Trigger context synchronization
        await enhanced_collaboration._synchronize_contexts(session)

        # Expired context should be removed
        assert len(session.shared_contexts) == 1  # Only initial project context remains

    @pytest.mark.asyncio
    async def test_collaboration_metrics_tracking(self, enhanced_collaboration):
        """Test that collaboration metrics are properly tracked."""
        initial_sessions = enhanced_collaboration.collaboration_metrics["sessions_created"]

        participants = ["agent1", "agent2", "agent3"]

        await enhanced_collaboration.start_multi_agent_session(
            participants=participants,
            mode=CollaborationMode.PARALLEL_CODING,
            project_context={},
            task_description="Test task",
        )

        # Check metrics updated
        assert enhanced_collaboration.collaboration_metrics["sessions_created"] == initial_sessions + 1

        # Share context and check metrics
        initial_contexts = enhanced_collaboration.collaboration_metrics["contexts_shared"]

        await enhanced_collaboration.share_context(
            session_id=list(enhanced_collaboration.active_sessions.keys())[0],
            context_type=ContextShareType.CODE_PATTERNS,
            content={"pattern": "test"},
            source_agent="agent1",
        )

        assert enhanced_collaboration.collaboration_metrics["contexts_shared"] == initial_contexts + 1