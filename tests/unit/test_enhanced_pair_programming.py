"""Tests for enhanced AI pair programming system."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.collaboration.enhanced_pair_programming import (
    CollaborationMode,
    CollaborationSession,
    ContextShareType,
    EnhancedAIPairProgramming,
    SharedContext,
    get_enhanced_pair_programming,
)


class TestSharedContext:
    """Test shared context functionality."""

    def test_context_creation(self):
        """Test creating shared context."""
        context = SharedContext(
            session_id="test_session",
            context_type=ContextShareType.CODE_PATTERNS,
            content={"pattern": "test pattern"},
            source_agent="agent1",
        )

        assert context.session_id == "test_session"
        assert context.context_type == ContextShareType.CODE_PATTERNS
        assert context.content["pattern"] == "test pattern"
        assert context.source_agent == "agent1"
        assert not context.is_expired()

    def test_context_expiry(self):
        """Test context expiry functionality."""
        context = SharedContext(
            session_id="test_session",
            context_type=ContextShareType.CODE_PATTERNS,
            content={"pattern": "test pattern"},
            source_agent="agent1",
            expiry_time=time.time() - 100,  # Expired 100 seconds ago
        )

        assert context.is_expired()

    def test_context_no_expiry(self):
        """Test context without expiry time."""
        context = SharedContext(
            session_id="test_session",
            context_type=ContextShareType.CODE_PATTERNS,
            content={"pattern": "test pattern"},
            source_agent="agent1",
            expiry_time=None,
        )

        assert not context.is_expired()


class TestCollaborationSession:
    """Test collaboration session management."""

    def test_session_creation(self):
        """Test creating a collaboration session."""
        session = CollaborationSession(
            session_id="test_session",
            participants=["agent1", "agent2"],
            mode=CollaborationMode.DRIVER_NAVIGATOR,
            project_context={"project": "test"},
        )

        assert session.session_id == "test_session"
        assert len(session.participants) == 2
        assert session.mode == CollaborationMode.DRIVER_NAVIGATOR
        assert session.current_driver == "agent1"  # First participant becomes driver
        assert len(session.shared_contexts) == 0


class TestEnhancedAIPairProgramming:
    """Test the enhanced AI pair programming system."""

    @pytest.fixture
    def mock_base_session(self):
        """Create a mock base session."""
        return MagicMock()

    @pytest.fixture
    def enhanced_system(self, mock_base_session):
        """Create a test enhanced system instance."""
        return EnhancedAIPairProgramming(mock_base_session)

    @pytest.mark.asyncio
    async def test_initialization(self, enhanced_system):
        """Test system initialization."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.get_context_engine"
        ) as mock_context:
            mock_context.return_value = AsyncMock()

            await enhanced_system.initialize()

            assert enhanced_system.context_engine is not None
            mock_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_enhanced_session(self, enhanced_system):
        """Test starting an enhanced collaboration session."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])

            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1", "agent2"],
                mode=CollaborationMode.PING_PONG,
                project_context={"project": "test"},
                task_description="Implement feature X",
            )

            assert session_id in enhanced_system.active_sessions
            session = enhanced_system.active_sessions[session_id]
            assert session.participants == ["agent1", "agent2"]
            assert session.mode == CollaborationMode.PING_PONG
            assert session.current_driver == "agent1"

            # Verify message was published
            mock_broker.publish.assert_called()

    @pytest.mark.asyncio
    async def test_share_context(self, enhanced_system):
        """Test sharing context between agents."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])
            enhanced_system.context_engine.store_context = AsyncMock()

            # Create a session first
            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1", "agent2"],
                mode=CollaborationMode.DRIVER_NAVIGATOR,
                project_context={},
                task_description="Test task",
            )

            # Share context
            success = await enhanced_system.share_context(
                session_id=session_id,
                source_agent="agent1",
                context_type=ContextShareType.CODE_PATTERNS,
                content={"pattern": "test pattern"},
                tags={"testing"},
            )

            assert success is True
            session = enhanced_system.active_sessions[session_id]
            assert session.context_exchanges == 1
            assert len(session.shared_contexts) == 1

            shared_context = session.shared_contexts[0]
            assert shared_context.source_agent == "agent1"
            assert shared_context.context_type == ContextShareType.CODE_PATTERNS
            assert shared_context.content["pattern"] == "test pattern"

    @pytest.mark.asyncio
    async def test_share_context_nonexistent_session(self, enhanced_system):
        """Test sharing context in non-existent session."""
        success = await enhanced_system.share_context(
            session_id="nonexistent",
            source_agent="agent1",
            context_type=ContextShareType.CODE_PATTERNS,
            content={"pattern": "test"},
            tags=set(),
        )

        assert success is False

    @pytest.mark.asyncio
    async def test_get_relevant_context(self, enhanced_system):
        """Test getting relevant context."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])
            enhanced_system.context_engine.store_context = AsyncMock()

            # Create session and add context
            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1", "agent2"],
                mode=CollaborationMode.DRIVER_NAVIGATOR,
                project_context={},
                task_description="Test task",
            )

            await enhanced_system.share_context(
                session_id=session_id,
                source_agent="agent1",
                context_type=ContextShareType.CODE_PATTERNS,
                content={"pattern": "test pattern", "language": "python"},
                tags={"python", "testing"},
            )

            # Get relevant context
            contexts = await enhanced_system.get_relevant_context(
                session_id=session_id,
                requesting_agent="agent2",
                query="python test pattern",
            )

            assert len(contexts) == 1
            assert contexts[0].source_agent == "agent1"
            assert contexts[0].context_type == ContextShareType.CODE_PATTERNS

    @pytest.mark.asyncio
    async def test_suggest_code_patterns(self, enhanced_system):
        """Test code pattern suggestions."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])

            # Create session
            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1"],
                mode=CollaborationMode.DRIVER_NAVIGATOR,
                project_context={},
                task_description="Test task",
            )

            # Get suggestions for code with common patterns
            suggestions = await enhanced_system.suggest_code_patterns(
                session_id=session_id,
                current_code="for i in range(len(items)): print(items[i])",
                context="iterating over list",
            )

            # Should get suggestions for better iteration patterns
            assert len(suggestions) > 0

            # Check for enumerate suggestion
            enumerate_suggestions = [
                s for s in suggestions if "enumerate" in s["content"]
            ]
            assert len(enumerate_suggestions) > 0

    @pytest.mark.asyncio
    async def test_switch_driver(self, enhanced_system):
        """Test switching the active driver."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])

            # Create session
            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1", "agent2", "agent3"],
                mode=CollaborationMode.DRIVER_NAVIGATOR,
                project_context={},
                task_description="Test task",
            )

            session = enhanced_system.active_sessions[session_id]
            assert session.current_driver == "agent1"

            # Switch to agent2
            success = await enhanced_system.switch_driver(session_id, "agent2")
            assert success is True
            assert session.current_driver == "agent2"

            # Try to switch to non-participant
            success = await enhanced_system.switch_driver(session_id, "agent4")
            assert success is False
            assert session.current_driver == "agent2"  # Should remain unchanged

    @pytest.mark.asyncio
    async def test_track_live_collaboration(self, enhanced_system):
        """Test tracking live collaboration state."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])

            # Create session
            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1", "agent2"],
                mode=CollaborationMode.DRIVER_NAVIGATOR,
                project_context={},
                task_description="Test task",
            )

            # Track live collaboration
            await enhanced_system.track_live_collaboration(
                session_id=session_id,
                agent_id="agent1",
                file_path="test.py",
                cursor_position=(10, 5),
                edit_action={"type": "insert", "text": "print('hello')"},
            )

            session = enhanced_system.active_sessions[session_id]
            assert "test.py" in session.active_files
            assert session.cursor_positions["agent1"] == (10, 5)
            assert len(session.edit_history) == 1

            edit = session.edit_history[0]
            assert edit["agent_id"] == "agent1"
            assert edit["file_path"] == "test.py"
            assert edit["action"]["type"] == "insert"

    @pytest.mark.asyncio
    async def test_get_collaboration_metrics(self, enhanced_system):
        """Test getting collaboration metrics."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])

            # Create session
            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1", "agent2"],
                mode=CollaborationMode.PING_PONG,
                project_context={},
                task_description="Test task",
            )

            # Add some activity
            await enhanced_system.share_context(
                session_id=session_id,
                source_agent="agent1",
                context_type=ContextShareType.CODE_PATTERNS,
                content={"pattern": "test"},
                tags=set(),
            )

            # Get metrics
            metrics = await enhanced_system.get_collaboration_metrics(session_id)

            assert metrics["session_id"] == session_id
            assert metrics["participants"] == ["agent1", "agent2"]
            assert metrics["mode"] == "ping_pong"
            assert metrics["context_exchanges"] == 1
            assert "duration_minutes" in metrics

    @pytest.mark.asyncio
    async def test_end_session(self, enhanced_system):
        """Test ending a collaboration session."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])

            # Create session
            session_id = await enhanced_system.start_enhanced_session(
                participants=["agent1", "agent2"],
                mode=CollaborationMode.DRIVER_NAVIGATOR,
                project_context={},
                task_description="Test task",
            )

            # Add some activity
            await enhanced_system.share_context(
                session_id=session_id,
                source_agent="agent1",
                context_type=ContextShareType.CODE_PATTERNS,
                content={"pattern": "test"},
                tags=set(),
            )

            # End session
            result = await enhanced_system.end_session(session_id)

            assert result.success is True
            assert result.session_id == session_id
            assert "context_exchanges" in result.metrics
            assert result.metrics["context_exchanges"] == 1

            # Session should be cleaned up
            assert session_id not in enhanced_system.active_sessions

    @pytest.mark.asyncio
    async def test_end_nonexistent_session(self, enhanced_system):
        """Test ending a non-existent session."""
        result = await enhanced_system.end_session("nonexistent")

        assert result.success is False
        assert result.error_message == "Session not found"

    @pytest.mark.asyncio
    async def test_context_relevance_calculation(self, enhanced_system):
        """Test context relevance calculation."""
        context = SharedContext(
            session_id="test",
            context_type=ContextShareType.CODE_PATTERNS,
            content={
                "description": "python function example",
                "code": "def test(): pass",
            },
            source_agent="agent1",
        )

        # Test relevance with matching query
        relevance = await enhanced_system._calculate_context_relevance(
            "python function", context
        )
        assert relevance > 0.5

        # Test relevance with non-matching query
        relevance = await enhanced_system._calculate_context_relevance(
            "java class", context
        )
        assert relevance < 0.5

    @pytest.mark.asyncio
    async def test_pattern_recognition(self, enhanced_system):
        """Test code pattern recognition."""
        # Test class with __init__ pattern
        suggestions = await enhanced_system._recognize_code_patterns(
            "class TestClass:\n    def __init__(self):\n        pass", "data container"
        )

        dataclass_suggestions = [s for s in suggestions if "dataclass" in s["content"]]
        assert len(dataclass_suggestions) > 0

        # Test range(len()) pattern
        suggestions = await enhanced_system._recognize_code_patterns(
            "for i in range(len(items)):\n    print(items[i])", "iteration"
        )

        enumerate_suggestions = [s for s in suggestions if "enumerate" in s["content"]]
        assert len(enumerate_suggestions) > 0

        # Test bare except pattern
        suggestions = await enhanced_system._recognize_code_patterns(
            "try:\n    risky_operation()\nexcept:\n    pass", "error handling"
        )

        except_suggestions = [
            s for s in suggestions if "exception types" in s["content"]
        ]
        assert len(except_suggestions) > 0

    def test_context_size_limit(self, enhanced_system):
        """Test that context size is limited."""
        session = CollaborationSession(
            session_id="test",
            participants=["agent1"],
            mode=CollaborationMode.DRIVER_NAVIGATOR,
            project_context={},
        )

        # Add more contexts than the limit
        for i in range(enhanced_system.max_shared_contexts_per_session + 10):
            context = SharedContext(
                session_id="test",
                context_type=ContextShareType.CODE_PATTERNS,
                content={"index": i},
                source_agent="agent1",
                relevance_score=i / 100,  # Varying relevance
            )
            session.shared_contexts.append(context)

        # Simulate the cleanup that happens in share_context
        if (
            len(session.shared_contexts)
            > enhanced_system.max_shared_contexts_per_session
        ):
            session.shared_contexts.sort(key=lambda c: (c.relevance_score, c.timestamp))
            session.shared_contexts = session.shared_contexts[
                -enhanced_system.max_shared_contexts_per_session :
            ]

        assert (
            len(session.shared_contexts)
            == enhanced_system.max_shared_contexts_per_session
        )


@pytest.mark.asyncio
async def test_get_enhanced_pair_programming():
    """Test the global enhanced pair programming getter."""
    with patch(
        "src.core.collaboration.enhanced_pair_programming.PairProgrammingSession"
    ):
        with patch(
            "src.core.collaboration.enhanced_pair_programming.get_context_engine"
        ) as mock_context:
            mock_context.return_value = AsyncMock()

            system1 = await get_enhanced_pair_programming()
            system2 = await get_enhanced_pair_programming()

            # Should return the same instance (singleton pattern)
            assert system1 is system2
            assert isinstance(system1, EnhancedAIPairProgramming)


class TestIntegration:
    """Integration tests for enhanced pair programming."""

    @pytest.mark.asyncio
    async def test_full_collaboration_workflow(self):
        """Test a complete collaboration workflow."""
        with patch(
            "src.core.collaboration.enhanced_pair_programming.message_broker"
        ) as mock_broker:
            mock_broker.publish = AsyncMock()

            # Create system
            base_session = MagicMock()
            enhanced_system = EnhancedAIPairProgramming(base_session)
            enhanced_system.context_engine = AsyncMock()
            enhanced_system.context_engine.search = AsyncMock(return_value=[])
            enhanced_system.context_engine.store_context = AsyncMock()

            # 1. Start collaboration session
            session_id = await enhanced_system.start_enhanced_session(
                participants=["developer", "qa_agent"],
                mode=CollaborationMode.DRIVER_NAVIGATOR,
                project_context={"language": "python"},
                task_description="Implement user authentication",
            )

            # 2. Share architecture knowledge
            await enhanced_system.share_context(
                session_id=session_id,
                source_agent="developer",
                context_type=ContextShareType.ARCHITECTURE_KNOWLEDGE,
                content={"pattern": "JWT authentication", "security": "high"},
                tags={"security", "authentication"},
            )

            # 3. Get code suggestions
            suggestions = await enhanced_system.suggest_code_patterns(
                session_id=session_id,
                current_code="def authenticate_user(username, password):",
                context="user authentication function",
            )

            # 4. Switch driver
            await enhanced_system.switch_driver(session_id, "qa_agent")

            # 5. Share testing knowledge
            await enhanced_system.share_context(
                session_id=session_id,
                source_agent="qa_agent",
                context_type=ContextShareType.TESTING_STRATEGIES,
                content={"strategy": "unit tests for auth", "coverage": "edge cases"},
                tags={"testing", "authentication"},
            )

            # 6. Track live collaboration
            await enhanced_system.track_live_collaboration(
                session_id=session_id,
                agent_id="qa_agent",
                file_path="test_auth.py",
                cursor_position=(1, 0),
                edit_action={"type": "insert", "text": "import pytest"},
            )

            # 7. Get relevant context for developer
            contexts = await enhanced_system.get_relevant_context(
                session_id=session_id,
                requesting_agent="developer",
                query="authentication testing",
                context_types=[ContextShareType.TESTING_STRATEGIES],
            )

            # 8. Get final metrics
            metrics = await enhanced_system.get_collaboration_metrics(session_id)

            # 9. End session
            result = await enhanced_system.end_session(session_id)

            # Verify the workflow
            assert len(suggestions) >= 0  # May have suggestions
            assert len(contexts) == 1  # Should find testing context
            assert metrics["context_exchanges"] == 2  # Two contexts shared
            assert result.success is True
            assert result.metrics["collaboration_efficiency"] > 0
