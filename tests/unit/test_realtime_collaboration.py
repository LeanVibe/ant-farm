"""Comprehensive tests for Real-time Collaboration System.

Tests the real-time collaboration features:
1. Shared workspace management and synchronization
2. Concurrent agent coordination and conflict resolution
3. Real-time document sharing and version control
4. Collaborative session management
5. Performance monitoring for collaboration workflows
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.realtime_collaboration import (
    CollaborationSession,
    RealTimeCollaborationManager,
    SharedWorkspace,
    SyncConflictResolver,
)


@pytest.fixture
def mock_redis():
    """Create mock Redis client for collaboration testing."""
    redis_mock = AsyncMock()

    # Basic operations
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.close = AsyncMock(return_value=None)

    # Workspace operations
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.hdel = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)

    # Session management
    redis_mock.sadd = AsyncMock(return_value=1)
    redis_mock.srem = AsyncMock(return_value=1)
    redis_mock.smembers = AsyncMock(return_value=set())
    redis_mock.sismember = AsyncMock(return_value=True)

    # Real-time updates
    redis_mock.publish = AsyncMock(return_value=1)
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.lrange = AsyncMock(return_value=[])
    redis_mock.ltrim = AsyncMock(return_value=True)

    # Lock operations for conflict resolution
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.get = AsyncMock(return_value=None)

    # Pub/sub
    pubsub_mock = AsyncMock()
    pubsub_mock.subscribe = AsyncMock(return_value=None)
    pubsub_mock.unsubscribe = AsyncMock(return_value=None)
    pubsub_mock.listen = AsyncMock()
    redis_mock.pubsub = Mock(return_value=pubsub_mock)

    return redis_mock


@pytest.fixture
async def collaboration_manager(mock_redis):
    """Create RealTimeCollaborationManager with mocked Redis."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        manager = RealTimeCollaborationManager("redis://localhost:6379/1")
        await manager.initialize()
        return manager


@pytest.fixture
def sample_workspace():
    """Create a sample shared workspace for testing."""
    return {
        "id": "workspace_123",
        "name": "Test Collaboration Workspace",
        "type": "code_development",
        "created_by": "agent_coordinator",
        "created_at": time.time(),
        "participants": ["agent_1", "agent_2", "agent_3"],
        "documents": {
            "main.py": {
                "content": "print('Hello, World!')",
                "version": 1,
                "last_modified": time.time(),
                "locked_by": None,
            }
        },
        "status": "active",
    }


class TestSharedWorkspace:
    """Test shared workspace functionality."""

    @pytest.mark.asyncio
    async def test_create_shared_workspace(self, collaboration_manager, mock_redis):
        """Test creating a new shared workspace."""
        # Act - Create shared workspace
        workspace = await collaboration_manager.create_workspace(
            name="Test Workspace",
            workspace_type="development",
            created_by="coordinator_agent",
            initial_participants=["agent_1", "agent_2"],
        )

        # Assert - Workspace created successfully
        assert workspace is not None
        assert isinstance(workspace, dict)
        assert "id" in workspace
        assert workspace["name"] == "Test Workspace"
        assert workspace["type"] == "development"
        assert "agent_1" in workspace["participants"]
        assert "agent_2" in workspace["participants"]

        # Verify Redis operations
        mock_redis.hset.assert_called()  # Workspace data stored

    @pytest.mark.asyncio
    async def test_join_workspace(
        self, collaboration_manager, sample_workspace, mock_redis
    ):
        """Test agent joining an existing workspace."""
        # Arrange - Mock existing workspace
        workspace_id = sample_workspace["id"]
        mock_redis.hgetall.return_value = {
            k: str(v) if not isinstance(v, (dict, list)) else str(v)
            for k, v in sample_workspace.items()
        }

        # Act - Agent joins workspace
        success = await collaboration_manager.join_workspace(
            workspace_id=workspace_id,
            agent_id="new_agent_4",
        )

        # Assert - Agent joined successfully
        assert success is True

        # Verify Redis operations
        mock_redis.hgetall.assert_called()  # Workspace data retrieved
        mock_redis.hset.assert_called()  # Updated workspace stored

    @pytest.mark.asyncio
    async def test_leave_workspace(
        self, collaboration_manager, sample_workspace, mock_redis
    ):
        """Test agent leaving a workspace."""
        # Arrange - Mock existing workspace with agent
        workspace_id = sample_workspace["id"]
        mock_redis.hgetall.return_value = {
            k: str(v) if not isinstance(v, (dict, list)) else str(v)
            for k, v in sample_workspace.items()
        }

        # Act - Agent leaves workspace
        success = await collaboration_manager.leave_workspace(
            workspace_id=workspace_id,
            agent_id="agent_2",
        )

        # Assert - Agent left successfully
        assert success is True

        # Verify workspace update operations
        mock_redis.hgetall.assert_called()
        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_workspace_document_sharing(self, collaboration_manager, mock_redis):
        """Test sharing documents within workspace."""
        # Act - Share document in workspace
        success = await collaboration_manager.share_document(
            workspace_id="workspace_123",
            document_name="shared_code.py",
            content="def hello(): return 'Hello from shared workspace'",
            shared_by="agent_1",
        )

        # Assert - Document shared successfully
        assert success is True

        # Verify document storage operations
        mock_redis.hset.assert_called()  # Document stored
        mock_redis.publish.assert_called()  # Real-time update published


class TestCollaborationSession:
    """Test collaboration session management."""

    @pytest.mark.asyncio
    async def test_start_collaboration_session(self, collaboration_manager, mock_redis):
        """Test starting a new collaboration session."""
        # Act - Start collaboration session
        session = await collaboration_manager.start_session(
            workspace_id="workspace_123",
            session_type="pair_programming",
            initiated_by="agent_lead",
            participants=["agent_1", "agent_2", "agent_3"],
        )

        # Assert - Session started successfully
        assert session is not None
        assert isinstance(session, dict)
        assert "session_id" in session
        assert session["workspace_id"] == "workspace_123"
        assert session["type"] == "pair_programming"
        assert len(session["participants"]) == 3

        # Verify session storage
        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_session_participant_management(
        self, collaboration_manager, mock_redis
    ):
        """Test adding and removing participants from session."""
        session_id = "session_456"

        # Act - Add participant to session
        success_add = await collaboration_manager.add_session_participant(
            session_id=session_id,
            agent_id="new_participant",
        )

        # Act - Remove participant from session
        success_remove = await collaboration_manager.remove_session_participant(
            session_id=session_id,
            agent_id="old_participant",
        )

        # Assert - Participant management successful
        assert success_add is True
        assert success_remove is True

        # Verify Redis operations for participant management
        assert mock_redis.hgetall.call_count >= 2  # Session data retrieved
        assert mock_redis.hset.call_count >= 2  # Session data updated

    @pytest.mark.asyncio
    async def test_session_real_time_updates(self, collaboration_manager, mock_redis):
        """Test real-time updates within collaboration session."""
        # Act - Send real-time update
        success = await collaboration_manager.send_session_update(
            session_id="session_789",
            update_type="cursor_position",
            update_data={
                "agent_id": "agent_1",
                "file": "main.py",
                "line": 42,
                "column": 15,
                "timestamp": time.time(),
            },
            sent_by="agent_1",
        )

        # Assert - Update sent successfully
        assert success is True

        # Verify real-time communication
        mock_redis.publish.assert_called()  # Real-time update published

    @pytest.mark.asyncio
    async def test_end_collaboration_session(self, collaboration_manager, mock_redis):
        """Test ending a collaboration session."""
        # Arrange - Mock existing session
        session_id = "session_end_test"
        mock_redis.hgetall.return_value = {
            "session_id": session_id,
            "status": "active",
            "participants": "['agent_1', 'agent_2']",
            "workspace_id": "workspace_123",
        }

        # Act - End session
        success = await collaboration_manager.end_session(
            session_id=session_id,
            ended_by="agent_1",
        )

        # Assert - Session ended successfully
        assert success is True

        # Verify session cleanup operations
        mock_redis.hgetall.assert_called()  # Session data retrieved
        mock_redis.hset.assert_called()  # Session status updated


class TestSyncConflictResolver:
    """Test synchronization and conflict resolution."""

    @pytest.mark.asyncio
    async def test_document_lock_acquisition(self, collaboration_manager, mock_redis):
        """Test acquiring lock on document for editing."""
        # Act - Acquire document lock
        lock_acquired = await collaboration_manager.acquire_document_lock(
            workspace_id="workspace_123",
            document_name="shared_file.py",
            agent_id="agent_1",
            timeout=30,
        )

        # Assert - Lock acquired successfully
        assert lock_acquired is True

        # Verify lock operations
        mock_redis.set.assert_called()  # Lock acquired

    @pytest.mark.asyncio
    async def test_document_lock_conflict(self, collaboration_manager, mock_redis):
        """Test handling document lock conflicts."""
        # Arrange - Mock existing lock
        mock_redis.get.return_value = b"agent_2"  # Document already locked by agent_2

        # Act - Attempt to acquire already locked document
        lock_acquired = await collaboration_manager.acquire_document_lock(
            workspace_id="workspace_123",
            document_name="locked_file.py",
            agent_id="agent_1",
            timeout=5,
        )

        # Assert - Lock acquisition failed due to conflict
        assert lock_acquired is False

        # Verify conflict check
        mock_redis.get.assert_called()  # Lock status checked

    @pytest.mark.asyncio
    async def test_document_lock_release(self, collaboration_manager, mock_redis):
        """Test releasing document lock."""
        # Act - Release document lock
        success = await collaboration_manager.release_document_lock(
            workspace_id="workspace_123",
            document_name="shared_file.py",
            agent_id="agent_1",
        )

        # Assert - Lock released successfully
        assert success is True

        # Verify lock release operations
        mock_redis.delete.assert_called()  # Lock removed

    @pytest.mark.asyncio
    async def test_concurrent_edit_conflict_resolution(
        self, collaboration_manager, mock_redis
    ):
        """Test resolving concurrent edit conflicts."""
        # Act - Handle concurrent edit conflict
        resolution = await collaboration_manager.resolve_edit_conflict(
            workspace_id="workspace_123",
            document_name="conflict_file.py",
            conflict_data={
                "agent_1_changes": {"line": 10, "content": "# Agent 1 change"},
                "agent_2_changes": {"line": 10, "content": "# Agent 2 change"},
                "base_content": "# Original content",
                "timestamp_1": time.time() - 10,
                "timestamp_2": time.time() - 5,
            },
        )

        # Assert - Conflict resolved successfully
        assert resolution is not None
        assert isinstance(resolution, dict)

        # Verify conflict resolution storage
        mock_redis.hset.assert_called()  # Resolution stored

    @pytest.mark.asyncio
    async def test_version_control_integration(self, collaboration_manager, mock_redis):
        """Test version control for collaborative documents."""
        # Act - Create new document version
        version = await collaboration_manager.create_document_version(
            workspace_id="workspace_123",
            document_name="versioned_file.py",
            content="def new_function(): pass",
            modified_by="agent_1",
            change_description="Added new function",
        )

        # Assert - Version created successfully
        assert version is not None
        assert isinstance(version, (int, str))

        # Verify version storage
        mock_redis.hset.assert_called()  # Version data stored


class TestRealTimeUpdates:
    """Test real-time update propagation."""

    @pytest.mark.asyncio
    async def test_live_cursor_tracking(self, collaboration_manager, mock_redis):
        """Test real-time cursor position tracking."""
        # Act - Update cursor position
        success = await collaboration_manager.update_agent_cursor(
            workspace_id="workspace_123",
            agent_id="agent_1",
            document="main.py",
            line=25,
            column=10,
        )

        # Assert - Cursor position updated
        assert success is True

        # Verify real-time update
        mock_redis.publish.assert_called()  # Cursor update published

    @pytest.mark.asyncio
    async def test_live_document_changes(self, collaboration_manager, mock_redis):
        """Test real-time document change propagation."""
        # Act - Propagate document change
        success = await collaboration_manager.propagate_document_change(
            workspace_id="workspace_123",
            document_name="main.py",
            change_type="insertion",
            change_data={
                "line": 15,
                "column": 0,
                "content": "# New comment added",
                "agent_id": "agent_2",
                "timestamp": time.time(),
            },
        )

        # Assert - Change propagated successfully
        assert success is True

        # Verify real-time propagation
        mock_redis.publish.assert_called()  # Change published

    @pytest.mark.asyncio
    async def test_workspace_awareness_updates(self, collaboration_manager, mock_redis):
        """Test workspace awareness (who's online, what they're doing)."""
        # Act - Update agent activity
        success = await collaboration_manager.update_agent_activity(
            workspace_id="workspace_123",
            agent_id="agent_1",
            activity="editing",
            details={
                "document": "main.py",
                "action": "typing",
                "last_seen": time.time(),
            },
        )

        # Assert - Activity updated successfully
        assert success is True

        # Verify activity tracking
        mock_redis.hset.assert_called()  # Activity stored
        mock_redis.publish.assert_called()  # Activity broadcast


class TestCollaborationPerformance:
    """Test performance aspects of collaboration system."""

    @pytest.mark.asyncio
    async def test_concurrent_session_management(
        self, collaboration_manager, mock_redis
    ):
        """Test managing multiple concurrent collaboration sessions."""
        # Act - Create multiple sessions concurrently
        tasks = []
        for i in range(5):
            task = collaboration_manager.start_session(
                workspace_id=f"workspace_{i}",
                session_type="development",
                initiated_by=f"agent_{i}",
                participants=[f"agent_{i}", f"agent_{i + 1}"],
            )
            tasks.append(task)

        sessions = await asyncio.gather(*tasks)

        # Assert - All sessions created successfully
        assert len(sessions) == 5
        assert all(session is not None for session in sessions)

        # Verify concurrent operations
        assert mock_redis.hset.call_count >= 5

    @pytest.mark.asyncio
    async def test_real_time_update_throughput(self, collaboration_manager, mock_redis):
        """Test throughput of real-time updates."""
        # Arrange - Prepare for throughput test
        start_time = time.time()
        update_count = 50

        # Act - Send rapid real-time updates
        tasks = []
        for i in range(update_count):
            task = collaboration_manager.send_session_update(
                session_id="performance_test_session",
                update_type="edit",
                update_data={
                    "change_id": i,
                    "timestamp": time.time(),
                    "agent": f"agent_{i % 3}",
                },
                sent_by=f"agent_{i % 3}",
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        # Assert - High throughput achieved
        duration = end_time - start_time
        updates_per_second = update_count / duration

        assert len(results) == update_count
        assert all(result is True for result in results)
        assert updates_per_second > 10  # Should handle at least 10 updates/second

    @pytest.mark.asyncio
    async def test_workspace_scaling(self, collaboration_manager, mock_redis):
        """Test system behavior with large number of workspace participants."""
        # Act - Create workspace with many participants
        large_participant_list = [f"agent_{i}" for i in range(20)]

        workspace = await collaboration_manager.create_workspace(
            name="Large Team Workspace",
            workspace_type="enterprise_development",
            created_by="team_lead",
            initial_participants=large_participant_list,
        )

        # Assert - Large workspace created successfully
        assert workspace is not None
        assert len(workspace["participants"]) == 20

        # Verify storage of large workspace
        mock_redis.hset.assert_called()


class TestCollaborationIntegration:
    """Test integration with other system components."""

    @pytest.mark.asyncio
    async def test_message_broker_integration(self, collaboration_manager, mock_redis):
        """Test integration with enhanced message broker for real-time communication."""
        # Act - Send collaboration message via message broker integration
        success = await collaboration_manager.send_collaboration_message(
            from_agent="collaborator_1",
            to_agents=["collaborator_2", "collaborator_3"],
            workspace_id="integration_workspace",
            message_type="collaboration_request",
            payload={
                "request_type": "pair_programming",
                "duration": 60,  # minutes
                "focus_area": "algorithm_optimization",
            },
        )

        # Assert - Integration message sent successfully
        assert success is True

        # Verify message broker operations
        mock_redis.publish.assert_called()  # Message published

    @pytest.mark.asyncio
    async def test_shared_knowledge_integration(
        self, collaboration_manager, mock_redis
    ):
        """Test integration with shared knowledge base."""
        # Act - Share knowledge learned during collaboration
        success = await collaboration_manager.contribute_session_knowledge(
            session_id="knowledge_session",
            knowledge_type="best_practice",
            knowledge_data={
                "practice": "Test-driven collaborative development",
                "context": "Multi-agent code development",
                "effectiveness_score": 0.92,
                "session_duration": 45,  # minutes
                "participants": ["agent_1", "agent_2"],
            },
        )

        # Assert - Knowledge contribution successful
        assert success is True

        # Verify knowledge storage
        mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_monitoring_integration(self, collaboration_manager, mock_redis):
        """Test integration with communication monitoring system."""
        # Act - Record collaboration metrics
        await collaboration_manager.record_collaboration_metrics(
            workspace_id="monitored_workspace",
            session_id="monitored_session",
            metrics={
                "session_duration": 30,  # minutes
                "active_participants": 3,
                "documents_edited": 5,
                "conflicts_resolved": 2,
                "productivity_score": 0.85,
                "communication_quality": 0.90,
            },
        )

        # Assert - Metrics recorded successfully
        # Verify metrics storage
        mock_redis.hset.assert_called()
