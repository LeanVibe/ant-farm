"""Integration tests for Enhanced Communication System.

Tests the integration between all communication components:
1. Enhanced Message Broker + Real-time Collaboration
2. Message Broker + Shared Knowledge Base
3. Real-time Collaboration + Knowledge Sharing
4. Communication Monitoring + Performance Optimization
5. End-to-end multi-agent coordination workflows
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.enhanced_message_broker import EnhancedMessageBroker, MessagePriority
from src.core.message_broker import MessageType
from src.core.realtime_collaboration import RealTimeCollaborationManager
from src.core.shared_knowledge_base import SharedKnowledgeBase


@pytest.fixture
def mock_redis():
    """Create comprehensive mock Redis for integration testing."""
    redis_mock = AsyncMock()

    # All Redis operations needed across components
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.close = AsyncMock(return_value=None)
    redis_mock.publish = AsyncMock(return_value=1)
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.hdel = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zrem = AsyncMock(return_value=1)
    redis_mock.zrange = AsyncMock(return_value=[])
    redis_mock.zrangebyscore = AsyncMock(return_value=[])
    redis_mock.sadd = AsyncMock(return_value=1)
    redis_mock.srem = AsyncMock(return_value=1)
    redis_mock.smembers = AsyncMock(return_value=set())
    redis_mock.sinter = AsyncMock(return_value=set())
    redis_mock.lpush = AsyncMock(return_value=1)
    redis_mock.lrange = AsyncMock(return_value=[])
    redis_mock.ltrim = AsyncMock(return_value=True)
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.set = AsyncMock(return_value=True)
    redis_mock.get = AsyncMock(return_value=None)
    redis_mock.delete = AsyncMock(return_value=1)
    redis_mock.hincrby = AsyncMock(return_value=1)
    redis_mock.execute_command = AsyncMock(return_value=[])

    # Pub/sub
    pubsub_mock = AsyncMock()
    pubsub_mock.subscribe = AsyncMock(return_value=None)
    pubsub_mock.unsubscribe = AsyncMock(return_value=None)
    pubsub_mock.listen = AsyncMock()
    redis_mock.pubsub = Mock(return_value=pubsub_mock)

    return redis_mock


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for knowledge base."""
    model_mock = Mock()
    model_mock.encode = Mock(return_value=[0.1] * 384)  # 384-dim embedding
    return model_mock


@pytest.fixture
async def communication_system(mock_redis, mock_embedding_model):
    """Create integrated communication system with all components."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_embedding_model,
        ):
            # Initialize all components
            message_broker = EnhancedMessageBroker("redis://localhost:6379/1")
            collaboration_manager = RealTimeCollaborationManager(
                "redis://localhost:6379/1"
            )
            knowledge_base = SharedKnowledgeBase("redis://localhost:6379/1")

            await message_broker.initialize()
            await collaboration_manager.initialize()
            await knowledge_base.initialize()

            return {
                "message_broker": message_broker,
                "collaboration": collaboration_manager,
                "knowledge_base": knowledge_base,
                "redis": mock_redis,
            }


class TestMessageBrokerCollaborationIntegration:
    """Test integration between message broker and collaboration system."""

    @pytest.mark.asyncio
    async def test_collaboration_session_messaging(self, communication_system):
        """Test messaging workflow for collaboration sessions."""
        broker = communication_system["message_broker"]
        collaboration = communication_system["collaboration"]

        # Step 1: Start collaboration session
        session = await collaboration.start_session(
            workspace_id="integration_workspace",
            session_type="pair_programming",
            initiated_by="agent_lead",
            participants=["agent_1", "agent_2", "agent_3"],
        )

        assert session is not None
        session_id = session["session_id"]

        # Step 2: Send collaboration invitation via message broker
        invite_message = await broker.send_priority_message(
            from_agent="agent_lead",
            to_agent="agent_4",
            topic="collaboration_invite",
            payload={
                "session_id": session_id,
                "workspace_id": "integration_workspace",
                "session_type": "pair_programming",
                "invitation_message": "Join us for algorithm optimization",
            },
            priority=MessagePriority.HIGH,
            message_type=MessageType.DIRECT,
        )

        assert invite_message is not None

        # Step 3: Handle acceptance via collaboration system
        join_success = await collaboration.join_workspace(
            workspace_id="integration_workspace",
            agent_id="agent_4",
        )

        assert join_success is True

        # Step 4: Send session update via message broker
        update_message = await broker.send_priority_message(
            from_agent="agent_4",
            to_agent="broadcast",
            topic="session_update",
            payload={
                "session_id": session_id,
                "update_type": "participant_joined",
                "new_participant": "agent_4",
                "timestamp": time.time(),
            },
            priority=MessagePriority.NORMAL,
            message_type=MessageType.BROADCAST,
        )

        assert update_message is not None

    @pytest.mark.asyncio
    async def test_real_time_document_collaboration(self, communication_system):
        """Test real-time document editing with message coordination."""
        broker = communication_system["message_broker"]
        collaboration = communication_system["collaboration"]

        # Step 1: Share document in workspace
        share_success = await collaboration.share_document(
            workspace_id="doc_workspace",
            document_name="algorithm.py",
            content="def optimize(): pass",
            shared_by="agent_1",
        )

        assert share_success is True

        # Step 2: Acquire document lock via collaboration system
        lock_acquired = await collaboration.acquire_document_lock(
            workspace_id="doc_workspace",
            document_name="algorithm.py",
            agent_id="agent_2",
            timeout=30,
        )

        assert lock_acquired is True

        # Step 3: Send real-time edit notifications via message broker
        edit_message = await broker.send_priority_message(
            from_agent="agent_2",
            to_agent="broadcast",
            topic="document_edit",
            payload={
                "workspace_id": "doc_workspace",
                "document": "algorithm.py",
                "edit_type": "insertion",
                "line": 1,
                "content": "    # TODO: Implement optimization algorithm",
                "timestamp": time.time(),
            },
            priority=MessagePriority.HIGH,
            message_type=MessageType.BROADCAST,
        )

        assert edit_message is not None

        # Step 4: Release lock and propagate final changes
        release_success = await collaboration.release_document_lock(
            workspace_id="doc_workspace",
            document_name="algorithm.py",
            agent_id="agent_2",
        )

        assert release_success is True


class TestMessageBrokerKnowledgeIntegration:
    """Test integration between message broker and knowledge base."""

    @pytest.mark.asyncio
    async def test_knowledge_sharing_workflow(self, communication_system):
        """Test knowledge sharing via message broker."""
        broker = communication_system["message_broker"]
        knowledge_base = communication_system["knowledge_base"]

        # Step 1: Store knowledge in knowledge base
        store_success = await knowledge_base.store_knowledge(
            title="Efficient Sorting Algorithm",
            content="Quick sort with median-of-three pivot selection",
            knowledge_type="algorithm",
            domain="computer_science",
            confidence=0.92,
            created_by="agent_researcher",
            tags=["sorting", "algorithms", "optimization"],
            context="algorithm_optimization",
        )

        assert store_success is True

        # Step 2: Share knowledge via message broker
        share_message = await broker.send_priority_message(
            from_agent="agent_researcher",
            to_agent="broadcast",
            topic="knowledge_share",
            payload={
                "knowledge_type": "algorithm",
                "title": "Efficient Sorting Algorithm",
                "domain": "computer_science",
                "confidence": 0.92,
                "urgency": "normal",
                "target_agents": ["agent_developer", "agent_optimizer"],
            },
            priority=MessagePriority.NORMAL,
            message_type=MessageType.BROADCAST,
        )

        assert share_message is not None

        # Step 3: Request specific knowledge via message broker
        request_message = await broker.send_priority_message(
            from_agent="agent_developer",
            to_agent="knowledge_base",
            topic="knowledge_request",
            payload={
                "query": "sorting algorithms for large datasets",
                "domain": "computer_science",
                "min_confidence": 0.8,
                "max_results": 5,
            },
            priority=MessagePriority.HIGH,
            message_type=MessageType.DIRECT,
        )

        assert request_message is not None

    @pytest.mark.asyncio
    async def test_knowledge_recommendation_messaging(self, communication_system):
        """Test knowledge recommendation distribution via messaging."""
        broker = communication_system["message_broker"]
        knowledge_base = communication_system["knowledge_base"]

        # Step 1: Agent subscribes to knowledge updates
        subscribe_success = await knowledge_base.subscribe_to_knowledge(
            agent_id="agent_learner",
            subscription_filters={
                "domains": ["machine_learning", "data_science"],
                "knowledge_types": ["best_practice", "algorithm"],
                "min_confidence": 0.8,
            },
        )

        assert subscribe_success is True

        # Step 2: Send personalized recommendations via message broker
        recommendation_message = await broker.send_priority_message(
            from_agent="knowledge_base",
            to_agent="agent_learner",
            topic="knowledge_recommendation",
            payload={
                "recommendations": [
                    {
                        "knowledge_id": "ml_algorithm_123",
                        "title": "Gradient Descent Optimization",
                        "relevance_score": 0.95,
                        "reason": "Matches your current ML project context",
                    },
                    {
                        "knowledge_id": "data_preprocessing_456",
                        "title": "Data Cleaning Best Practices",
                        "relevance_score": 0.87,
                        "reason": "Frequently used with gradient descent",
                    },
                ],
                "recommendation_context": "ml_project_work",
            },
            priority=MessagePriority.NORMAL,
            message_type=MessageType.DIRECT,
        )

        assert recommendation_message is not None


class TestCollaborationKnowledgeIntegration:
    """Test integration between collaboration and knowledge systems."""

    @pytest.mark.asyncio
    async def test_collaborative_knowledge_creation(self, communication_system):
        """Test creating knowledge from collaboration sessions."""
        collaboration = communication_system["collaboration"]
        knowledge_base = communication_system["knowledge_base"]

        # Step 1: Start collaborative problem-solving session
        session = await collaboration.start_session(
            workspace_id="problem_solving_workspace",
            session_type="brainstorming",
            initiated_by="agent_facilitator",
            participants=["agent_expert1", "agent_expert2", "agent_expert3"],
        )

        assert session is not None
        session_id = session["session_id"]

        # Step 2: Collaborate on solution
        update_success = await collaboration.send_session_update(
            session_id=session_id,
            update_type="solution_proposal",
            update_data={
                "solution": "Use hybrid approach combining A* and genetic algorithms",
                "reasoning": "A* for optimal path, genetic for parameter optimization",
                "confidence": 0.85,
                "proposed_by": "agent_expert1",
            },
            sent_by="agent_expert1",
        )

        assert update_success is True

        # Step 3: Capture collaborative knowledge
        knowledge_success = await knowledge_base.contribute_from_collaboration(
            session_id=session_id,
            knowledge_data={
                "title": "Hybrid Pathfinding Optimization",
                "content": "Combine A* algorithm with genetic algorithm for optimal pathfinding with parameter optimization",
                "knowledge_type": "solution_pattern",
                "domain": "algorithm_design",
                "confidence": 0.85,
                "participants": ["agent_expert1", "agent_expert2", "agent_expert3"],
                "session_type": "brainstorming",
                "collaboration_duration": 45,  # minutes
            },
        )

        assert knowledge_success is True

        # Step 4: Share discovered knowledge with broader community
        share_success = await knowledge_base.broadcast_knowledge(
            knowledge_id="hybrid_pathfinding_123",  # Mock ID
            broadcast_by="agent_facilitator",
            target_domains=["algorithm_design", "optimization"],
            message="New hybrid solution discovered through collaborative session",
        )

        assert share_success is True

    @pytest.mark.asyncio
    async def test_knowledge_guided_collaboration(self, communication_system):
        """Test using existing knowledge to guide collaboration."""
        collaboration = communication_system["collaboration"]
        knowledge_base = communication_system["knowledge_base"]

        # Step 1: Search for relevant knowledge before collaboration
        # (Mock search results in setup)

        # Step 2: Create knowledge-informed workspace
        workspace = await collaboration.create_workspace(
            name="Algorithm Optimization Workspace",
            workspace_type="optimization_project",
            created_by="agent_project_manager",
            initial_participants=["agent_optimizer", "agent_analyst"],
        )

        assert workspace is not None

        # Step 3: Share relevant knowledge in workspace
        knowledge_share_success = await collaboration.share_document(
            workspace_id=workspace["id"],
            document_name="relevant_knowledge.md",
            content="# Relevant Knowledge\n\n- Quick sort optimization techniques\n- Memory-efficient algorithms\n- Performance benchmarking methods",
            shared_by="agent_project_manager",
        )

        assert knowledge_share_success is True

        # Step 4: Update knowledge confidence based on successful application
        confidence_update = await knowledge_base.update_confidence(
            knowledge_id="quicksort_optimization_789",  # Mock ID
            feedback_type="positive",
            feedback_score=0.9,
            agent_id="agent_optimizer",
        )

        assert confidence_update is True


class TestEndToEndCommunicationWorkflows:
    """Test complete end-to-end communication workflows."""

    @pytest.mark.asyncio
    async def test_multi_agent_problem_solving_workflow(self, communication_system):
        """Test complete multi-agent problem-solving workflow."""
        broker = communication_system["message_broker"]
        collaboration = communication_system["collaboration"]
        knowledge_base = communication_system["knowledge_base"]

        # Phase 1: Problem Detection and Initial Communication
        problem_message = await broker.send_priority_message(
            from_agent="agent_monitor",
            to_agent="broadcast",
            topic="problem_detected",
            payload={
                "problem_type": "performance_degradation",
                "severity": "high",
                "affected_systems": ["database", "api_server"],
                "initial_analysis": "Query response time increased by 300%",
            },
            priority=MessagePriority.CRITICAL,
            message_type=MessageType.BROADCAST,
        )

        assert problem_message is not None

        # Phase 2: Assemble Expert Team via Collaboration
        workspace = await collaboration.create_workspace(
            name="Performance Investigation Workspace",
            workspace_type="incident_response",
            created_by="agent_monitor",
            initial_participants=["agent_dba", "agent_performance"],
        )

        assert workspace is not None

        # Phase 3: Knowledge Retrieval for Problem Context
        knowledge_store = await knowledge_base.store_knowledge(
            title="Database Performance Investigation",
            content="Check query execution plans and index usage",
            knowledge_type="troubleshooting_guide",
            domain="database_performance",
            confidence=0.9,
            created_by="agent_dba",
            tags=["performance", "database", "troubleshooting"],
            context="incident_response",
        )

        assert knowledge_store is True

        # Phase 4: Collaborative Investigation Session
        session = await collaboration.start_session(
            workspace_id=workspace["id"],
            session_type="incident_investigation",
            initiated_by="agent_dba",
            participants=["agent_dba", "agent_performance", "agent_monitor"],
        )

        assert session is not None

        # Phase 5: Real-time Investigation Updates
        investigation_update = await collaboration.send_session_update(
            session_id=session["session_id"],
            update_type="investigation_finding",
            update_data={
                "finding": "Missing index on frequently queried column",
                "evidence": "Query execution plan shows full table scan",
                "recommended_action": "CREATE INDEX idx_user_timestamp ON users(last_login)",
                "confidence": 0.95,
            },
            sent_by="agent_dba",
        )

        assert investigation_update is True

        # Phase 6: Solution Coordination via Messages
        solution_message = await broker.send_priority_message(
            from_agent="agent_dba",
            to_agent="agent_devops",
            topic="solution_implementation",
            payload={
                "solution_type": "database_optimization",
                "sql_command": "CREATE INDEX idx_user_timestamp ON users(last_login)",
                "estimated_impact": "75% performance improvement",
                "risk_level": "low",
                "approval_needed": True,
            },
            priority=MessagePriority.HIGH,
            message_type=MessageType.DIRECT,
        )

        assert solution_message is not None

        # Phase 7: Capture Learned Knowledge
        learned_knowledge = await knowledge_base.contribute_from_collaboration(
            session_id=session["session_id"],
            knowledge_data={
                "title": "Performance Issue Resolution Pattern",
                "content": "Query performance issues often resolved by adding missing indexes",
                "knowledge_type": "troubleshooting_pattern",
                "domain": "database_performance",
                "confidence": 0.9,
                "participants": ["agent_dba", "agent_performance", "agent_monitor"],
                "resolution_time": 25,  # minutes
                "effectiveness": 0.95,
            },
        )

        assert learned_knowledge is True

    @pytest.mark.asyncio
    async def test_continuous_learning_workflow(self, communication_system):
        """Test continuous learning and knowledge evolution workflow."""
        broker = communication_system["message_broker"]
        collaboration = communication_system["collaboration"]
        knowledge_base = communication_system["knowledge_base"]

        # Phase 1: Regular Knowledge Sharing Session
        learning_session = await collaboration.start_session(
            workspace_id="learning_workspace",
            session_type="knowledge_sharing",
            initiated_by="agent_teacher",
            participants=["agent_student1", "agent_student2", "agent_student3"],
        )

        assert learning_session is not None

        # Phase 2: Share Teaching Content
        teaching_content = await collaboration.share_document(
            workspace_id="learning_workspace",
            document_name="optimization_techniques.md",
            content="# Advanced Optimization Techniques\n\n1. Dynamic Programming\n2. Memoization\n3. Lazy Evaluation",
            shared_by="agent_teacher",
        )

        assert teaching_content is True

        # Phase 3: Interactive Q&A via Messages
        question_message = await broker.send_priority_message(
            from_agent="agent_student1",
            to_agent="agent_teacher",
            topic="learning_question",
            payload={
                "question": "When should I use dynamic programming vs memoization?",
                "context": "optimization_techniques",
                "session_id": learning_session["session_id"],
            },
            priority=MessagePriority.NORMAL,
            message_type=MessageType.DIRECT,
        )

        assert question_message is not None

        # Phase 4: Knowledge Base Query for Answer
        answer_knowledge = await knowledge_base.store_knowledge(
            title="Dynamic Programming vs Memoization",
            content="Use DP for optimal substructure problems, memoization for overlapping subproblems",
            knowledge_type="explanation",
            domain="algorithm_design",
            confidence=0.9,
            created_by="agent_teacher",
            tags=["dynamic_programming", "memoization", "optimization"],
            context="teaching_session",
        )

        assert answer_knowledge is True

        # Phase 5: Answer Distribution
        answer_message = await broker.send_priority_message(
            from_agent="agent_teacher",
            to_agent="broadcast",
            topic="learning_answer",
            payload={
                "question": "When should I use dynamic programming vs memoization?",
                "answer": "Use DP for optimal substructure problems, memoization for overlapping subproblems",
                "knowledge_id": "dp_vs_memoization_123",
                "session_id": learning_session["session_id"],
            },
            priority=MessagePriority.NORMAL,
            message_type=MessageType.BROADCAST,
        )

        assert answer_message is not None

        # Phase 6: Learning Progress Tracking
        progress_update = await collaboration.send_session_update(
            session_id=learning_session["session_id"],
            update_type="learning_progress",
            update_data={
                "concepts_covered": ["dynamic_programming", "memoization"],
                "understanding_level": 0.8,
                "questions_answered": 1,
                "active_participants": 4,
            },
            sent_by="agent_teacher",
        )

        assert progress_update is True


class TestCommunicationPerformanceIntegration:
    """Test performance monitoring across all communication components."""

    @pytest.mark.asyncio
    async def test_system_wide_performance_monitoring(self, communication_system):
        """Test monitoring performance across all communication components."""
        broker = communication_system["message_broker"]
        collaboration = communication_system["collaboration"]
        knowledge_base = communication_system["knowledge_base"]

        # Step 1: Monitor message throughput
        start_time = time.time()

        # Send batch of messages
        message_tasks = []
        for i in range(10):
            task = broker.send_priority_message(
                from_agent=f"agent_{i}",
                to_agent="performance_monitor",
                topic="performance_test",
                payload={"test_index": i, "timestamp": time.time()},
                priority=MessagePriority.NORMAL,
                message_type=MessageType.DIRECT,
            )
            message_tasks.append(task)

        message_results = await asyncio.gather(*message_tasks)
        message_time = time.time() - start_time

        # Step 2: Monitor collaboration operations
        collaboration_start = time.time()

        collab_workspace = await collaboration.create_workspace(
            name="Performance Test Workspace",
            workspace_type="performance_testing",
            created_by="performance_agent",
            initial_participants=[f"agent_{i}" for i in range(5)],
        )

        collaboration_time = time.time() - collaboration_start

        # Step 3: Monitor knowledge operations
        knowledge_start = time.time()

        knowledge_tasks = []
        for i in range(5):
            task = knowledge_base.store_knowledge(
                title=f"Performance Test Knowledge {i}",
                content=f"Test content for performance measurement {i}",
                knowledge_type="performance_test",
                domain="testing",
                confidence=0.8,
                created_by="performance_agent",
                tags=["performance", "testing"],
                context="performance_measurement",
            )
            knowledge_tasks.append(task)

        knowledge_results = await asyncio.gather(*knowledge_tasks)
        knowledge_time = time.time() - knowledge_start

        # Assert performance within acceptable bounds
        assert all(result is not None for result in message_results)
        assert collab_workspace is not None
        assert all(result is True for result in knowledge_results)

        # Performance benchmarks (adjust based on system requirements)
        assert message_time < 5.0  # 10 messages in under 5 seconds
        assert collaboration_time < 2.0  # Workspace creation in under 2 seconds
        assert knowledge_time < 3.0  # 5 knowledge entries in under 3 seconds

    @pytest.mark.asyncio
    async def test_load_balancing_effectiveness(self, communication_system):
        """Test load balancing across communication components."""
        broker = communication_system["message_broker"]

        # Simulate high-load scenario
        high_load_tasks = []
        for i in range(50):
            task = broker.send_priority_message(
                from_agent="load_generator",
                to_agent=f"worker_agent_{i % 5}",  # Distribute across 5 workers
                topic="load_test",
                payload={"load_index": i, "timestamp": time.time()},
                priority=MessagePriority.NORMAL,
                message_type=MessageType.DIRECT,
            )
            high_load_tasks.append(task)

        # Execute all tasks concurrently
        start_time = time.time()
        results = await asyncio.gather(*high_load_tasks)
        total_time = time.time() - start_time

        # Assert load balancing effectiveness
        assert len(results) == 50
        assert all(result is not None for result in results)
        assert total_time < 10.0  # Should handle 50 messages in under 10 seconds
