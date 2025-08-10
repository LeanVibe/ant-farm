"""Comprehensive tests for Shared Knowledge Base System.

Tests the shared knowledge base features:
1. Vector-based knowledge storage and retrieval
2. Context sharing across agents
3. Knowledge graph construction and traversal
4. Semantic search and similarity matching
5. Knowledge evolution and learning integration
"""

import asyncio
import time
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import pytest

from src.core.shared_knowledge_base import (
    KnowledgeEntry,
    KnowledgeGraph,
    SemanticSearchEngine,
    SharedKnowledgeBase,
)


@pytest.fixture
def mock_redis():
    """Create mock Redis client for knowledge base testing."""
    redis_mock = AsyncMock()

    # Basic operations
    redis_mock.ping = AsyncMock(return_value=True)
    redis_mock.close = AsyncMock(return_value=None)

    # Knowledge storage operations
    redis_mock.hset = AsyncMock(return_value=1)
    redis_mock.hget = AsyncMock(return_value=None)
    redis_mock.hgetall = AsyncMock(return_value={})
    redis_mock.hdel = AsyncMock(return_value=1)
    redis_mock.expire = AsyncMock(return_value=True)

    # Vector operations (for semantic search)
    redis_mock.execute_command = AsyncMock(return_value=[])  # For vector search

    # Knowledge graph operations
    redis_mock.sadd = AsyncMock(return_value=1)
    redis_mock.srem = AsyncMock(return_value=1)
    redis_mock.smembers = AsyncMock(return_value=set())
    redis_mock.sinter = AsyncMock(return_value=set())
    redis_mock.sunion = AsyncMock(return_value=set())

    # Indexing operations
    redis_mock.zadd = AsyncMock(return_value=1)
    redis_mock.zrem = AsyncMock(return_value=1)
    redis_mock.zrange = AsyncMock(return_value=[])
    redis_mock.zrangebyscore = AsyncMock(return_value=[])

    # Search operations
    redis_mock.keys = AsyncMock(return_value=[])
    redis_mock.scan = AsyncMock(return_value=(0, []))

    return redis_mock


@pytest.fixture
def mock_embedding_model():
    """Create mock embedding model for vector operations."""
    model_mock = Mock()
    model_mock.encode = Mock(
        return_value=[0.1, 0.2, 0.3, 0.4, 0.5] * 100
    )  # 500-dim vector
    return model_mock


@pytest.fixture
async def knowledge_base(mock_redis, mock_embedding_model):
    """Create SharedKnowledgeBase with mocked dependencies."""
    with patch("redis.asyncio.from_url", return_value=mock_redis):
        with patch(
            "sentence_transformers.SentenceTransformer",
            return_value=mock_embedding_model,
        ):
            kb = SharedKnowledgeBase("redis://localhost:6379/1")
            await kb.initialize()
            return kb


@pytest.fixture
def sample_knowledge_entries():
    """Create sample knowledge entries for testing."""
    return [
        {
            "id": "knowledge_1",
            "title": "Optimization Technique",
            "content": "Use caching to improve performance in database queries",
            "knowledge_type": "best_practice",
            "domain": "database",
            "confidence": 0.95,
            "created_by": "agent_1",
            "created_at": time.time(),
            "tags": ["performance", "database", "caching"],
            "context": "query_optimization",
        },
        {
            "id": "knowledge_2",
            "title": "Error Handling Pattern",
            "content": "Always use specific exception types instead of broad catches",
            "knowledge_type": "coding_standard",
            "domain": "software_development",
            "confidence": 0.90,
            "created_by": "agent_2",
            "created_at": time.time(),
            "tags": ["error_handling", "python", "best_practice"],
            "context": "code_quality",
        },
        {
            "id": "knowledge_3",
            "title": "Load Balancing Strategy",
            "content": "Round-robin works well for evenly distributed workloads",
            "knowledge_type": "system_design",
            "domain": "infrastructure",
            "confidence": 0.85,
            "created_by": "agent_3",
            "created_at": time.time(),
            "tags": ["load_balancing", "architecture", "performance"],
            "context": "system_scaling",
        },
    ]


class TestKnowledgeStorage:
    """Test knowledge storage and retrieval functionality."""

    @pytest.mark.asyncio
    async def test_store_knowledge_entry(
        self, knowledge_base, sample_knowledge_entries, mock_redis
    ):
        """Test storing a knowledge entry."""
        entry = sample_knowledge_entries[0]

        # Act - Store knowledge entry
        success = await knowledge_base.store_knowledge(
            title=entry["title"],
            content=entry["content"],
            knowledge_type=entry["knowledge_type"],
            domain=entry["domain"],
            confidence=entry["confidence"],
            created_by=entry["created_by"],
            tags=entry["tags"],
            context=entry["context"],
        )

        # Assert - Knowledge stored successfully
        assert success is True

        # Verify Redis operations
        mock_redis.hset.assert_called()  # Knowledge entry stored
        mock_redis.zadd.assert_called()  # Indexed for search

    @pytest.mark.asyncio
    async def test_retrieve_knowledge_by_id(self, knowledge_base, mock_redis):
        """Test retrieving knowledge entry by ID."""
        # Arrange - Mock stored knowledge
        knowledge_id = "test_knowledge_123"
        mock_redis.hgetall.return_value = {
            "id": knowledge_id,
            "title": "Test Knowledge",
            "content": "Test content for retrieval",
            "knowledge_type": "test",
            "domain": "testing",
            "confidence": "0.8",
            "created_by": "test_agent",
            "tags": "['testing', 'knowledge']",
        }

        # Act - Retrieve knowledge
        knowledge = await knowledge_base.get_knowledge(knowledge_id)

        # Assert - Knowledge retrieved successfully
        assert knowledge is not None
        assert knowledge["id"] == knowledge_id
        assert knowledge["title"] == "Test Knowledge"

        # Verify retrieval operation
        mock_redis.hgetall.assert_called()

    @pytest.mark.asyncio
    async def test_update_knowledge_entry(self, knowledge_base, mock_redis):
        """Test updating an existing knowledge entry."""
        # Arrange - Mock existing knowledge
        knowledge_id = "update_test_123"
        mock_redis.hgetall.return_value = {
            "id": knowledge_id,
            "title": "Original Title",
            "content": "Original content",
            "confidence": "0.7",
        }

        # Act - Update knowledge
        success = await knowledge_base.update_knowledge(
            knowledge_id=knowledge_id,
            updates={
                "title": "Updated Title",
                "content": "Updated content with new information",
                "confidence": 0.9,
            },
            updated_by="agent_updater",
        )

        # Assert - Knowledge updated successfully
        assert success is True

        # Verify update operations
        mock_redis.hgetall.assert_called()  # Original retrieved
        mock_redis.hset.assert_called()  # Updated stored

    @pytest.mark.asyncio
    async def test_delete_knowledge_entry(self, knowledge_base, mock_redis):
        """Test deleting a knowledge entry."""
        knowledge_id = "delete_test_123"

        # Act - Delete knowledge
        success = await knowledge_base.delete_knowledge(
            knowledge_id=knowledge_id,
            deleted_by="agent_cleaner",
        )

        # Assert - Knowledge deleted successfully
        assert success is True

        # Verify deletion operations
        mock_redis.hdel.assert_called()  # Knowledge removed


class TestSemanticSearch:
    """Test semantic search functionality."""

    @pytest.mark.asyncio
    async def test_semantic_search_by_content(
        self, knowledge_base, mock_redis, mock_embedding_model
    ):
        """Test semantic search based on content similarity."""
        # Arrange - Mock search results
        mock_redis.execute_command.return_value = [
            b"knowledge_1",
            b"0.95",  # ID and similarity score
            b"knowledge_2",
            b"0.87",
            b"knowledge_3",
            b"0.82",
        ]

        # Act - Perform semantic search
        results = await knowledge_base.semantic_search(
            query="database performance optimization techniques",
            top_k=3,
            min_similarity=0.8,
        )

        # Assert - Search results returned
        assert results is not None
        assert isinstance(results, list)

        # Verify embedding generation and search
        mock_embedding_model.encode.assert_called()  # Query embedded
        mock_redis.execute_command.assert_called()  # Vector search performed

    @pytest.mark.asyncio
    async def test_search_by_domain(self, knowledge_base, mock_redis):
        """Test searching knowledge by domain."""
        # Arrange - Mock domain search results
        mock_redis.smembers.return_value = {
            b"knowledge_1",
            b"knowledge_3",
            b"knowledge_5",
        }

        # Act - Search by domain
        results = await knowledge_base.search_by_domain(
            domain="database",
            limit=10,
        )

        # Assert - Domain search results returned
        assert results is not None
        assert isinstance(results, list)

        # Verify domain index query
        mock_redis.smembers.assert_called()

    @pytest.mark.asyncio
    async def test_search_by_tags(self, knowledge_base, mock_redis):
        """Test searching knowledge by tags."""
        # Arrange - Mock tag search results
        mock_redis.sinter.return_value = {b"knowledge_1", b"knowledge_2"}

        # Act - Search by tags
        results = await knowledge_base.search_by_tags(
            tags=["performance", "optimization"],
            match_all=True,
        )

        # Assert - Tag search results returned
        assert results is not None
        assert isinstance(results, list)

        # Verify tag intersection query
        mock_redis.sinter.assert_called()

    @pytest.mark.asyncio
    async def test_search_by_context(self, knowledge_base, mock_redis):
        """Test searching knowledge by context."""
        # Arrange - Mock context search results
        mock_redis.zrangebyscore.return_value = [
            b"knowledge_1",
            b"knowledge_4",
            b"knowledge_7",
        ]

        # Act - Search by context
        results = await knowledge_base.search_by_context(
            context="query_optimization",
            relevance_threshold=0.7,
        )

        # Assert - Context search results returned
        assert results is not None
        assert isinstance(results, list)

        # Verify context query
        mock_redis.zrangebyscore.assert_called()

    @pytest.mark.asyncio
    async def test_hybrid_search(
        self, knowledge_base, mock_redis, mock_embedding_model
    ):
        """Test hybrid search combining multiple search methods."""
        # Arrange - Mock hybrid search components
        mock_redis.execute_command.return_value = [b"knowledge_1", b"0.9"]
        mock_redis.smembers.return_value = {b"knowledge_1", b"knowledge_2"}
        mock_redis.sinter.return_value = {b"knowledge_1"}

        # Act - Perform hybrid search
        results = await knowledge_base.hybrid_search(
            query="optimization techniques",
            domain="database",
            tags=["performance"],
            context="query_optimization",
            weights={
                "semantic": 0.5,
                "domain": 0.2,
                "tags": 0.2,
                "context": 0.1,
            },
        )

        # Assert - Hybrid search results returned
        assert results is not None
        assert isinstance(results, list)

        # Verify multiple search methods used
        mock_embedding_model.encode.assert_called()  # Semantic search
        mock_redis.smembers.assert_called()  # Domain search
        mock_redis.sinter.assert_called()  # Tag search


class TestKnowledgeGraph:
    """Test knowledge graph functionality."""

    @pytest.mark.asyncio
    async def test_create_knowledge_relationship(self, knowledge_base, mock_redis):
        """Test creating relationships between knowledge entries."""
        # Act - Create relationship
        success = await knowledge_base.create_relationship(
            source_knowledge_id="knowledge_1",
            target_knowledge_id="knowledge_2",
            relationship_type="related_to",
            strength=0.8,
            created_by="agent_graph_builder",
        )

        # Assert - Relationship created successfully
        assert success is True

        # Verify graph operations
        mock_redis.sadd.assert_called()  # Relationship stored
        mock_redis.zadd.assert_called()  # Relationship strength indexed

    @pytest.mark.asyncio
    async def test_find_related_knowledge(self, knowledge_base, mock_redis):
        """Test finding related knowledge through graph traversal."""
        # Arrange - Mock related knowledge
        mock_redis.smembers.return_value = {
            b"knowledge_2",
            b"knowledge_3",
            b"knowledge_5",
        }

        # Act - Find related knowledge
        related = await knowledge_base.find_related_knowledge(
            knowledge_id="knowledge_1",
            max_depth=2,
            min_strength=0.5,
        )

        # Assert - Related knowledge found
        assert related is not None
        assert isinstance(related, list)

        # Verify graph traversal
        mock_redis.smembers.assert_called()

    @pytest.mark.asyncio
    async def test_knowledge_clustering(self, knowledge_base, mock_redis):
        """Test clustering related knowledge entries."""
        # Arrange - Mock clustering data
        mock_redis.smembers.side_effect = [
            {b"knowledge_1", b"knowledge_2"},  # Cluster 1
            {b"knowledge_3", b"knowledge_4"},  # Cluster 2
            {b"knowledge_5"},  # Cluster 3
        ]

        # Act - Get knowledge clusters
        clusters = await knowledge_base.get_knowledge_clusters(
            min_cluster_size=2,
            similarity_threshold=0.7,
        )

        # Assert - Clusters identified
        assert clusters is not None
        assert isinstance(clusters, list)

    @pytest.mark.asyncio
    async def test_knowledge_path_finding(self, knowledge_base, mock_redis):
        """Test finding connection paths between knowledge entries."""
        # Act - Find path between knowledge entries
        path = await knowledge_base.find_knowledge_path(
            source_id="knowledge_1",
            target_id="knowledge_5",
            max_path_length=3,
        )

        # Assert - Path found or properly handled
        assert path is not None or path == []  # Empty list if no path
        assert isinstance(path, list)


class TestKnowledgeEvolution:
    """Test knowledge evolution and learning features."""

    @pytest.mark.asyncio
    async def test_knowledge_confidence_update(self, knowledge_base, mock_redis):
        """Test updating knowledge confidence based on usage."""
        # Act - Update confidence based on positive feedback
        success = await knowledge_base.update_confidence(
            knowledge_id="knowledge_1",
            feedback_type="positive",
            feedback_score=0.9,
            agent_id="agent_feedback",
        )

        # Assert - Confidence updated successfully
        assert success is True

        # Verify confidence update
        mock_redis.hset.assert_called()  # Updated confidence stored

    @pytest.mark.asyncio
    async def test_knowledge_usage_tracking(self, knowledge_base, mock_redis):
        """Test tracking knowledge usage statistics."""
        # Act - Record knowledge usage
        await knowledge_base.record_usage(
            knowledge_id="knowledge_1",
            used_by="agent_user",
            usage_context="problem_solving",
            effectiveness_score=0.85,
        )

        # Assert - Usage recorded
        # Verify usage tracking operations
        mock_redis.hincrby.assert_called() or mock_redis.hset.assert_called()

    @pytest.mark.asyncio
    async def test_knowledge_popularity_ranking(self, knowledge_base, mock_redis):
        """Test ranking knowledge by popularity and effectiveness."""
        # Arrange - Mock popularity data
        mock_redis.zrevrange.return_value = [
            b"knowledge_1",
            b"knowledge_3",
            b"knowledge_2",
        ]

        # Act - Get popular knowledge
        popular = await knowledge_base.get_popular_knowledge(
            limit=10,
            time_window="7d",
        )

        # Assert - Popular knowledge retrieved
        assert popular is not None
        assert isinstance(popular, list)

        # Verify popularity query
        mock_redis.zrevrange.assert_called()

    @pytest.mark.asyncio
    async def test_knowledge_obsolescence_detection(self, knowledge_base, mock_redis):
        """Test detecting obsolete or outdated knowledge."""
        # Act - Find obsolete knowledge
        obsolete = await knowledge_base.find_obsolete_knowledge(
            age_threshold_days=90,
            usage_threshold=0.1,
            confidence_threshold=0.3,
        )

        # Assert - Obsolete knowledge detection performed
        assert obsolete is not None
        assert isinstance(obsolete, list)

    @pytest.mark.asyncio
    async def test_knowledge_refinement(self, knowledge_base, mock_redis):
        """Test automatic knowledge refinement based on feedback."""
        # Act - Refine knowledge based on feedback patterns
        success = await knowledge_base.refine_knowledge(
            knowledge_id="knowledge_1",
            refinement_data={
                "updated_content": "Enhanced content based on feedback",
                "confidence_adjustment": 0.05,
                "new_tags": ["verified", "enhanced"],
            },
            refined_by="agent_refiner",
        )

        # Assert - Knowledge refinement successful
        assert success is True

        # Verify refinement operations
        mock_redis.hset.assert_called()  # Refined knowledge stored


class TestKnowledgeSharing:
    """Test knowledge sharing between agents."""

    @pytest.mark.asyncio
    async def test_share_knowledge_with_agent(self, knowledge_base, mock_redis):
        """Test sharing specific knowledge with another agent."""
        # Act - Share knowledge
        success = await knowledge_base.share_knowledge(
            knowledge_id="knowledge_1",
            from_agent="agent_sharer",
            to_agent="agent_receiver",
            sharing_context="collaborative_task",
            urgency="normal",
        )

        # Assert - Knowledge shared successfully
        assert success is True

        # Verify sharing operations
        mock_redis.hset.assert_called()  # Sharing record stored
        mock_redis.publish.assert_called()  # Real-time notification

    @pytest.mark.asyncio
    async def test_broadcast_knowledge(self, knowledge_base, mock_redis):
        """Test broadcasting knowledge to all agents."""
        # Act - Broadcast knowledge
        success = await knowledge_base.broadcast_knowledge(
            knowledge_id="knowledge_1",
            broadcast_by="agent_broadcaster",
            target_domains=["database", "performance"],
            message="Important optimization technique discovered",
        )

        # Assert - Knowledge broadcast successfully
        assert success is True

        # Verify broadcast operations
        mock_redis.publish.assert_called()  # Broadcast message sent

    @pytest.mark.asyncio
    async def test_knowledge_subscription(self, knowledge_base, mock_redis):
        """Test agent subscription to knowledge updates."""
        # Act - Subscribe to knowledge updates
        success = await knowledge_base.subscribe_to_knowledge(
            agent_id="agent_subscriber",
            subscription_filters={
                "domains": ["database", "machine_learning"],
                "knowledge_types": ["best_practice", "algorithm"],
                "min_confidence": 0.8,
            },
        )

        # Assert - Subscription created successfully
        assert success is True

        # Verify subscription storage
        mock_redis.hset.assert_called()  # Subscription preferences stored

    @pytest.mark.asyncio
    async def test_knowledge_recommendation(
        self, knowledge_base, mock_redis, mock_embedding_model
    ):
        """Test recommending relevant knowledge to agents."""
        # Arrange - Mock recommendation data
        mock_redis.execute_command.return_value = [
            b"knowledge_2",
            b"0.92",
            b"knowledge_4",
            b"0.88",
        ]

        # Act - Get knowledge recommendations
        recommendations = await knowledge_base.recommend_knowledge(
            agent_id="agent_recipient",
            current_context="optimization_task",
            agent_interests=["performance", "database"],
            max_recommendations=5,
        )

        # Assert - Recommendations generated
        assert recommendations is not None
        assert isinstance(recommendations, list)

        # Verify recommendation generation
        mock_embedding_model.encode.assert_called()  # Context embedded
        mock_redis.execute_command.assert_called()  # Similar knowledge found


class TestKnowledgeIntegration:
    """Test integration with other system components."""

    @pytest.mark.asyncio
    async def test_collaboration_knowledge_integration(
        self, knowledge_base, mock_redis
    ):
        """Test integration with collaboration system for knowledge sharing."""
        # Act - Contribute knowledge from collaboration session
        success = await knowledge_base.contribute_from_collaboration(
            session_id="collab_session_123",
            knowledge_data={
                "title": "Collaborative Discovery",
                "content": "Learned during pair programming session",
                "participants": ["agent_1", "agent_2"],
                "session_duration": 45,
                "confidence": 0.8,
            },
        )

        # Assert - Collaboration knowledge contributed
        assert success is True

        # Verify integration operations
        mock_redis.hset.assert_called()  # Knowledge stored

    @pytest.mark.asyncio
    async def test_task_execution_knowledge_integration(
        self, knowledge_base, mock_redis
    ):
        """Test integration with task execution for automatic knowledge capture."""
        # Act - Capture knowledge from task execution
        success = await knowledge_base.capture_from_task_execution(
            task_id="task_456",
            execution_results={
                "approach": "divide_and_conquer",
                "performance": {"execution_time": 2.5, "memory_usage": "150MB"},
                "effectiveness": 0.9,
                "reusable_patterns": ["caching", "parallel_processing"],
            },
            captured_by="agent_executor",
        )

        # Assert - Task knowledge captured
        assert success is True

        # Verify knowledge capture
        mock_redis.hset.assert_called()  # Knowledge stored

    @pytest.mark.asyncio
    async def test_monitoring_knowledge_integration(self, knowledge_base, mock_redis):
        """Test integration with monitoring system for usage analytics."""
        # Act - Get knowledge usage analytics
        analytics = await knowledge_base.get_usage_analytics(
            time_period="7d",
            knowledge_types=["best_practice", "algorithm"],
            include_trends=True,
        )

        # Assert - Analytics retrieved
        assert analytics is not None
        assert isinstance(analytics, dict)

        # Verify analytics operations
        mock_redis.zrangebyscore.assert_called() or mock_redis.hgetall.assert_called()
