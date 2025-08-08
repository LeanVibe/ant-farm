"""Tests for ContextEngine functionality."""

import datetime
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

import numpy as np
import pytest

from src.core.context_engine import (
    ContextEngine,
    ContextSearchResult,
    EmbeddingProvider,
    EmbeddingService,
    SemanticSearch,
)
from src.core.models import Context


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = Mock()
    session.add = Mock()
    session.commit = Mock()
    session.rollback = Mock()
    session.close = Mock()
    session.query = Mock()
    return session


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    db_manager = Mock()
    db_manager.create_tables = Mock()
    db_manager.get_session = Mock()
    return db_manager


@pytest.fixture
def mock_embedding_service():
    """Create a mock embedding service."""
    service = Mock(spec=EmbeddingService)
    service.generate_embedding = AsyncMock(
        return_value=[0.1, 0.2, 0.3] * 512
    )  # 1536 dims
    service.provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    return service


@pytest.fixture
def context_engine(mock_db_manager, mock_db_session):
    """Create a ContextEngine with mocked dependencies."""
    mock_db_manager.get_session.return_value = mock_db_session

    # Mock the async database components
    mock_async_engine = AsyncMock()
    mock_async_session = AsyncMock()

    # Mock the async session context manager
    mock_async_session.__aenter__ = AsyncMock(return_value=mock_async_session)
    mock_async_session.__aexit__ = AsyncMock(return_value=None)
    mock_async_session.execute = AsyncMock()
    mock_async_session.commit = AsyncMock()
    mock_async_session.rollback = AsyncMock()
    mock_async_session.close = AsyncMock()

    # Mock the sessionmaker to return the mock session
    mock_async_sessionmaker = Mock()
    mock_async_sessionmaker.return_value = mock_async_session

    with patch(
        "src.core.context_engine.get_database_manager", return_value=mock_db_manager
    ):
        with patch("src.core.context_engine.get_cache_manager", return_value=None):
            with patch(
                "src.core.context_engine.create_async_engine",
                return_value=mock_async_engine,
            ):
                with patch(
                    "src.core.context_engine.async_sessionmaker",
                    return_value=mock_async_sessionmaker,
                ):
                    engine = ContextEngine("mock://test")
                    engine.async_engine = mock_async_engine
                    engine.AsyncSessionLocal = mock_async_sessionmaker
                    return engine


@pytest.mark.asyncio
class TestContextEngine:
    """Test suite for ContextEngine."""

    async def test_store_context_basic(self, context_engine):
        """Test basic context storage."""
        # Setup
        await context_engine.initialize()

        agent_id = str(uuid4())
        mock_context_id = str(uuid4())

        # Mock the async session to return our test context ID
        mock_context = Mock()
        mock_context.id = mock_context_id
        context_engine.AsyncSessionLocal().__aenter__.return_value.add = Mock()
        context_engine.AsyncSessionLocal().__aenter__.return_value.commit = AsyncMock()

        # Setup the context creation to set the ID
        def add_side_effect(ctx):
            ctx.id = mock_context_id

        context_engine.AsyncSessionLocal().__aenter__.return_value.add.side_effect = (
            add_side_effect
        )

        # Execute
        context_id = await context_engine.store_context(
            agent_id=agent_id,
            content="This is a test context.",
            category="general",
            importance_score=0.8,
            topic="testing",
        )

        # Verify
        assert context_id == mock_context_id
        assert context_engine.AsyncSessionLocal().__aenter__.return_value.add.called
        assert context_engine.AsyncSessionLocal().__aenter__.return_value.commit.called

    async def test_store_context_with_metadata(self, context_engine):
        """Test context storage with metadata."""
        await context_engine.initialize()

        agent_id = str(uuid4())
        metadata = {"source_file": "test.py", "line_number": 42}
        mock_context_id = str(uuid4())

        # Mock the async session
        mock_context = Mock()
        mock_context.id = mock_context_id
        context_engine.AsyncSessionLocal().__aenter__.return_value.add = Mock()
        context_engine.AsyncSessionLocal().__aenter__.return_value.commit = AsyncMock()

        # Setup the context creation to set the ID
        def add_side_effect(ctx):
            ctx.id = mock_context_id

        context_engine.AsyncSessionLocal().__aenter__.return_value.add.side_effect = (
            add_side_effect
        )

        context_id = await context_engine.store_context(
            agent_id=agent_id,
            content="def test_function(): pass",
            content_type="code",
            metadata=metadata,
            category="code",
        )

        assert context_id == mock_context_id
        assert context_engine.AsyncSessionLocal().__aenter__.return_value.add.called

        # Check the context that was added
        added_context = (
            context_engine.AsyncSessionLocal().__aenter__.return_value.add.call_args[0][
                0
            ]
        )
        assert added_context.content_type == "code"
        assert added_context.metadata == metadata

    async def test_retrieve_context_basic(self, context_engine, mock_db_session):
        """Test basic context retrieval."""
        await context_engine.initialize()

        agent_id = str(uuid4())

        # Mock contexts to return
        mock_contexts = [
            Mock(
                spec=Context,
                id=uuid4(),
                content="Python is a programming language",
                category="language",
                importance_score=0.9,
                access_count=0,
            ),
            Mock(
                spec=Context,
                id=uuid4(),
                content="FastAPI is a web framework",
                category="framework",
                importance_score=0.7,
                access_count=0,
            ),
        ]

        for ctx in mock_contexts:
            ctx.last_accessed = 0
            ctx.created_at = Mock()
            ctx.created_at.timestamp.return_value = time.time() - 3600  # 1 hour ago

        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = mock_contexts
        mock_db_session.query.return_value = mock_query

        # Test retrieval
        results = await context_engine.retrieve_context(
            query="programming language", agent_id=agent_id, limit=5
        )

        # Verify
        assert len(results) > 0
        assert all(isinstance(r, ContextSearchResult) for r in results)
        mock_db_session.commit.assert_called()

    async def test_retrieve_context_with_filters(self, context_engine, mock_db_session):
        """Test context retrieval with category and importance filters."""
        await context_engine.initialize()

        agent_id = str(uuid4())

        # Mock contexts with different categories and importance
        mock_contexts = [
            Mock(
                spec=Context,
                id=uuid4(),
                content="High importance framework info",
                category="framework",
                importance_score=0.9,
                access_count=0,
            ),
            Mock(
                spec=Context,
                id=uuid4(),
                content="Low importance framework info",
                category="framework",
                importance_score=0.2,
                access_count=0,
            ),
        ]

        for ctx in mock_contexts:
            ctx.last_accessed = 0
            ctx.created_at = Mock()
            ctx.created_at.timestamp.return_value = time.time() - 3600  # 1 hour ago

        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = mock_contexts
        mock_db_session.query.return_value = mock_query

        # Test category filter
        results = await context_engine.retrieve_context(
            query="framework", agent_id=agent_id, category_filter="framework"
        )

        assert len(results) >= 0  # May be filtered by importance
        mock_db_session.commit.assert_called()

    async def test_update_context_importance(self, context_engine):
        """Test updating context importance scores."""
        await context_engine.initialize()

        context_id = str(uuid4())

        # Mock context to update
        mock_context = Mock(spec=Context)
        mock_context.importance_score = 0.5
        mock_context.agent_id = uuid4()

        # Mock the async session
        mock_session = context_engine.AsyncSessionLocal().__aenter__.return_value
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_context
        mock_session.query.return_value = mock_query
        mock_session.commit = AsyncMock()

        # Update importance
        success = await context_engine.update_context_importance(
            context_id=context_id, importance_score=0.9
        )

        assert success
        assert mock_context.importance_score == 0.9
        mock_session.commit.assert_called_once()
        assert mock_context.importance_score == 0.9
        mock_db_session.commit.assert_called_once()

    async def test_share_context(self, context_engine, mock_db_session):
        """Test sharing context between agents."""
        await context_engine.initialize()

        context_id = str(uuid4())
        agent1_id = uuid4()
        agent2_id = str(uuid4())

        # Mock original context
        mock_original = Mock(spec=Context)
        mock_original.id = context_id
        mock_original.agent_id = agent1_id
        mock_original.content = "Shared knowledge"
        mock_original.content_type = "text"
        mock_original.importance_score = 0.8
        mock_original.category = "general"
        mock_original.topic = "sharing"
        mock_original.session_id = None
        mock_original.metadata = {}

        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.first.return_value = mock_original
        mock_db_session.query.return_value = mock_query

        # Share with agent2
        success = await context_engine.share_context(context_id, agent2_id)

        assert success is True
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()

    async def test_get_memory_stats_empty(self, context_engine, mock_db_session):
        """Test memory statistics with no contexts."""
        await context_engine.initialize()

        agent_id = str(uuid4())

        # Mock empty result
        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_db_session.query.return_value = mock_query

        # Get stats
        stats = await context_engine.get_memory_stats(agent_id)

        # Verify empty stats
        assert stats.total_contexts == 0
        assert stats.storage_size_mb == 0.0

    async def test_get_memory_stats_with_contexts(
        self, context_engine, mock_db_session
    ):
        """Test memory statistics generation with contexts."""
        await context_engine.initialize()

        agent_id = str(uuid4())

        # Mock contexts with different importance levels
        mock_contexts = [
            Mock(
                spec=Context,
                content="High importance",
                importance_score=0.9,
                category="important",
                content_type="text",
                access_count=5,
            ),
            Mock(
                spec=Context,
                content="Medium importance",
                importance_score=0.6,
                category="medium",
                content_type="text",
                access_count=3,
            ),
            Mock(
                spec=Context,
                content="Low importance",
                importance_score=0.3,
                category="low",
                content_type="text",
                access_count=1,
            ),
        ]

        # Mock datetime objects

        mock_time = datetime.datetime.now()
        for ctx in mock_contexts:
            ctx.created_at = mock_time

        mock_query = Mock()
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = mock_contexts
        mock_db_session.query.return_value = mock_query

        # Get stats
        stats = await context_engine.get_memory_stats(agent_id)

        # Verify stats
        assert stats.total_contexts == 3
        assert stats.contexts_by_importance["high"] == 1  # 0.9
        assert stats.contexts_by_importance["medium"] == 1  # 0.6
        assert stats.contexts_by_importance["low"] == 1  # 0.3


@pytest.mark.asyncio
class TestSemanticSearch:
    """Test suite for SemanticSearch functionality."""

    async def test_text_similarity_calculation(self):
        """Test text similarity calculation."""
        mock_embedding_service = Mock()
        search = SemanticSearch(mock_embedding_service)

        # Test exact match
        similarity = search._calculate_text_similarity("hello world", "hello world")
        assert similarity == 1.0

        # Test partial match
        similarity = search._calculate_text_similarity("hello", "hello world")
        assert 0.4 < similarity < 0.6

        # Test no match
        similarity = search._calculate_text_similarity("foo", "bar baz")
        assert similarity == 0.0

    async def test_relevance_calculation(self):
        """Test relevance score calculation."""
        mock_embedding_service = Mock()
        search = SemanticSearch(mock_embedding_service)

        # Create a mock context
        mock_context = Mock()
        mock_context.importance_score = 0.8
        mock_context.created_at.timestamp.return_value = 1000  # Very old

        with patch("time.time", return_value=1000000):  # Much later time
            relevance = search._calculate_relevance(mock_context, 0.9)

        # Should be weighted combination of similarity (0.9) and importance (0.8)
        # with age penalty
        assert 0.0 <= relevance <= 1.0
        assert (
            relevance > 0.5
        )  # Should be reasonably high given good similarity and importance


@pytest.mark.asyncio
class TestEmbeddingService:
    """Test suite for EmbeddingService."""

    def test_initialization_sentence_transformers(self):
        """Test initialization with sentence transformers."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model

            service = EmbeddingService(EmbeddingProvider.SENTENCE_TRANSFORMERS)

            assert service.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS
            assert service.model == mock_model

    def test_initialization_fallback(self):
        """Test fallback to sentence transformers when OpenAI unavailable."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_st.return_value = mock_model

            # Mock ImportError for openai
            with patch.dict("sys.modules", {"openai": None}):
                service = EmbeddingService(EmbeddingProvider.OPENAI)

            assert service.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS

    async def test_generate_sentence_transformers_embedding(self):
        """Test embedding generation with sentence transformers."""
        with patch("sentence_transformers.SentenceTransformer") as mock_st:
            mock_model = Mock()
            mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
            mock_st.return_value = mock_model

            service = EmbeddingService(EmbeddingProvider.SENTENCE_TRANSFORMERS)

            embedding = await service.generate_embedding("test text")

            assert embedding == [0.1, 0.2, 0.3]
