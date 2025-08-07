"""Advanced context engine for semantic memory and hierarchical knowledge management."""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog
from sqlalchemy.orm import Session

from .caching import CONTEXT_CACHE_CONFIG, CacheKey, get_cache_manager
from .models import Context, get_database_manager

logger = structlog.get_logger()


class EmbeddingProvider(Enum):
    """Supported embedding providers."""

    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence_transformers"
    OLLAMA = "ollama"


class MemoryLayer(Enum):
    """Hierarchical memory layers."""

    WORKING = "working"  # Current session, active tasks
    SHORT_TERM = "short_term"  # Recent activities, last 24 hours
    MEDIUM_TERM = "medium_term"  # Important events, last week
    LONG_TERM = "long_term"  # Core knowledge, permanent storage
    SEMANTIC = "semantic"  # Distilled patterns and rules


class ContextType(Enum):
    """Types of context for better organization."""

    TASK = "task"
    CONVERSATION = "conversation"
    CODE = "code"
    DECISION = "decision"
    LEARNING = "learning"
    ERROR = "error"
    SUCCESS = "success"
    PATTERN = "pattern"
    RULE = "rule"


@dataclass
class ContextSearchResult:
    """Result from context search."""

    context: Context
    similarity_score: float
    relevance_score: float
    layer: MemoryLayer
    recency_bonus: float = 0.0


@dataclass
class MemoryStats:
    """Comprehensive memory statistics."""

    total_contexts: int
    contexts_by_importance: dict[str, int]
    contexts_by_category: dict[str, int]
    contexts_by_layer: dict[str, int]
    contexts_by_type: dict[str, int]
    storage_size_mb: float
    oldest_context_age_days: float
    most_accessed_context_id: str
    compression_ratio: float
    search_performance_ms: float


@dataclass
class ConsolidationRule:
    """Rules for memory consolidation."""

    source_layer: MemoryLayer
    target_layer: MemoryLayer
    age_threshold_hours: float
    importance_threshold: float
    access_count_threshold: int
    compression_strategy: str  # "summarize", "merge", "distill"


@dataclass
class KnowledgePattern:
    """Discovered knowledge pattern."""

    id: str
    pattern_type: str
    confidence_score: float
    supporting_contexts: list[str]
    pattern_description: str
    discovered_at: float
    usage_count: int = 0


class EmbeddingService:
    """Service for generating text embeddings."""

    def __init__(
        self, provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS
    ):
        self.provider = provider
        self.model = None
        self.client = None
        self._initialize_provider()

    def _initialize_provider(self):
        """Initialize the embedding provider."""
        if self.provider == EmbeddingProvider.OPENAI:
            try:
                import openai

                self.client = openai.OpenAI()
                logger.info("OpenAI embedding service initialized")
            except ImportError:
                logger.warning(
                    "OpenAI not available, falling back to sentence-transformers"
                )
                self.provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
                self._initialize_sentence_transformers()

        elif self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            self._initialize_sentence_transformers()

        elif self.provider == EmbeddingProvider.OLLAMA:
            self._initialize_ollama()

    def _initialize_sentence_transformers(self):
        """Initialize sentence transformers (local)."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.provider = EmbeddingProvider.SENTENCE_TRANSFORMERS
            logger.info("Sentence transformers embedding service initialized")
        except ImportError:
            logger.error("sentence-transformers not available")
            raise

    def _initialize_ollama(self):
        """Initialize Ollama (local)."""
        try:
            import requests

            # Test Ollama connection
            response = requests.get("http://localhost:11434/api/version", timeout=5)
            if response.status_code == 200:
                self.provider = EmbeddingProvider.OLLAMA
                logger.info("Ollama embedding service initialized")
            else:
                raise ConnectionError("Ollama not available")
        except Exception:
            logger.warning(
                "Ollama not available, falling back to sentence-transformers"
            )
            self._initialize_sentence_transformers()

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate embedding for text."""
        if self.provider == EmbeddingProvider.OPENAI:
            return await self._generate_openai_embedding(text)
        elif self.provider == EmbeddingProvider.SENTENCE_TRANSFORMERS:
            return await self._generate_sentence_transformers_embedding(text)
        elif self.provider == EmbeddingProvider.OLLAMA:
            return await self._generate_ollama_embedding(text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _generate_openai_embedding(self, text: str) -> list[float]:
        """Generate embedding using OpenAI."""
        try:
            response = await asyncio.to_thread(
                lambda: self.client.embeddings.create(
                    model="text-embedding-ada-002", input=text
                )
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error("OpenAI embedding generation failed", error=str(e))
            raise

    async def _generate_sentence_transformers_embedding(self, text: str) -> list[float]:
        """Generate embedding using sentence transformers."""
        try:
            embedding = await asyncio.to_thread(self.model.encode, [text])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(
                "Sentence transformers embedding generation failed", error=str(e)
            )
            raise

    async def _generate_ollama_embedding(self, text: str) -> list[float]:
        """Generate embedding using Ollama."""
        try:
            import aiohttp
            from aiohttp import ClientError, ClientTimeout

            timeout = ClientTimeout(total=30)  # 30 second timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    "http://localhost:11434/api/embeddings",
                    json={"model": "nomic-embed-text", "prompt": text},
                ) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result["embedding"]
        except (TimeoutError, ClientError) as e:
            logger.error("Ollama API request failed", error=str(e))
            raise
        except KeyError as e:
            logger.error("Invalid Ollama API response format", error=str(e))
            raise
        except Exception as e:
            logger.error("Ollama embedding generation failed", error=str(e))
            raise


class ContextCompressor:
    """Service for compressing and summarizing contexts."""

    def __init__(self):
        self.compression_ratio = 0.3  # Target compression ratio
        self.max_chunk_size = 2000  # Max characters per chunk

    async def compress_contexts(self, contexts: list[Context]) -> Context:
        """Compress multiple contexts into a single summary context."""
        if not contexts:
            return None

        # Sort by importance
        contexts.sort(key=lambda c: c.importance_score, reverse=True)

        # Combine content
        combined_content = ""
        combined_metadata = {}
        max_importance = 0.0

        for ctx in contexts:
            combined_content += f"\n\n{ctx.content}"
            combined_metadata.update(ctx.metadata or {})
            max_importance = max(max_importance, ctx.importance_score)

        # Generate summary using available LLM
        summary_content = await self._generate_summary(combined_content)

        # Create compressed context
        compressed_context = Context(
            agent_id=contexts[0].agent_id,
            session_id=contexts[0].session_id,
            content=summary_content,
            content_type="summary",
            importance_score=max_importance * 0.9,  # Slightly lower than original
            category="compressed",
            topic=f"Summary of {len(contexts)} contexts",
            metadata={
                **combined_metadata,
                "compressed_from": [str(ctx.id) for ctx in contexts],
                "compression_ratio": len(summary_content) / len(combined_content),
            },
            source="context_compressor",
        )

        return compressed_context

    async def _generate_summary(self, content: str) -> str:
        """Generate summary of content."""
        # Simple extraction for now - in production would use LLM
        lines = content.split("\n")
        important_lines = []

        # Extract key information
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Simple heuristics for important content
            if any(
                keyword in line.lower()
                for keyword in [
                    "error",
                    "fail",
                    "success",
                    "complete",
                    "important",
                    "critical",
                    "key",
                    "result",
                    "conclusion",
                ]
            ):
                important_lines.append(line)

        # If we have extracted lines, use them
        if important_lines:
            summary = "\n".join(important_lines[:20])  # Max 20 lines
        else:
            # Fall back to first few lines
            summary = "\n".join(lines[:10])

        # Ensure reasonable length
        if len(summary) > 1000:
            summary = summary[:1000] + "..."

        return summary


class SemanticSearch:
    """Service for semantic search of contexts."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service

    async def search_contexts(
        self,
        query: str,
        agent_id: str,
        db_session: Session,
        limit: int = 10,
        min_similarity: float = 0.3,
        category_filter: str | None = None,
        time_filter_hours: int | None = None,
    ) -> list[ContextSearchResult]:
        """Search for contexts using semantic similarity."""

        # Generate query embedding
        await self.embedding_service.generate_embedding(query)

        # Get candidate contexts
        query_obj = db_session.query(Context).filter(Context.agent_id == agent_id)

        if category_filter:
            query_obj = query_obj.filter(Context.category == category_filter)

        if time_filter_hours:
            cutoff_time = time.time() - (time_filter_hours * 3600)
            query_obj = query_obj.filter(Context.created_at >= cutoff_time)

        contexts = query_obj.all()

        # Calculate similarities (simplified - would use actual vector similarity in production)
        results = []
        for context in contexts:
            # For now, use simple text similarity
            similarity = self._calculate_text_similarity(query, context.content)

            if similarity >= min_similarity:
                relevance = self._calculate_relevance(context, similarity)

                results.append(
                    ContextSearchResult(
                        context=context,
                        similarity_score=similarity,
                        relevance_score=relevance,
                        layer=MemoryLayer.WORKING,  # Default layer for search results
                    )
                )

        # Sort by relevance score
        results.sort(key=lambda r: r.relevance_score, reverse=True)

        return results[:limit]

    def _calculate_text_similarity(self, query: str, content: str) -> float:
        """Calculate simple text similarity (placeholder for vector similarity)."""
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        if not query_words or not content_words:
            return 0.0

        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_relevance(self, context: Context, similarity: float) -> float:
        """Calculate relevance score combining similarity and importance."""
        # Combine similarity with importance score and recency
        age_in_seconds = time.time() - context.created_at.timestamp()
        age_penalty = min(1.0, age_in_seconds / (7 * 24 * 3600))  # Week decay

        relevance = (
            similarity * 0.6  # 60% similarity
            + context.importance_score * 0.3  # 30% importance
            + (1.0 - age_penalty) * 0.1  # 10% recency
        )

        return min(1.0, relevance)


class MemoryConsolidator:
    """Service for consolidating and organizing memory during sleep cycles."""

    def __init__(self, context_engine):
        self.context_engine = context_engine
        self.compressor = ContextCompressor()

    async def consolidate_memory(
        self, agent_id: str, db_session: Session
    ) -> dict[str, Any]:
        """Consolidate memory for an agent."""
        logger.info("Starting memory consolidation", agent_id=agent_id)

        consolidation_stats = {
            "contexts_processed": 0,
            "contexts_compressed": 0,
            "contexts_archived": 0,
            "contexts_deleted": 0,
            "processing_time_seconds": 0,
        }

        start_time = time.time()

        try:
            # Get all contexts for agent
            contexts = (
                db_session.query(Context).filter(Context.agent_id == agent_id).all()
            )
            consolidation_stats["contexts_processed"] = len(contexts)

            # Group contexts by session and category
            context_groups = self._group_contexts(contexts)

            # Process each group
            for _group_key, group_contexts in context_groups.items():
                if len(group_contexts) > 5:  # Only compress if we have many contexts
                    compressed = await self.compressor.compress_contexts(group_contexts)
                    if compressed:
                        # Save compressed context
                        db_session.add(compressed)

                        # Mark original contexts as archived
                        for ctx in group_contexts:
                            if ctx.importance_score < 0.3:  # Low importance
                                db_session.delete(ctx)
                                consolidation_stats["contexts_deleted"] += 1
                            else:
                                ctx.metadata = ctx.metadata or {}
                                ctx.metadata["archived"] = True
                                consolidation_stats["contexts_archived"] += 1

                        consolidation_stats["contexts_compressed"] += 1

            # Clean up expired contexts
            await self._cleanup_expired_contexts(
                agent_id, db_session, consolidation_stats
            )

            # Update importance scores based on usage
            await self._update_importance_scores(agent_id, db_session)

            db_session.commit()

            consolidation_stats["processing_time_seconds"] = time.time() - start_time

            logger.info(
                "Memory consolidation completed",
                agent_id=agent_id,
                stats=consolidation_stats,
            )

            return consolidation_stats

        except Exception as e:
            db_session.rollback()
            logger.error("Memory consolidation failed", agent_id=agent_id, error=str(e))
            raise

    def _group_contexts(self, contexts: list[Context]) -> dict[str, list[Context]]:
        """Group contexts by session and category."""
        groups = {}

        for context in contexts:
            # Skip already archived contexts
            if context.metadata and context.metadata.get("archived"):
                continue

            key = f"{context.session_id}_{context.category}"
            if key not in groups:
                groups[key] = []
            groups[key].append(context)

        return groups

    async def _cleanup_expired_contexts(
        self, agent_id: str, db_session: Session, stats: dict[str, Any]
    ):
        """Clean up expired contexts."""
        current_time = time.time()

        expired_contexts = (
            db_session.query(Context)
            .filter(
                Context.agent_id == agent_id,
                Context.expires_at.isnot(None),
                Context.expires_at < current_time,
            )
            .all()
        )

        for context in expired_contexts:
            db_session.delete(context)
            stats["contexts_deleted"] += 1

    async def _update_importance_scores(self, agent_id: str, db_session: Session):
        """Update importance scores based on usage patterns."""
        contexts = db_session.query(Context).filter(Context.agent_id == agent_id).all()

        for context in contexts:
            # Calculate new importance based on access patterns
            age_days = (time.time() - context.created_at.timestamp()) / (24 * 3600)
            access_frequency = context.access_count / max(1, age_days)

            # Adjust importance score
            if access_frequency > 1.0:  # Frequently accessed
                context.importance_score = min(1.0, context.importance_score + 0.1)
            elif access_frequency < 0.1 and age_days > 7:  # Rarely accessed and old
                context.importance_score = max(0.0, context.importance_score - 0.1)


class ContextEngine:
    """Main context engine for semantic memory management."""

    def __init__(
        self,
        database_url: str,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
    ):
        self.db_manager = get_database_manager(database_url)
        self.embedding_service = EmbeddingService(embedding_provider)
        self.semantic_search = SemanticSearch(self.embedding_service)
        self.memory_consolidator = MemoryConsolidator(self)
        self.cache_manager = None

    async def initialize(self):
        """Initialize the context engine."""
        # Ensure database tables exist
        self.db_manager.create_tables()

        # Initialize cache manager
        self.cache_manager = await get_cache_manager()

        logger.info("Context engine initialized with caching support")

    async def store_context(
        self,
        agent_id: str,
        content: str,
        session_id: str | None = None,
        importance_score: float = 0.5,
        category: str = "general",
        topic: str = None,
        metadata: dict[str, Any] = None,
        content_type: str = "text",
    ) -> str:
        """Store a new context."""

        db_session = self.db_manager.get_session()
        try:
            # Generate embedding
            embedding_model = f"{self.embedding_service.provider.value}_model"

            # Create context
            context = Context(
                agent_id=agent_id,
                session_id=session_id,
                content=content,
                content_type=content_type,
                importance_score=importance_score,
                category=category,
                topic=topic or self._extract_topic(content),
                metadata=metadata or {},
                embedding_model=embedding_model,
                source="agent_input",
            )

            db_session.add(context)
            db_session.commit()

            context_id = str(context.id)

            # Invalidate related caches
            if self.cache_manager:
                cache_deps = [
                    f"agent:{agent_id}",
                    f"category:{category}",
                    "context_stats",
                ]
                if session_id:
                    cache_deps.append(f"session:{session_id}")

                await self.cache_manager.invalidate_dependency(f"agent:{agent_id}")

            logger.info(
                "Context stored",
                context_id=context_id,
                agent_id=agent_id,
                category=category,
            )

            return context_id

        except Exception as e:
            db_session.rollback()
            logger.error("Failed to store context", agent_id=agent_id, error=str(e))
            raise
        finally:
            db_session.close()

    async def retrieve_context(
        self,
        query: str,
        agent_id: str,
        limit: int = 10,
        category_filter: str = None,
        min_importance: float = 0.0,
    ) -> list[ContextSearchResult]:
        """Retrieve relevant contexts for a query with caching."""

        # Generate cache key
        cache_key = CacheKey.generate(
            "context_search",
            query=query,
            agent_id=agent_id,
            limit=limit,
            category_filter=category_filter,
            min_importance=min_importance,
        )

        # Try cache first
        if self.cache_manager:
            cache = self.cache_manager.get_cache("context_search", CONTEXT_CACHE_CONFIG)
            cached_results = await cache.get(cache_key)
            if cached_results is not None:
                logger.debug("Context search cache hit", query=query[:50])
                return cached_results

        db_session = self.db_manager.get_session()
        try:
            results = await self.semantic_search.search_contexts(
                query=query,
                agent_id=agent_id,
                db_session=db_session,
                limit=limit,
                category_filter=category_filter,
            )

            # Filter by minimum importance
            filtered_results = [
                r for r in results if r.context.importance_score >= min_importance
            ]

            # Update access tracking
            for result in filtered_results:
                result.context.access_count += 1
                result.context.last_accessed = time.time()

            db_session.commit()

            # Cache the results with dependencies
            if self.cache_manager:
                cache_deps = [f"agent:{agent_id}"]
                if category_filter:
                    cache_deps.append(f"category:{category_filter}")

                await cache.set(cache_key, filtered_results, dependencies=cache_deps)

            logger.info(
                "Context retrieved",
                agent_id=agent_id,
                query=query[:50],
                results_count=len(filtered_results),
            )

            return filtered_results

        except Exception as e:
            db_session.rollback()
            logger.error("Failed to retrieve context", agent_id=agent_id, error=str(e))
            raise
        finally:
            db_session.close()

    async def update_context_importance(
        self, context_id: str, new_importance: float
    ) -> bool:
        """Update the importance score of a context."""

        db_session = self.db_manager.get_session()
        try:
            context = db_session.query(Context).filter(Context.id == context_id).first()
            if context:
                context.importance_score = max(0.0, min(1.0, new_importance))
                db_session.commit()

                # Invalidate related caches
                if self.cache_manager:
                    await self.cache_manager.invalidate_dependency(
                        f"agent:{context.agent_id}"
                    )
                    await self.cache_manager.invalidate_dependency(
                        f"context:{context_id}"
                    )

                logger.info(
                    "Context importance updated",
                    context_id=context_id,
                    new_importance=new_importance,
                )
                return True
            return False
        except Exception as e:
            db_session.rollback()
            logger.error(
                "Failed to update context importance",
                context_id=context_id,
                error=str(e),
            )
            return False
        finally:
            db_session.close()

    async def share_context(self, context_id: str, target_agent_id: str) -> bool:
        """Share a context with another agent."""

        db_session = self.db_manager.get_session()
        try:
            original_context = (
                db_session.query(Context).filter(Context.id == context_id).first()
            )
            if not original_context:
                return False

            # Create a copy for the target agent
            shared_context = Context(
                agent_id=target_agent_id,
                session_id=original_context.session_id,
                content=original_context.content,
                content_type=original_context.content_type,
                importance_score=original_context.importance_score
                * 0.8,  # Slightly lower
                category=original_context.category,
                topic=original_context.topic,
                metadata={
                    **(original_context.metadata or {}),
                    "shared_from": str(original_context.id),
                    "original_agent": str(original_context.agent_id),
                },
                source="shared_context",
            )

            db_session.add(shared_context)
            db_session.commit()

            # Invalidate caches for both agents
            if self.cache_manager:
                await self.cache_manager.invalidate_dependency(
                    f"agent:{target_agent_id}"
                )
                await self.cache_manager.invalidate_dependency(
                    f"agent:{original_context.agent_id}"
                )

            logger.info(
                "Context shared", context_id=context_id, target_agent_id=target_agent_id
            )

            return True

        except Exception as e:
            db_session.rollback()
            logger.error("Failed to share context", context_id=context_id, error=str(e))
            return False
        finally:
            db_session.close()

    async def get_memory_stats(self, agent_id: str) -> MemoryStats:
        """Get memory statistics for an agent with caching."""

        # Cache key for memory stats
        cache_key = CacheKey.generate("memory_stats", agent_id=agent_id)

        # Try cache first
        if self.cache_manager:
            cache = self.cache_manager.get_cache("memory_stats", CONTEXT_CACHE_CONFIG)
            cached_stats = await cache.get(cache_key)
            if cached_stats is not None:
                logger.debug("Memory stats cache hit", agent_id=agent_id)
                return cached_stats

        db_session = self.db_manager.get_session()
        try:
            contexts = (
                db_session.query(Context).filter(Context.agent_id == agent_id).all()
            )

            if not contexts:
                empty_stats = MemoryStats(
                    total_contexts=0,
                    contexts_by_importance={},
                    contexts_by_category={},
                    contexts_by_layer={},
                    contexts_by_type={},
                    storage_size_mb=0.0,
                    oldest_context_age_days=0.0,
                    most_accessed_context_id="",
                    compression_ratio=1.0,
                    search_performance_ms=0.0,
                )

                # Cache empty results briefly
                if self.cache_manager:
                    await cache.set(cache_key, empty_stats, ttl=60)  # 1 minute

                return empty_stats

            # Calculate statistics
            importance_buckets = {"high": 0, "medium": 0, "low": 0}
            category_counts = {}
            layer_counts = {"working": 0, "short_term": 0, "long_term": 0}
            type_counts = {}
            total_size = 0
            most_accessed = contexts[0]

            for context in contexts:
                # Importance buckets
                if context.importance_score >= 0.7:
                    importance_buckets["high"] += 1
                elif context.importance_score >= 0.4:
                    importance_buckets["medium"] += 1
                else:
                    importance_buckets["low"] += 1

                # Category counts
                category = context.category or "uncategorized"
                category_counts[category] = category_counts.get(category, 0) + 1

                # Layer counts (simplified classification)
                age_hours = (time.time() - context.created_at.timestamp()) / 3600
                if age_hours < 24:
                    layer_counts["working"] += 1
                elif age_hours < 168:  # 1 week
                    layer_counts["short_term"] += 1
                else:
                    layer_counts["long_term"] += 1

                # Type counts
                content_type = context.content_type or "text"
                type_counts[content_type] = type_counts.get(content_type, 0) + 1

                # Size calculation
                total_size += len(context.content.encode("utf-8"))

                # Most accessed
                if context.access_count > most_accessed.access_count:
                    most_accessed = context

            # Calculate oldest context age
            oldest_context = min(contexts, key=lambda c: c.created_at)
            oldest_age_days = (time.time() - oldest_context.created_at.timestamp()) / (
                24 * 3600
            )

            stats = MemoryStats(
                total_contexts=len(contexts),
                contexts_by_importance=importance_buckets,
                contexts_by_category=category_counts,
                contexts_by_layer=layer_counts,
                contexts_by_type=type_counts,
                storage_size_mb=total_size / (1024 * 1024),
                oldest_context_age_days=oldest_age_days,
                most_accessed_context_id=str(most_accessed.id),
                compression_ratio=1.0,  # Default, would be calculated in production
                search_performance_ms=5.0,  # Default, would be measured
            )

            # Cache the results
            if self.cache_manager:
                await cache.set(cache_key, stats, dependencies=[f"agent:{agent_id}"])

            return stats

        except Exception as e:
            logger.error("Failed to get memory stats", agent_id=agent_id, error=str(e))
            raise
        finally:
            db_session.close()

    async def consolidate_memory(self, agent_id: str) -> dict[str, Any]:
        """Trigger memory consolidation for an agent."""

        db_session = self.db_manager.get_session()
        try:
            result = await self.memory_consolidator.consolidate_memory(
                agent_id, db_session
            )

            # Invalidate all caches for this agent after consolidation
            if self.cache_manager:
                await self.cache_manager.invalidate_dependency(f"agent:{agent_id}")

            return result
        finally:
            db_session.close()

    def _extract_topic(self, content: str) -> str:
        """Extract topic from content (simple implementation)."""
        # Simple topic extraction - in production would use NLP
        words = content.lower().split()

        # Look for important keywords
        keywords = []
        for word in words:
            if len(word) > 4 and word.isalpha():
                keywords.append(word)

        if keywords:
            return " ".join(keywords[:3])  # First 3 keywords
        else:
            return "general"


# Global context engine instance
context_engine = None


async def get_context_engine(database_url: str) -> ContextEngine:
    """Get the global context engine instance."""
    global context_engine
    if context_engine is None:
        context_engine = ContextEngine(database_url)
        await context_engine.initialize()
    return context_engine
