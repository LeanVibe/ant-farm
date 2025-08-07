"""Advanced context engine with hierarchical memory and pattern recognition."""

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import structlog
from sqlalchemy import and_, desc, func
from sqlalchemy.orm import Session

from .context_engine import EmbeddingProvider, EmbeddingService
from .models import Context, get_database_manager

logger = structlog.get_logger()


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
    """Enhanced result from context search."""

    context: Context
    similarity_score: float
    relevance_score: float
    layer: MemoryLayer
    recency_bonus: float = 0.0
    importance_bonus: float = 0.0
    access_frequency_bonus: float = 0.0


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


@dataclass
class MemoryConsolidationStats:
    """Statistics from memory consolidation."""

    contexts_processed: int
    contexts_compressed: int
    contexts_promoted: int
    contexts_archived: int
    storage_saved_mb: float
    patterns_discovered: int
    consolidation_duration_ms: float


class PatternRecognizer:
    """Recognizes patterns in stored contexts."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.discovered_patterns: dict[str, KnowledgePattern] = {}

    async def discover_patterns(
        self, contexts: list[Context]
    ) -> list[KnowledgePattern]:
        """Discover patterns in a set of contexts."""
        patterns = []

        # Group contexts by similarity
        similarity_clusters = await self._cluster_contexts(contexts)

        for cluster in similarity_clusters:
            if len(cluster) >= 3:  # Need at least 3 similar contexts
                pattern = await self._extract_pattern(cluster)
                if pattern:
                    patterns.append(pattern)

        # Temporal patterns
        temporal_patterns = await self._discover_temporal_patterns(contexts)
        patterns.extend(temporal_patterns)

        # Success/failure patterns
        outcome_patterns = await self._discover_outcome_patterns(contexts)
        patterns.extend(outcome_patterns)

        return patterns

    async def _cluster_contexts(self, contexts: list[Context]) -> list[list[Context]]:
        """Cluster contexts by semantic similarity."""
        if not contexts:
            return []

        # Calculate similarity matrix
        embeddings = []
        for context in contexts:
            if context.embedding:
                embeddings.append(np.array(context.embedding))
            else:
                # Generate embedding if missing
                emb = await self.embedding_service.generate_embedding(context.content)
                embeddings.append(np.array(emb))

        if not embeddings:
            return []

        # Simple clustering based on cosine similarity
        clusters = []
        used_indices = set()

        for i, emb1 in enumerate(embeddings):
            if i in used_indices:
                continue

            cluster = [contexts[i]]
            used_indices.add(i)

            for j, emb2 in enumerate(embeddings[i + 1 :], i + 1):
                if j in used_indices:
                    continue

                similarity = np.dot(emb1, emb2) / (
                    np.linalg.norm(emb1) * np.linalg.norm(emb2)
                )
                if similarity > 0.8:  # High similarity threshold
                    cluster.append(contexts[j])
                    used_indices.add(j)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    async def _extract_pattern(
        self, contexts: list[Context]
    ) -> KnowledgePattern | None:
        """Extract a knowledge pattern from similar contexts."""
        try:
            # Analyze common elements
            common_themes = self._find_common_themes(contexts)

            if common_themes:
                pattern_id = hashlib.md5(
                    f"{common_themes}_{len(contexts)}_{time.time()}".encode()
                ).hexdigest()[:12]

                pattern = KnowledgePattern(
                    id=pattern_id,
                    pattern_type="semantic_cluster",
                    confidence_score=min(0.9, len(contexts) / 10.0),
                    supporting_contexts=[ctx.id for ctx in contexts],
                    pattern_description=f"Common pattern: {common_themes}",
                    discovered_at=time.time(),
                )

                return pattern

        except Exception as e:
            logger.error("Failed to extract pattern", error=str(e))

        return None

    def _find_common_themes(self, contexts: list[Context]) -> str:
        """Find common themes in context content."""
        # Simple keyword extraction (in practice, would use more sophisticated NLP)
        word_counts = {}

        for context in contexts:
            words = context.content.lower().split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Find words that appear in most contexts
        threshold = len(contexts) * 0.6
        common_words = [
            word for word, count in word_counts.items() if count >= threshold
        ]

        return " ".join(common_words[:5])  # Top 5 common words

    async def _discover_temporal_patterns(
        self, contexts: list[Context]
    ) -> list[KnowledgePattern]:
        """Discover temporal patterns in contexts."""
        patterns = []

        # Sort contexts by timestamp
        sorted_contexts = sorted(contexts, key=lambda c: c.created_at)

        # Look for recurring sequences
        # This is a simplified implementation
        if len(sorted_contexts) >= 5:
            pattern = KnowledgePattern(
                id=f"temporal_{int(time.time())}",
                pattern_type="temporal_sequence",
                confidence_score=0.6,
                supporting_contexts=[ctx.id for ctx in sorted_contexts[:5]],
                pattern_description="Temporal sequence pattern detected",
                discovered_at=time.time(),
            )
            patterns.append(pattern)

        return patterns

    async def _discover_outcome_patterns(
        self, contexts: list[Context]
    ) -> list[KnowledgePattern]:
        """Discover patterns related to task outcomes."""
        patterns = []

        success_contexts = [ctx for ctx in contexts if "success" in ctx.content.lower()]
        failure_contexts = [
            ctx
            for ctx in contexts
            if "error" in ctx.content.lower() or "fail" in ctx.content.lower()
        ]

        if len(success_contexts) >= 3:
            pattern = KnowledgePattern(
                id=f"success_{int(time.time())}",
                pattern_type="success_pattern",
                confidence_score=0.7,
                supporting_contexts=[ctx.id for ctx in success_contexts[:5]],
                pattern_description="Success pattern identified",
                discovered_at=time.time(),
            )
            patterns.append(pattern)

        if len(failure_contexts) >= 3:
            pattern = KnowledgePattern(
                id=f"failure_{int(time.time())}",
                pattern_type="failure_pattern",
                confidence_score=0.8,  # Failures are important to learn from
                supporting_contexts=[ctx.id for ctx in failure_contexts[:5]],
                pattern_description="Failure pattern identified for avoidance",
                discovered_at=time.time(),
            )
            patterns.append(pattern)

        return patterns


class MemoryConsolidator:
    """Handles memory consolidation and compression."""

    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.consolidation_rules = self._initialize_consolidation_rules()

    def _initialize_consolidation_rules(self) -> list[ConsolidationRule]:
        """Initialize default consolidation rules."""
        return [
            ConsolidationRule(
                source_layer=MemoryLayer.WORKING,
                target_layer=MemoryLayer.SHORT_TERM,
                age_threshold_hours=1.0,
                importance_threshold=0.3,
                access_count_threshold=0,
                compression_strategy="summarize",
            ),
            ConsolidationRule(
                source_layer=MemoryLayer.SHORT_TERM,
                target_layer=MemoryLayer.MEDIUM_TERM,
                age_threshold_hours=24.0,
                importance_threshold=0.5,
                access_count_threshold=2,
                compression_strategy="merge",
            ),
            ConsolidationRule(
                source_layer=MemoryLayer.MEDIUM_TERM,
                target_layer=MemoryLayer.LONG_TERM,
                age_threshold_hours=168.0,  # 1 week
                importance_threshold=0.7,
                access_count_threshold=5,
                compression_strategy="distill",
            ),
            ConsolidationRule(
                source_layer=MemoryLayer.LONG_TERM,
                target_layer=MemoryLayer.SEMANTIC,
                age_threshold_hours=720.0,  # 1 month
                importance_threshold=0.8,
                access_count_threshold=10,
                compression_strategy="distill",
            ),
        ]

    async def consolidate_memory(
        self, db_session: Session, agent_id: str
    ) -> MemoryConsolidationStats:
        """Consolidate memory according to rules."""
        start_time = time.time()
        stats = MemoryConsolidationStats(
            contexts_processed=0,
            contexts_compressed=0,
            contexts_promoted=0,
            contexts_archived=0,
            storage_saved_mb=0.0,
            patterns_discovered=0,
            consolidation_duration_ms=0.0,
        )

        current_time = time.time()

        for rule in self.consolidation_rules:
            # Find contexts eligible for consolidation
            age_threshold_seconds = rule.age_threshold_hours * 3600
            cutoff_time = current_time - age_threshold_seconds

            contexts = (
                db_session.query(Context)
                .filter(
                    and_(
                        Context.agent_id == agent_id,
                        Context.created_at < cutoff_time,
                        Context.importance_score >= rule.importance_threshold,
                        Context.metadata["layer"].astext == rule.source_layer.value,
                    )
                )
                .all()
            )

            stats.contexts_processed += len(contexts)

            for context in contexts:
                if rule.compression_strategy == "summarize":
                    await self._summarize_context(context)
                    stats.contexts_compressed += 1
                elif rule.compression_strategy == "merge":
                    await self._merge_related_contexts(db_session, context)
                    stats.contexts_compressed += 1
                elif rule.compression_strategy == "distill":
                    await self._distill_context(context)
                    stats.contexts_compressed += 1

                # Promote to next layer
                context.metadata["layer"] = rule.target_layer.value
                stats.contexts_promoted += 1

        db_session.commit()

        stats.consolidation_duration_ms = (time.time() - start_time) * 1000
        return stats

    async def _summarize_context(self, context: Context) -> None:
        """Summarize context content."""
        if len(context.content) > 500:  # Only summarize long content
            # In practice, would use Claude or another LLM for summarization
            summary = context.content[:200] + "... [summarized]"
            context.content = summary
            context.metadata["compressed"] = True

    async def _merge_related_contexts(
        self, db_session: Session, context: Context
    ) -> None:
        """Merge context with related contexts."""
        # Find related contexts
        related = (
            db_session.query(Context)
            .filter(
                and_(
                    Context.agent_id == context.agent_id,
                    Context.category == context.category,
                    Context.id != context.id,
                    Context.created_at.between(
                        context.created_at - 3600,  # 1 hour before
                        context.created_at + 3600,  # 1 hour after
                    ),
                )
            )
            .limit(3)
            .all()
        )

        if related:
            # Merge content
            merged_content = context.content
            for rel_ctx in related:
                merged_content += (
                    f"\n\n[Merged from {rel_ctx.id}]: {rel_ctx.content[:100]}..."
                )
                db_session.delete(rel_ctx)

            context.content = merged_content
            context.metadata["merged_from"] = [ctx.id for ctx in related]

    async def _distill_context(self, context: Context) -> None:
        """Distill context to essential information."""
        # Extract key insights (simplified)
        lines = context.content.split("\n")
        key_lines = [
            line
            for line in lines
            if any(
                keyword in line.lower()
                for keyword in ["important", "key", "result", "conclusion", "decision"]
            )
        ]

        if key_lines:
            context.content = "\n".join(key_lines)
            context.metadata["distilled"] = True


class AdvancedContextEngine:
    """Advanced context engine with hierarchical memory and pattern recognition."""

    def __init__(
        self,
        database_url: str,
        embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
    ):
        self.database_url = database_url
        self.db_manager = get_database_manager(database_url)
        self.embedding_service = EmbeddingService(embedding_provider)
        self.pattern_recognizer = PatternRecognizer(self.embedding_service)
        self.memory_consolidator = MemoryConsolidator(self.embedding_service)

        # Cache for recent contexts
        self.working_memory_cache: dict[str, list[Context]] = {}
        self.cache_max_size = 1000

        logger.info("Advanced context engine initialized")

    async def store_context(
        self,
        agent_id: str,
        content: str,
        importance_score: float = 0.5,
        category: str = "general",
        context_type: ContextType = ContextType.CONVERSATION,
        topic: str = None,
        metadata: dict[str, Any] = None,
    ) -> str:
        """Store context with enhanced metadata and automatic layering."""

        # Generate embedding
        embedding = await self.embedding_service.generate_embedding(content)

        # Determine initial memory layer
        layer = self._determine_initial_layer(importance_score, context_type)

        # Create context metadata
        context_metadata = {
            "layer": layer.value,
            "context_type": context_type.value,
            "embedding_provider": self.embedding_service.provider.value,
            "auto_generated": True,
            **(metadata or {}),
        }

        # Create context record
        context = Context(
            id=hashlib.md5(f"{agent_id}_{content}_{time.time()}".encode()).hexdigest()[
                :16
            ],
            agent_id=agent_id,
            content=content,
            embedding=embedding,
            importance_score=importance_score,
            category=category,
            topic=topic,
            metadata=context_metadata,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=0,
        )

        # Store in database
        db_session = self.db_manager.get_session()
        try:
            db_session.add(context)
            db_session.commit()

            # Add to working memory cache
            if agent_id not in self.working_memory_cache:
                self.working_memory_cache[agent_id] = []

            self.working_memory_cache[agent_id].append(context)

            # Maintain cache size
            if len(self.working_memory_cache[agent_id]) > self.cache_max_size:
                self.working_memory_cache[agent_id] = self.working_memory_cache[
                    agent_id
                ][-self.cache_max_size :]

            logger.info(
                "Context stored",
                context_id=context.id,
                agent_id=agent_id,
                layer=layer.value,
                importance=importance_score,
            )

            return context.id

        except Exception as e:
            db_session.rollback()
            logger.error("Failed to store context", error=str(e))
            raise
        finally:
            db_session.close()

    def _determine_initial_layer(
        self, importance_score: float, context_type: ContextType
    ) -> MemoryLayer:
        """Determine initial memory layer for new context."""
        # High importance contexts start in short-term memory
        if importance_score >= 0.8:
            return MemoryLayer.SHORT_TERM

        # Pattern and rule contexts are important
        if context_type in [ContextType.PATTERN, ContextType.RULE]:
            return MemoryLayer.MEDIUM_TERM

        # Success and error contexts need attention
        if context_type in [ContextType.SUCCESS, ContextType.ERROR]:
            return MemoryLayer.SHORT_TERM

        # Default to working memory
        return MemoryLayer.WORKING

    async def retrieve_context(
        self,
        query: str,
        agent_id: str,
        limit: int = 10,
        layer_filter: MemoryLayer = None,
        category_filter: str = None,
        min_importance: float = 0.0,
        include_patterns: bool = True,
    ) -> list[ContextSearchResult]:
        """Advanced context retrieval with layer-aware search."""

        search_start = time.time()

        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(query)
        query_vector = np.array(query_embedding)

        # Search in cache first (working memory)
        cache_results = []
        if agent_id in self.working_memory_cache:
            cache_results = await self._search_cache(
                query_vector, self.working_memory_cache[agent_id], limit // 2
            )

        # Search in database
        db_results = await self._search_database(
            query_vector, agent_id, limit, layer_filter, category_filter, min_importance
        )

        # Combine and rank results
        all_results = cache_results + db_results

        # Apply advanced ranking
        ranked_results = self._rank_search_results(all_results, query)

        # Include relevant patterns if requested
        if include_patterns:
            pattern_results = await self._search_patterns(query, agent_id)
            ranked_results.extend(pattern_results)

        # Sort by final relevance score and limit
        final_results = sorted(
            ranked_results, key=lambda r: r.relevance_score, reverse=True
        )[:limit]

        # Update access statistics
        await self._update_access_stats(final_results)

        search_duration = (time.time() - search_start) * 1000
        logger.info(
            "Context search completed",
            query_length=len(query),
            results_count=len(final_results),
            search_duration_ms=search_duration,
        )

        return final_results

    async def _search_cache(
        self, query_vector: np.ndarray, cache_contexts: list[Context], limit: int
    ) -> list[ContextSearchResult]:
        """Search working memory cache."""
        results = []

        for context in cache_contexts[-100:]:  # Only search recent contexts
            if context.embedding:
                context_vector = np.array(context.embedding)
                similarity = np.dot(query_vector, context_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(context_vector)
                )

                if similarity > 0.3:  # Minimum similarity threshold
                    result = ContextSearchResult(
                        context=context,
                        similarity_score=similarity,
                        relevance_score=similarity,
                        layer=MemoryLayer(context.metadata.get("layer", "working")),
                        recency_bonus=0.1,  # Boost for being in working memory
                    )
                    results.append(result)

        return sorted(results, key=lambda r: r.similarity_score, reverse=True)[:limit]

    async def _search_database(
        self,
        query_vector: np.ndarray,
        agent_id: str,
        limit: int,
        layer_filter: MemoryLayer = None,
        category_filter: str = None,
        min_importance: float = 0.0,
    ) -> list[ContextSearchResult]:
        """Search database with vector similarity."""
        db_session = self.db_manager.get_session()

        try:
            # Build query filters
            filters = [Context.agent_id == agent_id]

            if layer_filter:
                filters.append(Context.metadata["layer"].astext == layer_filter.value)

            if category_filter:
                filters.append(Context.category == category_filter)

            if min_importance > 0.0:
                filters.append(Context.importance_score >= min_importance)

            # Get contexts (simplified - in practice would use pgvector for similarity search)
            contexts = (
                db_session.query(Context)
                .filter(and_(*filters))
                .order_by(desc(Context.importance_score))
                .limit(limit * 2)
                .all()
            )

            results = []
            for context in contexts:
                if context.embedding:
                    context_vector = np.array(context.embedding)
                    similarity = np.dot(query_vector, context_vector) / (
                        np.linalg.norm(query_vector) * np.linalg.norm(context_vector)
                    )

                    if similarity > 0.2:  # Database threshold can be lower
                        result = ContextSearchResult(
                            context=context,
                            similarity_score=similarity,
                            relevance_score=similarity,
                            layer=MemoryLayer(context.metadata.get("layer", "working")),
                        )
                        results.append(result)

            return sorted(results, key=lambda r: r.similarity_score, reverse=True)[
                :limit
            ]

        finally:
            db_session.close()

    def _rank_search_results(
        self, results: list[ContextSearchResult], query: str
    ) -> list[ContextSearchResult]:
        """Apply advanced ranking to search results."""
        current_time = time.time()

        for result in results:
            context = result.context

            # Recency bonus (more recent = higher score)
            age_hours = (current_time - context.created_at) / 3600
            result.recency_bonus = max(0, 1.0 - (age_hours / 168))  # Decay over a week

            # Importance bonus
            result.importance_bonus = context.importance_score

            # Access frequency bonus
            result.access_frequency_bonus = min(1.0, context.access_count / 10.0)

            # Calculate final relevance score
            result.relevance_score = (
                result.similarity_score * 0.4
                + result.importance_bonus * 0.3
                + result.recency_bonus * 0.2
                + result.access_frequency_bonus * 0.1
            )

        return results

    async def _search_patterns(
        self, query: str, agent_id: str
    ) -> list[ContextSearchResult]:
        """Search for relevant knowledge patterns."""
        # This would search discovered patterns
        # For now, return empty list
        return []

    async def _update_access_stats(self, results: list[ContextSearchResult]) -> None:
        """Update access statistics for retrieved contexts."""
        if not results:
            return

        db_session = self.db_manager.get_session()
        try:
            for result in results:
                context = result.context
                context.access_count += 1
                context.last_accessed = time.time()

            db_session.commit()
        except Exception as e:
            db_session.rollback()
            logger.error("Failed to update access stats", error=str(e))
        finally:
            db_session.close()

    async def consolidate_memory(self, agent_id: str) -> MemoryConsolidationStats:
        """Perform memory consolidation for an agent."""
        db_session = self.db_manager.get_session()

        try:
            # Perform consolidation
            stats = await self.memory_consolidator.consolidate_memory(
                db_session, agent_id
            )

            # Discover new patterns
            contexts = (
                db_session.query(Context)
                .filter(Context.agent_id == agent_id)
                .order_by(desc(Context.created_at))
                .limit(100)
                .all()
            )

            new_patterns = await self.pattern_recognizer.discover_patterns(contexts)
            stats.patterns_discovered = len(new_patterns)

            # Store discovered patterns
            for pattern in new_patterns:
                self.pattern_recognizer.discovered_patterns[pattern.id] = pattern

            logger.info(
                "Memory consolidation completed", agent_id=agent_id, **stats.__dict__
            )

            return stats

        finally:
            db_session.close()

    async def get_memory_stats(self, agent_id: str) -> dict[str, Any]:
        """Get comprehensive memory statistics."""
        db_session = self.db_manager.get_session()

        try:
            # Basic counts
            total_contexts = (
                db_session.query(func.count(Context.id))
                .filter(Context.agent_id == agent_id)
                .scalar()
            )

            # Layer distribution
            layer_stats = {}
            for layer in MemoryLayer:
                count = (
                    db_session.query(func.count(Context.id))
                    .filter(
                        and_(
                            Context.agent_id == agent_id,
                            Context.metadata["layer"].astext == layer.value,
                        )
                    )
                    .scalar()
                )
                layer_stats[layer.value] = count

            # Importance distribution
            importance_stats = {
                "high": db_session.query(func.count(Context.id))
                .filter(
                    and_(Context.agent_id == agent_id, Context.importance_score >= 0.7)
                )
                .scalar(),
                "medium": db_session.query(func.count(Context.id))
                .filter(
                    and_(
                        Context.agent_id == agent_id,
                        Context.importance_score >= 0.3,
                        Context.importance_score < 0.7,
                    )
                )
                .scalar(),
                "low": db_session.query(func.count(Context.id))
                .filter(
                    and_(Context.agent_id == agent_id, Context.importance_score < 0.3)
                )
                .scalar(),
            }

            return {
                "total_contexts": total_contexts,
                "layer_distribution": layer_stats,
                "importance_distribution": importance_stats,
                "patterns_discovered": len(self.pattern_recognizer.discovered_patterns),
                "cache_size": len(self.working_memory_cache.get(agent_id, [])),
                "embedding_provider": self.embedding_service.provider.value,
            }

        finally:
            db_session.close()


# Global instance
_advanced_context_engine: AdvancedContextEngine | None = None


async def get_advanced_context_engine(
    database_url: str,
    embedding_provider: EmbeddingProvider = EmbeddingProvider.SENTENCE_TRANSFORMERS,
) -> AdvancedContextEngine:
    """Get or create the advanced context engine singleton."""
    global _advanced_context_engine

    if _advanced_context_engine is None:
        _advanced_context_engine = AdvancedContextEngine(
            database_url, embedding_provider
        )

    return _advanced_context_engine
