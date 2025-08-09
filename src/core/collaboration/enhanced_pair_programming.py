"""
Enhanced AI pair programming with advanced context sharing and collaboration.

This module extends the basic pair programming system with:
- Advanced context sharing between agents
- Real-time collaboration synchronization
- Enhanced AI assistance with pattern recognition
- Shared knowledge base integration
- Live coding session management
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from ..constants import Intervals
from ..context_engine import get_context_engine
from ..message_broker import MessageType, message_broker
from .pair_programming import (
    CollaborationResult,
    PairProgrammingSession,
    SessionPhase,
)

logger = structlog.get_logger()


class ContextShareType(Enum):
    """Types of context that can be shared between agents."""

    CODE_PATTERNS = "code_patterns"
    ARCHITECTURE_KNOWLEDGE = "architecture_knowledge"
    BUG_PATTERNS = "bug_patterns"
    BEST_PRACTICES = "best_practices"
    PERFORMANCE_INSIGHTS = "performance_insights"
    TESTING_STRATEGIES = "testing_strategies"
    PROJECT_HISTORY = "project_history"
    DOMAIN_KNOWLEDGE = "domain_knowledge"


class CollaborationMode(Enum):
    """Different modes of AI pair programming collaboration."""

    DRIVER_NAVIGATOR = "driver_navigator"  # Classic pair programming
    PING_PONG = "ping_pong"  # Alternating implementation/testing
    PARALLEL_CODING = "parallel_coding"  # Simultaneous development
    MOB_PROGRAMMING = "mob_programming"  # Multiple agents, one driver
    ASYNC_COLLABORATION = "async_collaboration"  # Non-real-time collaboration


@dataclass
class SharedContext:
    """Shared context between collaborating agents."""

    session_id: str
    context_type: ContextShareType
    content: dict[str, Any]
    source_agent: str
    timestamp: float = field(default_factory=time.time)
    relevance_score: float = 1.0
    expiry_time: float | None = None
    tags: set[str] = field(default_factory=set)

    def is_expired(self) -> bool:
        """Check if this context has expired."""
        if self.expiry_time is None:
            return False
        return time.time() > self.expiry_time


@dataclass
class CollaborationSession:
    """Enhanced collaboration session with context sharing."""

    session_id: str
    participants: list[str]
    mode: CollaborationMode
    project_context: dict[str, Any]

    # Context sharing
    shared_contexts: list[SharedContext] = field(default_factory=list)
    context_sync_interval: float = 5.0

    # Session state
    current_driver: str | None = None
    active_phase: SessionPhase = SessionPhase.PLANNING
    start_time: float = field(default_factory=time.time)
    last_sync: float = field(default_factory=time.time)

    # Collaboration metrics
    context_exchanges: int = 0
    pattern_matches: int = 0
    suggestions_applied: int = 0

    # Live state tracking
    active_files: set[str] = field(default_factory=set)
    cursor_positions: dict[str, tuple[int, int]] = field(
        default_factory=dict
    )  # agent -> (line, col)
    edit_history: list[dict[str, Any]] = field(default_factory=list)


class EnhancedAIPairProgramming:
    """Enhanced AI pair programming with advanced context sharing."""

    def __init__(self, base_session: PairProgrammingSession):
        self.base_session = base_session
        self.context_engine = None
        self.active_sessions: dict[str, CollaborationSession] = {}

        # Context sharing configuration
        self.context_retention_hours = 24
        self.max_shared_contexts_per_session = 100
        self.context_relevance_threshold = 0.3

        # Pattern recognition
        self.pattern_cache: dict[str, Any] = {}
        self.pattern_match_threshold = 0.7

        # Performance metrics
        self.collaboration_metrics = {
            "sessions_created": 0,
            "contexts_shared": 0,
            "patterns_recognized": 0,
            "suggestions_generated": 0,
            "collaboration_efficiency": 0.0,
        }

    async def initialize(self):
        """Initialize the enhanced pair programming system."""
        self.context_engine = await get_context_engine()

        # Load existing patterns and best practices
        await self._load_collaboration_patterns()

        logger.info("Enhanced AI pair programming initialized")

    async def start_enhanced_session(
        self,
        participants: list[str],
        mode: CollaborationMode,
        project_context: dict[str, Any],
        task_description: str,
    ) -> str:
        """Start an enhanced collaboration session."""
        session_id = f"enhanced_session_{uuid.uuid4().hex[:8]}"

        # Create enhanced collaboration session
        session = CollaborationSession(
            session_id=session_id,
            participants=participants,
            mode=mode,
            project_context=project_context,
            current_driver=participants[0] if participants else None,
        )

        self.active_sessions[session_id] = session

        # Initialize context sharing
        await self._initialize_session_context(session, task_description)

        # Start context synchronization
        asyncio.create_task(self._context_sync_loop(session_id))

        # Notify participants
        await self._notify_session_start(session)

        logger.info(
            "Enhanced collaboration session started",
            session_id=session_id,
            participants=participants,
            mode=mode.value,
        )

        self.collaboration_metrics["sessions_created"] += 1

        return session_id

    async def _initialize_session_context(
        self, session: CollaborationSession, task_description: str
    ):
        """Initialize shared context for the session."""
        # Extract relevant context from the project
        if self.context_engine:
            # Search for relevant patterns and best practices
            pattern_results = await self.context_engine.search(
                f"code patterns {task_description}", limit=10
            )

            for result in pattern_results:
                context = SharedContext(
                    session_id=session.session_id,
                    context_type=ContextShareType.CODE_PATTERNS,
                    content={"pattern": result.content, "relevance": result.score},
                    source_agent="system",
                    relevance_score=result.score,
                    tags={"pattern", "initialization"},
                )
                session.shared_contexts.append(context)

            # Search for relevant architectural knowledge
            arch_results = await self.context_engine.search(
                f"architecture design {task_description}", limit=5
            )

            for result in arch_results:
                context = SharedContext(
                    session_id=session.session_id,
                    context_type=ContextShareType.ARCHITECTURE_KNOWLEDGE,
                    content={"knowledge": result.content, "relevance": result.score},
                    source_agent="system",
                    relevance_score=result.score,
                    tags={"architecture", "initialization"},
                )
                session.shared_contexts.append(context)

    async def share_context(
        self,
        session_id: str,
        source_agent: str,
        context_type: ContextShareType,
        content: dict[str, Any],
        tags: set[str] = None,
    ) -> bool:
        """Share context between agents in a session."""
        if session_id not in self.active_sessions:
            logger.warning(
                "Attempted to share context in non-existent session",
                session_id=session_id,
            )
            return False

        session = self.active_sessions[session_id]

        # Create shared context
        shared_context = SharedContext(
            session_id=session_id,
            context_type=context_type,
            content=content,
            source_agent=source_agent,
            tags=tags or set(),
            expiry_time=time.time() + (self.context_retention_hours * 3600),
        )

        # Add to session
        session.shared_contexts.append(shared_context)
        session.context_exchanges += 1

        # Maintain size limit
        if len(session.shared_contexts) > self.max_shared_contexts_per_session:
            # Remove oldest, least relevant contexts
            session.shared_contexts.sort(key=lambda c: (c.relevance_score, c.timestamp))
            session.shared_contexts = session.shared_contexts[
                -self.max_shared_contexts_per_session :
            ]

        # Store in context engine for future use
        if self.context_engine:
            await self.context_engine.store_context(
                f"collaboration_{context_type.value}",
                json.dumps(content),
                {"session_id": session_id, "source_agent": source_agent},
            )

        # Notify other participants
        await self._notify_context_shared(session, shared_context, source_agent)

        logger.info(
            "Context shared in session",
            session_id=session_id,
            source_agent=source_agent,
            context_type=context_type.value,
        )

        self.collaboration_metrics["contexts_shared"] += 1

        return True

    async def get_relevant_context(
        self,
        session_id: str,
        requesting_agent: str,
        query: str,
        context_types: list[ContextShareType] = None,
    ) -> list[SharedContext]:
        """Get relevant context for an agent based on current work."""
        if session_id not in self.active_sessions:
            return []

        session = self.active_sessions[session_id]
        relevant_contexts = []

        # Filter by context types if specified
        contexts_to_search = session.shared_contexts
        if context_types:
            contexts_to_search = [
                c for c in session.shared_contexts if c.context_type in context_types
            ]

        # Simple relevance scoring based on query matching
        for context in contexts_to_search:
            if context.is_expired():
                continue

            # Calculate relevance score
            relevance = await self._calculate_context_relevance(query, context)

            if relevance >= self.context_relevance_threshold:
                context.relevance_score = relevance
                relevant_contexts.append(context)

        # Sort by relevance
        relevant_contexts.sort(key=lambda c: c.relevance_score, reverse=True)

        return relevant_contexts[:10]  # Return top 10 most relevant

    async def _calculate_context_relevance(
        self, query: str, context: SharedContext
    ) -> float:
        """Calculate relevance score between query and context."""
        # Simple implementation - in practice, this would use more sophisticated NLP
        query_lower = query.lower()
        content_str = json.dumps(context.content).lower()

        # Check for keyword overlap
        query_words = set(query_lower.split())
        content_words = set(content_str.split())

        if not query_words:
            return 0.0

        overlap = len(query_words.intersection(content_words))
        relevance = overlap / len(query_words)

        # Boost relevance for recency and source agent
        time_factor = max(
            0.1, 1.0 - (time.time() - context.timestamp) / 3600
        )  # Decay over hour
        relevance *= time_factor

        return min(1.0, relevance)

    async def suggest_code_patterns(
        self, session_id: str, current_code: str, context: str = ""
    ) -> list[dict[str, Any]]:
        """Suggest relevant code patterns based on current work."""
        if session_id not in self.active_sessions:
            return []

        session = self.active_sessions[session_id]
        suggestions = []

        # Get relevant patterns from shared context
        pattern_contexts = await self.get_relevant_context(
            session_id,
            "system",
            f"{current_code} {context}",
            [ContextShareType.CODE_PATTERNS, ContextShareType.BEST_PRACTICES],
        )

        for pattern_context in pattern_contexts:
            if "pattern" in pattern_context.content:
                suggestions.append(
                    {
                        "type": "pattern",
                        "content": pattern_context.content["pattern"],
                        "relevance": pattern_context.relevance_score,
                        "source": pattern_context.source_agent,
                        "tags": list(pattern_context.tags),
                    }
                )

        # Add pattern recognition from cached patterns
        cached_suggestions = await self._recognize_code_patterns(current_code, context)
        suggestions.extend(cached_suggestions)

        # Sort by relevance
        suggestions.sort(key=lambda s: s["relevance"], reverse=True)

        session.pattern_matches += len(suggestions)
        self.collaboration_metrics["patterns_recognized"] += len(suggestions)

        return suggestions[:5]  # Return top 5 suggestions

    async def _recognize_code_patterns(
        self, code: str, context: str
    ) -> list[dict[str, Any]]:
        """Recognize patterns in code using cached pattern database."""
        suggestions = []

        # Simple pattern recognition - in practice, this would use ML
        code_lower = code.lower()

        # Check for common patterns
        if "class" in code_lower and "def __init__" in code_lower:
            suggestions.append(
                {
                    "type": "pattern",
                    "content": "Consider using dataclasses for simple data containers",
                    "relevance": 0.8,
                    "source": "pattern_recognition",
                    "tags": ["refactoring", "dataclass"],
                }
            )

        if "for" in code_lower and "range(len(" in code_lower:
            suggestions.append(
                {
                    "type": "pattern",
                    "content": "Use enumerate() instead of range(len()) for cleaner iteration",
                    "relevance": 0.9,
                    "source": "pattern_recognition",
                    "tags": ["optimization", "pythonic"],
                }
            )

        if "try:" in code_lower and "except:" in code_lower:
            suggestions.append(
                {
                    "type": "pattern",
                    "content": "Specify exception types instead of bare except clauses",
                    "relevance": 0.7,
                    "source": "pattern_recognition",
                    "tags": ["error_handling", "best_practice"],
                }
            )

        return suggestions

    async def track_live_collaboration(
        self,
        session_id: str,
        agent_id: str,
        file_path: str,
        cursor_position: tuple[int, int],
        edit_action: dict[str, Any] = None,
    ):
        """Track live collaboration state."""
        if session_id not in self.active_sessions:
            return

        session = self.active_sessions[session_id]

        # Update cursor position
        session.cursor_positions[agent_id] = cursor_position

        # Track active files
        session.active_files.add(file_path)

        # Record edit action
        if edit_action:
            edit_record = {
                "timestamp": time.time(),
                "agent_id": agent_id,
                "file_path": file_path,
                "action": edit_action,
                "cursor_position": cursor_position,
            }
            session.edit_history.append(edit_record)

            # Maintain edit history size
            if len(session.edit_history) > 1000:
                session.edit_history = session.edit_history[
                    -500:
                ]  # Keep last 500 edits

    async def switch_driver(self, session_id: str, new_driver: str) -> bool:
        """Switch the active driver in a collaboration session."""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        if new_driver not in session.participants:
            logger.warning(
                "Attempted to switch to non-participant driver",
                session_id=session_id,
                driver=new_driver,
            )
            return False

        old_driver = session.current_driver
        session.current_driver = new_driver

        # Notify participants of driver switch
        await message_broker.publish(
            MessageType.AGENT_COMMUNICATION,
            {
                "session_id": session_id,
                "event": "driver_switched",
                "old_driver": old_driver,
                "new_driver": new_driver,
                "timestamp": time.time(),
            },
        )

        logger.info(
            "Driver switched in collaboration session",
            session_id=session_id,
            old_driver=old_driver,
            new_driver=new_driver,
        )

        return True

    async def _context_sync_loop(self, session_id: str):
        """Background loop to synchronize context between agents."""
        while session_id in self.active_sessions:
            try:
                session = self.active_sessions[session_id]

                # Clean up expired contexts
                current_time = time.time()
                session.shared_contexts = [
                    c for c in session.shared_contexts if not c.is_expired()
                ]

                # Update last sync time
                session.last_sync = current_time

                # Sleep until next sync
                await asyncio.sleep(session.context_sync_interval)

            except Exception as e:
                logger.error(
                    "Error in context sync loop", session_id=session_id, error=str(e)
                )
                await asyncio.sleep(Intervals.AGENT_ERROR_DELAY)

    async def _notify_session_start(self, session: CollaborationSession):
        """Notify participants that a session has started."""
        await message_broker.publish(
            MessageType.AGENT_COMMUNICATION,
            {
                "event": "enhanced_collaboration_started",
                "session_id": session.session_id,
                "participants": session.participants,
                "mode": session.mode.value,
                "current_driver": session.current_driver,
            },
        )

    async def _notify_context_shared(
        self,
        session: CollaborationSession,
        shared_context: SharedContext,
        source_agent: str,
    ):
        """Notify participants when context is shared."""
        # Send to all participants except the source
        recipients = [p for p in session.participants if p != source_agent]

        for recipient in recipients:
            await message_broker.publish(
                MessageType.AGENT_COMMUNICATION,
                {
                    "to": recipient,
                    "from": source_agent,
                    "event": "context_shared",
                    "session_id": session.session_id,
                    "context_type": shared_context.context_type.value,
                    "content": shared_context.content,
                    "relevance": shared_context.relevance_score,
                    "tags": list(shared_context.tags),
                },
            )

    async def _load_collaboration_patterns(self):
        """Load existing collaboration patterns from the context engine."""
        if not self.context_engine:
            return

        try:
            # Load common code patterns
            pattern_results = await self.context_engine.search(
                "collaboration patterns", limit=50
            )

            for result in pattern_results:
                # Store in pattern cache for quick access
                self.pattern_cache[result.content[:50]] = {
                    "content": result.content,
                    "score": result.score,
                    "metadata": result.metadata,
                }

            logger.info(f"Loaded {len(self.pattern_cache)} collaboration patterns")

        except Exception as e:
            logger.warning("Could not load collaboration patterns", error=str(e))

    async def get_collaboration_metrics(self, session_id: str = None) -> dict[str, Any]:
        """Get collaboration metrics for a session or overall."""
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            return {
                "session_id": session_id,
                "participants": session.participants,
                "mode": session.mode.value,
                "duration_minutes": (time.time() - session.start_time) / 60,
                "context_exchanges": session.context_exchanges,
                "pattern_matches": session.pattern_matches,
                "suggestions_applied": session.suggestions_applied,
                "active_files": list(session.active_files),
                "edit_count": len(session.edit_history),
            }

        # Return overall metrics
        return self.collaboration_metrics

    async def end_session(self, session_id: str) -> CollaborationResult:
        """End an enhanced collaboration session."""
        if session_id not in self.active_sessions:
            return CollaborationResult(
                success=False, session_id=session_id, error_message="Session not found"
            )

        session = self.active_sessions[session_id]

        # Calculate collaboration efficiency
        duration_hours = (time.time() - session.start_time) / 3600
        contexts_per_hour = (
            session.context_exchanges / duration_hours if duration_hours > 0 else 0
        )
        efficiency = min(
            1.0, contexts_per_hour / 10
        )  # Normalize to 10 contexts/hour = 100% efficiency

        # Create result
        result = CollaborationResult(
            success=True,
            session_id=session_id,
            collaboration_summary=f"Enhanced collaboration session with {len(session.participants)} participants",
            metrics={
                "duration_hours": duration_hours,
                "context_exchanges": session.context_exchanges,
                "pattern_matches": session.pattern_matches,
                "suggestions_applied": session.suggestions_applied,
                "collaboration_efficiency": efficiency,
                "files_worked": len(session.active_files),
                "edits_made": len(session.edit_history),
            },
        )

        # Update global metrics
        self.collaboration_metrics["collaboration_efficiency"] = (
            self.collaboration_metrics["collaboration_efficiency"] * 0.9
            + efficiency * 0.1
        )

        # Clean up
        del self.active_sessions[session_id]

        logger.info(
            "Enhanced collaboration session ended",
            session_id=session_id,
            duration_hours=duration_hours,
            efficiency=efficiency,
        )

        return result


# Global instance
enhanced_pair_programming = None


async def get_enhanced_pair_programming() -> EnhancedAIPairProgramming:
    """Get or create the global enhanced pair programming instance."""
    global enhanced_pair_programming

    if enhanced_pair_programming is None:
        from .pair_programming import PairProgrammingSession

        # Create base session (placeholder)
        base_session = PairProgrammingSession("system", "system")
        enhanced_pair_programming = EnhancedAIPairProgramming(base_session)
        await enhanced_pair_programming.initialize()

        logger.info("Enhanced AI pair programming system initialized")

    return enhanced_pair_programming
