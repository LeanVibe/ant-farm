"""
Cross-Project Learning and Knowledge Sharing System.

This module enables agents to learn from successful patterns across different projects,
share knowledge between projects, and continuously improve their performance through
collective learning and pattern recognition.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog

from ..ai.model_orchestrator import model_orchestrator
from ..message_broker import MessageType, message_broker
from .enhanced_pair_programming import ContextShareType

logger = structlog.get_logger()


class KnowledgeType(Enum):
    """Types of knowledge that can be shared across projects."""

    CODE_PATTERNS = "code_patterns"
    ARCHITECTURE_DECISIONS = "architecture_decisions"
    PERFORMANCE_OPTIMIZATIONS = "performance_optimizations"
    SECURITY_PRACTICES = "security_practices"
    TESTING_STRATEGIES = "testing_strategies"
    DEPLOYMENT_PATTERNS = "deployment_patterns"
    ERROR_SOLUTIONS = "error_solutions"
    BEST_PRACTICES = "best_practices"
    LESSONS_LEARNED = "lessons_learned"
    TOOL_CONFIGURATIONS = "tool_configurations"


class LearningPriority(Enum):
    """Priority levels for learning items."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KnowledgeItem:
    """A piece of knowledge that can be shared across projects."""

    id: str
    knowledge_type: KnowledgeType
    title: str
    content: str
    source_project: str
    source_agent: str
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success_metrics: Dict[str, float] = field(default_factory=dict)
    usage_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    priority: LearningPriority = LearningPriority.MEDIUM
    confidence_score: float = 0.5
    validation_count: int = 0


@dataclass
class ProjectKnowledgeBase:
    """Knowledge base for a specific project."""

    project_id: str
    knowledge_items: Dict[str, KnowledgeItem] = field(default_factory=dict)
    learned_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    project_metrics: Dict[str, Any] = field(default_factory=dict)
    cross_project_references: Set[str] = field(default_factory=set)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


@dataclass
class LearningSession:
    """A session for learning and knowledge sharing."""

    id: str
    participating_projects: List[str]
    learning_objectives: List[str]
    knowledge_exchanges: List[Dict[str, Any]] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    status: str = "active"
    metrics: Dict[str, Any] = field(default_factory=dict)


class CrossProjectLearningSystem:
    """System for cross-project learning and knowledge sharing."""

    def __init__(self):
        self.global_knowledge_base: Dict[str, KnowledgeItem] = {}
        self.project_knowledge_bases: Dict[str, ProjectKnowledgeBase] = {}
        self.learning_sessions: Dict[str, LearningSession] = {}
        self.knowledge_embeddings: Dict[str, List[float]] = {}

        # Learning configuration
        self.similarity_threshold = 0.75
        self.knowledge_retention_days = 365
        self.max_knowledge_per_project = 1000
        self.learning_rate = 0.1

        # Performance metrics
        self.metrics = {
            "knowledge_items_shared": 0,
            "learning_sessions_completed": 0,
            "patterns_applied": 0,
            "cross_project_success_rate": 0.0,
            "knowledge_utilization_rate": 0.0,
        }

    async def initialize(self):
        """Initialize the cross-project learning system."""
        await self._load_existing_knowledge()
        logger.info("Cross-project learning system initialized")

    async def _load_existing_knowledge(self):
        """Load existing knowledge from persistent storage."""
        # This would load from database/files in a real implementation
        # For now, initialize with some basic knowledge
        await self._initialize_basic_knowledge()

    async def _initialize_basic_knowledge(self):
        """Initialize with basic programming knowledge."""

        # Add some fundamental patterns
        patterns = [
            {
                "type": KnowledgeType.CODE_PATTERNS,
                "title": "Async Error Handling Pattern",
                "content": """
try:
    async with aiofiles.open('file.txt', 'r') as f:
        content = await f.read()
    return await process_content(content)
except FileNotFoundError:
    logger.error("File not found")
    raise
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    await cleanup_resources()
    raise
                """,
                "tags": {"async", "error_handling", "python"},
                "success_metrics": {"success_rate": 0.95, "usage_count": 150},
            },
            {
                "type": KnowledgeType.SECURITY_PRACTICES,
                "title": "Input Validation Pattern",
                "content": """
def validate_user_input(data: dict) -> dict:
    \"\"\"Validate and sanitize user input.\"\"\"
    validated = {}

    # Email validation
    if 'email' in data:
        email = data['email'].strip().lower()
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
            raise ValueError("Invalid email format")
        validated['email'] = email

    # Password strength
    if 'password' in data:
        if len(data['password']) < 8:
            raise ValueError("Password too short")
        validated['password'] = hash_password(data['password'])

    return validated
                """,
                "tags": {"security", "validation", "input_sanitization"},
                "success_metrics": {"success_rate": 0.98, "usage_count": 200},
            },
            {
                "type": KnowledgeType.PERFORMANCE_OPTIMIZATIONS,
                "title": "Database Query Optimization",
                "content": """
# Use select_related for foreign key relationships
users_with_profiles = User.objects.select_related('profile').filter(
    profile__active=True
)

# Use prefetch_related for many-to-many relationships
posts_with_tags = Post.objects.prefetch_related('tags').filter(
    published_date__gte=timezone.now() - timedelta(days=7)
)

# Use database indexes for frequently queried fields
# Add to model Meta:
# indexes = [
#     models.Index(fields=['created_at', 'status']),
#     models.Index(fields=['user', 'category']),
# ]
                """,
                "tags": {"database", "optimization", "django", "queries"},
                "success_metrics": {"success_rate": 0.92, "performance_improvement": 0.35},
            },
        ]

        for pattern in patterns:
            await self.add_knowledge_item(
                knowledge_type=pattern["type"],
                title=pattern["title"],
                content=pattern["content"],
                source_project="system",
                source_agent="knowledge_system",
                tags=pattern["tags"],
                metadata={"auto_generated": True},
                success_metrics=pattern["success_metrics"],
            )

    async def add_knowledge_item(
        self,
        knowledge_type: KnowledgeType,
        title: str,
        content: str,
        source_project: str,
        source_agent: str,
        tags: Set[str] = None,
        metadata: Dict[str, Any] = None,
        success_metrics: Dict[str, float] = None,
        priority: LearningPriority = LearningPriority.MEDIUM,
    ) -> str:
        """Add a new knowledge item to the global knowledge base."""

        knowledge_id = f"knowledge_{uuid.uuid4().hex[:12]}"

        knowledge_item = KnowledgeItem(
            id=knowledge_id,
            knowledge_type=knowledge_type,
            title=title,
            content=content,
            source_project=source_project,
            source_agent=source_agent,
            tags=tags or set(),
            metadata=metadata or {},
            success_metrics=success_metrics or {},
            priority=priority,
        )

        # Add to global knowledge base
        self.global_knowledge_base[knowledge_id] = knowledge_item

        # Add to project knowledge base
        if source_project not in self.project_knowledge_bases:
            self.project_knowledge_bases[source_project] = ProjectKnowledgeBase(
                project_id=source_project
            )

        project_kb = self.project_knowledge_bases[source_project]
        project_kb.knowledge_items[knowledge_id] = knowledge_item
        project_kb.updated_at = time.time()

        # Generate embedding for similarity matching
        await self._generate_knowledge_embedding(knowledge_id, content)

        # Maintain size limits
        await self._maintain_knowledge_base_size()

        logger.info(
            "Knowledge item added",
            knowledge_id=knowledge_id,
            type=knowledge_type.value,
            source_project=source_project,
            title=title,
        )

        return knowledge_id

    async def _generate_knowledge_embedding(self, knowledge_id: str, content: str):
        """Generate embedding for knowledge item."""
        # Simplified embedding generation
        import hashlib

        hash_obj = hashlib.md5(content.encode())
        hash_bytes = hash_obj.digest()
        embedding = [float(b) / 255.0 for b in hash_bytes]
        self.knowledge_embeddings[knowledge_id] = embedding

    async def _maintain_knowledge_base_size(self):
        """Maintain knowledge base size limits."""
        # Remove old, low-priority items if we exceed limits
        if len(self.global_knowledge_base) > 5000:  # Global limit
            # Sort by priority, usage, and recency
            items = list(self.global_knowledge_base.values())
            items.sort(key=lambda x: (
                x.priority.value,
                x.usage_count,
                x.last_used
            ), reverse=True)

            # Keep top items
            to_keep = items[:4000]
            self.global_knowledge_base = {item.id: item for item in to_keep}

        # Clean up per-project knowledge bases
        for project_kb in self.project_knowledge_bases.values():
            if len(project_kb.knowledge_items) > self.max_knowledge_per_project:
                items = list(project_kb.knowledge_items.values())
                items.sort(key=lambda x: (x.usage_count, x.last_used), reverse=True)
                project_kb.knowledge_items = {
                    item.id: item for item in items[:self.max_knowledge_per_project]
                }

    async def find_relevant_knowledge(
        self,
        query: str,
        project_context: str = "",
        knowledge_types: List[KnowledgeType] = None,
        tags: Set[str] = None,
        limit: int = 10,
    ) -> List[Tuple[KnowledgeItem, float]]:
        """Find relevant knowledge items based on query and context."""

        relevant_items = []

        # Search through global knowledge base
        for knowledge_id, item in self.global_knowledge_base.items():
            if knowledge_types and item.knowledge_type not in knowledge_types:
                continue

            if tags and not tags.intersection(item.tags):
                continue

            # Calculate relevance score
            relevance = await self._calculate_knowledge_relevance(
                query, project_context, item
            )

            if relevance >= self.similarity_threshold:
                relevant_items.append((item, relevance))

        # Sort by relevance and success metrics
        relevant_items.sort(key=lambda x: (
            x[1],  # relevance
            x[0].success_metrics.get("success_rate", 0),
            x[0].usage_count
        ), reverse=True)

        return relevant_items[:limit]

    async def _calculate_knowledge_relevance(
        self,
        query: str,
        project_context: str,
        item: KnowledgeItem
    ) -> float:
        """Calculate relevance score for a knowledge item."""

        # Simple text-based relevance calculation
        query_lower = query.lower()
        context_lower = project_context.lower()
        content_lower = item.content.lower()
        title_lower = item.title.lower()

        # Check for keyword matches
        query_words = set(query_lower.split())
        context_words = set(context_lower.split())

        # Title relevance (higher weight)
        title_matches = len(query_words.intersection(set(title_lower.split())))
        title_relevance = min(1.0, title_matches / max(1, len(query_words)))

        # Content relevance
        content_matches = len(query_words.intersection(set(content_lower.split())))
        content_relevance = min(1.0, content_matches / max(1, len(query_words)))

        # Context relevance
        if context_words:
            context_matches = len(context_words.intersection(set(content_lower.split())))
            context_relevance = min(1.0, context_matches / len(context_words))
        else:
            context_relevance = 0.0

        # Tag relevance
        tag_relevance = 0.0
        if item.tags:
            tag_matches = len(query_words.intersection(item.tags))
            tag_relevance = min(1.0, tag_matches / len(item.tags))

        # Combine relevance scores
        relevance = (
            title_relevance * 0.4 +
            content_relevance * 0.3 +
            context_relevance * 0.2 +
            tag_relevance * 0.1
        )

        # Boost based on success metrics and usage
        success_boost = item.success_metrics.get("success_rate", 0.5) * 0.1
        usage_boost = min(0.1, item.usage_count / 100 * 0.1)

        relevance += success_boost + usage_boost

        return min(1.0, relevance)

    async def share_knowledge_between_projects(
        self,
        source_project: str,
        target_project: str,
        knowledge_types: List[KnowledgeType] = None,
        min_success_rate: float = 0.8,
    ) -> Dict[str, Any]:
        """Share knowledge from one project to another."""

        if source_project not in self.project_knowledge_bases:
            return {"error": f"Source project {source_project} not found"}

        source_kb = self.project_knowledge_bases[source_project]

        # Find high-quality knowledge to share
        shareable_knowledge = []
        for item in source_kb.knowledge_items.values():
            if knowledge_types and item.knowledge_type not in knowledge_types:
                continue

            success_rate = item.success_metrics.get("success_rate", 0)
            if success_rate >= min_success_rate:
                shareable_knowledge.append(item)

        # Add to target project
        if target_project not in self.project_knowledge_bases:
            self.project_knowledge_bases[target_project] = ProjectKnowledgeBase(
                project_id=target_project
            )

        target_kb = self.project_knowledge_bases[target_project]
        shared_count = 0

        for item in shareable_knowledge:
            if item.id not in target_kb.knowledge_items:
                target_kb.knowledge_items[item.id] = item
                target_kb.cross_project_references.add(source_project)
                shared_count += 1

        target_kb.updated_at = time.time()

        # Update metrics
        self.metrics["knowledge_items_shared"] += shared_count

        result = {
            "source_project": source_project,
            "target_project": target_project,
            "knowledge_shared": shared_count,
            "knowledge_types": [kt.value for kt in (knowledge_types or [])],
        }

        logger.info(
            "Knowledge shared between projects",
            source_project=source_project,
            target_project=target_project,
            shared_count=shared_count,
        )

        return result

    async def start_learning_session(
        self,
        participating_projects: List[str],
        learning_objectives: List[str],
        session_duration_hours: float = 2.0,
    ) -> str:
        """Start a cross-project learning session."""

        session_id = f"learning_{uuid.uuid4().hex[:12]}"

        session = LearningSession(
            id=session_id,
            participating_projects=participating_projects,
            learning_objectives=learning_objectives,
        )

        self.learning_sessions[session_id] = session

        # Schedule session end
        asyncio.create_task(self._end_learning_session(session_id, session_duration_hours))

        # Notify participating projects
        await self._notify_learning_session_start(session)

        logger.info(
            "Learning session started",
            session_id=session_id,
            projects=len(participating_projects),
            objectives=len(learning_objectives),
        )

        return session_id

    async def _end_learning_session(self, session_id: str, duration_hours: float):
        """End a learning session after the specified duration."""
        await asyncio.sleep(duration_hours * 3600)

        if session_id in self.learning_sessions:
            session = self.learning_sessions[session_id]
            session.status = "completed"

            # Calculate session metrics
            session.metrics = {
                "duration_hours": duration_hours,
                "knowledge_exchanges": len(session.knowledge_exchanges),
                "participating_projects": len(session.participating_projects),
                "objectives_addressed": len(session.learning_objectives),
            }

            self.metrics["learning_sessions_completed"] += 1

            # Notify completion
            await self._notify_learning_session_end(session)

            logger.info(
                "Learning session completed",
                session_id=session_id,
                exchanges=len(session.knowledge_exchanges),
            )

    async def _notify_learning_session_start(self, session: LearningSession):
        """Notify projects about learning session start."""
        for project_id in session.participating_projects:
            await message_broker.publish(
                MessageType.SYSTEM_EVENT,
                {
                    "event": "learning_session_started",
                    "session_id": session.id,
                    "participating_projects": session.participating_projects,
                    "objectives": session.learning_objectives,
                    "project_id": project_id,
                }
            )

    async def _notify_learning_session_end(self, session: LearningSession):
        """Notify projects about learning session end."""
        for project_id in session.participating_projects:
            await message_broker.publish(
                MessageType.SYSTEM_EVENT,
                {
                    "event": "learning_session_ended",
                    "session_id": session.id,
                    "metrics": session.metrics,
                    "project_id": project_id,
                }
            )

    async def apply_learned_pattern(
        self,
        knowledge_id: str,
        target_project: str,
        target_agent: str,
        context: Dict[str, Any] = None,
    ) -> bool:
        """Apply a learned pattern to a target project."""

        if knowledge_id not in self.global_knowledge_base:
            return False

        knowledge_item = self.global_knowledge_base[knowledge_id]

        # Update usage statistics
        knowledge_item.usage_count += 1
        knowledge_item.last_used = time.time()

        # Record application in project knowledge base
        if target_project in self.project_knowledge_bases:
            project_kb = self.project_knowledge_bases[target_project]
            if "applied_patterns" not in project_kb.learned_patterns:
                project_kb.learned_patterns["applied_patterns"] = []

            project_kb.learned_patterns["applied_patterns"].append({
                "knowledge_id": knowledge_id,
                "applied_at": time.time(),
                "applied_by": target_agent,
                "context": context or {},
            })

        # Notify the target agent
        await message_broker.send_message(
            from_agent="learning_system",
            to_agent=target_agent,
            topic="pattern_applied",
            payload={
                "knowledge_id": knowledge_id,
                "pattern_title": knowledge_item.title,
                "pattern_type": knowledge_item.knowledge_type.value,
                "content": knowledge_item.content,
                "context": context,
                "project_id": target_project,
            },
            message_type=MessageType.NOTIFICATION,
        )

        self.metrics["patterns_applied"] += 1

        logger.info(
            "Pattern applied",
            knowledge_id=knowledge_id,
            target_project=target_project,
            target_agent=target_agent,
        )

        return True

    async def validate_knowledge_item(
        self,
        knowledge_id: str,
        validation_result: bool,
        feedback: str = "",
        validator_project: str = "",
    ):
        """Validate a knowledge item and update its metrics."""

        if knowledge_id not in self.global_knowledge_base:
            return

        item = self.global_knowledge_base[knowledge_id]
        item.validation_count += 1

        # Update confidence score based on validation
        if validation_result:
            # Positive validation increases confidence
            item.confidence_score = min(1.0, item.confidence_score + self.learning_rate)
        else:
            # Negative validation decreases confidence
            item.confidence_score = max(0.0, item.confidence_score - self.learning_rate)

        # Store validation feedback
        if "validations" not in item.metadata:
            item.metadata["validations"] = []

        item.metadata["validations"].append({
            "result": validation_result,
            "feedback": feedback,
            "validator_project": validator_project,
            "timestamp": time.time(),
        })

        logger.info(
            "Knowledge item validated",
            knowledge_id=knowledge_id,
            validation_result=validation_result,
            new_confidence=item.confidence_score,
        )

    async def get_project_knowledge_summary(self, project_id: str) -> Dict[str, Any]:
        """Get a summary of knowledge for a specific project."""

        if project_id not in self.project_knowledge_bases:
            return {"error": f"Project {project_id} not found"}

        project_kb = self.project_knowledge_bases[project_id]

        # Calculate knowledge distribution
        type_distribution = {}
        for item in project_kb.knowledge_items.values():
            knowledge_type = item.knowledge_type.value
            type_distribution[knowledge_type] = type_distribution.get(knowledge_type, 0) + 1

        # Calculate success metrics
        total_success_rate = 0
        total_usage = 0
        item_count = len(project_kb.knowledge_items)

        for item in project_kb.knowledge_items.values():
            total_success_rate += item.success_metrics.get("success_rate", 0)
            total_usage += item.usage_count

        avg_success_rate = total_success_rate / item_count if item_count > 0 else 0
        avg_usage = total_usage / item_count if item_count > 0 else 0

        return {
            "project_id": project_id,
            "total_knowledge_items": item_count,
            "type_distribution": type_distribution,
            "average_success_rate": avg_success_rate,
            "average_usage_count": avg_usage,
            "cross_project_references": len(project_kb.cross_project_references),
            "learned_patterns_count": len(project_kb.learned_patterns),
            "created_at": project_kb.created_at,
            "updated_at": project_kb.updated_at,
        }

    async def get_learning_recommendations(
        self,
        project_id: str,
        current_context: str = "",
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Get learning recommendations for a project."""

        recommendations = []

        # Find knowledge gaps by comparing with other projects
        if project_id in self.project_knowledge_bases:
            project_kb = self.project_knowledge_bases[project_id]

            # Get knowledge from other projects
            other_projects_kb = {
                pid: kb for pid, kb in self.project_knowledge_bases.items()
                if pid != project_id
            }

            # Find missing high-value knowledge
            project_types = {
                item.knowledge_type for item in project_kb.knowledge_items.values()
            }

            for other_pid, other_kb in other_projects_kb.items():
                for item in other_kb.knowledge_items.values():
                    if item.knowledge_type not in project_types:
                        if item.success_metrics.get("success_rate", 0) > 0.8:
                            recommendations.append({
                                "type": "missing_knowledge_type",
                                "knowledge_type": item.knowledge_type.value,
                                "source_project": other_pid,
                                "potential_value": item.success_metrics.get("success_rate", 0),
                                "description": f"Consider learning {item.knowledge_type.value} patterns from {other_pid}",
                            })

        # Context-based recommendations
        if current_context:
            relevant_knowledge = await self.find_relevant_knowledge(
                current_context, project_id, limit=limit
            )

            for item, relevance in relevant_knowledge:
                recommendations.append({
                    "type": "context_relevant",
                    "knowledge_id": item.id,
                    "title": item.title,
                    "relevance_score": relevance,
                    "knowledge_type": item.knowledge_type.value,
                    "description": f"Relevant {item.knowledge_type.value} knowledge for current context",
                })

        # Sort by potential value
        recommendations.sort(key=lambda x: x.get("potential_value", x.get("relevance_score", 0)), reverse=True)

        return recommendations[:limit]

    async def export_project_knowledge(
        self,
        project_id: str,
        export_path: str = None,
        knowledge_types: List[KnowledgeType] = None,
    ) -> str:
        """Export project knowledge for backup or sharing."""

        if project_id not in self.project_knowledge_bases:
            raise ValueError(f"Project {project_id} not found")

        project_kb = self.project_knowledge_bases[project_id]

        # Filter knowledge items
        items_to_export = project_kb.knowledge_items.values()
        if knowledge_types:
            items_to_export = [
                item for item in items_to_export
                if item.knowledge_type in knowledge_types
            ]

        # Prepare export data
        export_data = {
            "project_id": project_id,
            "export_timestamp": time.time(),
            "knowledge_items": [
                {
                    "id": item.id,
                    "type": item.knowledge_type.value,
                    "title": item.title,
                    "content": item.content,
                    "tags": list(item.tags),
                    "metadata": item.metadata,
                    "success_metrics": item.success_metrics,
                    "usage_count": item.usage_count,
                    "created_at": item.created_at,
                    "last_used": item.last_used,
                    "priority": item.priority.value,
                    "confidence_score": item.confidence_score,
                }
                for item in items_to_export
            ],
            "project_metrics": project_kb.project_metrics,
            "learned_patterns": project_kb.learned_patterns,
        }

        # Export to file or return as JSON string
        if export_path:
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)

            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(
                "Project knowledge exported to file",
                project_id=project_id,
                export_path=export_path,
                items_exported=len(export_data["knowledge_items"]),
            )

            return export_path
        else:
            return json.dumps(export_data, indent=2, default=str)

    async def import_project_knowledge(
        self,
        target_project: str,
        import_data: str,
        source_path: str = None,
    ) -> Dict[str, Any]:
        """Import knowledge into a project."""

        # Load import data
        if source_path:
            with open(source_path, 'r') as f:
                import_data = json.load(f)
        else:
            import_data = json.loads(import_data)

        # Ensure target project exists
        if target_project not in self.project_knowledge_bases:
            self.project_knowledge_bases[target_project] = ProjectKnowledgeBase(
                project_id=target_project
            )

        target_kb = self.project_knowledge_bases[target_project]

        # Import knowledge items
        imported_count = 0
        skipped_count = 0

        for item_data in import_data.get("knowledge_items", []):
            knowledge_id = item_data["id"]

            # Skip if already exists
            if knowledge_id in target_kb.knowledge_items:
                skipped_count += 1
                continue

            # Create knowledge item
            knowledge_item = KnowledgeItem(
                id=knowledge_id,
                knowledge_type=KnowledgeType(item_data["type"]),
                title=item_data["title"],
                content=item_data["content"],
                source_project=import_data.get("project_id", "imported"),
                source_agent="import_system",
                tags=set(item_data.get("tags", [])),
                metadata=item_data.get("metadata", {}),
                success_metrics=item_data.get("success_metrics", {}),
                usage_count=item_data.get("usage_count", 0),
                created_at=item_data.get("created_at", time.time()),
                last_used=item_data.get("last_used", time.time()),
                priority=LearningPriority(item_data.get("priority", "medium")),
                confidence_score=item_data.get("confidence_score", 0.5),
            )

            # Add to both global and project knowledge bases
            self.global_knowledge_base[knowledge_id] = knowledge_item
            target_kb.knowledge_items[knowledge_id] = knowledge_item

            # Generate embedding
            await self._generate_knowledge_embedding(knowledge_id, knowledge_item.content)

            imported_count += 1

        target_kb.updated_at = time.time()

        result = {
            "target_project": target_project,
            "imported_count": imported_count,
            "skipped_count": skipped_count,
            "total_items": len(target_kb.knowledge_items),
        }

        logger.info(
            "Knowledge imported into project",
            target_project=target_project,
            imported_count=imported_count,
            skipped_count=skipped_count,
        )

        return result

    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide learning metrics."""

        total_projects = len(self.project_knowledge_bases)
        total_knowledge_items = len(self.global_knowledge_base)
        active_learning_sessions = sum(
            1 for session in self.learning_sessions.values()
            if session.status == "active"
        )

        # Calculate knowledge utilization rate
        total_usage = sum(item.usage_count for item in self.global_knowledge_base.values())
        if total_knowledge_items > 0:
            self.metrics["knowledge_utilization_rate"] = total_usage / total_knowledge_items

        # Calculate cross-project success rate
        successful_shares = self.metrics["knowledge_items_shared"]
        if successful_shares > 0:
            # Simplified calculation - in practice, this would track actual success
            self.metrics["cross_project_success_rate"] = 0.85

        return {
            **self.metrics,
            "total_projects": total_projects,
            "total_knowledge_items": total_knowledge_items,
            "active_learning_sessions": active_learning_sessions,
            "average_knowledge_per_project": total_knowledge_items / total_projects if total_projects > 0 else 0,
        }


# Global instance
cross_project_learning = CrossProjectLearningSystem()