"""SQLAlchemy models for LeanVibe Agent Hive 2.0."""

import uuid
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID

try:
    from pgvector.sqlalchemy import Vector

    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

    # Fallback for when pgvector is not available
    class Vector:
        def __init__(self, dim):
            self.dim = dim

        def __call__(self, *args, **kwargs):
            return Text  # Fallback to TEXT column


from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker

Base = declarative_base()


class Agent(Base):
    """Agent model for tracking all agents in the system."""

    __tablename__ = "agents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False, unique=True, index=True)
    type = Column(String(100), nullable=False, index=True)
    role = Column(String(100), nullable=False)
    capabilities = Column(JSONB, default={})
    system_prompt = Column(Text)
    status = Column(String(50), default="inactive", index=True)
    tmux_session = Column(String(255))
    last_heartbeat = Column(DateTime)

    # Performance metrics
    tasks_completed = Column(Integer, default=0)
    tasks_failed = Column(Integer, default=0)
    average_task_time = Column(Float, default=0.0)
    load_factor = Column(Float, default=0.0)

    # Configuration
    config = Column(JSONB, default={})
    preferences = Column(JSONB, default={})

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)

    # Relationships
    tasks = relationship("Task", back_populates="agent", cascade="all, delete-orphan")
    contexts = relationship(
        "Context", back_populates="agent", cascade="all, delete-orphan"
    )
    sent_messages = relationship(
        "Conversation",
        foreign_keys="Conversation.from_agent_id",
        back_populates="from_agent",
    )
    received_messages = relationship(
        "Conversation",
        foreign_keys="Conversation.to_agent_id",
        back_populates="to_agent",
    )
    checkpoints = relationship("SystemCheckpoint", back_populates="agent")

    def __repr__(self):
        return (
            f"<Agent(name='{self.name}', type='{self.type}', status='{self.status}')>"
        )

    def to_dict(self):
        """Convert agent to dictionary."""
        return {
            "id": str(self.id),
            "name": self.name,
            "type": self.type,
            "role": self.role,
            "status": self.status,
            "capabilities": self.capabilities,
            "tasks_completed": self.tasks_completed,
            "tasks_failed": self.tasks_failed,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat()
            if self.last_heartbeat
            else None,
        }


class Task(Base):
    """Task model for tracking all tasks in the system."""

    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    description = Column(Text)
    type = Column(String(100), nullable=False, index=True)
    payload = Column(JSONB, default={})
    priority = Column(Integer, default=5, index=True)
    status = Column(String(50), default="pending", index=True)

    # Assignment
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), index=True)
    assigned_by = Column(String(255))  # Who/what assigned this task

    # Dependencies and relationships
    dependencies = Column(JSONB, default=[])  # List of task IDs this task depends on
    parent_task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"))

    # Execution details
    result = Column(JSONB)
    error_message = Column(Text)
    retry_count = Column(Integer, default=0)
    max_retries = Column(Integer, default=3)
    timeout_seconds = Column(Integer, default=300)

    # Performance metrics
    estimated_duration = Column(Integer)  # Estimated duration in seconds
    actual_duration = Column(Integer)  # Actual duration in seconds
    complexity_score = Column(Float, default=1.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, index=True)
    completed_at = Column(DateTime, index=True)
    deadline = Column(DateTime)

    # Relationships
    agent = relationship("Agent", back_populates="tasks")
    parent_task = relationship("Task", remote_side=[id])
    subtasks = relationship("Task", back_populates="parent_task")

    def __repr__(self):
        return f"<Task(id='{self.id}', title='{self.title}', status='{self.status}')>"

    def to_dict(self):
        """Convert task to dictionary."""
        return {
            "id": str(self.id),
            "title": self.title,
            "description": self.description,
            "type": self.type,
            "status": self.status,
            "priority": self.priority,
            "agent_id": str(self.agent_id) if self.agent_id else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "payload": self.payload,
            "result": self.result,
        }


class Session(Base):
    """Session model for tracking work sessions and collaboration."""

    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    type = Column(String(100), nullable=False, index=True)
    description = Column(Text)

    # Session participants
    agents = Column(JSONB, default=[])  # List of agent IDs participating
    owner_agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"))

    # Session state
    state = Column(JSONB, default={})
    status = Column(String(50), default="active", index=True)

    # Session configuration
    config = Column(JSONB, default={})
    goals = Column(JSONB, default=[])  # List of session goals

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_active = Column(DateTime, default=datetime.utcnow, index=True)
    ended_at = Column(DateTime)

    # Relationships
    owner = relationship("Agent")
    contexts = relationship("Context", back_populates="session")
    conversations = relationship("Conversation", back_populates="session")

    def __repr__(self):
        return (
            f"<Session(name='{self.name}', type='{self.type}', status='{self.status}')>"
        )


class Context(Base):
    """Context model for semantic memory and knowledge storage."""

    __tablename__ = "contexts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True
    )
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), index=True)

    # Content
    content = Column(Text, nullable=False)
    content_type = Column(String(50), default="text")  # text, code, image, etc.

    # Vector embeddings for semantic search
    embedding = Column(
        Vector(1536) if PGVECTOR_AVAILABLE else Text
    )  # OpenAI ada-002 dimension
    embedding_model = Column(String(100))  # Which model created the embedding

    # Classification and importance
    importance_score = Column(Float, default=0.5, index=True)
    category = Column(String(100), index=True)
    topic = Column(String(200))

    # Hierarchy
    parent_id = Column(UUID(as_uuid=True), ForeignKey("contexts.id"))
    level = Column(Integer, default=0)  # Depth in hierarchy

    # Metadata
    tags = Column(JSONB, default=[])
    meta_data = Column(JSONB, default={})
    source = Column(String(200))  # Where this context came from

    # Access tracking
    access_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Expiration
    expires_at = Column(DateTime)  # For temporary contexts

    # Relationships
    agent = relationship("Agent", back_populates="contexts")
    session = relationship("Session", back_populates="contexts")
    parent = relationship("Context", remote_side=[id])
    children = relationship("Context", back_populates="parent")

    def __repr__(self):
        return f"<Context(id='{self.id}', agent='{self.agent_id}', importance={self.importance_score})>"


class Conversation(Base):
    """Conversation model for tracking inter-agent communication."""

    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True
    )

    # Message details
    from_agent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id"), nullable=False, index=True
    )
    to_agent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id"), index=True
    )  # Nullable for broadcasts

    message_type = Column(String(100), nullable=False, index=True)
    topic = Column(String(200), index=True)
    content = Column(Text, nullable=False)

    # Vector embeddings for semantic search of conversations
    embedding = Column(
        Vector(1536) if PGVECTOR_AVAILABLE else Text
    )  # For semantic search

    # Message threading
    reply_to = Column(UUID(as_uuid=True), ForeignKey("conversations.id"))
    thread_id = Column(UUID(as_uuid=True), index=True)  # Group related messages

    # Message metadata
    msg_metadata = Column(JSONB, default={})
    attachments = Column(JSONB, default=[])  # File attachments, links, etc.

    # Message status
    status = Column(String(50), default="sent")  # sent, delivered, read, processed
    priority = Column(Integer, default=5)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    delivered_at = Column(DateTime)
    processed_at = Column(DateTime)

    # Relationships
    session = relationship("Session", back_populates="conversations")
    from_agent = relationship(
        "Agent", foreign_keys=[from_agent_id], back_populates="sent_messages"
    )
    to_agent = relationship(
        "Agent", foreign_keys=[to_agent_id], back_populates="received_messages"
    )
    reply_to_message = relationship("Conversation", remote_side=[id])

    def __repr__(self):
        return f"<Conversation(from='{self.from_agent_id}', to='{self.to_agent_id}', type='{self.message_type}')>"


class SystemCheckpoint(Base):
    """System checkpoint model for tracking system state and changes."""

    __tablename__ = "system_checkpoints"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Checkpoint metadata
    type = Column(
        String(100), nullable=False, index=True
    )  # manual, auto, pre_change, post_change
    description = Column(Text)
    version = Column(String(50))  # System version at checkpoint

    # System state
    state = Column(JSONB, nullable=False)  # Complete system state snapshot
    agent_states = Column(JSONB, default={})  # Individual agent states
    task_queue_state = Column(JSONB, default={})  # Task queue state

    # Git integration
    git_commit_hash = Column(String(64), index=True)
    git_branch = Column(String(100))
    git_tag = Column(String(100))

    # Checkpoint metadata
    agent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id")
    )  # Who created this checkpoint

    # Performance data
    performance_metrics = Column(JSONB, default={})
    system_health = Column(JSONB, default={})

    # Rollback data
    rollback_data = Column(JSONB)  # Data needed to rollback to previous state
    rollback_script = Column(Text)  # Script to execute rollback

    # Status
    status = Column(
        String(50), default="created", index=True
    )  # created, applied, rolled_back

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    applied_at = Column(DateTime)
    rolled_back_at = Column(DateTime)

    # Relationships
    agent = relationship("Agent", back_populates="checkpoints")

    def __repr__(self):
        return f"<SystemCheckpoint(type='{self.type}', status='{self.status}', created_at='{self.created_at}')>"


class SystemMetric(Base):
    """System metrics model for performance monitoring."""

    __tablename__ = "system_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Metric identification
    metric_name = Column(String(200), nullable=False, index=True)
    metric_type = Column(
        String(50), nullable=False, index=True
    )  # counter, gauge, histogram

    # Metric data
    value = Column(Float, nullable=False)
    unit = Column(String(50))  # seconds, bytes, count, etc.

    # Context
    agent_id = Column(UUID(as_uuid=True), ForeignKey("agents.id"), index=True)
    task_id = Column(UUID(as_uuid=True), ForeignKey("tasks.id"), index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.id"), index=True)

    # Labels for grouping
    labels = Column(JSONB, default={})  # Key-value pairs for filtering

    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    agent = relationship("Agent")
    task = relationship("Task")
    session = relationship("Session")

    def __repr__(self):
        return f"<SystemMetric(name='{self.metric_name}', value={self.value}, timestamp='{self.timestamp}')>"


class CodeArtifact(Base):
    """Code artifacts model for tracking generated code and modifications."""

    __tablename__ = "code_artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Artifact identification
    name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False, index=True)
    artifact_type = Column(
        String(100), nullable=False, index=True
    )  # file, function, class, module

    # Content
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), index=True)  # SHA-256 hash of content

    # Metadata
    language = Column(String(50))
    framework = Column(String(100))
    dependencies = Column(JSONB, default=[])

    # Generation context
    generated_by_agent_id = Column(
        UUID(as_uuid=True), ForeignKey("agents.id"), index=True
    )
    generated_for_task_id = Column(
        UUID(as_uuid=True), ForeignKey("tasks.id"), index=True
    )
    generation_prompt = Column(Text)

    # Version control
    version = Column(String(50), default="1.0.0")
    parent_artifact_id = Column(UUID(as_uuid=True), ForeignKey("code_artifacts.id"))

    # Quality metrics
    complexity_score = Column(Float)
    test_coverage = Column(Float)
    quality_score = Column(Float)

    # Status
    status = Column(
        String(50), default="draft", index=True
    )  # draft, reviewed, approved, deployed

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    deployed_at = Column(DateTime)

    # Relationships
    generated_by_agent = relationship("Agent")
    generated_for_task = relationship("Task")
    parent_artifact = relationship("CodeArtifact", remote_side=[id])
    child_artifacts = relationship("CodeArtifact", back_populates="parent_artifact")

    def __repr__(self):
        return f"<CodeArtifact(name='{self.name}', type='{self.artifact_type}', status='{self.status}')>"


# Database utility functions
class DatabaseManager:
    """Database manager for handling connections and operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self):
        """Create all tables using sync engine."""
        # Convert async URL to sync URL for table creation
        sync_url = self.database_url.replace("postgresql+asyncpg://", "postgresql://")
        sync_engine = create_engine(sync_url, echo=False)
        Base.metadata.create_all(bind=sync_engine)
        sync_engine.dispose()

    def drop_tables(self):
        """Drop all tables using sync engine."""
        # Convert async URL to sync URL for table operations
        sync_url = self.database_url.replace("postgresql+asyncpg://", "postgresql://")
        sync_engine = create_engine(sync_url, echo=False)
        Base.metadata.drop_all(bind=sync_engine)
        sync_engine.dispose()

    def get_session(self):
        """Get a database session."""
        return self.SessionLocal()

    def init_database(self):
        """Initialize database with default data."""
        self.create_tables()

        # Add any default data here
        session = self.get_session()
        try:
            # Create system agent if not exists
            system_agent = session.query(Agent).filter_by(name="system").first()
            if not system_agent:
                system_agent = Agent(
                    name="system",
                    type="system",
                    role="system",
                    capabilities={"system_management": True, "monitoring": True},
                    status="active",
                )
                session.add(system_agent)
                session.commit()
        finally:
            session.close()


# Event listeners for automatic updates
@event.listens_for(Agent, "before_update")
def receive_before_update(mapper, connection, target):
    """Update the updated_at timestamp on agent updates."""
    target.updated_at = datetime.utcnow()


@event.listens_for(Context, "before_update")
def receive_before_update_context(mapper, connection, target):
    """Update timestamps and access tracking on context updates."""
    target.updated_at = datetime.utcnow()
    target.access_count += 1
    target.last_accessed = datetime.utcnow()


# Global database manager instance
db_manager = None


def get_database_manager(database_url: str) -> DatabaseManager:
    """Get the global database manager instance."""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager(database_url)
    return db_manager
