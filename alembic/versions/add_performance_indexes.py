"""Add performance optimization indexes

Revision ID: add_performance_indexes
Revises: optimize_database_indexes
Create Date: 2024-12-10

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers
revision = "add_performance_indexes"
down_revision = "optimize_database_indexes"
branch_labels = None
depends_on = None


def upgrade():
    """Add performance optimization indexes for common query patterns."""

    # Contexts table indexes (for semantic search and filtering)
    op.create_index(
        "idx_contexts_importance",
        "contexts",
        ["importance_score"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_contexts_agent_category",
        "contexts",
        ["agent_id", "category"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_contexts_created_at", "contexts", ["created_at"], postgresql_using="btree"
    )

    # Tasks table indexes (for queue management and filtering)
    op.create_index(
        "idx_tasks_priority_status",
        "tasks",
        ["priority", "status"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_tasks_agent_status",
        "tasks",
        ["agent_id", "status"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_tasks_created_at", "tasks", ["created_at"], postgresql_using="btree"
    )

    op.create_index(
        "idx_tasks_type_status",
        "tasks",
        ["task_type", "status"],
        postgresql_using="btree",
    )

    # Agents table indexes (for orchestration and monitoring)
    op.create_index(
        "idx_agents_type_status", "agents", ["type", "status"], postgresql_using="btree"
    )

    op.create_index(
        "idx_agents_status_heartbeat",
        "agents",
        ["status", "last_heartbeat"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_agents_created_at", "agents", ["created_at"], postgresql_using="btree"
    )

    # Conversations table indexes (for context retrieval)
    op.create_index(
        "idx_conversations_agent_created",
        "conversations",
        ["agent_id", "created_at"],
        postgresql_using="btree",
    )

    op.create_index(
        "idx_conversations_session_id",
        "conversations",
        ["session_id"],
        postgresql_using="btree",
    )

    # Vector search optimization for embeddings (if using pgvector)
    try:
        # HNSW index for vector similarity search
        op.execute("""
            CREATE INDEX CONCURRENTLY idx_conversations_embedding_hnsw 
            ON conversations USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 64);
        """)

        op.execute("""
            CREATE INDEX CONCURRENTLY idx_contexts_embedding_hnsw 
            ON contexts USING hnsw (embedding vector_cosine_ops) 
            WITH (m = 16, ef_construction = 64);
        """)
    except Exception:
        # Fallback to IVFFlat if HNSW not available
        op.execute("""
            CREATE INDEX CONCURRENTLY idx_conversations_embedding_ivfflat 
            ON conversations USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)

        op.execute("""
            CREATE INDEX CONCURRENTLY idx_contexts_embedding_ivfflat 
            ON contexts USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
        """)

    # Performance monitoring tables (if they exist)
    try:
        op.create_index(
            "idx_performance_metrics_timestamp",
            "performance_metrics",
            ["timestamp"],
            postgresql_using="btree",
        )

        op.create_index(
            "idx_performance_metrics_metric_name",
            "performance_metrics",
            ["metric_name", "timestamp"],
            postgresql_using="btree",
        )
    except Exception:
        # Table might not exist yet
        pass

    # Modify search indexes for better performance
    try:
        # Full-text search index for contexts content
        op.execute("""
            CREATE INDEX CONCURRENTLY idx_contexts_content_fts 
            ON contexts USING gin(to_tsvector('english', content));
        """)

        # Full-text search index for tasks
        op.execute("""
            CREATE INDEX CONCURRENTLY idx_tasks_description_fts 
            ON tasks USING gin(to_tsvector('english', description || ' ' || title));
        """)
    except Exception:
        # GIN extension might not be available
        pass


def downgrade():
    """Remove performance optimization indexes."""

    # Drop performance indexes
    indexes_to_drop = [
        "idx_contexts_importance",
        "idx_contexts_agent_category",
        "idx_contexts_created_at",
        "idx_tasks_priority_status",
        "idx_tasks_agent_status",
        "idx_tasks_created_at",
        "idx_tasks_type_status",
        "idx_agents_type_status",
        "idx_agents_status_heartbeat",
        "idx_agents_created_at",
        "idx_conversations_agent_created",
        "idx_conversations_session_id",
        "idx_conversations_embedding_hnsw",
        "idx_conversations_embedding_ivfflat",
        "idx_contexts_embedding_hnsw",
        "idx_contexts_embedding_ivfflat",
        "idx_performance_metrics_timestamp",
        "idx_performance_metrics_metric_name",
        "idx_contexts_content_fts",
        "idx_tasks_description_fts",
    ]

    for index_name in indexes_to_drop:
        try:
            op.drop_index(index_name)
        except Exception:
            # Index might not exist
            pass
