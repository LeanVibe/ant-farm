"""Database optimization migrations for improved query performance.

This module contains database optimizations including:
- Strategic index creation for frequent queries
- Query optimization patterns
- Performance monitoring utilities

Target: <50ms p95 database response time

Revision ID: optimize_indexes
Revises: 7b71b9a0ac9e
Create Date: 2024-01-01 12:00:00.000000
"""

from alembic import op
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision = "optimize_indexes"
down_revision = "7b71b9a0ac9e"
branch_labels = None
depends_on = None


def upgrade():
    """Apply database optimizations."""

    # Context table optimizations
    op.create_index(
        "idx_contexts_agent_importance_category",
        "contexts",
        ["agent_id", "importance_score", "category"],
        postgresql_where=text(
            "importance_score >= 0.3"
        ),  # Partial index for important contexts
    )

    op.create_index(
        "idx_contexts_agent_session_created",
        "contexts",
        ["agent_id", "session_id", "created_at"],
    )

    op.create_index("idx_contexts_category_topic", "contexts", ["category", "topic"])

    op.create_index(
        "idx_contexts_access_tracking",
        "contexts",
        ["agent_id", "access_count", "last_accessed"],
    )

    # Task table optimizations
    op.create_index(
        "idx_tasks_agent_status_priority", "tasks", ["agent_id", "status", "priority"]
    )

    op.create_index("idx_tasks_status_created", "tasks", ["status", "created_at"])

    op.create_index(
        "idx_tasks_priority_created",
        "tasks",
        ["priority", "created_at"],
        postgresql_where=text("status IN ('pending', 'assigned')"),  # Only active tasks
    )

    op.create_index(
        "idx_tasks_completion_tracking",
        "tasks",
        ["status", "started_at", "completed_at"],
        postgresql_where=text("completed_at IS NOT NULL"),
    )

    # Agent table optimizations
    op.create_index("idx_agents_status_type", "agents", ["status", "type"])

    op.create_index(
        "idx_agents_performance",
        "agents",
        ["load_factor", "tasks_completed", "last_heartbeat"],
    )

    # Session table optimizations
    op.create_index(
        "idx_sessions_status_last_active", "sessions", ["status", "last_active"]
    )

    op.create_index("idx_sessions_type_created", "sessions", ["type", "created_at"])

    # Conversation table optimizations
    op.create_index(
        "idx_conversations_agents_created",
        "conversations",
        ["from_agent_id", "to_agent_id", "created_at"],
    )

    op.create_index(
        "idx_conversations_session_type",
        "conversations",
        ["session_id", "message_type", "created_at"],
    )

    op.create_index(
        "idx_conversations_thread", "conversations", ["thread_id", "created_at"]
    )

    # System metrics optimizations
    op.create_index(
        "idx_system_metrics_name_timestamp",
        "system_metrics",
        ["metric_name", "timestamp"],
    )

    op.create_index(
        "idx_system_metrics_agent_timestamp",
        "system_metrics",
        ["agent_id", "timestamp"],
        postgresql_where=text("agent_id IS NOT NULL"),
    )

    # Code artifacts optimizations
    op.create_index(
        "idx_code_artifacts_path_type", "code_artifacts", ["file_path", "artifact_type"]
    )

    op.create_index(
        "idx_code_artifacts_agent_task",
        "code_artifacts",
        ["generated_by_agent_id", "generated_for_task_id"],
    )


def downgrade():
    """Remove database optimizations."""

    # Drop all created indexes
    indexes_to_drop = [
        "idx_contexts_agent_importance_category",
        "idx_contexts_agent_session_created",
        "idx_contexts_category_topic",
        "idx_contexts_access_tracking",
        "idx_tasks_agent_status_priority",
        "idx_tasks_status_created",
        "idx_tasks_priority_created",
        "idx_tasks_completion_tracking",
        "idx_agents_status_type",
        "idx_agents_performance",
        "idx_sessions_status_last_active",
        "idx_sessions_type_created",
        "idx_conversations_agents_created",
        "idx_conversations_session_type",
        "idx_conversations_thread",
        "idx_system_metrics_name_timestamp",
        "idx_system_metrics_agent_timestamp",
        "idx_code_artifacts_path_type",
        "idx_code_artifacts_agent_task",
    ]

    for index_name in indexes_to_drop:
        op.drop_index(index_name)
