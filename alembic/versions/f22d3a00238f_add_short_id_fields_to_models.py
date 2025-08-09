"""add_short_id_fields_to_models

Revision ID: f22d3a00238f
Revises: optimize_indexes
Create Date: 2025-08-08 15:56:33.131624

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "f22d3a00238f"
down_revision: str | Sequence[str] | None = "optimize_indexes"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add short_id columns to relevant tables
    op.add_column("agents", sa.Column("short_id", sa.String(length=10), nullable=True))
    op.add_column("tasks", sa.Column("short_id", sa.String(length=10), nullable=True))
    op.add_column(
        "sessions", sa.Column("short_id", sa.String(length=10), nullable=True)
    )

    # Create unique indexes on short_id columns
    op.create_index("ix_agents_short_id", "agents", ["short_id"], unique=True)
    op.create_index("ix_tasks_short_id", "tasks", ["short_id"], unique=True)
    op.create_index("ix_sessions_short_id", "sessions", ["short_id"], unique=True)


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes first
    op.drop_index("ix_sessions_short_id", table_name="sessions")
    op.drop_index("ix_tasks_short_id", table_name="tasks")
    op.drop_index("ix_agents_short_id", table_name="agents")

    # Drop columns
    op.drop_column("sessions", "short_id")
    op.drop_column("tasks", "short_id")
    op.drop_column("agents", "short_id")
