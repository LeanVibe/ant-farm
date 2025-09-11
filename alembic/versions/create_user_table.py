"""create_user_table

Revision ID: create_user_table
Revises: f22d3a00238f
Create Date: 2025-09-04 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "create_user_table"
down_revision: str | Sequence[str] | None = "f22d3a00238f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create users table for authentication and profile management."""
    op.create_table(
        "users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=sa.text("gen_random_uuid()")),
        sa.Column("username", sa.String(length=100), nullable=False, unique=True),
        sa.Column("email", sa.String(length=255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),

        # Profile fields
        sa.Column("first_name", sa.String(length=100)),
        sa.Column("last_name", sa.String(length=100)),
        sa.Column("bio", sa.Text()),
        sa.Column("avatar_url", sa.String(length=500)),
        sa.Column("preferences", JSONB(), default={}),
        sa.Column("timezone", sa.String(length=50), default="UTC"),
        sa.Column("language", sa.String(length=10), default="en"),

        # Account status
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("is_admin", sa.Boolean(), default=False),
        sa.Column("email_verified", sa.Boolean(), default=False),

        # Security
        sa.Column("failed_login_attempts", sa.Integer(), default=0),
        sa.Column("locked_until", sa.DateTime(timezone=True)),
        sa.Column("last_password_change", sa.DateTime(timezone=True)),

        # Permissions
        sa.Column("permissions", JSONB(), default=[]),

        # Timestamps
        sa.Column("created_at", sa.DateTime(timezone=True), default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), default=sa.text("NOW()")),
        sa.Column("last_login", sa.DateTime(timezone=True)),
    )

    # Create indexes
    op.create_index("ix_users_username", "users", ["username"], unique=True)
    op.create_index("ix_users_email", "users", ["email"], unique=True)
    op.create_index("ix_users_created_at", "users", ["created_at"])
    op.create_index("ix_users_is_active", "users", ["is_active"])


def downgrade() -> None:
    """Drop users table."""
    op.drop_index("ix_users_is_active", table_name="users")
    op.drop_index("ix_users_created_at", table_name="users")
    op.drop_index("ix_users_email", table_name="users")
    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")