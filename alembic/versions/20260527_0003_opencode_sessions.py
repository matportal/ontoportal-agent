"""add assistant opencode sessions

Revision ID: 20260527_0003
Revises: 20260521_0002
Create Date: 2026-05-27
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260527_0003"
down_revision = "20260521_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_opencode_sessions",
        sa.Column("session_id", sa.String(length=128), primary_key=True),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("opencode_session_id", sa.String(length=255), nullable=True),
        sa.Column("latest_run_id", sa.String(length=128), nullable=True),
        sa.Column("workspace", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("model", sa.String(length=255), nullable=True),
        sa.Column("auth_source", sa.String(length=64), nullable=True),
        sa.Column("auth_kind", sa.String(length=64), nullable=True),
        sa.Column("objective", sa.Text(), nullable=True),
        sa.Column("latest_execution_json", sa.JSON(), nullable=True),
        sa.Column("validation_summary_json", sa.JSON(), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["thread_id"], ["assistant_threads.thread_id"], ondelete="CASCADE"),
    )
    op.create_index("ix_assistant_opencode_sessions_user_id", "assistant_opencode_sessions", ["user_id"])
    op.create_index("ix_assistant_opencode_sessions_thread_id", "assistant_opencode_sessions", ["thread_id"])
    op.create_index("ix_assistant_opencode_sessions_opencode_session_id", "assistant_opencode_sessions", ["opencode_session_id"])
    op.create_index("ix_assistant_opencode_sessions_latest_run_id", "assistant_opencode_sessions", ["latest_run_id"])
    op.create_index("ix_assistant_opencode_sessions_status", "assistant_opencode_sessions", ["status"])
    op.create_index("ix_assistant_opencode_sessions_expires_at", "assistant_opencode_sessions", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_assistant_opencode_sessions_expires_at", table_name="assistant_opencode_sessions")
    op.drop_index("ix_assistant_opencode_sessions_status", table_name="assistant_opencode_sessions")
    op.drop_index("ix_assistant_opencode_sessions_latest_run_id", table_name="assistant_opencode_sessions")
    op.drop_index("ix_assistant_opencode_sessions_opencode_session_id", table_name="assistant_opencode_sessions")
    op.drop_index("ix_assistant_opencode_sessions_thread_id", table_name="assistant_opencode_sessions")
    op.drop_index("ix_assistant_opencode_sessions_user_id", table_name="assistant_opencode_sessions")
    op.drop_table("assistant_opencode_sessions")
