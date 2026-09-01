"""assistant v2 schema

Revision ID: 20260304_0001
Revises:
Create Date: 2026-03-04
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260304_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "assistant_user_settings",
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("settings_encrypted", sa.Text(), nullable=False),
        sa.Column("key_version", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_table(
        "assistant_mcp_servers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("url", sa.String(length=2048), nullable=False),
        sa.Column("api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("api_key_key_version", sa.String(length=64), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("timeout_ms", sa.Integer(), nullable=False, server_default=sa.text("10000")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "name", "url", name="uq_assistant_mcp_server_user_name_url"),
    )
    op.create_index("ix_assistant_mcp_servers_user_id", "assistant_mcp_servers", ["user_id"], unique=False)

    op.create_table(
        "assistant_threads",
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("thread_id"),
    )
    op.create_index("ix_assistant_threads_user_id", "assistant_threads", ["user_id"], unique=False)

    op.create_table(
        "assistant_messages",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("thread_id", sa.String(length=128), nullable=False),
        sa.Column("user_id", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=32), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("reasoning_summary", sa.Text(), nullable=True),
        sa.Column("usage_json", sa.JSON(), nullable=True),
        sa.Column("citations_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["thread_id"], ["assistant_threads.thread_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_assistant_messages_thread_id", "assistant_messages", ["thread_id"], unique=False)
    op.create_index("ix_assistant_messages_user_id", "assistant_messages", ["user_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_assistant_messages_user_id", table_name="assistant_messages")
    op.drop_index("ix_assistant_messages_thread_id", table_name="assistant_messages")
    op.drop_table("assistant_messages")

    op.drop_index("ix_assistant_threads_user_id", table_name="assistant_threads")
    op.drop_table("assistant_threads")

    op.drop_index("ix_assistant_mcp_servers_user_id", table_name="assistant_mcp_servers")
    op.drop_table("assistant_mcp_servers")

    op.drop_table("assistant_user_settings")
