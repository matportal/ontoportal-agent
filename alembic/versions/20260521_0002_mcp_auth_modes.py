"""add mcp auth mode columns

Revision ID: 20260521_0002
Revises: 20260304_0001
Create Date: 2026-05-21
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20260521_0002"
down_revision = "20260304_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    dialect_name = op.get_bind().dialect.name
    op.add_column(
        "assistant_mcp_servers",
        sa.Column("auth_mode", sa.String(length=32), nullable=False, server_default="api_key"),
    )
    op.add_column(
        "assistant_mcp_servers",
        sa.Column("username", sa.String(length=255), nullable=True),
    )
    op.add_column(
        "assistant_mcp_servers",
        sa.Column("password_encrypted", sa.Text(), nullable=True),
    )
    op.add_column(
        "assistant_mcp_servers",
        sa.Column("password_key_version", sa.String(length=64), nullable=True),
    )
    if dialect_name != "sqlite":
        op.alter_column("assistant_mcp_servers", "auth_mode", server_default=None)


def downgrade() -> None:
    op.drop_column("assistant_mcp_servers", "password_key_version")
    op.drop_column("assistant_mcp_servers", "password_encrypted")
    op.drop_column("assistant_mcp_servers", "username")
    op.drop_column("assistant_mcp_servers", "auth_mode")
