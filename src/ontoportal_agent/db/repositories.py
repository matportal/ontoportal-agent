from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from .models import AssistantMcpServer, AssistantMessage, AssistantThread, AssistantUserSettings


def list_threads(session: Session, *, user_id: str, limit: int = 100) -> list[AssistantThread]:
    stmt = (
        select(AssistantThread)
        .where(AssistantThread.user_id == user_id)
        .order_by(AssistantThread.updated_at.desc())
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())


def create_thread(
    session: Session,
    *,
    user_id: str,
    title: str | None = None,
    thread_id: str | None = None,
) -> AssistantThread:
    resolved_thread_id = (thread_id or str(uuid.uuid4())).strip()
    if not resolved_thread_id:
        raise ValueError("Thread id cannot be blank.")

    thread = AssistantThread(
        thread_id=resolved_thread_id,
        user_id=user_id,
        title=title.strip() if isinstance(title, str) and title.strip() else None,
    )
    session.add(thread)
    session.commit()
    session.refresh(thread)
    return thread


def ensure_thread(
    session: Session,
    *,
    user_id: str,
    thread_id: str | None,
    title: str | None = None,
) -> AssistantThread:
    if thread_id:
        existing = get_thread(session, user_id=user_id, thread_id=thread_id)
        if existing:
            return existing
    return create_thread(session, user_id=user_id, title=title, thread_id=thread_id)


def get_thread(session: Session, *, user_id: str, thread_id: str) -> Optional[AssistantThread]:
    stmt = select(AssistantThread).where(
        AssistantThread.thread_id == thread_id,
        AssistantThread.user_id == user_id,
    )
    return session.execute(stmt).scalars().first()


def delete_thread(session: Session, *, user_id: str, thread_id: str) -> bool:
    thread = get_thread(session, user_id=user_id, thread_id=thread_id)
    if not thread:
        return False
    session.delete(thread)
    session.commit()
    return True


def update_thread_title(
    session: Session,
    *,
    user_id: str,
    thread_id: str,
    title: str | None,
) -> Optional[AssistantThread]:
    thread = get_thread(session, user_id=user_id, thread_id=thread_id)
    if thread is None:
        return None

    clean_title = title.strip() if isinstance(title, str) and title.strip() else None
    if clean_title is None:
        return thread

    thread.title = clean_title[:255]
    thread.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(thread)
    return thread


def list_thread_messages(
    session: Session,
    *,
    user_id: str,
    thread_id: str,
    limit: int = 1000,
) -> list[AssistantMessage]:
    stmt = (
        select(AssistantMessage)
        .where(
            AssistantMessage.user_id == user_id,
            AssistantMessage.thread_id == thread_id,
        )
        .order_by(AssistantMessage.created_at.asc())
        .limit(limit)
    )
    return list(session.execute(stmt).scalars().all())


def get_thread_execution(
    session: Session,
    *,
    user_id: str,
    thread_id: str,
    run_id: str,
) -> dict[str, Any] | None:
    messages = list_thread_messages(session, user_id=user_id, thread_id=thread_id)
    for message in reversed(messages):
        usage = message.usage_json if isinstance(message.usage_json, dict) else {}
        execution = usage.get("execution") if isinstance(usage, dict) else None
        if isinstance(execution, dict) and str(execution.get("run_id") or "") == str(run_id):
            return execution
    return None


def get_latest_thread_execution(
    session: Session,
    *,
    user_id: str,
    thread_id: str,
) -> dict[str, Any] | None:
    messages = list_thread_messages(session, user_id=user_id, thread_id=thread_id)
    for message in reversed(messages):
        usage = message.usage_json if isinstance(message.usage_json, dict) else {}
        execution = usage.get("execution") if isinstance(usage, dict) else None
        if isinstance(execution, dict):
            return execution
    return None


def create_message(
    session: Session,
    *,
    user_id: str,
    thread_id: str,
    role: str,
    content: str,
    reasoning_summary: str | None = None,
    usage_json: dict[str, Any] | None = None,
    citations_json: list[Any] | None = None,
) -> AssistantMessage:
    message = AssistantMessage(
        user_id=user_id,
        thread_id=thread_id,
        role=role,
        content=content,
        reasoning_summary=reasoning_summary,
        usage_json=usage_json,
        citations_json=citations_json,
    )
    session.add(message)
    thread = get_thread(session, user_id=user_id, thread_id=thread_id)
    if thread:
        thread.updated_at = datetime.now(timezone.utc)
    session.commit()
    session.refresh(message)
    return message


def get_user_settings(session: Session, *, user_id: str) -> Optional[AssistantUserSettings]:
    stmt = select(AssistantUserSettings).where(AssistantUserSettings.user_id == user_id)
    return session.execute(stmt).scalars().first()


def upsert_user_settings(
    session: Session,
    *,
    user_id: str,
    settings_encrypted: str,
    key_version: str,
) -> AssistantUserSettings:
    row = get_user_settings(session, user_id=user_id)
    if row is None:
        row = AssistantUserSettings(
            user_id=user_id,
            settings_encrypted=settings_encrypted,
            key_version=key_version,
        )
        session.add(row)
    else:
        row.settings_encrypted = settings_encrypted
        row.key_version = key_version
    session.commit()
    session.refresh(row)
    return row


def replace_mcp_servers(
    session: Session,
    *,
    user_id: str,
    mcp_servers: list[dict[str, Any]],
    encrypt_secret: Callable[[dict[str, str]], tuple[str, str]] | None = None,
) -> list[AssistantMcpServer]:
    session.execute(delete(AssistantMcpServer).where(AssistantMcpServer.user_id == user_id))
    created: list[AssistantMcpServer] = []
    for item in mcp_servers:
        auth_mode = str(item.get("auth_mode") or "api_key").strip().lower() or "api_key"
        username = str(item.get("username") or "").strip() or None
        raw_api_key = (item.get("api_key") or "").strip()
        raw_password = (item.get("password") or "").strip()
        api_key_encrypted = None
        api_key_key_version = None
        if raw_api_key and encrypt_secret is not None:
            api_key_encrypted, api_key_key_version = encrypt_secret({"api_key": raw_api_key})
        password_encrypted = None
        password_key_version = None
        if raw_password and encrypt_secret is not None:
            password_encrypted, password_key_version = encrypt_secret({"password": raw_password})

        server = AssistantMcpServer(
            user_id=user_id,
            name=(item.get("name") or "").strip() or "MCP",
            url=(item.get("url") or "").strip(),
            auth_mode=auth_mode,
            username=username,
            api_key_encrypted=api_key_encrypted,
            api_key_key_version=api_key_key_version,
            password_encrypted=password_encrypted,
            password_key_version=password_key_version,
            enabled=bool(item.get("enabled", True)),
            timeout_ms=max(1000, int(item.get("timeout_ms", 30000))),
        )
        session.add(server)
        created.append(server)
    session.commit()
    for row in created:
        session.refresh(row)
    return created


def list_mcp_servers(session: Session, *, user_id: str) -> list[AssistantMcpServer]:
    stmt = (
        select(AssistantMcpServer)
        .where(AssistantMcpServer.user_id == user_id)
        .order_by(AssistantMcpServer.id.asc())
    )
    return list(session.execute(stmt).scalars().all())
