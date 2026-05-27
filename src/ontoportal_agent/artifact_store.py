from __future__ import annotations

import io
import mimetypes
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any


TEXT_ARTIFACT_SUFFIXES = {".ttl", ".rdf", ".owl", ".json", ".yaml", ".yml", ".md", ".txt"}
MAX_TEXT_VIEW_BYTES = 2_000_000
_SENSITIVE_ARTIFACT_DIRS = {
    ".git",
    ".opencode-home",
    ".opencode-state",
    ".opencode-cache",
    ".opencode",
    ".codex-home",
    ".codex",
    ".pi",
    ".cache",
    ".config",
    ".local",
}
_SENSITIVE_ARTIFACT_FILES = {
    ".env",
    "auth.json",
    "opencode.json",
    "pi.json",
    "credentials",
    "credentials.json",
    "secrets.json",
    "secret.json",
    "token",
    "token.json",
}
_SENSITIVE_ARTIFACT_SUFFIXES = {".key", ".pem", ".p12", ".pfx", ".pkcs12"}
_SECRET_SCAN_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    re.compile(r"(?i)authorization\s*:\s*(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]{10,}"),
    re.compile(
        r"(?i)(?:api[_-]?key|access[_-]?token|refresh[_-]?token|client[_-]?secret|password)"
        r"\s*[=:]\s*[\"']?[A-Za-z0-9_./+=:-]{12,}"
    ),
)


class ArtifactAccessError(ValueError):
    """Raised when a requested workspace artifact path is not safe or available."""


def parse_expires_at(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def artifact_expired(execution: dict[str, Any], *, now: datetime | None = None) -> bool:
    expires_at = parse_expires_at(execution.get("expires_at"))
    if expires_at is None:
        return False
    return (now or datetime.now(timezone.utc)) > expires_at


def _sensitive_artifact_reason(parts: tuple[str, ...]) -> str:
    for part in parts:
        name = str(part or "")
        lower = name.lower()
        if lower in _SENSITIVE_ARTIFACT_DIRS:
            return f"Runtime directory '{name}' is not available as an artifact."
        if lower in _SENSITIVE_ARTIFACT_FILES:
            return f"Runtime credential file '{name}' is not available as an artifact."
        if Path(lower).suffix in _SENSITIVE_ARTIFACT_SUFFIXES:
            return f"Sensitive key material '{name}' is not available as an artifact."
        if lower.startswith(".env"):
            return f"Environment file '{name}' is not available as an artifact."
        if any(marker in lower for marker in ("token", "secret", "credential")):
            return f"Sensitive token or credential file '{name}' is not available as an artifact."
    return ""


def sanitize_artifact_path(raw_path: str) -> Path:
    raw = str(raw_path or "").strip().replace("\\", "/")
    if not raw:
        raise ArtifactAccessError("Artifact path is required.")
    if raw.startswith("/") or raw.startswith("~"):
        raise ArtifactAccessError("Absolute artifact paths are not allowed.")

    pure = PurePosixPath(raw)
    if pure.is_absolute():
        raise ArtifactAccessError("Absolute artifact paths are not allowed.")
    parts = pure.parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ArtifactAccessError("Unsafe artifact path.")
    reason = _sensitive_artifact_reason(parts)
    if reason:
        raise ArtifactAccessError(reason)
    return Path(*parts)


def resolve_workspace(workspace: str | Path) -> Path:
    try:
        root = Path(workspace).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ArtifactAccessError("Artifact workspace is no longer available.") from exc
    if not root.is_dir():
        raise ArtifactAccessError("Artifact workspace is no longer available.")
    return root


def resolve_safe_workspace(root: str | Path, workspace: str | Path) -> Path:
    try:
        allowed_root = Path(root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ArtifactAccessError("Workspace root is not available.") from exc
    try:
        candidate = Path(workspace).expanduser().resolve(strict=True)
    except OSError as exc:
        raise ArtifactAccessError("Workspace is no longer available.") from exc
    if not allowed_root.is_dir() or not candidate.is_dir():
        raise ArtifactAccessError("Workspace is no longer available.")
    try:
        candidate.relative_to(allowed_root)
    except ValueError as exc:
        raise ArtifactAccessError("Workspace path escapes the assistant workspace root.") from exc
    if candidate == allowed_root:
        raise ArtifactAccessError("Assistant workspace root cannot be resumed directly.")
    return candidate


def resolve_artifact_file(workspace: str | Path, raw_path: str) -> Path:
    root = resolve_workspace(workspace)
    relative = sanitize_artifact_path(raw_path)
    try:
        candidate = (root / relative).resolve(strict=True)
    except OSError as exc:
        raise ArtifactAccessError("Artifact file is no longer available.") from exc
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactAccessError("Artifact path escapes the workspace.") from exc
    if not candidate.is_file():
        raise ArtifactAccessError("Artifact path is not a file.")
    return candidate


def _scan_text_for_secret(content: str) -> bool:
    return any(pattern.search(content) for pattern in _SECRET_SCAN_PATTERNS)


def assert_artifact_safe_for_exposure(file_path: str | Path) -> None:
    path = Path(file_path)
    try:
        sample = path.read_bytes()[:MAX_TEXT_VIEW_BYTES]
    except OSError as exc:
        raise ArtifactAccessError("Artifact file is no longer available.") from exc
    if _scan_text_for_secret(sample.decode("utf-8", errors="replace")):
        raise ArtifactAccessError("Artifact appears to contain credentials or tokens and cannot be exposed.")


def text_language(path: str) -> str:
    suffix = Path(path).suffix.lower()
    return {
        ".ttl": "turtle",
        ".rdf": "xml",
        ".owl": "xml",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
    }.get(suffix, "text")


def is_text_artifact(path: str) -> bool:
    return Path(str(path or "")).suffix.lower() in TEXT_ARTIFACT_SUFFIXES


def normalize_execution_files(execution: dict[str, Any]) -> list[dict[str, Any]]:
    changed_files = execution.get("changed_files")
    items = changed_files if isinstance(changed_files, list) else []
    artifact_paths = {
        str(item.get("path") or "")
        for item in (execution.get("artifact_candidates") if isinstance(execution.get("artifact_candidates"), list) else [])
        if isinstance(item, dict)
    }
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        try:
            safe_path = str(sanitize_artifact_path(path).as_posix())
        except ArtifactAccessError:
            continue
        status = str(item.get("status") or "M").strip() or "M"
        kind = str(item.get("kind") or Path(safe_path).suffix.lstrip(".") or "file")
        normalized.append(
            {
                "path": safe_path,
                "status": status,
                "kind": kind,
                "artifact": safe_path in artifact_paths or is_text_artifact(safe_path),
                "viewable": is_text_artifact(safe_path),
                "language": text_language(safe_path),
            }
        )
    return normalized


def execution_allows_path(execution: dict[str, Any], path: str) -> bool:
    try:
        safe_path = str(sanitize_artifact_path(path).as_posix())
    except ArtifactAccessError:
        return False
    return any(item.get("path") == safe_path for item in normalize_execution_files(execution))


def file_metadata(workspace: str | Path, path: str, base: dict[str, Any] | None = None) -> dict[str, Any]:
    item = dict(base or {})
    safe_path = str(sanitize_artifact_path(path).as_posix())
    item["path"] = safe_path
    item.setdefault("kind", Path(safe_path).suffix.lstrip(".") or "file")
    item["viewable"] = is_text_artifact(safe_path)
    item["language"] = text_language(safe_path)

    try:
        file_path = resolve_artifact_file(workspace, safe_path)
        stat = file_path.stat()
        item["available"] = True
        item["size"] = stat.st_size
        item["mtime"] = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
        item["content_type"] = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    except ArtifactAccessError:
        item["available"] = False
        item["size"] = None
        item["mtime"] = None
        item["content_type"] = None
    return item


def list_artifact_files(execution: dict[str, Any]) -> list[dict[str, Any]]:
    workspace = execution.get("workspace")
    return [file_metadata(workspace, item["path"], item) for item in normalize_execution_files(execution)]


def read_artifact_text(workspace: str | Path, path: str) -> dict[str, Any]:
    safe_path = str(sanitize_artifact_path(path).as_posix())
    if not is_text_artifact(safe_path):
        raise ArtifactAccessError("Artifact file type is not viewable as text.")
    file_path = resolve_artifact_file(workspace, safe_path)
    assert_artifact_safe_for_exposure(file_path)
    size = file_path.stat().st_size
    if size > MAX_TEXT_VIEW_BYTES:
        raise ArtifactAccessError("Artifact file is too large to view inline.")
    content = file_path.read_text(encoding="utf-8", errors="replace")
    if _scan_text_for_secret(content):
        raise ArtifactAccessError("Artifact appears to contain credentials or tokens and cannot be exposed.")
    return {
        "path": safe_path,
        "language": text_language(safe_path),
        "content": content,
        "size": size,
        "content_type": mimetypes.guess_type(file_path.name)[0] or "text/plain",
    }


def read_artifact_diff(workspace: str | Path, path: str, *, max_chars: int = 120_000) -> dict[str, Any]:
    root = resolve_workspace(workspace)
    safe_path = str(sanitize_artifact_path(path).as_posix())
    completed = subprocess.run(
        ["git", "diff", "--cached", "--no-color", "--", safe_path],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    diff = completed.stdout or completed.stderr or ""
    if _scan_text_for_secret(diff):
        raise ArtifactAccessError("Artifact diff appears to contain credentials or tokens and cannot be exposed.")
    truncated = len(diff) > max_chars
    return {
        "path": safe_path,
        "language": "diff",
        "content": diff[:max_chars] if truncated else diff,
        "size": len(diff.encode("utf-8")),
        "truncated": truncated,
        "content_type": "text/x-diff",
    }


def build_artifact_bundle(workspace: str | Path, execution: dict[str, Any]) -> bytes:
    files = [item for item in list_artifact_files(execution) if item.get("available")]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in files:
            safe_path = item["path"]
            file_path = resolve_artifact_file(workspace, safe_path)
            assert_artifact_safe_for_exposure(file_path)
            archive.write(file_path, arcname=safe_path)
    return buffer.getvalue()


def cleanup_expired_workspaces(root: str | Path, *, retention_days: int) -> int:
    retention_seconds = max(1, int(retention_days)) * 24 * 60 * 60
    try:
        base = Path(root).expanduser().resolve(strict=True)
    except OSError:
        return 0
    if not base.is_dir():
        return 0

    now = datetime.now(timezone.utc).timestamp()
    removed = 0
    for child in base.iterdir():
        if not child.is_dir():
            continue
        try:
            if not (child / "opencode.json").exists():
                continue
            age = now - child.stat().st_mtime
            if age <= retention_seconds:
                continue
            import shutil

            shutil.rmtree(child, ignore_errors=True)
            removed += 1
        except OSError:
            continue
    return removed
