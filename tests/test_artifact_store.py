import importlib
import os
import time
import zipfile
from io import BytesIO

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.artifact_store import (
    ArtifactAccessError,
    build_artifact_bundle,
    cleanup_expired_workspaces,
    execution_allows_path,
    list_artifact_files,
    read_artifact_text,
    resolve_artifact_file,
    sanitize_artifact_path,
)


@pytest.mark.parametrize("path", ["../secret.ttl", "/tmp/secret.ttl", "~/secret.ttl", "ok/../secret.ttl", ""])
def test_sanitize_artifact_path_rejects_unsafe_paths(path):
    with pytest.raises(ArtifactAccessError):
        sanitize_artifact_path(path)


def test_artifact_store_lists_only_declared_files_and_bundles_available_files(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "proposal.ttl").write_text("@prefix ex: <https://example.org/> .\n", encoding="utf-8")
    (workspace / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (workspace / "opencode.json").write_text('{"api_key":"secret"}', encoding="utf-8")
    execution = {
        "workspace": str(workspace),
        "changed_files": [
            {"status": "A", "path": "proposal.ttl", "kind": "ttl"},
            {"status": "A", "path": "notes.md", "kind": "md"},
            {"status": "A", "path": "../ignored.ttl", "kind": "ttl"},
        ],
        "artifact_candidates": [{"status": "A", "path": "proposal.ttl", "kind": "ttl"}],
    }

    files = list_artifact_files(execution)

    assert [item["path"] for item in files] == ["proposal.ttl", "notes.md"]
    assert files[0]["viewable"] is True
    assert files[0]["artifact"] is True
    assert execution_allows_path(execution, "proposal.ttl") is True
    assert execution_allows_path(execution, "opencode.json") is False

    payload = read_artifact_text(workspace, "proposal.ttl")
    assert payload["language"] == "turtle"
    assert "@prefix ex:" in payload["content"]

    with zipfile.ZipFile(BytesIO(build_artifact_bundle(workspace, execution))) as archive:
        assert sorted(archive.namelist()) == ["notes.md", "proposal.ttl"]


def test_resolve_artifact_file_blocks_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.ttl"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "leak.ttl").symlink_to(outside)

    with pytest.raises(ArtifactAccessError):
        resolve_artifact_file(workspace, "leak.ttl")


def test_cleanup_expired_workspaces_removes_only_old_opencode_runs(tmp_path):
    root = tmp_path / "opencode-runs"
    root.mkdir()
    old_workspace = root / "thread-old-run"
    fresh_workspace = root / "thread-fresh-run"
    unrelated = root / "not-an-opencode-run"
    for path in (old_workspace, fresh_workspace, unrelated):
        path.mkdir()
    (old_workspace / "opencode.json").write_text("{}", encoding="utf-8")
    (fresh_workspace / "opencode.json").write_text("{}", encoding="utf-8")
    (unrelated / "notes.md").write_text("keep", encoding="utf-8")

    old_time = time.time() - (3 * 24 * 60 * 60)
    os.utime(old_workspace, (old_time, old_time))

    removed = cleanup_expired_workspaces(root, retention_days=1)

    assert removed == 1
    assert not old_workspace.exists()
    assert fresh_workspace.exists()
    assert unrelated.exists()
