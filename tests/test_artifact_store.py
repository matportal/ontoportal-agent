import importlib
import json
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
    ontology_artifact_summary,
    read_artifact_text,
    resolve_artifact_file,
    resolve_safe_workspace,
    sanitize_artifact_path,
)


@pytest.mark.parametrize("path", ["../secret.ttl", "/tmp/secret.ttl", "~/secret.ttl", "ok/../secret.ttl", ""])
def test_sanitize_artifact_path_rejects_unsafe_paths(path):
    with pytest.raises(ArtifactAccessError):
        sanitize_artifact_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "opencode.json",
        "pi.json",
        ".git/config",
        ".opencode-home/auth.json",
        ".codex-home/auth.json",
        ".env",
        "nested/token.json",
        "private.pem",
    ],
)
def test_sanitize_artifact_path_rejects_runtime_and_secret_files(path):
    with pytest.raises(ArtifactAccessError):
        sanitize_artifact_path(path)


def test_resolve_safe_workspace_stays_under_runtime_root(tmp_path):
    root = tmp_path / "opencode-runs"
    workspace = root / "thread-1"
    outside = tmp_path / "outside"
    workspace.mkdir(parents=True)
    outside.mkdir()

    assert resolve_safe_workspace(root, workspace) == workspace.resolve()
    with pytest.raises(ArtifactAccessError):
        resolve_safe_workspace(root, root)
    with pytest.raises(ArtifactAccessError):
        resolve_safe_workspace(root, outside)
    with pytest.raises(ArtifactAccessError):
        resolve_safe_workspace(root, root / "missing")


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


def test_ontology_artifact_summary_extracts_safe_structured_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ontology-proposal.json").write_text(
        json.dumps(
            {
                "schema_version": "ontology-copilot/v1",
                "title": "Add processing method",
                "summary": "Draft proposal only.",
                "goals": ["Represent processing methods."],
                "scope": "Review-only scope.",
                "competency_questions": [{"id": "CQ1", "question": "Which materials use the method?"}],
                "reuse_candidates": [
                    {
                        "label": "Sintering",
                        "iri": "https://example.org/Sintering",
                        "source_ontology": "EX",
                        "confidence": 0.9,
                        "recommended_action": "reuse",
                        "rationale": "Existing candidate found.",
                    }
                ],
                "operations": [
                    {
                        "operation": "create_class",
                        "entity_type": "class",
                        "iri": "https://example.org/ProcessingMethod",
                        "label": "Processing method",
                        "rationale": "Needed for review.",
                    }
                ],
                "assumptions": ["Operator review required."],
                "risks": ["IRI policy needs review."],
            }
        ),
        encoding="utf-8",
    )
    (workspace / "validation-summary.json").write_text(
        json.dumps({"schema_version": "ontology-copilot/v1", "status": "warning", "summary": "Review warnings."}),
        encoding="utf-8",
    )
    execution = {
        "workspace": str(workspace),
        "changed_files": [
            {"status": "A", "path": "ontology-proposal.json", "kind": "json"},
            {"status": "A", "path": "validation-summary.json", "kind": "json"},
        ],
        "validation_report": {
            "diagnostics": [{"status": "warning", "path": "ontology-proposal.json", "message": "Review needed."}],
            "diagnostic_summary": {"warnings": 1, "failed": 0},
        },
    }

    summary = ontology_artifact_summary(execution)

    assert summary["available"] is True
    assert summary["proposal"]["title"] == "Add processing method"
    assert summary["proposal"]["operations_count"] == 1
    assert summary["workspace"]["competency_questions_count"] == 1
    assert summary["reuse"]["candidates"][0]["recommended_action"] == "reuse"
    assert summary["validation"]["status"] == "warning"
    assert summary["validation"]["diagnostic_summary"]["warnings"] == 1
    assert "content" not in json.dumps(summary)


def test_ontology_artifact_summary_rejects_unsafe_structured_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "ontology-proposal.json").write_text(
        json.dumps(
            {
                "schema_version": "ontology-copilot/v1",
                "title": "Unsafe proposal",
                "summary": "References /tmp/private.ttl",
                "competency_questions": [{"id": "CQ1", "question": "What is proposed?"}],
                "operations": [
                    {
                        "operation": "create_class",
                        "entity_type": "class",
                        "iri": "https://example.org/Thing",
                        "label": "Thing",
                        "rationale": "Needed for review.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    execution = {
        "workspace": str(workspace),
        "changed_files": [{"status": "A", "path": "ontology-proposal.json", "kind": "json"}],
    }

    summary = ontology_artifact_summary(execution)

    assert summary["available"] is False
    assert summary["proposal"] == {}
    assert summary["errors"] == [{"path": "ontology-proposal.json", "message": "Artifact schema validation failed."}]


def test_artifact_secret_scanner_blocks_view_and_bundle(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    secret_artifact = workspace / "proposal.ttl"
    secret_artifact.write_text(
        "@prefix ex: <https://example.org/> .\n# api_key = sk-testsecretvalue12345\n",
        encoding="utf-8",
    )
    execution = {
        "workspace": str(workspace),
        "changed_files": [{"status": "A", "path": "proposal.ttl", "kind": "ttl"}],
    }

    with pytest.raises(ArtifactAccessError):
        read_artifact_text(workspace, "proposal.ttl")
    with pytest.raises(ArtifactAccessError):
        build_artifact_bundle(workspace, execution)


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
