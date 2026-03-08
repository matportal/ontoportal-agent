import json
from pathlib import Path

import pytest

from ontoportal_agent.agent.graph import _extract_json_object, _format_pending_approval_response, _parse_edit_plan


def test_parse_edit_plan_accepts_fenced_json():
    response = """```json
{
  "workspace": "material_science_ontology_dev",
  "actions": [
    {
      "description": "Create TensileStrengthMeasurement",
      "artifact": "mso.ttl",
      "create": true,
      "format": "turtle",
      "code": "ontology_repo.save_graph(graph, workspace, artifact)"
    }
  ],
  "publish": {
    "acronym": "MSO",
    "artifact": "mso.ttl",
    "contact_email": "ontology-curator@example.com",
    "notes": "Initial creation",
    "private": false
  }
}
```"""

    plan = _parse_edit_plan(response)

    assert plan["workspace"] == "material_science_ontology_dev"
    assert plan["actions"][0]["artifact"] == "mso.ttl"
    assert plan["publish"]["acronym"] == "MSO"


def test_extract_json_object_accepts_prose_wrapped_payload():
    response = (
        "Here is the plan.\n\n"
        "{\n"
        '  "workspace": "session",\n'
        '  "actions": [],\n'
        '  "publish": null\n'
        "}\n"
        "\nPlease review it."
    )

    extracted = _extract_json_object(response)
    parsed = json.loads(extracted)

    assert parsed["workspace"] == "session"
    assert parsed["actions"] == []
    assert parsed["publish"] is None


def test_parse_edit_plan_rejects_non_json_text():
    with pytest.raises(json.JSONDecodeError):
        _parse_edit_plan("not a plan")


def test_format_pending_approval_response_escapes_workspace_and_artifact_details():
    response = _format_pending_approval_response(
        summary_lines=["Create a new ontology class called `TensileStrengthMeasurement`."],
        workspace="tensile_strength_ontology_v1",
        sandbox_output="Sandbox executed without output.",
        artifact_path=Path("/tmp/ontoportal-agent/tensile_strength_ontology_v1/tensile_strength.ttl"),
    )

    assert "## Proposed ontology edits (pending approval)" in response
    assert "- Workspace: `tensile_strength_ontology_v1`" in response
    assert "`/tmp/ontoportal-agent/tensile_strength_ontology_v1/tensile_strength.ttl`" in response
    assert "```text" in response
