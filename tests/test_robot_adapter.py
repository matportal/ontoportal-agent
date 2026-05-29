import importlib
import subprocess

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.ontology.tools.robot import RobotAdapter


def test_robot_adapter_builds_whitelisted_jar_commands(tmp_path):
    robot_jar = tmp_path / "robot.jar"
    robot_jar.write_text("fake", encoding="utf-8")
    artifact = tmp_path / "proposal.ttl"
    artifact.write_text("@prefix ex: <https://example.org/> .\n", encoding="utf-8")
    output = tmp_path / "proposal.report.tsv"

    adapter = RobotAdapter(enabled=True, java_path="java", robot_jar_path=robot_jar)

    assert adapter.command("verify", artifact) == ["java", "-jar", str(robot_jar), "verify", "--input", str(artifact)]
    assert adapter.command("report", artifact, output_path=output) == [
        "java",
        "-jar",
        str(robot_jar),
        "report",
        "--input",
        str(artifact),
        "--output",
        str(output),
    ]
    assert adapter.command("merge", artifact) is None


def test_robot_adapter_verify_runs_report_and_redacts_workspace(monkeypatch, tmp_path):
    robot_jar = tmp_path / "robot.jar"
    robot_jar.write_text("fake", encoding="utf-8")
    artifact = tmp_path / "proposal.ttl"
    artifact.write_text("@prefix ex: <https://example.org/> .\n", encoding="utf-8")
    commands = []

    def fake_run(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("ontoportal_agent.ontology.tools.robot.subprocess.run", fake_run)
    adapter = RobotAdapter(enabled=True, java_path="java", robot_jar_path=robot_jar)

    result = adapter.verify(path=artifact, display_path="proposal.ttl")

    assert result["status"] == "passed"
    assert result["command"] == ["java", "-jar", "<workspace>/robot.jar", "verify", "--input", "<workspace>/proposal.ttl"]
    assert result["report"]["status"] == "passed"
    assert result["report"]["path"] == "proposal.ttl.robot-report.tsv"
    assert commands[0][3] == "verify"
    assert commands[1][3] == "report"


def test_robot_adapter_disabled_is_unavailable(tmp_path):
    artifact = tmp_path / "proposal.ttl"
    artifact.write_text("@prefix ex: <https://example.org/> .\n", encoding="utf-8")
    adapter = RobotAdapter(enabled=False, java_path="java", robot_jar_path=None)

    result = adapter.verify(path=artifact, display_path="proposal.ttl")

    assert result == {"status": "unavailable", "message": "ROBOT is not configured in this runtime."}


def test_robot_adapter_timeout_is_failed(monkeypatch, tmp_path):
    robot_jar = tmp_path / "robot.jar"
    robot_jar.write_text("fake", encoding="utf-8")
    artifact = tmp_path / "proposal.ttl"
    artifact.write_text("@prefix ex: <https://example.org/> .\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        raise subprocess.TimeoutExpired(command, timeout=60)

    monkeypatch.setattr("ontoportal_agent.ontology.tools.robot.subprocess.run", fake_run)
    adapter = RobotAdapter(enabled=True, java_path="java", robot_jar_path=robot_jar)

    result = adapter.verify(path=artifact, display_path="proposal.ttl")

    assert result["status"] == "failed"
    assert result["message"] == "ROBOT verify timed out after 60 seconds."
