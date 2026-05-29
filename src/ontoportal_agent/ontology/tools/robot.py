from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Callable


class RobotAdapter:
    """Whitelisted ROBOT verify/report adapter for ontology proposal artifacts."""

    def __init__(
        self,
        *,
        enabled: bool,
        java_path: str,
        robot_jar_path: Path | None,
        timeout_seconds: int = 60,
        truncate: Callable[[str, int], str] | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.java_path = str(java_path or "java")
        self.robot_jar_path = robot_jar_path
        self.timeout_seconds = int(timeout_seconds)
        self._truncate = truncate or self._default_truncate

    def verify(self, *, path: Path, display_path: str) -> dict[str, object]:
        command = self.command("verify", path)
        if command is None:
            return {
                "status": "unavailable",
                "message": "ROBOT is not configured in this runtime.",
            }
        try:
            completed = subprocess.run(
                command,
                cwd=str(path.parent),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "tool": command[0],
                "command": self.redacted_command(command, workspace=path.parent),
                "message": f"ROBOT verify timed out after {self.timeout_seconds} seconds.",
            }
        output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
        message = self._truncate(output, 400) if output else ""
        if completed.returncode == 0:
            result: dict[str, object] = {
                "status": "passed",
                "tool": command[0],
                "command": self.redacted_command(command, workspace=path.parent),
                "message": message or "ROBOT verify passed.",
            }
            report_result = self.report(path=path)
            if report_result:
                result["report"] = report_result
            return result
        return {
            "status": "failed",
            "tool": command[0],
            "command": self.redacted_command(command, workspace=path.parent),
            "message": message or f"ROBOT verify failed for {display_path}.",
        }

    def report(self, *, path: Path) -> dict[str, object] | None:
        output_path = path.with_name(f"{path.name}.robot-report.tsv")
        command = self.command("report", path, output_path=output_path)
        if command is None:
            return None
        try:
            completed = subprocess.run(
                command,
                cwd=str(path.parent),
                capture_output=True,
                text=True,
                check=False,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "failed",
                "path": output_path.name,
                "message": f"ROBOT report timed out after {self.timeout_seconds} seconds.",
            }
        output = "\n".join(part for part in [completed.stdout.strip(), completed.stderr.strip()] if part)
        message = self._truncate(output, 400) if output else ""
        if completed.returncode == 0:
            return {
                "status": "passed",
                "path": output_path.name,
                "message": message or "ROBOT report generated.",
            }
        return {
            "status": "failed",
            "path": output_path.name,
            "message": message or "ROBOT report failed.",
        }

    def command(self, action: str, path: Path, *, output_path: Path | None = None) -> list[str] | None:
        if not self.enabled:
            return None
        action = str(action or "").strip()
        if action not in {"verify", "report"}:
            return None
        args = [action, "--input", str(path)]
        if output_path is not None:
            args.extend(["--output", str(output_path)])
        robot_jar = self.robot_jar_path
        if robot_jar and robot_jar.exists():
            return [self.java_path, "-jar", str(robot_jar), *args]
        robot_path = shutil.which("robot")
        if robot_path:
            return [robot_path, *args]
        return None

    @staticmethod
    def redacted_command(command: list[str], *, workspace: Path) -> list[str]:
        workspace_text = str(workspace)
        return [str(item).replace(workspace_text, "<workspace>") for item in command]

    @staticmethod
    def _default_truncate(value: str, max_chars: int) -> str:
        clean = " ".join(str(value).split())
        if len(clean) <= max_chars:
            return clean
        return f"{clean[: max_chars - 3]}..."
