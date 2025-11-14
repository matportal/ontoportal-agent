from __future__ import annotations

from pathlib import Path

import typer

from .agent.runtime import OntoPortalAgent
from .config import get_settings
from .publishing import OntoPortalPublisher

app = typer.Typer(add_completion=False)


@app.command()
def chat():
    """Launch an interactive REPL for the OntoPortal agent."""
    agent = OntoPortalAgent()
    typer.echo("OntoPortal Agent ready. Type 'exit' to quit.\n")
    while True:
        prompt = typer.prompt("user")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        response = agent.invoke(prompt)
        typer.echo(f"agent: {response}\n")


@app.command()
def publish(
    acronym: str = typer.Argument(..., help="Ontology acronym to update."),
    artifact: Path = typer.Argument(..., help="Path to the ontology artifact to upload."),
    contact_email: str = typer.Option(..., "--contact-email", help="Contact email for the submission."),
    notes: str = typer.Option("Submitted via OntoPortal Agent", "--notes", help="Submission notes."),
    private: bool = typer.Option(False, "--private/--public", help="Submit the artifact as a private MatPortal submission."),
):
    """Publish a prepared ontology artifact back to OntoPortal."""
    get_settings()  # ensure env validation
    publisher = OntoPortalPublisher()
    result = publisher.submit_ontology(
        acronym=acronym,
        artifact_path=artifact,
        contact_email=contact_email,
        notes=notes,
        is_private=private,
    )
    typer.echo(f"Submission queued with id: {result.get('submissionId')}")


if __name__ == "__main__":
    app()
