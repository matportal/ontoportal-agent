from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

import requests

from .config import get_settings


class OntoPortalPublisher:
    """Handles submission of ontology updates back to OntoPortal."""

    def __init__(self, *, api_base: Optional[str] = None, api_key: Optional[str] = None):
        settings = get_settings()
        self.api_base = api_base or settings.ontoportal_api_base
        self.api_key = api_key or settings.ontoportal_api_key

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"apikey token={self.api_key}",
            "Content-Type": "application/json",
        }

    def submit_ontology(
        self,
        acronym: str,
        artifact_path: Path,
        *,
        contact_email: str,
        notes: str = "Updated via OntoPortal Agent",
        is_private: bool = False,
    ) -> dict:
        """Creates a new submission for the given ontology artifact."""
        url = f"{self.api_base.rstrip('/')}/ontologies/{acronym}/submissions"
        payload = {
            "contactEmail": contact_email,
            "description": notes,
            "uploadedFile": base64.b64encode(artifact_path.read_bytes()).decode("utf-8"),
            "filename": artifact_path.name,
            "contentType": "application/rdf+xml" if artifact_path.suffix == ".rdf" else "text/turtle",
        }
        if is_private:
            payload["isPrivate"] = True
        response = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=120)
        response.raise_for_status()
        return response.json()
