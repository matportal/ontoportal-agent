from __future__ import annotations

from copy import deepcopy
from typing import Any

ANTIGRAVITY_PROVIDER_ID = "google"
DEFAULT_ANTIGRAVITY_MODEL_REF = "google/antigravity-gemini-3-pro"

_DEFAULT_MODALITIES = {
    "input": ["text", "image", "pdf"],
    "output": ["text"],
}

ANTIGRAVITY_MODEL_DEFINITIONS: dict[str, dict[str, Any]] = {
    "antigravity-gemini-3-pro": {
        "name": "Gemini 3 Pro (Antigravity)",
        "limit": {"context": 1048576, "output": 65535},
        "modalities": _DEFAULT_MODALITIES,
        "variants": {
            "low": {"thinkingLevel": "low"},
            "high": {"thinkingLevel": "high"},
        },
    },
    "antigravity-gemini-3.1-pro": {
        "name": "Gemini 3.1 Pro (Antigravity)",
        "limit": {"context": 1048576, "output": 65535},
        "modalities": _DEFAULT_MODALITIES,
        "variants": {
            "low": {"thinkingLevel": "low"},
            "high": {"thinkingLevel": "high"},
        },
    },
    "antigravity-gemini-3-flash": {
        "name": "Gemini 3 Flash (Antigravity)",
        "limit": {"context": 1048576, "output": 65536},
        "modalities": _DEFAULT_MODALITIES,
        "variants": {
            "minimal": {"thinkingLevel": "minimal"},
            "low": {"thinkingLevel": "low"},
            "medium": {"thinkingLevel": "medium"},
            "high": {"thinkingLevel": "high"},
        },
    },
    "antigravity-claude-sonnet-4-6": {
        "name": "Claude Sonnet 4.6 (Antigravity)",
        "limit": {"context": 200000, "output": 64000},
        "modalities": _DEFAULT_MODALITIES,
    },
    "antigravity-claude-opus-4-6-thinking": {
        "name": "Claude Opus 4.6 Thinking (Antigravity)",
        "limit": {"context": 200000, "output": 64000},
        "modalities": _DEFAULT_MODALITIES,
        "variants": {
            "low": {"thinkingConfig": {"thinkingBudget": 8192}},
            "max": {"thinkingConfig": {"thinkingBudget": 32768}},
        },
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash (Gemini CLI)",
        "limit": {"context": 1048576, "output": 65536},
        "modalities": _DEFAULT_MODALITIES,
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro (Gemini CLI)",
        "limit": {"context": 1048576, "output": 65536},
        "modalities": _DEFAULT_MODALITIES,
    },
    "gemini-3-flash-preview": {
        "name": "Gemini 3 Flash Preview (Gemini CLI)",
        "limit": {"context": 1048576, "output": 65536},
        "modalities": _DEFAULT_MODALITIES,
    },
    "gemini-3-pro-preview": {
        "name": "Gemini 3 Pro Preview (Gemini CLI)",
        "limit": {"context": 1048576, "output": 65535},
        "modalities": _DEFAULT_MODALITIES,
    },
    "gemini-3.1-pro-preview": {
        "name": "Gemini 3.1 Pro Preview (Gemini CLI)",
        "limit": {"context": 1048576, "output": 65535},
        "modalities": _DEFAULT_MODALITIES,
    },
    "gemini-3.1-pro-preview-customtools": {
        "name": "Gemini 3.1 Pro Preview Custom Tools (Gemini CLI)",
        "limit": {"context": 1048576, "output": 65535},
        "modalities": _DEFAULT_MODALITIES,
    },
}


def normalize_antigravity_model_ref(value: str | None, *, default: str | None = None) -> str:
    fallback = default or DEFAULT_ANTIGRAVITY_MODEL_REF
    raw = str(value or "").strip()
    if not raw:
        return fallback
    model_id = raw.split("/", 1)[1] if raw.startswith(f"{ANTIGRAVITY_PROVIDER_ID}/") else raw
    if model_id in ANTIGRAVITY_MODEL_DEFINITIONS:
        return f"{ANTIGRAVITY_PROVIDER_ID}/{model_id}"
    return fallback


def antigravity_opencode_provider_config() -> dict[str, Any]:
    return {
        ANTIGRAVITY_PROVIDER_ID: {
            "models": deepcopy(ANTIGRAVITY_MODEL_DEFINITIONS),
        }
    }


def antigravity_model_options(*, selected_model_ref: str | None = None) -> list[dict[str, Any]]:
    selected = normalize_antigravity_model_ref(selected_model_ref)
    options: list[dict[str, Any]] = []
    for model_id, definition in ANTIGRAVITY_MODEL_DEFINITIONS.items():
        model_ref = f"{ANTIGRAVITY_PROVIDER_ID}/{model_id}"
        variants = sorted((definition.get("variants") or {}).keys())
        options.append(
            {
                "id": model_id,
                "model_ref": model_ref,
                "name": str(definition.get("name") or model_id),
                "variants": variants,
                "context": int((definition.get("limit") or {}).get("context") or 0),
                "output": int((definition.get("limit") or {}).get("output") or 0),
                "selected": model_ref == selected,
            }
        )
    return options
