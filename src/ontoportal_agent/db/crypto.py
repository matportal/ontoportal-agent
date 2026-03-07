from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def _normalize_b64(value: str) -> str:
    padded = value.strip()
    missing = len(padded) % 4
    if missing:
        padded += "=" * (4 - missing)
    return padded


def _decode_key(raw: Optional[str]) -> Optional[bytes]:
    if not raw:
        return None

    raw = raw.strip()
    # Accept raw 32-byte keys as plain text for local/dev usage.
    if len(raw.encode("utf-8")) == 32:
        return raw.encode("utf-8")

    try:
        decoded = base64.urlsafe_b64decode(_normalize_b64(raw))
    except Exception:
        return None

    if len(decoded) != 32:
        return None
    return decoded


def _key_fingerprint(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()[:12]


@dataclass(frozen=True)
class EncryptionKeyMaterial:
    key_id: str
    key: bytes


class EncryptionService:
    def __init__(self, current_key_raw: Optional[str], previous_key_raw: Optional[str] = None):
        current_key = _decode_key(current_key_raw)
        previous_key = _decode_key(previous_key_raw)

        self.current: Optional[EncryptionKeyMaterial] = None
        self.previous: Optional[EncryptionKeyMaterial] = None

        if current_key:
            self.current = EncryptionKeyMaterial(
                key_id=f"cur-{_key_fingerprint(current_key)}",
                key=current_key,
            )
        if previous_key:
            self.previous = EncryptionKeyMaterial(
                key_id=f"prev-{_key_fingerprint(previous_key)}",
                key=previous_key,
            )

    @property
    def enabled(self) -> bool:
        return self.current is not None

    def encrypt_json(self, payload: Any) -> tuple[str, str]:
        if not self.current:
            raise RuntimeError("Encryption key is not configured.")
        plaintext = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        nonce = os.urandom(12)
        cipher = AESGCM(self.current.key).encrypt(nonce, plaintext, None)
        envelope = {
            "key_id": self.current.key_id,
            "nonce": base64.urlsafe_b64encode(nonce).decode("ascii"),
            "ciphertext": base64.urlsafe_b64encode(cipher).decode("ascii"),
        }
        return json.dumps(envelope, separators=(",", ":")), self.current.key_id

    def decrypt_json(self, encrypted: str) -> tuple[Any, str]:
        envelope = json.loads(encrypted)
        key_id = envelope.get("key_id", "")
        nonce = base64.urlsafe_b64decode(_normalize_b64(envelope["nonce"]))
        ciphertext = base64.urlsafe_b64decode(_normalize_b64(envelope["ciphertext"]))
        candidates: list[EncryptionKeyMaterial] = []
        if self.current:
            candidates.append(self.current)
        if self.previous:
            candidates.append(self.previous)

        # Prefer key-id match first, then try fallbacks.
        ordered_candidates = []
        for candidate in candidates:
            if key_id and candidate.key_id == key_id:
                ordered_candidates.insert(0, candidate)
            else:
                ordered_candidates.append(candidate)

        last_error: Exception | None = None
        for candidate in ordered_candidates:
            try:
                plaintext = AESGCM(candidate.key).decrypt(nonce, ciphertext, None)
                return json.loads(plaintext.decode("utf-8")), candidate.key_id
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError("Failed to decrypt payload with available keys.") from last_error
