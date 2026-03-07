import importlib

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.db.crypto import EncryptionService


def test_encryption_roundtrip():
    service = EncryptionService(current_key_raw="A" * 32)
    encrypted, key_version = service.encrypt_json({"api_key": "secret", "provider": "openai"})
    assert key_version.startswith("cur-")

    decrypted, decrypted_key_version = service.decrypt_json(encrypted)
    assert decrypted == {"api_key": "secret", "provider": "openai"}
    assert decrypted_key_version == key_version


def test_encryption_key_rotation_uses_previous_key():
    old_service = EncryptionService(current_key_raw="A" * 32)
    encrypted, _ = old_service.encrypt_json({"token": "abc"})

    rotated_service = EncryptionService(
        current_key_raw="B" * 32,
        previous_key_raw="A" * 32,
    )
    decrypted, used_key = rotated_service.decrypt_json(encrypted)
    assert decrypted == {"token": "abc"}
    assert used_key.endswith(old_service.current.key_id.split("-", 1)[1])
