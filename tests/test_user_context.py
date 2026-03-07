import importlib
import time

import pytest

if importlib.util.find_spec("ontoportal_agent") is None:
    pytest.skip("ontoportal_agent package not available", allow_module_level=True)

from ontoportal_agent.db.user_context import build_signature, verify_user_context_headers


def test_verify_user_context_signature_ok():
    secret = "test-secret"
    timestamp = str(int(time.time()))
    headers = {
        "X-Assistant-User-Id": "u-1",
        "X-Assistant-Username": "alice",
        "X-Assistant-User-Email": "alice@example.org",
        "X-Assistant-User-Timestamp": timestamp,
        "X-Assistant-User-Signature": build_signature(
            secret, "u-1", "alice", "alice@example.org", timestamp
        ),
    }

    context = verify_user_context_headers(headers, secret=secret, ttl_seconds=300)
    assert context.user_id == "u-1"
    assert context.username == "alice"


def test_verify_user_context_signature_expired():
    secret = "test-secret"
    timestamp = str(int(time.time()) - 3600)
    headers = {
        "X-Assistant-User-Id": "u-1",
        "X-Assistant-Username": "alice",
        "X-Assistant-User-Email": "alice@example.org",
        "X-Assistant-User-Timestamp": timestamp,
        "X-Assistant-User-Signature": build_signature(
            secret, "u-1", "alice", "alice@example.org", timestamp
        ),
    }

    with pytest.raises(ValueError):
        verify_user_context_headers(headers, secret=secret, ttl_seconds=300)
