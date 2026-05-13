"""Security tests — input sanitisation, validation, and API hardening."""

from __future__ import annotations

import pytest

from orchestrator.core.sanitize import (
    MAX_QUERY_LENGTH,
    sanitize_context,
    sanitize_query,
    sanitize_text,
    validate_history,
    validate_model_name,
    validate_session_id,
)


class TestSanitizeText:
    def test_strips_null_bytes(self):
        assert sanitize_text("hello\x00world") == "helloworld"

    def test_strips_control_chars(self):
        assert sanitize_text("a\x01b\x02c\x7f") == "abc"

    def test_preserves_newlines_and_tabs(self):
        assert sanitize_text("line1\nline2\ttab") == "line1\nline2\ttab"

    def test_preserves_unicode(self):
        assert sanitize_text("olá café résumé 日本語") == "olá café résumé 日本語"

    def test_empty_string(self):
        assert sanitize_text("") == ""


class TestSanitizeQuery:
    def test_strips_control_chars(self):
        assert sanitize_query("hello\x00world") == "helloworld"

    def test_strips_whitespace(self):
        assert sanitize_query("  hello  ") == "hello"

    def test_enforces_max_length(self):
        long = "a" * (MAX_QUERY_LENGTH + 100)
        result = sanitize_query(long)
        assert len(result) == MAX_QUERY_LENGTH

    def test_empty_after_strip(self):
        assert sanitize_query("   ") == ""


class TestValidateHistory:
    def test_none_returns_none(self):
        assert validate_history(None) is None

    def test_empty_returns_none(self):
        assert validate_history([]) is None

    def test_valid_history(self):
        history = [
            {"role": "user", "content": "olá"},
            {"role": "assistant", "content": "oi"},
        ]
        result = validate_history(history)
        assert result == history

    def test_filters_invalid_role(self):
        history = [
            {"role": "hacker", "content": "inject"},
            {"role": "user", "content": "valid"},
        ]
        result = validate_history(history)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_filters_missing_content(self):
        history = [
            {"role": "user"},
            {"role": "user", "content": "ok"},
        ]
        result = validate_history(history)
        assert len(result) == 1

    def test_filters_empty_content(self):
        history = [{"role": "user", "content": "  "}]
        assert validate_history(history) is None

    def test_strips_control_chars_from_content(self):
        history = [{"role": "user", "content": "hello\x00world"}]
        result = validate_history(history)
        assert result[0]["content"] == "helloworld"

    def test_drops_extra_keys(self):
        history = [{"role": "user", "content": "test", "evil": "payload"}]
        result = validate_history(history)
        assert set(result[0].keys()) == {"role", "content"}

    def test_caps_messages(self):
        history = [{"role": "user", "content": f"msg{i}"} for i in range(100)]
        result = validate_history(history)
        assert len(result) == 50  # MAX_HISTORY_MESSAGES

    def test_non_dict_entries_skipped(self):
        history = ["not a dict", {"role": "user", "content": "ok"}]
        result = validate_history(history)
        assert len(result) == 1


class TestValidateSessionId:
    def test_valid_uuid(self):
        sid = "550e8400-e29b-41d4-a716-446655440000"
        assert validate_session_id(sid) == sid

    def test_invalid_string(self):
        assert validate_session_id("not-a-uuid") is None

    def test_sql_injection_attempt(self):
        assert validate_session_id("'; DROP TABLE sessions; --") is None

    def test_none(self):
        assert validate_session_id(None) is None

    def test_empty(self):
        assert validate_session_id("") is None


class TestValidateModelName:
    def test_valid_model(self):
        assert validate_model_name("qwen3:8b") == "qwen3:8b"

    def test_valid_with_dots(self):
        assert validate_model_name("qwen2.5-coder:7b") == "qwen2.5-coder:7b"

    def test_none_returns_none(self):
        assert validate_model_name(None) is None

    def test_empty_returns_none(self):
        assert validate_model_name("") is None

    def test_rejects_shell_injection(self):
        assert validate_model_name("model; rm -rf /") is None

    def test_rejects_newline(self):
        assert validate_model_name("model\ninjection") is None

    def test_rejects_too_long(self):
        assert validate_model_name("a" * 200) is None

    def test_strips_whitespace(self):
        assert validate_model_name("  qwen3:8b  ") == "qwen3:8b"


class TestSanitizeContext:
    def test_strips_control_chars(self):
        assert sanitize_context("data\x00value") == "datavalue"

    def test_truncates_long_content(self):
        big = "x" * 50_000
        result = sanitize_context(big)
        assert len(result) == 32_000

    def test_preserves_normal_content(self):
        text = "## Docker\nConfiguração do container.\n```python\nprint('ok')\n```"
        assert sanitize_context(text) == text


class TestAPISecurityHeaders:
    """Test security headers via the FastAPI test client."""

    @pytest.fixture()
    def client(self):
        from fastapi.testclient import TestClient

        from orchestrator.api.app import app
        from orchestrator.config import _reset_settings

        _reset_settings()
        with TestClient(app) as c:
            yield c

    def test_health_has_security_headers(self, client):
        resp = client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Cache-Control"] == "no-store"

    def test_error_handler_hides_internals(self, client):
        # A request to a non-existent endpoint should not leak stack traces
        resp = client.get("/nonexistent")
        assert resp.status_code in (404, 405)
        body = resp.text
        assert "Traceback" not in body
        assert "File " not in body
