"""Tests for the CLI."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from orchestrator.cli.main import main
from orchestrator.context.base import Complexity, Intent, OrchestratorResult, RoutingResult


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "AI Orchestrator" in result.output

    def test_classify(self, runner):
        with patch("orchestrator.factory.create_engine") as mock_factory:
            engine = MagicMock()
            engine.classify.return_value = RoutingResult(
                intent=Intent.CODE, complexity=Complexity.COMPLEX,
            )
            mock_factory.return_value = engine
            result = runner.invoke(main, ["classify", "debug", "this", "function"])
        assert result.exit_code == 0
        assert "code" in result.output
        assert "complex" in result.output

    def test_ask_json(self, runner):
        with patch("orchestrator.factory.create_engine") as mock_factory:
            engine = MagicMock()
            engine.run.return_value = OrchestratorResult(
                response="answer",
                model_used="qwen3:8b",
                intent=Intent.GENERAL,
                complexity=Complexity.NORMAL,
                sources_used=[],
                context_tokens=0,
                latency_ms=10.0,
            )
            mock_factory.return_value = engine
            result = runner.invoke(main, ["ask", "--json-output", "hello"])
        assert result.exit_code == 0
        assert '"response": "answer"' in result.output

    def test_config(self, runner):
        result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        assert "default:" in result.output

    def test_stream_with_debug(self, runner):
        """--stream --debug should emit routing decisions to stderr before streaming."""
        with patch("orchestrator.factory.create_engine") as mock_factory:
            engine = MagicMock()
            engine.classify.return_value = RoutingResult(
                intent=Intent.CODE, complexity=Complexity.COMPLEX,
            )
            engine.stream.return_value = iter(["hello ", "world"])
            mock_factory.return_value = engine
            result = runner.invoke(
                main,
                ["ask", "--stream", "--debug", "explain this"],
                catch_exceptions=False,
            )
        assert result.exit_code == 0
        # engine.classify should have been called (for the debug header)
        engine.classify.assert_called_once()
        # engine.stream should have been called (for actual output)
        engine.stream.assert_called_once()


# ------------------------------------------------------------------
# Session integration tests
# ------------------------------------------------------------------


def _fake_result(response: str = "answer") -> OrchestratorResult:
    return OrchestratorResult(
        response=response,
        model_used="qwen3:8b",
        intent=Intent.GENERAL,
        complexity=Complexity.NORMAL,
        sources_used=[],
        context_tokens=0,
        latency_ms=10.0,
    )


def _session_cfg(enabled: bool = True):
    """Return a Settings mock with session enabled/disabled."""
    cfg = MagicMock()
    cfg.session.enabled = enabled
    cfg.session.db_path = ""
    cfg.session.max_messages = 20
    return cfg


class TestCLISession:
    """Tests for --session / -s flag."""

    def test_ask_without_session_is_stateless(self, runner):
        """Without -s the engine receives history=None (regression guard)."""
        with patch("orchestrator.factory.create_engine") as mock_factory:
            engine = MagicMock()
            engine.run.return_value = _fake_result()
            mock_factory.return_value = engine
            result = runner.invoke(main, ["ask", "hello"])

        assert result.exit_code == 0
        engine.run.assert_called_once_with("hello", history=None, model_override=None)

    def test_ask_new_session_generates_id(self, runner, tmp_path):
        """``-s`` (no value) generates a UUID and shows it in stderr."""
        db = str(tmp_path / "test.db")
        cfg = _session_cfg(enabled=True)
        cfg.session.db_path = db

        with (
            patch("orchestrator.factory.create_engine") as mock_factory,
            patch("orchestrator.cli.main.get_settings", return_value=cfg),
        ):
            engine = MagicMock()
            engine.run.return_value = _fake_result()
            mock_factory.return_value = engine

            result = runner.invoke(main, ["ask", "-s", "new", "hello"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "Session:" in result.output  # CliRunner mixes stdout+stderr
        # Engine received history (None for first message in new session)
        engine.run.assert_called_once()
        _, kwargs = engine.run.call_args
        assert kwargs["history"] is None  # new session, no prior messages

    def test_ask_continue_session_loads_history(self, runner, tmp_path):
        """``-s <id>`` loads prior history and passes it to the engine."""
        db = str(tmp_path / "test.db")
        cfg = _session_cfg(enabled=True)
        cfg.session.db_path = db

        # Pre-populate the session store with one exchange
        from orchestrator.core.session import SessionStore
        store = SessionStore(db_path=db)
        store.append("test-sess", "user", "olá")
        store.append("test-sess", "assistant", "Olá! Como posso ajudar?")
        store.close()

        with (
            patch("orchestrator.factory.create_engine") as mock_factory,
            patch("orchestrator.cli.main.get_settings", return_value=cfg),
        ):
            engine = MagicMock()
            engine.run.return_value = _fake_result("Claro!")
            mock_factory.return_value = engine

            result = runner.invoke(
                main, ["ask", "-s", "test-sess", "como te chamas?"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0
        # Engine must have received the 2-message history
        engine.run.assert_called_once()
        _, kwargs = engine.run.call_args
        history = kwargs["history"]
        assert history is not None
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_ask_session_persists_exchange(self, runner, tmp_path):
        """After a successful ask with -s, user+assistant are persisted."""
        db = str(tmp_path / "test.db")
        cfg = _session_cfg(enabled=True)
        cfg.session.db_path = db

        with (
            patch("orchestrator.factory.create_engine") as mock_factory,
            patch("orchestrator.cli.main.get_settings", return_value=cfg),
        ):
            engine = MagicMock()
            engine.run.return_value = _fake_result("world")
            mock_factory.return_value = engine

            result = runner.invoke(
                main, ["ask", "-s", "persist-test", "hello"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0

        # Verify persistence by reading the DB directly
        from orchestrator.core.session import SessionStore
        store = SessionStore(db_path=db)
        msgs = store.get("persist-test")
        store.close()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "world"}

    def test_stream_with_session_persists(self, runner, tmp_path):
        """Streaming with -s accumulates chunks and persists."""
        db = str(tmp_path / "test.db")
        cfg = _session_cfg(enabled=True)
        cfg.session.db_path = db

        with (
            patch("orchestrator.factory.create_engine") as mock_factory,
            patch("orchestrator.cli.main.get_settings", return_value=cfg),
        ):
            engine = MagicMock()
            engine.stream.return_value = iter(["hello ", "world"])
            mock_factory.return_value = engine

            result = runner.invoke(
                main, ["ask", "--stream", "-s", "stream-sess", "hi"],
                catch_exceptions=False,
            )

        assert result.exit_code == 0

        from orchestrator.core.session import SessionStore
        store = SessionStore(db_path=db)
        msgs = store.get("stream-sess")
        store.close()
        assert len(msgs) == 2
        assert msgs[0]["content"] == "hi"
        assert msgs[1]["content"] == "hello world"

    def test_session_disabled_shows_warning(self, runner):
        """When session.enabled=false, -s shows a warning and runs stateless."""
        cfg = _session_cfg(enabled=False)

        with (
            patch("orchestrator.factory.create_engine") as mock_factory,
            patch("orchestrator.cli.main.get_settings", return_value=cfg),
        ):
            engine = MagicMock()
            engine.run.return_value = _fake_result()
            mock_factory.return_value = engine

            result = runner.invoke(main, ["ask", "-s", "new", "hello"], catch_exceptions=False)

        assert result.exit_code == 0
        assert "Sessions disabled" in result.output
        engine.run.assert_called_once_with("hello", history=None, model_override=None)
