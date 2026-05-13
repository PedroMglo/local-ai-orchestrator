"""Tests for config loading."""

import os

from orchestrator.config import Settings, _reset_settings, load_settings


class TestConfig:
    def setup_method(self):
        _reset_settings()

    def test_loads_defaults(self):
        settings = load_settings()
        assert isinstance(settings, Settings)
        assert settings.orchestrator.port == 8585

    def test_env_override(self):
        os.environ["ORC_ORCHESTRATOR_PORT"] = "9999"
        try:
            settings = load_settings()
            assert settings.orchestrator.port == 9999
        finally:
            del os.environ["ORC_ORCHESTRATOR_PORT"]

    def test_models_config(self):
        settings = load_settings()
        assert settings.models.default
        assert settings.models.fast
        assert settings.models.code
        assert settings.models.deep

    def test_security_allowed_commands(self):
        settings = load_settings()
        assert "free" in settings.security.allowed_commands
        assert "rm" not in settings.security.allowed_commands

    def test_repos_paths_tuple(self):
        settings = load_settings()
        assert isinstance(settings.repos.paths, tuple)
