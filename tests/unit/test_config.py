import os

import pytest

from src.core import config

# TDD Step 1: Failing test for config loading and validation


def test_settings_loads_env_vars(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SECRET_KEY", "change-me-in-production")
    monkeypatch.setenv("DATABASE_URL", "not-a-postgres-url")
    monkeypatch.setenv("REDIS_URL", "not-a-redis-url")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Re-instantiate settings to pick up monkeypatched env vars
    settings = config.Settings()
    issues = settings.validate_configuration()
    assert "SECRET_KEY must be changed in production" in issues
    assert (
        "At least one API key (Anthropic or OpenAI) should be set in production"
        in issues
    )
    assert "DATABASE_URL must be a PostgreSQL URL" in issues
    assert "REDIS_URL must be a Redis URL" in issues


def test_settings_defaults():
    settings = config.Settings()
    assert settings.environment == config.EnvironmentType.DEVELOPMENT
    assert settings.api_port == 9001
    # Accept either os.getcwd() or '/app' (containerized env)
    assert settings.project_root in [os.getcwd(), "/app", "/tmp"]


def test_get_api_config():
    settings = config.Settings()
    api_cfg = config.get_api_config()
    assert api_cfg["port"] == settings.api_port
    assert api_cfg["host"] == settings.api_host
    assert api_cfg["workers"] == settings.api_workers
    assert api_cfg["debug"] == settings.debug
    assert api_cfg["cors_origins"] == settings.cors_origins


def test_validate_environment_raises(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SECRET_KEY", "change-me-in-production")
    monkeypatch.setenv("DATABASE_URL", "not-a-postgres-url")
    monkeypatch.setenv("REDIS_URL", "not-a-redis-url")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    # Re-instantiate settings to pick up monkeypatched env vars
    config.settings = config.Settings()
    with pytest.raises(ValueError) as exc:
        config.validate_environment()
    assert "Configuration validation failed" in str(exc.value)
    assert "SECRET_KEY must be changed in production" in str(exc.value)
    assert "At least one API key" in str(exc.value)
    assert "DATABASE_URL must be a PostgreSQL URL" in str(exc.value)
    assert "REDIS_URL must be a Redis URL" in str(exc.value)
