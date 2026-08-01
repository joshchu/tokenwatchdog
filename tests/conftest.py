"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tokenwatchdog.config import load_config
from tokenwatchdog.providers import claude_cli, codex_app_server
from tokenwatchdog.store import Store


@pytest.fixture(autouse=True)
def _no_real_cli_spawns(monkeypatch):
    """No test may shell out to a real `claude` or `codex` binary — it's
    slow and every spawn hits the vendor's rate-limited usage surface.
    Tests that exercise these sources inject their own fake callables."""
    monkeypatch.setattr(claude_cli, "_spawn_claude_usage", lambda cfg: None)
    monkeypatch.setattr(codex_app_server, "_spawn_and_query", lambda cfg: None)


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path / "config.toml")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "history.db")
    yield s
    s.close()
