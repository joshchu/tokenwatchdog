"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tokenwatchdog.config import load_config
from tokenwatchdog.providers import claude_cli
from tokenwatchdog.store import Store


@pytest.fixture(autouse=True)
def _no_real_claude_spawn(monkeypatch):
    """No test may shell out to a real `claude` binary — it's slow and every
    spawn hits Anthropic's rate-limited usage endpoint. Tests that exercise
    the CLI source inject their own fake spawn callable instead."""
    monkeypatch.setattr(claude_cli, "_spawn_claude_usage", lambda cfg: None)


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path / "config.toml")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "history.db")
    yield s
    s.close()
