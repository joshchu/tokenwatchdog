"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tokenwatchdog.config import load_config
from tokenwatchdog.store import Store


@pytest.fixture
def cfg(tmp_path):
    return load_config(tmp_path / "config.toml")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "history.db")
    yield s
    s.close()
