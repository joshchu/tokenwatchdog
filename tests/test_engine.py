"""Engine wiring smoke tests — fake providers, real store/predictor/alerts.

This exercises the tick() pipeline (read -> store -> predict -> alert)
end-to-end without touching real Codex/Claude data on disk.
"""

from __future__ import annotations

from tokenwatchdog.config import load_config
from tokenwatchdog.engine import Engine
from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.store import Store


class _FakeProvider:
    name = "fake"

    def __init__(self, windows):
        self._windows = windows

    def read(self, cfg, store):
        return self._windows


def _make_engine(tmp_path, windows, watch=("weekly", "w5h")):
    config_path = tmp_path / "config.toml"
    watch_toml = ", ".join(f'"{w}"' for w in watch)
    config_path.write_text(f"[windows]\nwatch = [{watch_toml}]\n")
    cfg = load_config(config_path)
    return Engine(
        cfg=cfg,
        store=Store(tmp_path / "history.db"),
        providers=[_FakeProvider(windows)],
    )


def test_tick_returns_no_data_when_no_provider_reports_a_window(tmp_path):
    engine = _make_engine(tmp_path, windows=[])
    state = engine.tick(now=1000.0)
    assert {f.status for f in state.forecasts} == {"NO_DATA"}
    assert state.alerts == ()


def test_tick_persists_samples_and_forecasts(tmp_path):
    window = Window(
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        used_percent=50.0,
        window_minutes=10080,
        resets_at=2000.0,
        source_ts=999.0,
        is_estimated=False,
        source_file="fake",
    )
    engine = _make_engine(tmp_path, windows=[window], watch=("weekly",))
    state = engine.tick(now=1000.0)
    assert len(state.windows) == 1

    forecast = next(
        f
        for f in state.forecasts
        if f.window.kind is WindowKind.WEEKLY and f.window.provider is Provider.CODEX
    )
    assert forecast.status == "OK"

    rows = engine.store.recent_samples(Provider.CODEX, WindowKind.WEEKLY, since_ts=0.0)
    assert len(rows) == 1


def test_tick_fires_threshold_alert_once(tmp_path):
    window = Window(
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        used_percent=95.0,
        window_minutes=10080,
        resets_at=2000.0,
        source_ts=999.0,
        is_estimated=False,
        source_file="fake",
    )
    engine = _make_engine(tmp_path, windows=[window], watch=("weekly",))
    first = engine.tick(now=1000.0)
    assert len(first.alerts) == 1

    second = engine.tick(now=1010.0)
    assert len(second.alerts) == 0
