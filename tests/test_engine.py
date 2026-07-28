"""Engine wiring smoke tests — fake providers, real store/predictor/alerts.

This exercises the tick() pipeline (read -> store -> predict -> alert)
end-to-end without touching real Codex/Claude data on disk.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from tokenwatchdog.config import load_config
from tokenwatchdog.engine import Engine
from tokenwatchdog.models import Forecast, Provider, Window, WindowKind
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
    # Regression: NO_DATA has no ETA either -- a "low" confidence sitting
    # next to a blank ETA read as confidence in a forecast that was never
    # made at all.
    assert all(f.confidence is None for f in state.forecasts)


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


def test_tick_surfaces_tokens_burned_past_quota_once_saturated(tmp_path):
    """End-to-end: a provider that has ingested real token_events (Codex)
    gets tokens_burned_past_quota threaded onto its Forecast once
    used_percent pins at 100 -- proves the engine.py wiring, not just the
    pure predictor function in isolation."""
    window = Window(
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        used_percent=100.0,
        window_minutes=10080,
        resets_at=2000.0,
        source_ts=999.0,
        is_estimated=False,
        source_file="fake",
    )
    engine = _make_engine(tmp_path, windows=[window], watch=("weekly",))
    engine.store.upsert_token_event(
        provider=Provider.CODEX,
        request_id="session-1",
        message_id="2026-07-24T00:00:00.000Z",
        ts=999.0,
        model="codex",
        input_tokens=500,
        output_tokens=100,
        cache_creation=0,
        cache_read=0,
    )
    state = engine.tick(now=1000.0)

    forecast = next(
        f
        for f in state.forecasts
        if f.window.kind is WindowKind.WEEKLY and f.window.provider is Provider.CODEX
    )
    assert forecast.tokens_burned_past_quota == 600


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


def _saved_prediction(window: Window, *, status: str = "OK") -> Forecast:
    eta_p50 = datetime.fromtimestamp(21_000.0, tz=timezone.utc)
    return Forecast(
        window=window,
        status=status,
        model_name="montecarlo",
        burn_per_hour=1.0,
        time_to_reset_h=None,
        eta_calendar=eta_p50 if status == "OK" else None,
        eta_workhours=None,
        eta_p50=eta_p50 if status == "OK" else None,
        eta_p90=(
            datetime.fromtimestamp(22_000.0, tz=timezone.utc)
            if status == "OK"
            else None
        ),
        prob_exhaust_before_reset=None,
        confidence="medium",
        exhausts_before_reset=False,
        n_samples=5,
    )


def test_idle_tick_recovers_saved_prediction_after_restart(tmp_path):
    """The fallback comes from SQLite, not process memory, and repeated idle
    ticks do not bury it under newer blank IDLE rows."""
    db_path = tmp_path / "history.db"
    current_window = Window(
        provider=Provider.CLAUDE,
        kind=WindowKind.W5H,
        used_percent=73.0,
        window_minutes=300,
        resets_at=None,
        source_ts=19_000.0,
        is_estimated=False,
        source_file="fake",
    )
    store = Store(db_path)
    store.insert_forecast(made_at=19_500.0, forecast=_saved_prediction(current_window))
    store.close()

    config_path = tmp_path / "config.toml"
    config_path.write_text('[windows]\nwatch = ["w5h"]\n')
    engine = Engine(
        cfg=load_config(config_path),
        store=Store(db_path),
        providers=[_FakeProvider([current_window])],
    )

    first = engine.tick(now=20_000.0)
    current_mc = first.forecast_from("montecarlo", current_window)
    assert current_mc is not None
    assert current_mc.status == "IDLE"
    assert current_mc.eta_p50 is None
    retained = first.retained_prediction_for(current_window)
    assert retained is not None
    assert retained.used_percent == 73.0
    assert retained.eta_p50.timestamp() == 21_000.0
    assert retained.eta_p90 is not None
    assert retained.eta_p90.timestamp() == 22_000.0

    second = engine.tick(now=20_060.0)
    assert second.retained_prediction_for(current_window) == retained


def test_newer_reset_pending_result_blocks_an_old_saved_prediction(tmp_path):
    db_path = tmp_path / "history.db"
    current_window = Window(
        provider=Provider.CLAUDE,
        kind=WindowKind.W5H,
        used_percent=73.0,
        window_minutes=300,
        resets_at=None,
        source_ts=19_000.0,
        is_estimated=False,
        source_file="fake",
    )
    store = Store(db_path)
    saved = _saved_prediction(current_window)
    store.insert_forecast(made_at=19_400.0, forecast=saved)
    store.insert_forecast(
        made_at=19_500.0,
        forecast=dataclasses.replace(
            saved,
            status="RESET_PENDING",
            eta_calendar=None,
            eta_p50=None,
            eta_p90=None,
        ),
    )

    config_path = tmp_path / "config.toml"
    config_path.write_text('[windows]\nwatch = ["w5h"]\n')
    engine = Engine(
        cfg=load_config(config_path),
        store=store,
        providers=[_FakeProvider([current_window])],
    )

    state = engine.tick(now=20_000.0)
    assert state.retained_prediction_for(current_window) is None


def test_saved_prediction_requires_the_same_usage_level_and_a_future_p50(tmp_path):
    db_path = tmp_path / "history.db"
    saved_window = Window(
        provider=Provider.CLAUDE,
        kind=WindowKind.W5H,
        used_percent=72.0,
        window_minutes=300,
        resets_at=None,
        source_ts=19_000.0,
        is_estimated=False,
        source_file="fake",
    )
    current_window = dataclasses.replace(saved_window, used_percent=73.0)
    store = Store(db_path)
    store.insert_forecast(made_at=19_500.0, forecast=_saved_prediction(saved_window))

    config_path = tmp_path / "config.toml"
    config_path.write_text('[windows]\nwatch = ["w5h"]\n')
    engine = Engine(
        cfg=load_config(config_path),
        store=store,
        providers=[_FakeProvider([current_window])],
    )

    state = engine.tick(now=20_000.0)
    assert state.retained_prediction_for(current_window) is None

    matching_window = dataclasses.replace(current_window, used_percent=72.0)
    engine._providers = [_FakeProvider([matching_window])]
    after_eta = engine.tick(now=21_001.0)
    assert after_eta.retained_prediction_for(matching_window) is None


def test_current_censored_idle_result_beats_an_older_saved_band(tmp_path):
    """A blank P50 from a simulation that actually ran means the median
    future survives. It must not be replaced with yesterday's rosier band."""
    window = Window(
        provider=Provider.CLAUDE,
        kind=WindowKind.W5H,
        used_percent=73.0,
        window_minutes=300,
        resets_at=None,
        source_ts=19_000.0,
        is_estimated=False,
        source_file="fake",
    )
    engine = _make_engine(tmp_path, windows=[window], watch=("w5h",))
    engine.store.insert_forecast(made_at=19_500.0, forecast=_saved_prediction(window))
    current = dataclasses.replace(
        _saved_prediction(window),
        status="IDLE",
        eta_calendar=None,
        eta_p50=None,
        eta_p90=None,
        confidence="medium",
        prob_exhaust_before_reset=0.3,
    )

    assert engine._retained_prediction_for(current, now=20_000.0) is None
