"""Predictor tests: idle/reset gates, steady-burn ETA, 100%-used, W5H
cap-at-reset, resets_at derivation, and working-hours math.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.predictor import LinearPredictor, project_workhours_exhaustion
from tokenwatchdog.store import SampleRow

UTC = ZoneInfo("UTC")


def _sample(source_ts, used_percent, resets_at=None):
    return SampleRow(
        captured_at=source_ts,
        source_ts=source_ts,
        used_percent=used_percent,
        resets_at=resets_at,
        is_estimated=False,
    )


def _window(
    provider, kind, used_percent, source_ts, resets_at=None, is_estimated=False
):
    window_minutes = 300 if kind is WindowKind.W5H else 10080
    return Window(
        provider=provider,
        kind=kind,
        used_percent=used_percent,
        window_minutes=window_minutes,
        resets_at=resets_at,
        source_ts=source_ts,
        is_estimated=is_estimated,
        source_file="test",
    )


def test_steady_weekly_burn_produces_sane_eta_and_exhausts_before_reset(cfg):
    now = 100_000.0
    resets_at = now + 20 * 3600
    history = [
        _sample(now - 3600, 40.0, resets_at),
        _sample(now - 1800, 50.0, resets_at),
        _sample(now, 60.0, resets_at),
    ]
    window = _window(Provider.CODEX, WindowKind.WEEKLY, 60.0, now, resets_at)
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "OK"
    assert forecast.burn_per_hour == pytest.approx(20.0, rel=0.01)
    assert forecast.exhausts_before_reset is True
    assert forecast.eta_calendar is not None
    assert (forecast.eta_calendar.timestamp() - now) == pytest.approx(
        2 * 3600, rel=0.01
    )


def test_stale_sample_is_idle(cfg):
    now = 100_000.0
    stale_source_ts = now - cfg.thresholds.stale_after_minutes * 60 - 60
    window = _window(Provider.CLAUDE, WindowKind.W5H, 50.0, stale_source_ts)
    forecast = LinearPredictor().forecast(window, [], [], cfg, now)
    assert forecast.status == "IDLE"
    assert forecast.eta_calendar is None


def test_negative_delta_is_reset_pending(cfg):
    now = 100_000.0
    history = [_sample(now - 120, 95.0), _sample(now, 5.0)]
    window = _window(Provider.CODEX, WindowKind.WEEKLY, 5.0, now)
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "RESET_PENDING"
    assert forecast.burn_per_hour == 0.0


def test_100_percent_used_exhausts_immediately(cfg):
    now = 100_000.0
    resets_at = now + 5 * 3600
    history = [_sample(now - 60, 99.0, resets_at), _sample(now, 100.0, resets_at)]
    window = _window(Provider.CLAUDE, WindowKind.W5H, 100.0, now, resets_at)
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "OK"
    assert forecast.eta_calendar is not None
    assert (forecast.eta_calendar.timestamp() - now) == pytest.approx(0.0, abs=1.0)


def test_w5h_caps_exhaustion_claim_at_reset_when_burn_is_slow(cfg):
    now = 100_000.0
    resets_at = now + 1 * 3600  # resets in 1h
    history = [_sample(now - 900, 10.0, resets_at), _sample(now, 11.0, resets_at)]
    window = _window(Provider.CLAUDE, WindowKind.W5H, 11.0, now, resets_at)
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "OK"
    assert forecast.exhausts_before_reset is False
    assert forecast.time_to_reset_h == pytest.approx(1.0, rel=0.01)


def test_derives_w5h_resets_at_from_observed_block_start(cfg):
    now = 100_000.0
    block_start = now - 3600
    history = [
        _sample(block_start - 60, 95.0, None),
        _sample(block_start, 2.0, None),  # reset happened here
        _sample(now, 10.0, None),
    ]
    window = _window(
        Provider.CLAUDE, WindowKind.W5H, 10.0, now, resets_at=None, is_estimated=True
    )
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    expected_resets_at = block_start + 5 * 3600
    expected_h = (expected_resets_at - now) / 3600
    assert forecast.time_to_reset_h == pytest.approx(expected_h, rel=0.01)
    # Regression: the derived value must ride along on Forecast.window too
    # (not just inform time_to_reset_h locally) — alerts.py's re-arm logic
    # reads window.resets_at, and it stays None forever for Claude
    # otherwise, capping a burn alert to firing once for the database's
    # entire lifetime.
    assert forecast.window.resets_at == pytest.approx(expected_resets_at, rel=0.01)


def test_unknown_w5h_block_start_leaves_reset_unknown(cfg):
    now = 100_000.0
    history = [_sample(now - 60, 8.0, None), _sample(now, 10.0, None)]
    window = _window(
        Provider.CLAUDE, WindowKind.W5H, 10.0, now, resets_at=None, is_estimated=True
    )
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.time_to_reset_h is None
    assert forecast.exhausts_before_reset is False


def test_weekly_resets_at_derived_as_next_monday_midnight_utc(cfg):
    # Verified: 2026-07-24 is a Friday, 2026-07-27 is the following Monday.
    now_dt = datetime(2026, 7, 24, 15, 0, tzinfo=UTC)
    now = now_dt.timestamp()
    history = [_sample(now - 60, 8.0, None), _sample(now, 10.0, None)]
    window = _window(
        Provider.CLAUDE, WindowKind.WEEKLY, 10.0, now, resets_at=None, is_estimated=True
    )
    cfg_utc = dataclasses.replace(cfg, timezone="UTC")
    forecast = LinearPredictor().forecast(window, history, [], cfg_utc, now)
    expected_monday = datetime(2026, 7, 27, 0, 0, tzinfo=UTC)
    assert forecast.time_to_reset_h == pytest.approx(
        (expected_monday.timestamp() - now) / 3600, rel=0.001
    )


def test_working_hours_projection_skips_the_weekend(cfg):
    # Verified independently: Fri 2026-07-24 15:00 + 20 working-hours (9-5,
    # Mon-Fri) = Fri 2h + Mon 8h + Tue 8h + Wed 2h -> Wed 2026-07-29 11:00.
    start = datetime(2026, 7, 24, 15, 0, tzinfo=UTC)
    result = project_workhours_exhaustion(start, 20.0, cfg.working_hours)
    assert result == datetime(2026, 7, 29, 11, 0, tzinfo=UTC)


def test_working_hours_disabled_matches_plain_24_7_projection(cfg):
    disabled = dataclasses.replace(cfg.working_hours, enabled=False)
    start = datetime(2026, 7, 24, 15, 0, tzinfo=UTC)
    result = project_workhours_exhaustion(start, 5.0, disabled)
    assert result == datetime(2026, 7, 24, 20, 0, tzinfo=UTC)
