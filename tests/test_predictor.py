"""Predictor tests: idle/reset gates, steady-burn ETA, 100%-used, W5H
cap-at-reset, resets_at derivation, and working-hours math.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.predictor import (
    LinearPredictor,
    _is_reset,
    project_workhours_exhaustion,
)
from tokenwatchdog.store import SampleRow

UTC = ZoneInfo("UTC")


def _sample(source_ts, used_percent, resets_at=None, is_estimated=False):
    return SampleRow(
        captured_at=source_ts,
        source_ts=source_ts,
        used_percent=used_percent,
        resets_at=resets_at,
        is_estimated=is_estimated,
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


def test_flat_burn_reports_no_confidence_not_a_number_for_a_missing_eta(cfg):
    """Regression: confidence rates the burn-rate estimate, not whether an
    ETA is shown. With zero burn there's no exhaustion trajectory at all —
    showing e.g. "medium" next to a blank ETA read as confidence in a
    forecast that doesn't exist."""
    now = 100_000.0
    history = [_sample(now - 3600, 40.0), _sample(now - 1800, 40.0), _sample(now, 40.0)]
    window = _window(Provider.CODEX, WindowKind.WEEKLY, 40.0, now)
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.burn_per_hour == pytest.approx(0.0)
    assert forecast.eta_calendar is None
    assert forecast.confidence is None


def test_stale_sample_is_idle(cfg):
    now = 100_000.0
    stale_source_ts = now - cfg.thresholds.stale_after_minutes * 60 - 60
    window = _window(Provider.CLAUDE, WindowKind.W5H, 50.0, stale_source_ts)
    forecast = LinearPredictor().forecast(window, [], [], cfg, now)
    assert forecast.status == "IDLE"
    assert forecast.confidence is None
    assert forecast.eta_calendar is None


def test_negative_delta_is_reset_pending(cfg):
    now = 100_000.0
    history = [_sample(now - 120, 95.0), _sample(now, 5.0)]
    window = _window(Provider.CODEX, WindowKind.WEEKLY, 5.0, now)
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "RESET_PENDING"
    assert forecast.burn_per_hour == 0.0


def test_large_drop_on_authoritative_source_is_always_a_reset(cfg):
    # Codex/Desktop percentages are authoritative -- any real drop counts.
    prev = _sample(0.0, 60.0, is_estimated=False)
    curr = _sample(60.0, 40.0, is_estimated=False)  # drop of 20, lands at 40
    assert _is_reset(prev, curr) is True


def test_large_drop_on_estimated_source_needs_to_land_near_zero(cfg):
    """Regression: Claude's token-compute percentage is a trailing
    rolling-window sum, so a single old burst aging out can produce a
    large drop that isn't a real reset. A magnitude threshold alone
    false-positives on that; requiring the landing point to also be near
    zero (what a true reset always looks like) cuts most of those."""
    prev = _sample(0.0, 60.0, is_estimated=True)
    curr = _sample(60.0, 40.0, is_estimated=True)  # big drop, but lands at 40
    assert _is_reset(prev, curr) is False

    curr_near_zero = _sample(60.0, 3.0, is_estimated=True)  # big drop, lands near 0
    assert _is_reset(prev, curr_near_zero) is True


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
    """Regression: a linear projection that would only reach 100% well
    after a known, imminent reset must not be reported as an ETA at all —
    the window refills before usage ever gets there, so there's nothing
    to warn about. Reported previously as: the 5h ETA didn't account for
    an imminent reset, showing a nonsensical far-future "exhaustion" time
    for a window that was about to refresh in an hour."""
    now = 100_000.0
    resets_at = now + 1 * 3600  # resets in 1h
    history = [_sample(now - 900, 10.0, resets_at), _sample(now, 11.0, resets_at)]
    window = _window(Provider.CLAUDE, WindowKind.W5H, 11.0, now, resets_at)
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "OK"
    assert forecast.exhausts_before_reset is False
    assert forecast.eta_calendar is None
    assert forecast.eta_workhours is None
    assert forecast.time_to_reset_h == pytest.approx(1.0, rel=0.01)
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


def test_weekly_resets_at_derived_from_observed_reset_not_a_calendar_guess(cfg):
    """Regression: the weekly reset time must come from an actually
    observed reset (block_started_at + 7 days), never a hardcoded
    calendar assumption like "next Monday" — Anthropic doesn't publish
    the true anchor, and guessing one would violate "never fabricate.\""""
    now = 100_000.0
    block_start = now - 3 * 24 * 3600  # a reset was observed 3 days ago
    history = [
        _sample(block_start - 60, 98.0, None),
        _sample(block_start, 1.0, None),  # reset happened here
        _sample(now, 20.0, None),
    ]
    window = _window(
        Provider.CLAUDE, WindowKind.WEEKLY, 20.0, now, resets_at=None, is_estimated=True
    )
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    expected_resets_at = block_start + 7 * 24 * 3600
    assert forecast.window.resets_at == pytest.approx(expected_resets_at, rel=0.01)
    assert forecast.time_to_reset_h == pytest.approx(
        (expected_resets_at - now) / 3600, rel=0.01
    )


def test_stale_derived_reset_is_advanced_to_a_future_cycle_not_left_in_the_past(cfg):
    """Regression: if the observed reset is more than a full cycle old —
    e.g. a lightly used window whose real reset produced a drop too small
    for `_is_reset` to catch — the derived resets_at must not come out in
    the past. It should advance by whole cycles instead, since the same
    fixed-cycle assumption already being made implies the window kept
    resetting on schedule even though we didn't see the boundary."""
    now = 100_000.0
    block_start = now - 13 * 3600  # W5H (5h) cycle: 2 whole cycles have elapsed
    history = [
        _sample(block_start - 60, 98.0, None),
        _sample(block_start, 1.0, None),  # the one reset we DID observe
        _sample(now, 30.0, None),
    ]
    window = _window(
        Provider.CLAUDE, WindowKind.W5H, 30.0, now, resets_at=None, is_estimated=True
    )
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.window.resets_at is not None
    assert forecast.window.resets_at > now
    # Anchored to the same block_start + a whole number of 5h cycles.
    assert (forecast.window.resets_at - block_start) % (5 * 3600) == pytest.approx(
        0.0, abs=0.01
    )


def test_weekly_resets_at_unknown_when_no_reset_ever_observed(cfg):
    now = 100_000.0
    history = [_sample(now - 60, 18.0, None), _sample(now, 20.0, None)]
    window = _window(
        Provider.CLAUDE, WindowKind.WEEKLY, 20.0, now, resets_at=None, is_estimated=True
    )
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    assert forecast.window.resets_at is None
    assert forecast.time_to_reset_h is None


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
