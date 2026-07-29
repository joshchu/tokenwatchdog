"""Codex's weekly window rolls — old usage ages out continuously — and a
rolling window has no reset events, so a drop alone proves nothing.

Measured over 10.5 days of real history before this existed: 2 of 3 drops
flagged as "resets" were roll-off (17→0 in 12 minutes when a week-old burst
aged out; 14→1 across a 52.7-hour idle gap), and each falsely split a cycle —
corrupting block boundaries, truth scans, and risk grading.
"""

from __future__ import annotations

from tokenwatchdog.models import Provider, Window, WindowKind, window_is_rolling
from tokenwatchdog.predictor import (
    LinearPredictor,
    _is_reset,
    _is_rolloff_clear,
    _reset_predicate,
    _split_into_blocks,
)
from tokenwatchdog.scoring import realized_exhaustion_hours
from tokenwatchdog.store import SampleRow

HOUR = 3600.0


def _sample(source_ts, used_percent, resets_at=None):
    return SampleRow(
        captured_at=source_ts,
        source_ts=source_ts,
        used_percent=used_percent,
        resets_at=resets_at,
        is_estimated=False,
    )


def _codex_weekly(used_percent, source_ts, resets_at=None):
    return Window(
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        used_percent=used_percent,
        window_minutes=10080,
        resets_at=resets_at,
        source_ts=source_ts,
        is_estimated=False,
        source_file="test",
    )


def test_only_codex_weekly_rolls():
    assert window_is_rolling(Provider.CODEX, WindowKind.WEEKLY)
    assert not window_is_rolling(Provider.CODEX, WindowKind.W5H)
    assert not window_is_rolling(Provider.CLAUDE, WindowKind.WEEKLY)
    assert not window_is_rolling(Provider.CLAUDE, WindowKind.W5H)


def test_reset_predicate_selects_by_window_shape():
    assert _reset_predicate(Provider.CODEX, WindowKind.WEEKLY) is _is_rolloff_clear
    assert _reset_predicate(Provider.CODEX, WindowKind.W5H) is _is_reset
    assert _reset_predicate(Provider.CLAUDE, WindowKind.WEEKLY) is _is_reset
    assert _reset_predicate(Provider.CLAUDE, WindowKind.W5H) is _is_reset


def test_a_rolloff_drop_before_the_declared_clear_is_not_a_boundary():
    """The measured false positive: 17→0 in 12 minutes, a week-old burst
    aging out, while the provider's own clear-time was still days away."""
    now = 1_000_000.0
    clears_at = now + 3 * 24 * HOUR
    prev = _sample(now, 17.0, resets_at=clears_at)
    curr = _sample(now + 720.0, 0.0, resets_at=clears_at)

    assert _is_rolloff_clear(prev, curr) is False
    assert _is_reset(prev, curr) is True  # what the old predicate wrongly said


def test_a_drop_past_the_declared_clear_landing_near_zero_is_a_boundary():
    """The one genuine clear in the same history: 100→3, minutes after the
    provider-declared clear-time passed."""
    now = 1_000_000.0
    cleared_at = now - 240.0  # declared clear-time just passed
    prev = _sample(now - 900.0, 100.0, resets_at=cleared_at)
    curr = _sample(now, 3.0, resets_at=cleared_at)

    assert _is_rolloff_clear(prev, curr) is True


def test_a_drop_landing_high_is_never_a_boundary():
    """Even past the declared clear, a window that still reads 40% did not
    clear — partial roll-off, still the same rolling accumulation."""
    now = 1_000_000.0
    prev = _sample(now - 900.0, 100.0, resets_at=now - 240.0)
    curr = _sample(now, 40.0, resets_at=now - 240.0)

    assert _is_rolloff_clear(prev, curr) is False


def test_resets_at_advancing_is_aging_not_a_boundary():
    """On a rolling window the provider's resets_at slides forward as usage
    ages out — _is_reset treats that advance as a reset, _is_rolloff_clear
    deliberately does not."""
    now = 1_000_000.0
    prev = _sample(now, 30.0, resets_at=now + 24 * HOUR)
    curr = _sample(now + 300.0, 30.0, resets_at=now + 30 * HOUR)

    assert _is_reset(prev, curr) is True
    assert _is_rolloff_clear(prev, curr) is False


def test_week_old_burst_aging_out_no_longer_splits_the_cycle():
    now = 1_000_000.0
    clears_at = now + 3 * 24 * HOUR
    history = [
        _sample(now - 2 * HOUR, 17.0, clears_at),
        _sample(now - HOUR, 17.0, clears_at),
        _sample(now - HOUR + 720.0, 0.0, clears_at),  # roll-off, not a clear
        _sample(now, 1.0, clears_at),
    ]
    blocks = _split_into_blocks(
        history, _reset_predicate(Provider.CODEX, WindowKind.WEEKLY)
    )
    assert len(blocks) == 1


def test_gentle_rolloff_decline_is_ok_with_no_eta_not_an_error(cfg):
    """A rolling window drifting down is a calm fact: negative burn, no
    exhaustion trajectory, nothing pending. _robust_slope_per_hour has no
    positivity assumption and _project_forecast maps burn <= 0 to a quiet
    OK row."""
    now = 1_000_000.0
    clears_at = now + 2 * 24 * HOUR
    history = [
        _sample(now - 3 * HOUR, 30.0, clears_at),
        _sample(now - 2 * HOUR, 28.0, clears_at),
        _sample(now - HOUR, 26.0, clears_at),
        _sample(now, 24.0, clears_at),
    ]
    window = _codex_weekly(24.0, now, clears_at)

    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "OK"
    assert forecast.burn_per_hour <= 0.0
    assert forecast.eta_calendar is None
    assert forecast.exhausts_before_reset is False


def test_truth_scan_is_bounded_by_the_origins_own_cycle_end():
    """An exhaustion after the boundary the forecast was scoped to belongs
    to a different cycle even if no drop was caught in the act — the only
    honest scoping on a rolling window, where drops can't be trusted to
    mark boundaries."""
    now = 1_000_000.0
    cycle_end = now + HOUR
    is_reset = _reset_predicate(Provider.CODEX, WindowKind.WEEKLY)

    beyond = [
        _sample(now, 50.0, cycle_end),
        _sample(now + 2 * HOUR, 100.0, cycle_end),  # past the declared clear
    ]
    assert realized_exhaustion_hours(beyond, 0, is_reset) is None

    within = [
        _sample(now, 50.0, cycle_end),
        _sample(now + 0.5 * HOUR, 100.0, cycle_end),
    ]
    assert realized_exhaustion_hours(within, 0, is_reset) == 0.5
