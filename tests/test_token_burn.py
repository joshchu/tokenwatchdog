"""Token-velocity burn, deterministic 5h resets, and the consequences.

These cover the chain that was broken end to end: a whole-number percentage
can't resolve a slow window, so burn read 0.00%/h and no ETA was produced;
and with no observed reset there was no `resets_at`, so `exhausts_before_
reset` was False by construction and the burn alert could never fire.
"""

from __future__ import annotations

import dataclasses

import pytest

from tokenwatchdog.alerts import evaluate
from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.predictor import LinearPredictor, _percent_per_token
from tokenwatchdog.store import SampleRow, TokenEventRow

HOUR = 3600.0


def _sample(source_ts, used_percent, resets_at=None, is_estimated=False):
    return SampleRow(
        captured_at=source_ts,
        source_ts=source_ts,
        used_percent=used_percent,
        resets_at=resets_at,
        is_estimated=is_estimated,
    )


def _event(ts, total):
    return TokenEventRow(
        ts=ts,
        model="claude-fable-5",
        input_tokens=total,
        output_tokens=0,
        cache_creation=0,
        cache_read=0,
    )


def _window(kind, used_percent, source_ts, *, resets_at=None, provider=Provider.CLAUDE):
    return Window(
        provider=provider,
        kind=kind,
        used_percent=used_percent,
        window_minutes=300 if kind is WindowKind.W5H else 10080,
        resets_at=resets_at,
        source_ts=source_ts,
        is_estimated=False,
        source_file="test",
    )


def _quantized_weekly_history(now, *, hours, rate_per_hour, poll_seconds=300.0):
    """What the store actually holds for a weekly window: a percentage
    reported in WHOLE numbers, polled every few minutes."""
    samples = []
    ts = now - hours * HOUR
    while ts <= now:
        elapsed_h = (ts - (now - hours * HOUR)) / HOUR
        samples.append(_sample(ts, float(int(elapsed_h * rate_per_hour))))
        ts += poll_seconds
    return samples


def test_a_one_hour_lookback_cannot_resolve_a_quantized_weekly_burn(cfg):
    """The measured failure that moved the default. A real 0.6%/h weekly
    burn advances the integer percentage by 0.6 of a step per hour, so a
    60-minute lookback sees a change of 0 or 1 and nothing in between; the
    honest slope of an unchanging series is 0.00, and no ETA follows."""
    now = 2_000_000.0
    narrow = dataclasses.replace(
        cfg, burn=dataclasses.replace(cfg.burn, lookback_weekly_minutes=60.0)
    )
    history = _quantized_weekly_history(now, hours=48, rate_per_hour=0.6)
    window = _window(WindowKind.WEEKLY, history[-1].used_percent, now)

    forecast = LinearPredictor().forecast(window, history, [], narrow, now)

    assert forecast.burn_basis == "percent"
    assert forecast.burn_per_hour == pytest.approx(0.0)
    assert forecast.eta_calendar is None


def test_the_wider_weekly_lookback_resolves_it_coarsely(cfg):
    """Same series at the shipped default: hours of span cover several
    integer steps, so a usable — if coarse — rate comes back. This is what
    the widened lookback buys on its own, before any token data."""
    now = 2_000_000.0
    history = _quantized_weekly_history(now, hours=48, rate_per_hour=0.6)
    window = _window(WindowKind.WEEKLY, history[-1].used_percent, now)

    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.burn_basis == "percent"
    assert forecast.burn_per_hour == pytest.approx(0.6, rel=0.25)
    assert forecast.eta_calendar is not None


def test_token_throughput_is_more_accurate_than_the_quantized_slope(cfg):
    """Same series, now with the token log that produced it. Calibrating
    percent-per-token across the block converts throughput into %/h, and
    lands closer to the true rate than the integer slope does."""
    now = 2_000_000.0
    rate = 0.6
    history = _quantized_weekly_history(now, hours=48, rate_per_hour=rate)
    events = [_event(now - h * HOUR, 100_000) for h in range(48, 0, -1)]
    window = _window(WindowKind.WEEKLY, history[-1].used_percent, now)

    with_tokens = LinearPredictor().forecast(window, history, events, cfg, now)
    without = LinearPredictor().forecast(window, history, [], cfg, now)

    assert with_tokens.burn_basis == "tokens"
    assert with_tokens.burn_per_hour == pytest.approx(rate, rel=0.1)
    assert abs(with_tokens.burn_per_hour - rate) < abs(without.burn_per_hour - rate)

    # And it's a real projection, not a coincidence: remaining/rate hours out.
    expected_h = (100.0 - window.used_percent) / rate
    actual_h = (with_tokens.eta_calendar.timestamp() - now) / HOUR
    assert actual_h == pytest.approx(expected_h, rel=0.2)


def test_tokens_catch_a_fresh_burst_the_integer_percentage_has_not_ticked(cfg):
    """The case the percentage simply cannot report: a heavy job started an
    hour ago, spending real tokens, but the whole-number percentage hasn't
    rolled over to its next value yet. Its slope is flat — genuinely zero
    information — while throughput already knows."""
    now = 2_000_000.0
    history = [
        _sample(now - 48 * HOUR, 40.0),
        _sample(now - 24 * HOUR, 50.0),
        _sample(now - 8 * HOUR, 60.0),
        _sample(now - 4 * HOUR, 60.0),  # plateau across the whole lookback
        _sample(now, 60.0),
    ]
    quiet_then_busy = [_event(now - 48 * HOUR, 1_000_000)] + [
        _event(now - 0.5 * HOUR, 400_000)
    ]
    window = _window(WindowKind.WEEKLY, 60.0, now)

    flat = LinearPredictor().forecast(window, history, [], cfg, now)
    assert flat.burn_per_hour == pytest.approx(0.0)
    assert flat.eta_calendar is None

    live = LinearPredictor().forecast(window, history, quiet_then_busy, cfg, now)
    assert live.burn_basis == "tokens"
    assert live.burn_per_hour > 0
    assert live.eta_calendar is not None


def test_calibration_needs_real_percentage_movement(cfg):
    """One integer step of movement carries ±50% error, so it is refused
    rather than used to scale every future token into a confident rate."""
    now = 2_000_000.0
    events = [_event(now - h * HOUR, 100_000) for h in range(10, 0, -1)]
    barely_moved = [_sample(now - 10 * HOUR, 40.0), _sample(now, 41.0)]
    assert _percent_per_token(barely_moved, events) is None

    moved = [_sample(now - 10 * HOUR, 40.0), _sample(now, 50.0)]
    assert _percent_per_token(moved, events) == pytest.approx(10.0 / 1_000_000)


def test_saturated_block_is_not_calibrated_from(cfg):
    """A percentage clipped at 100 understates how much was really spent, so
    the ratio derived from it would be wrong. tokens_burned_past_quota is
    what covers the saturated case."""
    now = 2_000_000.0
    events = [_event(now - h * HOUR, 100_000) for h in range(10, 0, -1)]
    saturated = [_sample(now - 10 * HOUR, 80.0), _sample(now, 100.0)]
    assert _percent_per_token(saturated, events) is None


def test_percent_slope_wins_when_the_token_log_cannot_see_the_usage(cfg):
    """An account-wide percentage also counts usage that never reaches this
    machine's token log (Claude Desktop, agent mode). If the percentage is
    climbing while the log is silent, a confident zero would be worse than a
    coarse slope."""
    now = 2_000_000.0
    # Calibratable history, but no token events in the recent lookback.
    history = [
        _sample(now - 10 * HOUR, 40.0),
        _sample(now - 9 * HOUR, 50.0),
        _sample(now - HOUR, 60.0),
        _sample(now, 62.0),
    ]
    stale_events = [_event(now - h * HOUR, 500_000) for h in range(10, 7, -1)]
    window = _window(WindowKind.WEEKLY, 62.0, now)

    forecast = LinearPredictor().forecast(window, history, stale_events, cfg, now)

    assert forecast.burn_basis == "percent"
    assert forecast.burn_per_hour > 0


def test_5h_reset_is_derived_from_the_token_log_with_no_observed_reset(cfg):
    """Claude never reports resets_at, and before this the only way to learn
    one was to catch a percentage drop between two polls. The 5-hour block
    anchor makes it computable from activity alone."""
    now = 2_000_000.0
    anchor = now - 2 * HOUR
    events = [_event(anchor + i * 1800.0, 10_000) for i in range(5)]
    history = [_sample(now - HOUR, 30.0), _sample(now, 40.0)]
    window = _window(WindowKind.W5H, 40.0, now)
    assert window.resets_at is None

    forecast = LinearPredictor().forecast(window, history, events, cfg, now)

    assert forecast.window.resets_at == pytest.approx(anchor + 5 * HOUR)
    assert forecast.time_to_reset_h == pytest.approx(3.0, abs=0.01)


def test_burn_alert_becomes_reachable_once_a_reset_is_known(cfg, store):
    """The downstream consequence. `exhausts_before_reset` is False whenever
    time_to_reset_h is None, and the burn alert requires it — so with no
    derivable reset the alert was unreachable for Claude no matter how fast
    quota was burning."""
    now = 2_000_000.0
    anchor = now - HOUR
    events = [_event(anchor + i * 600.0, 200_000) for i in range(6)]
    # Climbing hard: 60% -> 90% over the last hour.
    history = [
        _sample(anchor, 60.0),
        _sample(anchor + 1800.0, 75.0),
        _sample(now, 90.0),
    ]
    window = _window(WindowKind.W5H, 90.0, now)

    forecast = LinearPredictor().forecast(window, history, events, cfg, now)
    assert forecast.window.resets_at is not None
    assert forecast.exhausts_before_reset is True

    fired = evaluate(forecast, cfg, store, now)
    assert any(a.alert_kind == "burn" for a in fired)

    # Same forecast with the reset unknown: structurally silent.
    without_reset = dataclasses.replace(
        forecast,
        window=dataclasses.replace(forecast.window, resets_at=None),
        time_to_reset_h=None,
        exhausts_before_reset=False,
    )
    assert all(a.alert_kind != "burn" for a in evaluate(without_reset, cfg, store, now))
