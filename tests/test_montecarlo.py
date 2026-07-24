"""Monte Carlo predictor tests: hour-of-week bucketing + simulated
exhaustion bands.

Assertions are structural (status, field presence, valid ranges, ordering)
rather than exact numeric values, since the model is intentionally
stochastic — that's the point of a probabilistic forecaster. `random` is
seeded per-test for reproducible CI runs, not to pin exact outputs.
"""

from __future__ import annotations

import random

import pytest

from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.predictor import MonteCarloPredictor, _percentile_within_horizon
from tokenwatchdog.store import SampleRow


def _sample(source_ts, used_percent, resets_at=None):
    return SampleRow(
        captured_at=source_ts,
        source_ts=source_ts,
        used_percent=used_percent,
        resets_at=resets_at,
        is_estimated=False,
    )


def _window(kind, used_percent, source_ts, resets_at=None):
    window_minutes = 300 if kind is WindowKind.W5H else 10080
    return Window(
        provider=Provider.CODEX,
        kind=kind,
        used_percent=used_percent,
        window_minutes=window_minutes,
        resets_at=resets_at,
        source_ts=source_ts,
        is_estimated=False,
        source_file="test",
    )


@pytest.fixture(autouse=True)
def _seeded_random():
    random.seed(20260724)


def test_produces_ordered_p50_p90_band_with_enough_history(cfg):
    now = 1_000_000.0
    # Reset far enough out that a ~2%/h burn from the last block's used%
    # (well under 50%) genuinely exhausts before it -- simulations are
    # capped AT the reset, so a too-close resets_at would correctly (but
    # unhelpfully for this test) report "no ETA, doesn't exhaust in time."
    resets_at = now + 60 * 3600
    history = []
    ts = now - 5 * 24 * 3600
    percent = 0.0
    while ts <= now:
        history.append(_sample(ts, percent, resets_at if ts == now else None))
        percent += 2.0
        if percent >= 95.0:
            percent = 0.0  # a reset every ~48h of climbing
        ts += 3600
    # The window's current reading must match the history's own last point —
    # any other value risks looking like an (unintended) additional reset.
    window = _window(WindowKind.WEEKLY, history[-1].used_percent, now, resets_at)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "OK"
    assert forecast.model_name == "montecarlo"
    assert forecast.eta_p50 is not None
    assert forecast.eta_p90 is not None
    assert forecast.eta_p50 <= forecast.eta_p90
    assert forecast.prob_exhaust_before_reset is not None
    assert 0.0 <= forecast.prob_exhaust_before_reset <= 1.0
    assert forecast.confidence in ("low", "medium", "high")


def test_simulation_is_capped_at_an_imminent_reset(cfg):
    """Regression: simulations must run only up to a KNOWN reset, not
    beyond it. A slow, steady burn with a reset an hour away should
    report no ETA (not at risk this cycle) rather than a far-future
    exhaustion time that ignores the refill happening first."""
    now = 1_000_000.0
    resets_at = now + 3600  # resets in 1h
    # +0.07%/h, oldest first -- at this rate it would take roughly 1200
    # hours to exhaust from 10%, nowhere close to the 1h reset.
    hours_ago = list(range(20, -1, -1))
    history = [
        _sample(now - h * 3600, 10.0 + (20 - h) * 0.07, resets_at) for h in hours_ago
    ]
    window = _window(WindowKind.W5H, history[-1].used_percent, now, resets_at)
    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "OK"
    assert forecast.eta_p50 is None
    assert forecast.eta_calendar is None
    assert forecast.exhausts_before_reset is False


def test_idle_short_circuits_before_simulating(cfg):
    now = 1_000_000.0
    stale_ts = now - cfg.thresholds.stale_after_minutes * 60 - 60
    window = _window(WindowKind.W5H, 50.0, stale_ts)
    forecast = MonteCarloPredictor().forecast(window, [], [], cfg, now)
    assert forecast.status == "IDLE"
    assert forecast.eta_p50 is None


def test_reset_pending_short_circuits_before_simulating(cfg):
    now = 1_000_000.0
    history = [_sample(now - 120, 95.0), _sample(now, 5.0)]
    window = _window(WindowKind.WEEKLY, 5.0, now)
    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "RESET_PENDING"


def test_no_history_falls_back_to_ok_low_confidence(cfg):
    now = 1_000_000.0
    window = _window(WindowKind.WEEKLY, 10.0, now)
    forecast = MonteCarloPredictor().forecast(window, [], [], cfg, now)
    assert forecast.status == "OK"
    assert forecast.confidence == "low"
    assert forecast.eta_p50 is None


def test_flat_zero_burn_has_no_exhaustion_but_still_ok(cfg):
    now = 1_000_000.0
    history = [_sample(now - i * 3600, 10.0) for i in range(5, -1, -1)]
    window = _window(WindowKind.WEEKLY, 10.0, now)
    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "OK"
    assert forecast.burn_per_hour == pytest.approx(0.0)
    assert forecast.eta_p50 is None  # no simulation ever exhausted -> no band


def test_percentile_within_horizon_is_censored_not_conditional():
    """Regression: percentiles must be computed over ALL simulated
    outcomes (non-exhausting runs counted as right-censored at the
    horizon), not just the subset that happened to exhaust — otherwise a
    "P50" computed from a small exhausting minority reads as an early,
    confident ETA when the true unconditional median never exhausts at
    all within the horizon.
    """
    horizon = 100.0
    # 3 of 10 simulated futures exhaust; 7 run out the clock (censored).
    outcomes = sorted([10.0, 20.0, 30.0] + [horizon] * 7)

    # Most futures don't exhaust within the horizon -> no median ETA to
    # report, not the median of the exhausting minority (30.0 or lower).
    assert _percentile_within_horizon(outcomes, 0.5, horizon) is None
    # The fast tail (bottom 10%) is still knowable even though P50 isn't.
    assert _percentile_within_horizon(outcomes, 0.1, horizon) == 20.0


def test_percentile_within_horizon_returns_a_value_when_most_exhaust():
    horizon = 100.0
    outcomes = sorted([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0] + [horizon] * 3)
    assert _percentile_within_horizon(outcomes, 0.5, horizon) == 60.0


def test_selectable_via_config(tmp_path):
    from tokenwatchdog.config import load_config
    from tokenwatchdog.predictor import select_predictor

    config_path = tmp_path / "config.toml"
    config_path.write_text('[predictor]\nmodel = "montecarlo"\n')
    cfg = load_config(config_path)
    assert select_predictor(cfg).name == "montecarlo"
