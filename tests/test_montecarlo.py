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


def _window(kind, used_percent, source_ts, resets_at=None, provider=Provider.CODEX):
    window_minutes = 300 if kind is WindowKind.W5H else 10080
    return Window(
        provider=provider,
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


def test_idle_with_a_valid_current_level_still_predicts_from_learned_rhythm(cfg):
    # CLAUDE deliberately: a stale level is durable only on a FIXED window.
    # A rolling window's level decays unobserved, so it is never vouched
    # for (see test_alerts.test_a_stale_rolling_level_is_not_vouched_for).
    now = 1_000_000.0
    stale_ts = now - cfg.thresholds.stale_after_minutes_weekly * 60 - 60
    resets_at = now + 60 * 3600
    history = []
    ts = stale_ts - 5 * 24 * 3600
    percent = 0.0
    while ts <= stale_ts:
        history.append(_sample(ts, percent))
        percent += 2.0
        if percent >= 95.0:
            percent = 0.0
        ts += 3600
    window = _window(
        WindowKind.WEEKLY,
        history[-1].used_percent,
        stale_ts,
        resets_at,
        provider=Provider.CLAUDE,
    )

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "IDLE"
    assert forecast.eta_p50 is not None
    assert forecast.eta_p90 is not None
    assert forecast.prob_exhaust_before_reset is not None
    assert forecast.confidence is not None


def test_idle_with_a_level_from_an_old_cycle_still_short_circuits(cfg):
    now = 1_000_000.0
    resets_at = now + 60 * 3600
    old_source_ts = now - 8 * 24 * 3600
    window = _window(WindowKind.WEEKLY, 50.0, old_source_ts, resets_at)

    forecast = MonteCarloPredictor().forecast(window, [], [], cfg, now)

    assert forecast.status == "IDLE"
    assert forecast.eta_p50 is None
    assert forecast.prob_exhaust_before_reset is None


def test_reset_pending_short_circuits_before_simulating(cfg):
    # CLAUDE deliberately: fixed-cycle semantics. A codex weekly drop is
    # roll-off, not a reset (see tests/test_rolling_window.py).
    now = 1_000_000.0
    history = [_sample(now - 120, 95.0), _sample(now, 5.0)]
    window = _window(WindowKind.WEEKLY, 5.0, now, provider=Provider.CLAUDE)
    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)
    assert forecast.status == "RESET_PENDING"


def _burst_history(now, *, level_now, burst_rate, resets_at):
    """Hours of slow ~1%/h climb, then one final hour at `burst_rate` —
    the live-burst shape, sized so the burst fills the w5h lookback (60
    min): the robust slope deliberately smooths a burst SHORTER than the
    lookback as an outlier, which is its own pinned behavior."""
    slow_hours = 3
    start = level_now - burst_rate - slow_hours * 1.0
    history = [
        _sample(now - (slow_hours - i) * 3600.0 - 3600.0, start + i * 1.0, resets_at)
        for i in range(slow_hours + 1)
    ]
    history.append(_sample(now - 1800.0, level_now - burst_rate / 2.0, resets_at))
    history.append(_sample(now, level_now, resets_at))
    return history


def test_live_rate_dominates_the_first_simulated_hour(cfg):
    """The original measured failure (F2): at 80-95% used mid-burst the
    model predicted 2.28h when the truth was minutes, because its first
    simulated hour drew from "what this hour-of-week is generally like"
    rather than from the burst happening right now. With the live rate
    seeding the blend, remaining quota inside the first hour resolves at
    the live pace."""
    now = 1_000_000.0
    resets_at = now + 3 * 3600
    history = _burst_history(now, level_now=92.0, burst_rate=10.0, resets_at=resets_at)
    window = _window(WindowKind.W5H, 92.0, now, resets_at, provider=Provider.CLAUDE)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.eta_p50 is not None
    eta_h = (forecast.eta_p50.timestamp() - now) / 3600.0
    # 8% remaining at ~10%/h live -> around the hour mark; nowhere near
    # the ~8h the slow profile alone would say.
    assert eta_h < 2.0
    assert forecast.burn_per_hour == pytest.approx(10.0, rel=0.4)


def test_live_rate_decays_into_the_profile_instead_of_extrapolating(cfg):
    """The blend is a blend, not a linear clone: far from the cap, the
    live burst hands off to the profile within a few hours (the geometric
    sum contributes only ~2x the live rate in total), so the median future
    reflects the slow history — censored here — rather than the burst
    extrapolated for hours."""
    now = 1_000_000.0
    resets_at = now + 4.5 * 3600
    history = _burst_history(now, level_now=20.0, burst_rate=10.0, resets_at=resets_at)
    window = _window(WindowKind.W5H, 20.0, now, resets_at, provider=Provider.CLAUDE)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    # Pure live-rate extrapolation would claim exhaustion in ~8h — inside
    # a naive 8h view but far past what the blend sustains: ~20% of quota
    # from the decaying burst plus ~1%/h of profile never reaches 100%
    # before the reset. The median run censors.
    assert forecast.eta_p50 is None
    # The rate REPORTED is still the live one — one field, one meaning.
    assert forecast.burn_per_hour == pytest.approx(10.0, rel=0.4)


def test_a_stale_rate_is_not_blended(cfg):
    """An hours-old burst must not dominate hour one of an idle window's
    future: on a stale rate the simulation runs from the profile alone,
    and the reported rate falls back to the profile mean."""
    now = 1_000_000.0
    stale_ts = now - cfg.thresholds.stale_after_minutes_weekly * 60 - 60
    resets_at = now + 48 * 3600
    history = [
        _sample(stale_ts - h * 3600, max(0.0, 50.0 - 10.0 - (h - 1) * 0.2), resets_at)
        for h in range(72, 0, -1)
    ]
    history.append(_sample(stale_ts, 50.0, resets_at))  # burst, then silence
    window = _window(
        WindowKind.WEEKLY, 50.0, stale_ts, resets_at, provider=Provider.CLAUDE
    )

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "IDLE"  # level valid, rate stale -> still simulates
    # Reported rate is the profile mean, nowhere near the stale 10%/h burst.
    assert forecast.burn_per_hour < 2.0


def test_unknown_reset_simulates_only_the_windows_own_duration(cfg):
    """With no derivable reset, the horizon is the window's own duration —
    a 5-hour window refills at least every 5 hours, so nothing past that can
    be this cycle's exhaustion. The old 14-day fallback paid for up to 672k
    draws a tick and then blanked everything past the 5-hour output cap.

    The wiring is observable through confidence: this profile knows exactly
    the hour-of-week slots the coming five hours occupy (learned one week
    ago), so judged against the window's own span coverage is complete and
    the medium rating from n stands; judged against 336 mostly-unknown
    hours it was 5/168 ≈ 3% coverage and rated "low"."""
    now = 1_000_000.0
    week = 168 * 3600.0
    # Eight hourly readings from exactly one week ago — their hour-of-week
    # slots cover now..now+7, so the 5h horizon is fully known — plus three
    # fresh readings inside the w5h lookback (the live rate's evidence,
    # which caps confidence at medium). Burn is far too slow to exhaust a
    # 5h horizon and no reset is observed or derivable.
    history = [_sample(now - week + k * 3600, 20.0 + k) for k in range(8)]
    history += [
        _sample(now - 2400.0, 29.0),
        _sample(now - 1200.0, 29.5),
        _sample(now, 30.0),
    ]
    window = _window(WindowKind.W5H, 30.0, now)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "OK"
    assert forecast.eta_p50 is None  # censored at 5h, honestly
    assert forecast.prob_exhaust_before_reset is None  # reset unknown
    assert forecast.confidence == "medium"


def test_workhours_eta_is_the_p50_not_a_double_count(cfg):
    """The p50 already walks a learned week — overnight idling is in the
    profile. Re-projecting it through the working-hours budget gate counted
    time-of-day twice (that gate's contract is a 24/7 burn-hours budget,
    which only linear's constant-rate hours satisfy)."""
    now = 1_000_000.0
    resets_at = now + 60 * 3600
    history = []
    ts = now - 5 * 24 * 3600
    percent = 0.0
    while ts <= now:
        history.append(_sample(ts, percent, resets_at if ts == now else None))
        percent += 2.0
        if percent >= 95.0:
            percent = 0.0
        ts += 3600
    window = _window(WindowKind.WEEKLY, history[-1].used_percent, now, resets_at)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.eta_p50 is not None
    assert forecast.eta_workhours == forecast.eta_p50


def test_no_history_falls_back_to_ok_with_no_confidence(cfg):
    now = 1_000_000.0
    window = _window(WindowKind.WEEKLY, 10.0, now)
    forecast = MonteCarloPredictor().forecast(window, [], [], cfg, now)
    assert forecast.status == "OK"
    assert forecast.confidence is None  # nothing forecast -- nothing to rate
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
    predictor, reason = select_predictor(cfg)
    assert predictor.name == "montecarlo"
    assert "config" in reason  # an explicit choice is never silently overridden


def test_confidence_is_capped_by_hour_of_week_coverage(cfg):
    """Observation COUNT alone overstates confidence badly. History packed
    into a couple of days can hold hundreds of observations while leaving
    most of a week unlearned — and every unlearned hour of the simulated
    horizon gets drawn from the all-hours pool instead. Measured on real
    data: 896 observations covering 94 of 168 buckets, rated "high"."""
    now = 1_000_000.0
    # Two days of dense history: plenty of observations, ~48/168 coverage.
    history = []
    ts = now - 2 * 24 * 3600
    percent = 0.0
    while ts <= now:
        history.append(_sample(ts, percent))
        percent = min(percent + 0.1, 60.0)
        ts += 600.0
    assert len(history) > 100  # volume alone would say "high"

    # A full week of horizon, most of which has no bucket of its own.
    weekly = _window(WindowKind.WEEKLY, 60.0, now, resets_at=now + 7 * 24 * 3600)
    forecast = MonteCarloPredictor().forecast(weekly, history, [], cfg, now)

    assert forecast.n_samples > 100
    assert forecast.confidence in ("low", "medium")


def test_simulation_resolves_exhaustion_inside_the_current_hour(cfg):
    """The simulation walks in whole hours, so without interpolating the
    hour it runs out in, it could not express any ETA sooner than 60 minutes
    away — and a window resetting sooner than that had every outcome
    censored past its horizon, silencing the band exactly when it mattered
    most."""
    now = 1_000_000.0
    # 99% used with a steady, brisk burn and only 40 minutes to the reset.
    history = []
    ts = now - 6 * 3600
    percent = 60.0
    while ts <= now:
        history.append(_sample(ts, min(percent, 99.0)))
        percent += 1.0
        ts += 600.0
    window = _window(WindowKind.W5H, 99.0, now, resets_at=now + 2400.0)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.eta_p50 is not None
    minutes_out = (forecast.eta_p50.timestamp() - now) / 60.0
    assert 0.0 < minutes_out < 40.0


def test_risk_is_reported_even_when_the_median_future_does_not_exhaust(cfg):
    """The censored-P50 path. "The median future is fine, but some aren't"
    is the entire reason to simulate rather than extrapolate, so dropping
    the probability here threw away the model's most distinctive output."""
    now = 1_000_000.0
    # A slow, noisy burn against a near reset: most simulated futures won't
    # exhaust in time, so there's no point ETA -- but the tail is real.
    history = []
    ts = now - 4 * 24 * 3600
    percent = 0.0
    while ts <= now:
        history.append(_sample(ts, percent))
        percent = min(percent + 0.05, 80.0)
        ts += 600.0
    window = _window(WindowKind.WEEKLY, 80.0, now, resets_at=now + 8 * 3600)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.eta_p50 is None  # median future survives to the reset
    assert forecast.prob_exhaust_before_reset is not None
    assert 0.0 <= forecast.prob_exhaust_before_reset <= 1.0
