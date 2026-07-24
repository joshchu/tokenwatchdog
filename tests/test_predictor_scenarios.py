"""Longer, more realistic scenario tests for the predictors' "secret sauce"
claim -- robust to a single burst outlier, responsive to a genuine pace
change, and numerically sane over a long, low-and-slow history. Unlike
test_predictor.py/test_montecarlo.py's focused unit tests, these build
multi-day synthetic time series and check properties (closeness to a known
true rate, one predictor vs. another) rather than exact hand-computed
numbers.
"""

from __future__ import annotations

import pytest

from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.predictor import LinearPredictor, MonteCarloPredictor
from tokenwatchdog.store import SampleRow


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


def _climb(
    now,
    *,
    start_seconds_ago,
    end_seconds_ago,
    step_seconds,
    start_percent,
    rate_per_hour,
):
    """Monotonically rising samples, oldest first, from `start_seconds_ago`
    to `end_seconds_ago` before `now`, climbing at a constant rate. Never
    reverts downward -- a real quota percentage only drops at a reset, so a
    "spike" here means one bigger step, never a value that un-happens."""
    samples = []
    ts = now - start_seconds_ago
    percent = start_percent
    end_ts = now - end_seconds_ago
    while ts <= end_ts:
        samples.append(_sample(ts, percent))
        percent += rate_per_hour * (step_seconds / 3600.0)
        ts += step_seconds
    return samples


def test_low_slow_burn_is_reported_as_no_exhaustion_not_a_distant_date(cfg):
    """The "low but slow burn" case: a genuinely low, steady pace sustained
    for ten days must read as calm -- and specifically must not put a date
    in the ETA column.

    At 0.05%/h a window sitting at ~22% needs ~1500 hours to reach 100%,
    which is nine times a weekly window's own 168-hour life. It will reset
    long before it ever gets there, so that date describes an event that
    cannot happen; a bound of one window duration is implied by what the
    window IS, independent of whether a reset time has been derived yet.
    (Found by scripts/backtest.py, which caught a FIVE-hour window being
    handed a 3311-hour ETA on exactly this path.)"""
    now = 2_000_000.0
    rate = 0.05  # slow: ~1.2%/day
    history = _climb(
        now,
        start_seconds_ago=10 * 24 * 3600,
        end_seconds_ago=0,
        step_seconds=1800,  # every 30 min, realistic polling cadence
        start_percent=10.0,
        rate_per_hour=rate,
    )
    window = _window(
        Provider.CLAUDE,
        WindowKind.WEEKLY,
        history[-1].used_percent,
        now,
        is_estimated=True,
    )
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    # The rate is still measured and reported -- this is "you won't run
    # out," not "we don't know anything."
    assert forecast.status == "OK"
    assert forecast.burn_per_hour == pytest.approx(rate, rel=0.1)
    assert (100.0 - window.used_percent) / rate > 9 * 168  # far past a week
    assert forecast.eta_calendar is None
    assert forecast.eta_workhours is None
    assert forecast.exhausts_before_reset is False


def test_a_burn_that_does_exhaust_within_the_window_still_gets_an_eta(cfg):
    """The other side of the bound: it suppresses the impossible, not the
    merely far-off. A pace that genuinely runs the window out inside its own
    duration must still produce a date."""
    now = 2_000_000.0
    rate = 2.0  # ~50h to burn 100% -- comfortably inside a 168h window
    history = _climb(
        now,
        start_seconds_ago=24 * 3600,
        end_seconds_ago=0,
        step_seconds=1800,
        start_percent=10.0,
        rate_per_hour=rate,
    )
    window = _window(Provider.CLAUDE, WindowKind.WEEKLY, history[-1].used_percent, now)

    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.eta_calendar is not None
    hours_out = (forecast.eta_calendar.timestamp() - now) / 3600.0
    assert hours_out == pytest.approx((100.0 - window.used_percent) / rate, rel=0.15)
    assert hours_out < 168.0


def test_vanishingly_small_burn_with_unknown_reset_does_not_overflow(cfg):
    """Regression: a burn rate close enough to zero (but not exactly zero,
    e.g. floating-point noise off two nearly-identical readings) divides
    into a remaining-percent to produce an exhaustion horizon of literally
    thousands of years. Projected forward from `now`, that overflows
    Python's datetime range -- the fix must report "no meaningful ETA"
    (None) instead of crashing the whole tick."""
    now = 2_000_000.0
    history = [
        _sample(now - 3600, 40.0),
        _sample(now, 40.0 + 1e-9),  # a hair above flat -- not caught by burn<=0
    ]
    window = _window(
        Provider.CODEX,
        WindowKind.WEEKLY,
        history[-1].used_percent,
        now,
        is_estimated=False,
    )
    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "OK"
    assert forecast.burn_per_hour > 0.0
    assert forecast.eta_calendar is None
    assert forecast.eta_workhours is None


def test_extremely_distant_burn_does_not_blow_the_workhours_convergence_guard(cfg):
    """Regression: a second, distinct overflow path. ~684 years out is
    comfortably inside datetime's range (no OverflowError from plain
    calendar arithmetic), but working-hours projection advances at most
    ~1 working day per loop iteration, so it blows its own 100,000-
    iteration convergence guard (RuntimeError) long before that. Same
    "no meaningful ETA" answer applies."""
    now = 2_000_000.0
    history = [
        _sample(now - 3600, 40.0),
        _sample(now, 40.0 + 1e-5),  # ~684-year exhaustion horizon from here
    ]
    window = _window(
        Provider.CODEX,
        WindowKind.WEEKLY,
        history[-1].used_percent,
        now,
        is_estimated=False,
    )
    assert cfg.working_hours.enabled  # this path only exists when it's on

    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "OK"
    assert forecast.burn_per_hour > 0.0
    assert forecast.eta_calendar is None
    assert forecast.eta_workhours is None


def test_robust_slope_resists_an_outlier_near_the_edge_of_the_window(cfg):
    """One enormous one-off jump (e.g. a single huge request), positioned
    near the START of the lookback window, then climbing steadily at the
    old pace for the rest of it. Only a small minority of fixed-lag
    slopes involve the outlier point here, so the median comfortably
    lands on the steady rate."""
    now = 2_000_000.0
    steady_rate = 0.75  # %/h
    step_seconds = 240  # every 4 min, well inside the 60-min weekly lookback
    samples = []
    ts = now - 3600
    percent = 30.0
    spike_ts = now - 3600 + step_seconds  # the second sample: near the start
    while ts <= now:
        percent += steady_rate * (step_seconds / 3600.0)
        if ts == spike_ts:
            percent += 20.0  # one enormous one-off burst, never reverted
        samples.append(_sample(ts, percent))
        ts += step_seconds
    window = _window(
        Provider.CODEX,
        WindowKind.WEEKLY,
        samples[-1].used_percent,
        now,
        resets_at=now + 5 * 24 * 3600,
    )

    forecast = LinearPredictor().forecast(window, samples, [], cfg, now)

    naive_slope = (samples[-1].used_percent - samples[0].used_percent) / (
        (samples[-1].source_ts - samples[0].source_ts) / 3600.0
    )
    assert naive_slope > steady_rate * 2  # the spike visibly drags the naive number up

    assert forecast.status == "OK"
    assert forecast.burn_per_hour == pytest.approx(steady_rate, rel=0.5)
    assert forecast.burn_per_hour < naive_slope / 2


def test_robust_slope_resists_an_outlier_near_the_middle_of_the_window(cfg):
    """Regression: the median of ALL C(n,2) pairwise slopes (this
    predictor's previous algorithm, "Theil-Sen") failed exactly this
    case -- a permanent step positioned near the middle of the window
    contaminates close to half of all pairs (before-count * after-count
    is maximized at the midpoint), which exceeds the median's breakdown
    point. Measured at the time: it produced an estimate *worse* than
    the naive endpoint-to-endpoint slope, not better. Fixed by switching
    to the median of fixed-lag-k slopes (see _robust_slope_per_hour) --
    a step can corrupt at most k of the (n-k) lag-k slopes regardless of
    where it falls, so resistance no longer depends on the outlier's
    position in the window."""
    now = 2_000_000.0
    steady_rate = 0.75  # %/h
    step_seconds = 240  # every 4 min, well inside the 60-min weekly lookback
    samples = []
    ts = now - 3600
    percent = 30.0
    spike_ts = now - 32 * 60  # roughly the middle of the window
    while ts <= now:
        percent += steady_rate * (step_seconds / 3600.0)
        if ts == spike_ts:
            percent += 20.0  # one enormous one-off burst, never reverted
        samples.append(_sample(ts, percent))
        ts += step_seconds
    window = _window(
        Provider.CODEX,
        WindowKind.WEEKLY,
        samples[-1].used_percent,
        now,
        resets_at=now + 5 * 24 * 3600,
    )

    forecast = LinearPredictor().forecast(window, samples, [], cfg, now)

    naive_slope = (samples[-1].used_percent - samples[0].used_percent) / (
        (samples[-1].source_ts - samples[0].source_ts) / 3600.0
    )
    assert naive_slope > steady_rate * 2  # the spike visibly drags the naive number up

    assert forecast.status == "OK"
    assert forecast.burn_per_hour == pytest.approx(steady_rate, rel=0.5)
    assert forecast.burn_per_hour < naive_slope / 2


def _pace_change_history(now, *, old_rate, new_rate, new_phase_hours):
    """Days at `old_rate`, then `new_phase_hours` at `new_rate`."""
    old_phase = _climb(
        now,
        start_seconds_ago=96 * 3600,
        end_seconds_ago=new_phase_hours * 3600 + 1800,  # one step before the change
        step_seconds=1800,
        start_percent=5.0,
        rate_per_hour=old_rate,
    )
    new_phase = _climb(
        now,
        start_seconds_ago=new_phase_hours * 3600,
        end_seconds_ago=0,
        step_seconds=900,
        start_percent=old_phase[-1].used_percent,
        rate_per_hour=new_rate,
    )
    return old_phase + new_phase


def test_linear_tracks_a_sustained_pace_change_monte_carlo_lags(cfg):
    """The flip side of outlier-resistance: a REAL, sustained pace change
    (not a one-off blip) must still be picked up, not smoothed away
    forever. Once the new pace has held for the whole lookback, linear
    reports it. Monte Carlo instead pools burn observations from the entire
    history (recency-decayed, not excluded), so its mean stays anchored near
    the old rate until the new rate has had time to outweigh days of old
    data -- a real tradeoff between the two models, not a bug in either."""
    now = 2_000_000.0
    old_rate, new_rate = 0.3, 10.0  # %/h -- a genuine ~33x step up
    history = _pace_change_history(
        now,
        old_rate=old_rate,
        new_rate=new_rate,
        new_phase_hours=cfg.burn.lookback_weekly_minutes / 60.0,
    )
    window = _window(
        Provider.CLAUDE,
        WindowKind.WEEKLY,
        history[-1].used_percent,
        now,
        is_estimated=True,
    )

    linear_forecast = LinearPredictor().forecast(window, history, [], cfg, now)
    mc_forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert linear_forecast.burn_per_hour == pytest.approx(new_rate, rel=0.3)
    # Monte Carlo's mean is volume-dominated by four days of old-rate data
    # against a few hours of new-rate data -- it must land far closer to the
    # old rate than to the new one.
    assert mc_forecast.burn_per_hour < (old_rate + new_rate) / 2
    assert linear_forecast.burn_per_hour > mc_forecast.burn_per_hour * 5


def test_weekly_burn_smooths_a_pace_change_shorter_than_its_lookback(cfg):
    """The weekly lookback is hours wide on purpose, so a pace change that
    has only just started is reported partially rather than in full -- a
    7-day budget shouldn't have its ETA yanked around by a burst that
    started twenty minutes ago.

    That width is what makes the weekly window readable at all: its
    percentage is reported in whole numbers, and an evenly-spread week burns
    ~0.6%/h, so a one-hour lookback observes a change of 0 or 1 and nothing
    in between. Pinned here because it is a deliberate trade against
    responsiveness, not a free win."""
    now = 2_000_000.0
    old_rate, new_rate = 0.3, 10.0
    lookback_hours = cfg.burn.lookback_weekly_minutes / 60.0
    elapsed_hours = lookback_hours / 3.0
    history = _pace_change_history(
        now, old_rate=old_rate, new_rate=new_rate, new_phase_hours=elapsed_hours
    )
    window = _window(Provider.CLAUDE, WindowKind.WEEKLY, history[-1].used_percent, now)

    burn = LinearPredictor().forecast(window, history, [], cfg, now).burn_per_hour

    # Between the two rates, and visibly moved off the old one: the change
    # registers immediately, it just isn't believed in full yet.
    assert old_rate < burn < new_rate
    assert burn > old_rate * 5
