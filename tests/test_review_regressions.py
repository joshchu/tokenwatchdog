"""Defects found by an independent review of the prediction changes.

Each of these reproduced against real or realistic data before being fixed,
and each is a case where the code confidently reported something false rather
than reporting nothing.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

from tokenwatchdog.alerts import evaluate, level_still_in_cycle
from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.predictor import LinearPredictor, MonteCarloPredictor
from tokenwatchdog.store import SampleRow, TokenEventRow

UTC = ZoneInfo("UTC")
HOUR = 3600.0


def _sample(ts, used_percent, resets_at=None):
    return SampleRow(
        captured_at=ts,
        source_ts=ts,
        used_percent=used_percent,
        resets_at=resets_at,
        is_estimated=False,
    )


def _event(ts, total=10_000):
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


def test_a_reading_from_an_expired_cycle_does_not_alert(cfg, store):
    """A 26-hour-old 95% reading of a FIVE-hour window fired a threshold
    alert, then re-fired every five hours indefinitely.

    `level_still_in_cycle` checked only `now < resets_at` — but a derived
    reset is advanced by whole cycles once its projection elapses, so the very
    passage of a boundary manufactured a future reset time that made a
    long-dead reading look live, and advanced again on each re-arm."""
    now = 2_000_000.0
    history = [
        _sample(now - 31 * HOUR, 90.0),
        _sample(now - 30 * HOUR, 2.0),  # an observed reset, 30h ago
        _sample(now - 29 * HOUR, 40.0),
        _sample(now - 27 * HOUR, 80.0),
        _sample(now - 26 * HOUR, 95.0),  # last reading, 26h old
    ]
    window = _window(WindowKind.W5H, 95.0, now - 26 * HOUR)

    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert forecast.status == "IDLE"
    assert level_still_in_cycle(forecast.window, now) is False
    assert evaluate(forecast, cfg, store, now) == []
    # And still silent after the phantom reset would have advanced again.
    later = LinearPredictor().forecast(window, history, [], cfg, now + 6 * HOUR)
    assert evaluate(later, cfg, store, now + 6 * HOUR) == []


def test_a_fresh_reading_over_the_threshold_still_alerts(cfg, store):
    """The other side: tightening the cycle check must not silence the real
    case it was added for — a stale RATE with a level that genuinely is
    still this cycle's."""
    now = 2_000_000.0
    window = _window(
        WindowKind.WEEKLY, 95.0, now - 4 * HOUR, resets_at=now + 3 * 24 * HOUR
    )
    history = [_sample(now - 5 * HOUR, 80.0), _sample(now - 4 * HOUR, 95.0)]

    forecast = LinearPredictor().forecast(window, history, [], cfg, now)

    assert level_still_in_cycle(forecast.window, now) is True
    assert [a.alert_kind for a in evaluate(forecast, cfg, store, now)] == ["threshold"]


def test_no_reset_is_invented_for_an_expired_5h_block(cfg):
    """With a token log to read, no anchor means the block has already
    expired — the window is empty and the next one doesn't exist until the
    next request. Reporting a reset time there described a block that wasn't
    running."""
    now = 2_000_000.0
    history = [
        _sample(now - 31 * HOUR, 90.0),
        _sample(now - 30 * HOUR, 2.0),
        _sample(now - 26 * HOUR, 95.0),
    ]
    stale_events = [_event(now - 26 * HOUR)]  # nothing for 26h
    window = _window(WindowKind.W5H, 95.0, now - 26 * HOUR)

    forecast = LinearPredictor().forecast(window, history, stale_events, cfg, now)

    assert forecast.window.resets_at is None
    assert forecast.time_to_reset_h is None


def test_an_already_exhausted_window_reports_full_risk_not_zero(cfg):
    """`_simulate_exhaustion_hours` returns 0.0 for a window already at 100%,
    and `outcome or horizon` treated that falsy 0.0 as "never exhausted" —
    rewriting every run as censored and reporting a 0% chance of exhausting
    on a window that is already exhausted. Reproduced on real data: the
    saturated Codex weekly window read "Risk 0%" on a red 100% row."""
    now = 2_000_000.0
    history = [
        _sample(now - 6 * HOUR, 70.0),
        _sample(now - 3 * HOUR, 85.0),
        _sample(now, 100.0),
    ]
    window = _window(
        WindowKind.WEEKLY,
        100.0,
        now,
        resets_at=now + 4 * 24 * HOUR,
        provider=Provider.CODEX,
    )

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    assert forecast.prob_exhaust_before_reset == 1.0


def test_a_censored_median_does_not_synthesize_a_mean_rate_eta(cfg):
    """When most simulated futures survive to the reset there is no point ETA
    — and none may be invented from the all-hours mean burn. That fallback
    produced a confident ETA and a "burning" status at a simulated 0% risk,
    contradicting the simulation in the same Forecast, and those ETAs were
    then graded as though the model had made them."""
    now = 2_000_000.0
    # A slow steady climb with a reset close enough that few futures exhaust.
    history = []
    ts = now - 4 * 24 * HOUR
    percent = 0.0
    while ts <= now:
        history.append(_sample(ts, min(percent, 60.0)))
        percent += 0.05
        ts += 600.0
    window = _window(WindowKind.WEEKLY, 60.0, now, resets_at=now + 2 * HOUR)

    forecast = MonteCarloPredictor().forecast(window, history, [], cfg, now)

    if forecast.eta_p50 is None:  # the censored path this test is about
        assert forecast.eta_calendar is None
        assert forecast.eta_workhours is None
        assert forecast.exhausts_before_reset is False
        # The rate and the risk are still reported -- this is "you probably
        # won't run out," not "nothing is known."
        assert forecast.burn_per_hour > 0
        assert forecast.prob_exhaust_before_reset is not None
