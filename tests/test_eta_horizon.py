"""What an ETA is allowed to claim.

Two bounds, both about not printing a confident date for an event that
can't happen:

- each ETA is capped against the reset **independently** — the working-hours
  one is always the later of the two, so a single decision made from the
  24/7 projection let it sit in the table pointing days past a reset;
- with no reset derived yet, the window's own duration still bounds it.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime
from zoneinfo import ZoneInfo

from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.predictor import LinearPredictor
from tokenwatchdog.store import SampleRow

UTC = ZoneInfo("UTC")
HOUR = 3600.0
# A Friday afternoon, inside working hours with the weekend immediately
# after -- the arrangement that makes the two ETAs diverge the most.
FRIDAY_1600 = datetime(2026, 7, 24, 16, 0, tzinfo=UTC).timestamp()


def _utc_cfg(cfg):
    return dataclasses.replace(cfg, timezone="UTC")


def _sample(source_ts, used_percent):
    return SampleRow(
        captured_at=source_ts,
        source_ts=source_ts,
        used_percent=used_percent,
        resets_at=None,
        is_estimated=False,
    )


def _window(kind, used_percent, source_ts, resets_at):
    return Window(
        provider=Provider.CLAUDE,
        kind=kind,
        used_percent=used_percent,
        window_minutes=300 if kind is WindowKind.W5H else 10080,
        resets_at=resets_at,
        source_ts=source_ts,
        is_estimated=False,
        source_file="test",
    )


def _forecast_at(cfg, *, used_percent, rate_per_hour, resets_at, kind):
    """A forecast with a known burn rate: two samples an hour apart whose
    difference IS the rate."""
    now = FRIDAY_1600
    history = [
        _sample(now - HOUR, used_percent - rate_per_hour),
        _sample(now, used_percent),
    ]
    window = _window(kind, used_percent, now, resets_at)
    return LinearPredictor().forecast(window, history, [], _utc_cfg(cfg), now)


def test_working_hours_eta_is_capped_at_the_reset_independently(cfg):
    """20 burn-hours from Friday 16:00 lands Wednesday once the weekend is
    skipped — but this window resets Saturday. The 24/7 projection (Saturday
    noon) legitimately beats the reset, and deciding suppression from that
    one number is what published the Wednesday date."""
    now = FRIDAY_1600
    forecast = _forecast_at(
        cfg,
        used_percent=50.0,
        rate_per_hour=2.5,  # 50% remaining / 2.5 = 20 burn-hours
        resets_at=now + 24 * HOUR,  # Saturday 16:00
        kind=WindowKind.WEEKLY,
    )

    assert forecast.burn_per_hour == 2.5
    # 24/7: Saturday noon, four hours inside the reset -- a real ETA.
    assert forecast.eta_calendar is not None
    assert (forecast.eta_calendar.timestamp() - now) / HOUR == 20.0
    # Working hours: Wednesday, four days PAST the reset -- not an ETA.
    assert forecast.eta_workhours is None
    assert forecast.exhausts_before_reset is True


def test_both_etas_survive_when_both_land_before_the_reset(cfg):
    """The cap suppresses what's impossible, not whatever is inconvenient:
    with a whole week until the reset, the Wednesday working-hours date is
    entirely reachable and must still be shown."""
    now = FRIDAY_1600
    forecast = _forecast_at(
        cfg,
        used_percent=50.0,
        rate_per_hour=2.5,
        resets_at=now + 7 * 24 * HOUR,
        kind=WindowKind.WEEKLY,
    )

    assert forecast.eta_calendar is not None
    assert forecast.eta_workhours is not None
    # The working-hours projection is the later of the two, by design.
    assert forecast.eta_workhours > forecast.eta_calendar


def test_unknown_reset_still_bounds_the_eta_by_the_window_duration(cfg):
    """A 5-hour window cannot take 30 hours to exhaust; it refills five
    times over first. Before this bound, exactly this path handed a 5-hour
    window a 3311-hour ETA (found by scripts/backtest.py)."""
    forecast = _forecast_at(
        cfg,
        used_percent=40.0,
        rate_per_hour=2.0,  # 60% remaining / 2 = 30 hours out
        resets_at=None,
        kind=WindowKind.W5H,
    )

    assert forecast.burn_per_hour == 2.0
    assert forecast.time_to_reset_h is None
    assert forecast.eta_calendar is None
    assert forecast.eta_workhours is None


def test_unknown_reset_keeps_an_eta_that_fits_inside_the_duration(cfg):
    """Same unknown reset, but a rate that exhausts the window in two hours
    — well inside its five-hour life, so the date is possible and stands."""
    forecast = _forecast_at(
        cfg,
        used_percent=80.0,
        rate_per_hour=10.0,  # 20% remaining / 10 = 2 hours out
        resets_at=None,
        kind=WindowKind.W5H,
    )

    assert forecast.eta_calendar is not None
    assert (forecast.eta_calendar.timestamp() - FRIDAY_1600) / HOUR == 2.0
    # Still not claimed as "before the reset" -- there's no known reset to
    # have beaten, and the duration bound is not a reset time.
    assert forecast.exhausts_before_reset is False
