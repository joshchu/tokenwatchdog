"""Model selection — and, mostly, its refusal to select.

`predictor.model = "auto"` grades every model against this machine's own
stored forecasts. The interesting property is not that it picks a winner but
that it usually declines to: with one user's history, a small MAE difference
is as likely to be that person's luck as a real improvement, and switching on
it would dress up noise as a measurement.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tokenwatchdog.config import load_config
from tokenwatchdog.models import Forecast, Provider, Window, WindowKind
from tokenwatchdog.predictor import _is_reset, select_predictor
from tokenwatchdog.scoring import (
    MIN_EPISODES_TO_GRADUATE,
    MIN_RELATIVE_IMPROVEMENT,
    ModelScore,
    better_model,
    realized_exhaustion_hours,
    score_model,
)
from tokenwatchdog.store import ForecastRow, SampleRow

HOUR = 3600.0


def _score(name, *, episodes, mae):
    return ModelScore(
        model_name=name,
        moments=episodes * 2,
        with_eta=episodes,
        episodes=episodes,
        mae_hours=mae,
        bias_hours=0.0,
    )


def _sample(ts, used_percent):
    return SampleRow(
        captured_at=ts,
        source_ts=ts,
        used_percent=used_percent,
        resets_at=None,
        is_estimated=False,
    )


def _row(made_at, model_name, eta_calendar, status="OK"):
    return ForecastRow(
        made_at=made_at,
        model_name=model_name,
        eta_calendar=eta_calendar,
        status=status,
    )


def _forecast(made_at, used_percent, *, eta, model_name="montecarlo"):
    """A minimal Forecast for round-tripping through store.insert_forecast."""
    tz = UTC
    return Forecast(
        window=Window(
            provider=Provider.CLAUDE,
            kind=WindowKind.WEEKLY,
            used_percent=used_percent,
            window_minutes=10080,
            resets_at=None,
            source_ts=made_at,
            is_estimated=False,
            source_file="test",
        ),
        status="OK",
        model_name=model_name,
        burn_per_hour=20.0,
        time_to_reset_h=None,
        eta_calendar=datetime.fromtimestamp(eta, tz=tz),
        eta_workhours=None,
        eta_p50=datetime.fromtimestamp(eta, tz=tz),
        eta_p90=None,
        prob_exhaust_before_reset=0.5,
        confidence="high",
        exhausts_before_reset=True,
        n_samples=4,
        burn_basis="tokens",
    )


def test_thin_history_keeps_the_default_however_good_a_challenger_looks():
    """A challenger with a tenth of the default's error still doesn't win on
    three episodes. This is the guard against exactly the failure mode of
    tuning to one person's data and calling the result measured."""
    chosen, reason = better_model(
        "linear",
        {
            "linear": _score("linear", episodes=3, mae=10.0),
            "montecarlo": _score("montecarlo", episodes=3, mae=1.0),
        },
    )
    assert chosen == "linear"
    assert "not enough scored history" in reason
    assert str(MIN_EPISODES_TO_GRADUATE) in reason


def test_a_marginal_win_on_ample_history_still_keeps_the_default():
    """Enough episodes, but the margin is inside the noise floor. Churning
    the model for a few percent isn't an improvement, it's a coin flip with
    extra steps."""
    n = MIN_EPISODES_TO_GRADUATE + 10
    chosen, reason = better_model(
        "linear",
        {
            "linear": _score("linear", episodes=n, mae=10.0),
            "montecarlo": _score("montecarlo", episodes=n, mae=9.5),  # 5% better
        },
    )
    assert chosen == "linear"
    assert "margin" in reason


def test_a_decisive_win_on_ample_history_switches_and_says_why():
    n = MIN_EPISODES_TO_GRADUATE + 10
    chosen, reason = better_model(
        "linear",
        {
            "linear": _score("linear", episodes=n, mae=10.0),
            "montecarlo": _score("montecarlo", episodes=n, mae=4.0),  # 60% better
        },
    )
    assert chosen == "montecarlo"
    assert "60%" in reason and "linear" in reason
    assert 0.6 > MIN_RELATIVE_IMPROVEMENT  # sanity: the margin really was cleared


def test_scoring_ignores_forecasts_belonging_to_other_models():
    """Both models write a forecast every tick, so grading one must not pick
    up the other's rows."""
    now = 1_000_000.0
    samples = [_sample(now, 50.0), _sample(now + HOUR, 100.0)]
    rows = [
        _row(now, "linear", now + 2 * HOUR),  # 1h late
        _row(now, "montecarlo", now + 10 * HOUR),  # 9h late
    ]
    pairs = [(rows, samples, _is_reset)]

    linear = score_model("linear", pairs)
    montecarlo = score_model("montecarlo", pairs)

    assert linear.episodes == montecarlo.episodes == 1
    assert linear.mae_hours is not None and montecarlo.mae_hours is not None
    assert linear.mae_hours < montecarlo.mae_hours


def test_a_forecast_whose_window_never_exhausted_is_not_scored():
    """Only moments where exhaustion actually happened yield an error. A
    window that reset first says nothing about whether the ETA was right."""
    now = 1_000_000.0
    samples = [_sample(now, 50.0), _sample(now + HOUR, 2.0)]  # reset, never hit 100
    assert realized_exhaustion_hours(samples, 0, _is_reset) is None

    score = score_model(
        "linear", [([_row(now, "linear", now + HOUR)], samples, _is_reset)]
    )
    assert score.with_eta == 1  # it did produce an ETA...
    assert score.episodes == 0  # ...but there's nothing to grade it against
    assert score.mae_hours is None


def test_auto_on_a_fresh_store_reports_why_it_stayed_on_the_default(tmp_path, store):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.predictor.model == "auto"

    predictor, reason = select_predictor(cfg, store)

    assert predictor.name == "linear"
    assert "linear" in reason


def test_auto_grades_real_stored_rows(tmp_path, store):
    """End to end through the store: written forecasts come back out, get
    graded, and still don't clear the bar on a handful of episodes.

    Writes forecast rows as well as samples. With samples alone this asserted
    the right answer for the wrong reason — grading nothing also returns the
    default — so it never actually exercised the stored-grading path."""
    cfg = load_config(tmp_path / "config.toml")
    now = 1_000_000.0
    for i in range(4):
        ts = now + i * HOUR
        used = 40.0 + i * 20.0  # reaches 100 on the last one
        store.insert_sample(
            captured_at=ts,
            provider=Provider.CLAUDE,
            window_kind=WindowKind.WEEKLY,
            source_ts=ts,
            used_percent=used,
            window_minutes=10080,
            resets_at=None,
            is_estimated=False,
            source_file="test",
        )
        store.insert_forecast(
            made_at=ts,
            forecast=_forecast(ts, used, eta=ts + 2 * HOUR),
        )

    graded = score_model(
        "montecarlo",
        [
            (
                store.recent_forecasts(Provider.CLAUDE, WindowKind.WEEKLY, 0.0),
                store.recent_samples(Provider.CLAUDE, WindowKind.WEEKLY, 0.0),
                _is_reset,
            )
        ],
    )
    assert graded.with_eta == 4  # the rows really did come back out...
    assert graded.episodes > 0  # ...and really were gradeable

    predictor, reason = select_predictor(cfg, store)
    assert predictor.name == "linear"
    assert "not enough" in reason or "no stored forecasts" in reason


def test_a_moment_already_at_the_cap_is_not_an_episode():
    """There is nothing left to forecast at 100%, so "exhausted now" is not a
    prediction and must not be scored as one. Before the guard, the scan
    started at index+1 and returned the distance to the NEXT 100% sample,
    turning the poll cadence into the measured error."""
    now = 1_000_000.0
    samples = [_sample(now, 100.0), _sample(now + 300.0, 100.0)]

    assert realized_exhaustion_hours(samples, 0, _is_reset) is None


def test_an_unclamped_reading_above_the_cap_is_also_not_an_episode():
    """The token-computed percentage isn't clamped, so it can read past 100;
    the guard is `>=` rather than `== 100` for exactly that reason."""
    now = 1_000_000.0
    samples = [_sample(now, 104.0), _sample(now + 300.0, 100.0)]

    assert realized_exhaustion_hours(samples, 0, _is_reset) is None


def test_sitting_at_the_cap_cannot_mint_episodes_for_one_model():
    """The defect this guard exists for, end to end.

    While a window is pinned at 100%, montecarlo keeps answering "exhausted
    now" every tick while linear falls silent once its lookback clears the
    climb. Each of those answers used to score as an episode worth roughly one
    sample cadence of error, so two hours of saturation minted ~120 of them
    for montecarlo against ~9 for linear — enough to clear the graduation bar
    and hand `auto` to the model that is worse on real episodes, via
    better_model's no-baseline branch."""
    now = 1_000_000.0
    # A pinned-at-100 stretch, sampled every five minutes.
    samples = [_sample(now + i * 300.0, 100.0) for i in range(24)]
    rows = [_row(now + i * 300.0, "montecarlo", now + i * 300.0) for i in range(24)]

    score = score_model("montecarlo", [(rows, samples, _is_reset)])

    assert score.with_eta == 24  # it did answer every tick...
    assert score.episodes == 0  # ...and not one of them was gradeable
    assert score.mae_hours is None

    chosen, reason = better_model(
        "linear",
        {
            "linear": _score("linear", episodes=0, mae=None),
            "montecarlo": score,
        },
    )
    assert chosen == "linear"
    assert "not enough scored history" in reason


def test_a_forecast_is_graded_against_the_reading_it_actually_saw():
    """Matching a forecast to the *nearest* sample could pick one that didn't
    exist yet, grading a prediction against its own future. The origin is the
    newest sample at or before `made_at`."""
    now = 1_000_000.0
    samples = [_sample(now, 50.0), _sample(now + HOUR, 100.0)]
    # Made a minute before the exhaustion sample landed: "nearest" is that
    # future sample, which would make the origin already-exhausted.
    made_at = now + HOUR - 60.0
    rows = [_row(made_at, "linear", now + HOUR)]

    score = score_model("linear", [(rows, samples, _is_reset)])

    assert score.episodes == 1  # graded against the 50% reading, not the 100%
    assert score.mae_hours is not None
    # ETA landed exactly on the real exhaustion instant, so the error is 0 --
    # and would NOT be, if the horizon were measured from `made_at` while
    # truth was measured from the origin sample's own timestamp.
    assert score.mae_hours == 0.0


def test_a_deliberately_withheld_forecast_is_not_a_coverage_miss():
    """A forecast the predictor declined to make (no live reading to
    extrapolate from) isn't the same as one it got wrong — counting IDLE and
    NO_DATA rows as moments-without-an-ETA would penalize a model for
    correctly staying quiet."""
    now = 1_000_000.0
    samples = [_sample(now, 50.0), _sample(now + HOUR, 100.0)]
    rows = [
        _row(now, "linear", None, status="IDLE"),
        _row(now, "linear", None, status="NO_DATA"),
        _row(now, "linear", now + 2 * HOUR, status="OK"),
    ]

    score = score_model("linear", [(rows, samples, _is_reset)])

    assert score.moments == 1  # only the OK row counts
    assert score.with_eta == 1
    assert score.coverage == 1.0
