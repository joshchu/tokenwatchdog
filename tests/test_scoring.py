"""Model selection — and, mostly, its refusal to select.

`predictor.model = "auto"` grades every model against this machine's own
stored forecasts. The interesting property is not that it picks a winner but
that it usually declines to: with one user's history, a small MAE difference
is as likely to be that person's luck as a real improvement, and switching on
it would dress up noise as a measurement.
"""

from __future__ import annotations

from tokenwatchdog.config import load_config
from tokenwatchdog.models import Provider, WindowKind
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
    pairs = [(rows, samples)]

    linear = score_model("linear", pairs, _is_reset)
    montecarlo = score_model("montecarlo", pairs, _is_reset)

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
        "linear", [([_row(now, "linear", now + HOUR)], samples)], _is_reset
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
    graded, and still don't clear the bar on a handful of episodes."""
    cfg = load_config(tmp_path / "config.toml")
    now = 1_000_000.0
    for i in range(4):
        ts = now + i * HOUR
        store.insert_sample(
            captured_at=ts,
            provider=Provider.CLAUDE,
            window_kind=WindowKind.WEEKLY,
            source_ts=ts,
            used_percent=40.0 + i * 20.0,  # reaches 100 on the last one
            window_minutes=10080,
            resets_at=None,
            is_estimated=False,
            source_file="test",
        )

    predictor, reason = select_predictor(cfg, store)
    assert predictor.name == "linear"
    assert "not enough" in reason or "no stored forecasts" in reason


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

    score = score_model("linear", [(rows, samples)], _is_reset)

    assert score.moments == 1  # only the OK row counts
    assert score.with_eta == 1
    assert score.coverage == 1.0
