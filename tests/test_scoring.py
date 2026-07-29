"""Model selection — and, mostly, its refusal to select.

`predictor.model = "auto"` grades every model against this machine's own
stored forecasts. The interesting property is not that it picks a winner but
that it usually declines to: with one user's history, a small difference is
as likely to be that person's luck as a real improvement, and switching on
it would dress up noise as a measurement.

The decision runs on the DENSE metric — used% error at a fixed horizon,
hundreds of scorable moments per day — with a double bar: beat the default
AND beat persistence ("nothing changes"), each by a real margin. Exhaustion
episodes (a below-cap moment whose window then hit 100%) stay as a printed
diagnostic; at a couple of real crossings per week they can't decide
anything.
"""

from __future__ import annotations

from datetime import UTC, datetime

from tokenwatchdog.config import load_config
from tokenwatchdog.models import Forecast, Provider, Window, WindowKind
from tokenwatchdog.predictor import _is_reset, select_predictor
from tokenwatchdog.scoring import (
    MIN_DENSE_MOMENTS,
    MIN_RELATIVE_IMPROVEMENT,
    DenseScore,
    better_model,
    pool_dense,
    realized_exhaustion_hours,
    score_model,
    score_model_dense,
    used_percent_at,
)
from tokenwatchdog.store import ForecastRow, SampleRow

HOUR = 3600.0


def _dense(name, *, moments, mae, bias=0.0, persistence=None):
    return DenseScore(
        model_name=name,
        moments=moments,
        mae_points=mae,
        bias_points=bias,
        persistence_mae_points=persistence,
    )


def _sample(ts, used_percent, resets_at=None):
    return SampleRow(
        captured_at=ts,
        source_ts=ts,
        used_percent=used_percent,
        resets_at=resets_at,
        is_estimated=False,
    )


def _row(
    made_at,
    model_name,
    eta_calendar,
    status="OK",
    used_percent=50.0,
    burn_per_hour=None,
    prob=None,
):
    return ForecastRow(
        made_at=made_at,
        model_name=model_name,
        eta_calendar=eta_calendar,
        status=status,
        used_percent=used_percent,
        burn_per_hour=burn_per_hour,
        prob_exhaust_before_reset=prob,
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


# -- the gate: better_model on dense scores ----------------------------------


def test_thin_dense_history_keeps_the_default_however_good_a_challenger_looks():
    """A challenger with a tenth of the default's error still doesn't win on
    a hundred moments. This is the guard against tuning to one person's
    data and calling the result measured."""
    chosen, reason = better_model(
        "linear",
        {
            "linear": _dense("linear", moments=800, mae=1.0),
            "montecarlo": _dense("montecarlo", moments=100, mae=0.1, persistence=1.0),
        },
    )
    assert chosen == "linear"
    assert "not enough dense history" in reason
    assert str(MIN_DENSE_MOMENTS) in reason


def test_beating_the_default_but_not_persistence_keeps_the_default():
    """The load-bearing case. Measured on real history, persistence beat
    both models at a 1h horizon — so a challenger that merely edges out the
    default while losing to "nothing changes" has learned noise, not usage,
    and doesn't get to alert anyone."""
    chosen, reason = better_model(
        "linear",
        {
            "linear": _dense("linear", moments=900, mae=1.0),
            "montecarlo": _dense(
                "montecarlo",
                moments=900,
                mae=0.8,  # 20% better than the default...
                persistence=0.82,  # ...but persistence already does 0.82
            ),
        },
    )
    assert chosen == "linear"
    assert "persistence" in reason


def test_a_marginal_win_on_ample_history_still_keeps_the_default():
    """Enough moments, but the margin is inside the noise floor — measured:
    sub-margin orderings flip sign between independent halves of the same
    user's history."""
    chosen, reason = better_model(
        "linear",
        {
            "linear": _dense("linear", moments=900, mae=1.0),
            "montecarlo": _dense("montecarlo", moments=900, mae=0.95, persistence=2.0),
        },
    )
    assert chosen == "linear"
    assert "margin" in reason
    assert 0.05 < MIN_RELATIVE_IMPROVEMENT  # sanity: 5% really is sub-margin


def test_a_decisive_win_over_both_bars_switches_and_says_why():
    chosen, reason = better_model(
        "linear",
        {
            "linear": _dense("linear", moments=900, mae=1.0),
            "montecarlo": _dense("montecarlo", moments=900, mae=0.5, persistence=1.0),
        },
    )
    assert chosen == "montecarlo"
    assert "50%" in reason and "persistence" in reason


def test_a_default_with_no_dense_score_is_not_displaced():
    """No you-win-by-default branch: with nothing to compare against, the
    default stands. The old episode gate's no-baseline branch was exactly
    how junk episodes could promote the worse model."""
    chosen, reason = better_model(
        "linear",
        {
            "montecarlo": _dense("montecarlo", moments=900, mae=0.1, persistence=1.0),
        },
    )
    assert chosen == "linear"
    assert "no dense score of its own" in reason


def test_pooling_weights_by_moments():
    pooled = pool_dense(
        "montecarlo",
        [
            _dense("montecarlo", moments=100, mae=1.0, bias=1.0, persistence=2.0),
            _dense("montecarlo", moments=300, mae=3.0, bias=-1.0, persistence=4.0),
        ],
    )
    assert pooled.moments == 400
    assert pooled.mae_points == 2.5
    assert pooled.bias_points == -0.5
    assert pooled.persistence_mae_points == 3.5


# -- the dense metric itself --------------------------------------------------


def test_dense_scores_predicted_used_percent_at_the_horizon():
    now = 1_000_000.0
    samples = [
        _sample(now, 50.0),
        _sample(now + HOUR, 60.0),
        _sample(now + 2 * HOUR, 61.0),
    ]
    rows = [_row(now, "linear", None, used_percent=50.0, burn_per_hour=8.0)]

    score = score_model_dense("linear", [(rows, samples, _is_reset)], 1.0)

    assert score.moments == 1
    assert score.mae_points == 2.0  # predicted 58, truth 60
    assert score.bias_points == -2.0
    assert score.persistence_mae_points == 10.0  # "nothing changes" said 50


def test_dense_clamps_predictions_at_the_cap():
    now = 1_000_000.0
    samples = [
        _sample(now, 90.0),
        _sample(now + HOUR, 95.0),
        _sample(now + 2 * HOUR, 96.0),
    ]
    rows = [_row(now, "linear", None, used_percent=90.0, burn_per_hour=50.0)]

    score = score_model_dense("linear", [(rows, samples, _is_reset)], 1.0)

    assert score.mae_points == 5.0  # clamped to 100, truth 95


def test_dense_skips_moments_whose_cycle_does_not_reach_the_horizon():
    """A reset inside the horizon — or the origin's own declared resets_at
    passing — makes "used% at +h in this cycle" unanswerable, not zero."""
    now = 1_000_000.0
    rows = [_row(now, "linear", None, burn_per_hour=8.0)]

    reset_inside = [
        _sample(now, 50.0),
        _sample(now + 0.5 * HOUR, 2.0),  # reset before the horizon
        _sample(now + 2 * HOUR, 10.0),
    ]
    score = score_model_dense("linear", [(rows, reset_inside, _is_reset)], 1.0)
    assert score.moments == 0

    declared_end_inside = [
        _sample(now, 50.0, resets_at=now + 0.5 * HOUR),
        _sample(now + 2 * HOUR, 60.0, resets_at=now + 0.5 * HOUR),
    ]
    score = score_model_dense("linear", [(rows, declared_end_inside, _is_reset)], 1.0)
    assert score.moments == 0


def test_dense_truth_is_the_newest_sample_at_or_before_the_horizon():
    """The level is durable between samples, so truth at +1h is the newest
    reading at or before it — not an interpolation toward a later one."""
    now = 1_000_000.0
    samples = [
        _sample(now, 50.0),
        _sample(now + 0.5 * HOUR, 55.0),
        _sample(now + 2 * HOUR, 99.0),
    ]
    assert used_percent_at(samples, 0, now + HOUR, _is_reset) == 55.0

    rows = [_row(now, "linear", None, used_percent=50.0, burn_per_hour=5.0)]
    score = score_model_dense("linear", [(rows, samples, _is_reset)], 1.0)
    assert score.moments == 1
    assert score.mae_points == 0.0  # predicted 55, truth 55


def test_dense_skips_rows_that_cannot_be_scored():
    """No burn recorded (pre-migration NULL), already at the cap, or not an
    OK answer — none of these are moments the model can be graded on."""
    now = 1_000_000.0
    samples = [
        _sample(now, 50.0),
        _sample(now + HOUR, 60.0),
        _sample(now + 2 * HOUR, 61.0),
    ]
    rows = [
        _row(now, "linear", None, burn_per_hour=None),
        _row(now, "linear", None, used_percent=100.0, burn_per_hour=1.0),
        _row(now, "linear", None, status="IDLE", burn_per_hour=1.0),
        _row(now, "linear", None, status=None, burn_per_hour=1.0),
    ]

    assert score_model_dense("linear", [(rows, samples, _is_reset)], 1.0).moments == 0


# -- episode scoring (the diagnostic) -----------------------------------------


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
    assert score.answered == 1  # it did produce an ETA...
    assert score.episodes == 0  # ...but there's nothing to grade it against
    assert score.mae_hours is None


def test_auto_on_a_fresh_store_reports_why_it_stayed_on_the_default(tmp_path, store):
    cfg = load_config(tmp_path / "config.toml")
    assert cfg.predictor.model == "auto"

    predictor, reason = select_predictor(cfg, store)

    assert predictor.name == "linear"
    assert "linear" in reason


def test_auto_grades_real_stored_rows(tmp_path, store):
    """End to end through the store: written forecasts come back out — with
    the fields the dense metric reads — get graded, and still don't clear
    the bar on a handful of moments."""
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

    rows = store.recent_forecasts(Provider.CLAUDE, WindowKind.WEEKLY, 0.0)
    assert rows[0].used_percent == 40.0  # the dense metric's inputs round-trip
    assert rows[0].burn_per_hour == 20.0
    assert rows[0].prob_exhaust_before_reset == 0.5

    graded = score_model(
        "montecarlo",
        [
            (
                rows,
                store.recent_samples(Provider.CLAUDE, WindowKind.WEEKLY, 0.0),
                _is_reset,
            )
        ],
    )
    assert graded.answered == 4  # the rows really did come back out...
    assert graded.episodes > 0  # ...and really were gradeable

    predictor, reason = select_predictor(cfg, store)
    assert predictor.name == "linear"


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
    """While a window is pinned at 100%, montecarlo keeps answering
    "exhausted now" every tick. Each of those answers used to score as an
    episode worth roughly one sample cadence of error — ~120 in two hours —
    and the old episode-based gate then promoted the model that is worse on
    real episodes via its no-baseline branch. The at-cap guard keeps them
    unscorable, and the gate now decides on dense moments, never episodes."""
    now = 1_000_000.0
    # A pinned-at-100 stretch, sampled every five minutes; each row records
    # the 100% it was made from, as real rows do.
    samples = [_sample(now + i * 300.0, 100.0) for i in range(24)]
    rows = [
        _row(
            now + i * 300.0,
            "montecarlo",
            now + i * 300.0,
            used_percent=100.0,
            burn_per_hour=0.0,
        )
        for i in range(24)
    ]

    score = score_model("montecarlo", [(rows, samples, _is_reset)])

    assert score.answered == 24  # it did answer every tick...
    assert score.episodes == 0  # ...and not one of them was gradeable
    assert score.mae_hours is None

    # Nor can those moments feed the dense metric: at the cap there is
    # nothing left to predict, so saturation mints nothing anywhere.
    dense = score_model_dense("montecarlo", [(rows, samples, _is_reset)], 1.0)
    assert dense.moments == 0


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
    """A forecast the predictor declined to make — no live reading (IDLE),
    nothing reported (NO_DATA), a just-turned-over cycle (RESET_PENDING) —
    isn't the same as one it got wrong. And a risk-only answer (censored
    median, known tail probability) is an answer, not a miss."""
    now = 1_000_000.0
    samples = [_sample(now, 50.0), _sample(now + HOUR, 100.0)]
    rows = [
        _row(now, "linear", None, status="IDLE"),
        _row(now, "linear", None, status="NO_DATA"),
        _row(now, "linear", None, status="RESET_PENDING"),
        _row(now, "linear", None, status="OK", prob=0.2),  # risk-only answer
        _row(now, "linear", now + 2 * HOUR, status="OK"),
    ]

    score = score_model("linear", [(rows, samples, _is_reset)])

    assert score.moments == 2  # only the two OK rows count
    assert score.answered == 2  # ...and both answered, one via risk alone
    assert score.coverage == 1.0
