"""Was a forecast any good — scored against what the store already knows.

Every tick writes every model's forecast to `store.forecasts`, and the same
store holds the samples that say what actually happened next. That is enough
to grade a model retrospectively with no re-simulation: read the forecasts,
read the samples, and check each ETA against the exhaustion it was
predicting.

Two consumers, one definition of "correct" so they can never drift:
`scripts/backtest.py` (which replays predictors from scratch, to test
*changes*) and `predictor.select_predictor` (which grades what already ran,
to decide which model `predictor.model = "auto"` should trust).

The bar for switching models is deliberately high. With one user's history
there is no way to distinguish a real improvement from a lucky fit, so a
model has to beat the default across a real number of scored episodes -- not
by a hair, and not on a handful.
"""

from __future__ import annotations

import bisect
import statistics
from collections.abc import Callable
from dataclasses import dataclass

from tokenwatchdog.store import ForecastRow, SampleRow

_EXHAUSTED_PERCENT = 100.0

# Scored episodes a challenger needs before it may displace the default.
# An "episode" is a moment that was BELOW 100% and whose window then hit it
# in the same cycle, so an error is computable -- and those are RARE: a weekly
# window supplies at most a couple a week no matter how long the tool runs.
# Measured over this repo's first ~6.5 days: 14 episodes for linear and 3 for
# montecarlo on the 5-hour window, 0 on either weekly one. Tuning anything on
# that is fitting noise, which is what this threshold exists to prevent.
#
# Counting caveat, unfixed: this counts forecast MOMENTS, and many correlated
# forecasts made while climbing toward one exhaustion each count separately --
# those 14 point at just 2 real crossings. So 30 is a floor on evidence, not
# on independent events. Grouping by cycle (or a denser, properly independent
# metric) is what would make `auto` genuinely decidable; until then it is
# expected to keep reporting "not enough" and staying on the default.
MIN_EPISODES_TO_GRADUATE = 30

# And it has to win by a real margin, not a rounding difference.
MIN_RELATIVE_IMPROVEMENT = 0.15


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    moments: int  # forecasts examined
    with_eta: int  # ...that produced an ETA at all
    episodes: int  # ...whose window then actually exhausted, so error is real
    mae_hours: float | None
    bias_hours: float | None

    @property
    def coverage(self) -> float:
        return self.with_eta / self.moments if self.moments else 0.0


def realized_exhaustion_hours(
    samples: list[SampleRow], index: int, is_reset
) -> float | None:
    """Hours from `samples[index]` until its window actually hit 100%, or
    None if it never did before the cycle turned over.

    Scoped to the cycle on purpose: exhaustion in a *later* one is not the
    event the forecast was about, and counting it would reward a wildly
    pessimistic model for eventually being right about a different week.

    A moment already AT the cap is not an episode at all: there is nothing
    left to forecast, every model trivially answers "exhausted now," and the
    scan below -- which starts at index + 1 -- would score that against the
    NEXT sample reading 100%, one sample cadence away. That turned the whole
    metric into a measurement of poll cadence: a Monte Carlo MAE of 0.1h that
    reflected no forecasting skill, a P90 that could never contain a truth
    strictly greater than its own 0.0, and, worst of all, a saturated stretch
    quietly minting one scorable "episode" per tick for whichever model still
    answers at the cap. Two hours of sitting at 100% produced 120 such
    episodes for montecarlo against 9 for linear (whose slope goes flat once
    its lookback clears the climb), enough to clear
    MIN_EPISODES_TO_GRADUATE and promote the model that is markedly WORSE on
    real episodes. An episode has to start below the cap and cross it.

    The scan is additionally bounded by the origin sample's own declared
    `resets_at`: an exhaustion after the boundary the forecast was scoped to
    belongs to a different cycle even if no drop was caught in the act. This
    is a no-op for Claude (its samples never carry resets_at) and is what
    makes grading on a ROLLING window (codex weekly) independent of drop
    detection — roll-off drops there aren't resets, so the scan can't rely
    on seeing one.

    `is_reset` is injected rather than imported to keep this module free of
    a circular dependency on predictor.py. It must be the same predicate the
    predictors use -- a threshold of its own here read a 5-hour window as
    taking 54 hours to exhaust, because a cycle starting at 1% drops by only
    1 point when it turns over and the scan sailed straight past it.
    """
    # `>=`, matching the predicates everywhere else: the token-compute
    # percentage is unclamped and can read above 100.
    if samples[index].used_percent >= _EXHAUSTED_PERCENT:
        return None
    cycle_end = samples[index].resets_at
    for offset in range(index + 1, len(samples)):
        if cycle_end is not None and samples[offset].source_ts > cycle_end:
            return None
        if is_reset(samples[offset - 1], samples[offset]):
            return None
        if samples[offset].used_percent >= _EXHAUSTED_PERCENT:
            return (samples[offset].source_ts - samples[index].source_ts) / 3600.0
    return None


ResetPredicate = Callable[[SampleRow, SampleRow], bool]
ScoringPair = tuple[list[ForecastRow], list[SampleRow], ResetPredicate]


def score_model(model_name: str, pairs: list[ScoringPair]) -> ModelScore:
    """Grade one model across every window's stored forecasts.

    `pairs` is one (forecasts, samples, reset-predicate) triple per window —
    the predicate rides with its window because "what counts as a cycle
    boundary" is a per-window fact now that rolling windows exist (see
    predictor._reset_predicate). Errors are pooled across all of them rather
    than scored per window, because exhaustion episodes are the scarce
    resource: split per window, none of them ever reaches a decidable sample
    size.

    Each forecast is matched to the newest sample that existed when it was
    made (see `_origin_sample_index`), and errors are computed between two
    absolute instants rather than two durations, so nothing inherits the lag
    between `made_at` and the reading it was built from.
    """
    errors: list[float] = []
    moments = 0
    with_eta = 0
    for rows, samples, is_reset in pairs:
        # Extracted once per window, not per row -- the whole point of the
        # binary search below.
        source_times = [sample.source_ts for sample in samples]
        own = [row for row in rows if row.model_name == model_name]
        # A forecast the predictor deliberately withheld (no live reading to
        # extrapolate from) is not a coverage failure, so it isn't counted as
        # a moment at all. Rows written before `status` existed are NULL and
        # stay counted, since we genuinely don't know which they were.
        own = [row for row in own if row.status not in ("IDLE", "NO_DATA")]
        moments += len(own)
        for row in own:
            if row.eta_calendar is None:
                continue
            with_eta += 1
            index = _origin_sample_index(source_times, row.made_at)
            if index is None:
                continue
            truth_h = realized_exhaustion_hours(samples, index, is_reset)
            if truth_h is None:
                continue
            # Both sides as absolute instants. Measuring the forecast's
            # horizon from `made_at` while measuring truth from the origin
            # sample's `source_ts` folded the gap between them into every
            # error, which is a lag in when we polled, not a forecasting
            # mistake.
            actual = samples[index].source_ts + truth_h * 3600.0
            errors.append((row.eta_calendar - actual) / 3600.0)
    return ModelScore(
        model_name=model_name,
        moments=moments,
        with_eta=with_eta,
        episodes=len(errors),
        mae_hours=statistics.fmean(abs(e) for e in errors) if errors else None,
        bias_hours=statistics.fmean(errors) if errors else None,
    )


def _origin_sample_index(source_times: list[float], made_at: float) -> int | None:
    """Index of the newest sample at or before `made_at` -- the reading the
    forecast was actually built from -- or None if the forecast predates
    every stored sample.

    Deliberately not "nearest": a forecast made a few seconds before the next
    sample arrived is nearest to a reading it had never seen, which grades it
    against information from its own future. The lag runs one way, so the
    sample at or before `made_at` is the honest origin.

    Binary search, over `source_times` extracted once by the caller. This ran
    at Engine construction across every stored forecast x every stored
    sample: fine at 0.02s on six days of history, but two models writing a
    forecast per window per minute reaches ~161k rows in one retention
    period, where the linear version measured 133s for a single window --
    minutes of startup to conclude "not enough history to compare models."
    """
    position = bisect.bisect_right(source_times, made_at)
    return position - 1 if position else None


def better_model(default: str, scores: dict[str, ModelScore]) -> tuple[str, str]:
    """(chosen model, why) — the model `auto` should trust, and a sentence
    saying how that was decided, because a silent switch is worse than no
    switch.

    Returns `default` unless a challenger has cleared both bars above. Both
    exist because the alternative is picking whichever model happened to
    look better on a couple of episodes of one person's usage, and then
    presenting that as a measurement."""
    ranked = sorted(
        (
            (score.mae_hours, score)
            for score in scores.values()
            if score.mae_hours is not None
            and score.episodes >= MIN_EPISODES_TO_GRADUATE
        ),
        key=lambda pair: pair[0],
    )
    if not ranked:
        available = max((s.episodes for s in scores.values()), default=0)
        return default, (
            f"{default}: not enough scored history to compare models "
            f"({available} exhaustion episode(s), need {MIN_EPISODES_TO_GRADUATE})"
        )
    winner_mae, winner = ranked[0]
    if winner.model_name == default:
        return default, f"{default}: best measured MAE ({winner_mae:.1f}h)"

    baseline = scores.get(default)
    if baseline is None or baseline.mae_hours is None:
        return winner.model_name, (
            f"{winner.model_name}: {winner_mae:.1f}h MAE over {winner.episodes} "
            f"episodes; {default} produced no scorable ETA to compare against"
        )
    improvement = (baseline.mae_hours - winner_mae) / baseline.mae_hours
    if improvement < MIN_RELATIVE_IMPROVEMENT:
        return default, (
            f"{default}: {winner.model_name} is only {improvement:.0%} better "
            f"({winner_mae:.1f}h vs {baseline.mae_hours:.1f}h), under the "
            f"{MIN_RELATIVE_IMPROVEMENT:.0%} margin worth switching for"
        )
    return winner.model_name, (
        f"{winner.model_name}: {improvement:.0%} lower MAE than {default} "
        f"({winner_mae:.1f}h vs {baseline.mae_hours:.1f}h) over "
        f"{winner.episodes} episodes"
    )
