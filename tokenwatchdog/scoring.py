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
from dataclasses import dataclass

from tokenwatchdog.store import ForecastRow, SampleRow

_EXHAUSTED_PERCENT = 100.0

# Scored episodes a challenger needs before it may displace the default.
# An "episode" is a moment whose window genuinely did hit 100% later in the
# same cycle, so an error is computable -- and those are RARE: a weekly
# window supplies at most a couple a week no matter how long the tool runs.
# Measured on this repo's first ~6 days of history: 4-15 episodes for a
# 5-hour window and 0 for a weekly one. Tuning anything on that is fitting
# noise, which is exactly what this threshold exists to prevent.
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

    `is_reset` is injected rather than imported to keep this module free of
    a circular dependency on predictor.py. It must be the same predicate the
    predictors use -- a threshold of its own here read a 5-hour window as
    taking 54 hours to exhaust, because a cycle starting at 1% drops by only
    1 point when it turns over and the scan sailed straight past it.
    """
    for offset in range(index + 1, len(samples)):
        if is_reset(samples[offset - 1], samples[offset]):
            return None
        if samples[offset].used_percent >= _EXHAUSTED_PERCENT:
            return (samples[offset].source_ts - samples[index].source_ts) / 3600.0
    return None


def score_model(
    model_name: str,
    pairs: list[tuple[list[ForecastRow], list[SampleRow]]],
    is_reset,
) -> ModelScore:
    """Grade one model across every window's stored forecasts.

    `pairs` is one (forecasts, samples) pair per window. Errors are pooled
    across all of them rather than scored per window, because exhaustion
    episodes are the scarce resource: split per window, none of them ever
    reaches a decidable sample size.

    Each forecast is matched to the sample nearest in time to when it was
    made — a forecast is built from the newest sample available at the time,
    so its `source_ts` lags `made_at` by up to a poll interval, and matching
    exactly would drop every row to a timestamp mismatch.
    """
    errors: list[float] = []
    moments = 0
    with_eta = 0
    for rows, samples in pairs:
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
            index = _nearest_sample_index(samples, source_times, row.made_at)
            if index is None:
                continue
            truth_h = realized_exhaustion_hours(samples, index, is_reset)
            if truth_h is None:
                continue
            errors.append((row.eta_calendar - row.made_at) / 3600.0 - truth_h)
    return ModelScore(
        model_name=model_name,
        moments=moments,
        with_eta=with_eta,
        episodes=len(errors),
        mae_hours=statistics.fmean(abs(e) for e in errors) if errors else None,
        bias_hours=statistics.fmean(errors) if errors else None,
    )


def _nearest_sample_index(
    samples: list[SampleRow], source_times: list[float], made_at: float
) -> int | None:
    """Index of the sample closest to `made_at`. A forecast is made from the
    newest sample available at the time, so its `source_ts` can lag
    `made_at` by up to a poll interval -- matching on nearest rather than
    exact avoids dropping every row to a timestamp mismatch.

    Binary search, over `source_times` extracted once by the caller. This ran
    at Engine construction across every stored forecast x every stored
    sample: fine at 0.02s on six days of history, but two models writing a
    forecast per window per minute reaches ~161k rows in one retention
    period, where the linear version measured 133s for a single window --
    minutes of startup to conclude "not enough history to compare models."
    """
    if not samples:
        return None
    position = bisect.bisect_left(source_times, made_at)
    if position == 0:
        return 0
    if position == len(source_times):
        return len(source_times) - 1
    before, after = source_times[position - 1], source_times[position]
    return position - 1 if made_at - before <= after - made_at else position


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
