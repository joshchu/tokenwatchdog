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

from tokenwatchdog.models import WindowKind
from tokenwatchdog.store import ForecastRow, SampleRow

_EXHAUSTED_PERCENT = 100.0

# Exhaustion "episodes" (a below-100% moment whose window then hit 100% in
# the same cycle) stay as a DIAGNOSTIC only: they are far too rare to decide
# anything -- measured at 2 real crossings in 10.5 days, with the 14-17
# scorable moments all pointing at those same two events. The decision metric
# is the dense fixed-horizon score below; no episode count gates anything.

# A challenger has to win by a real margin, not a rounding difference --
# measured: sub-margin orderings between models flip sign between independent
# halves of the same user's history, so anything under this is period noise.
MIN_RELATIVE_IMPROVEMENT = 0.15

# Dense moments a challenger needs before the comparison is decidable. Both
# models forecast every watched window every tick, so pooled OK moments
# accumulate at hundreds per day -- decidable within days rather than never
# (episodes). The real anti-noise guard is the double bar in better_model:
# correlated moments cannot fake a win over PERSISTENCE, a zero-parameter
# baseline that exploits the same autocorrelation in full.
MIN_DENSE_MOMENTS = 500

# The horizon each window kind is scored at: predicted used% at +h versus
# what the store later recorded. 1h fits the 5-hour window's decision ("do I
# stop now"); 24h fits the weekly one ("do I ration today") -- at 1h a weekly
# window's integer percentage mostly measures quantization, not skill.
DENSE_HORIZON_H = {WindowKind.W5H: 1.0, WindowKind.WEEKLY: 24.0}

# The zero-parameter baseline: predicted used% at +h = used% now. Measured
# on 10.5 days of history it beat BOTH models at a 1h horizon on every
# window, so a model that cannot beat it has no claim to be selected.
PERSISTENCE = "persistence"


@dataclass(frozen=True)
class ModelScore:
    model_name: str
    moments: int  # forecasts examined
    answered: int  # ...that gave an ETA or a deliberate risk-only answer
    episodes: int  # ...whose window then actually exhausted, so error is real
    mae_hours: float | None
    bias_hours: float | None

    @property
    def coverage(self) -> float:
        return self.answered / self.moments if self.moments else 0.0


@dataclass(frozen=True)
class DenseScore:
    """Fixed-horizon used% error, plus the persistence baseline measured on
    the SAME moments — the pairing that makes "beats persistence" a matched
    comparison rather than a cross-sample one."""

    model_name: str
    moments: int
    mae_points: float | None
    bias_points: float | None
    persistence_mae_points: float | None


def pool_dense(model_name: str, scores: list[DenseScore]) -> DenseScore:
    """One DenseScore across window kinds scored at different horizons.

    Moment-weighted means — exact, since each MAE is itself a mean of
    absolute errors. Pooling across horizons is deliberate: the unit is
    "used% points of error at that window's own decision horizon," and a
    per-window split never reaches a decidable sample size (the same reason
    score_model pools episodes)."""
    mae = 0.0
    bias = 0.0
    persistence = 0.0
    total = 0
    for score in scores:
        # A scored entry carries all three means or none: they are built
        # from the same per-moment error lists in score_model_dense.
        if (
            not score.moments
            or score.mae_points is None
            or score.bias_points is None
            or score.persistence_mae_points is None
        ):
            continue
        total += score.moments
        mae += score.mae_points * score.moments
        bias += score.bias_points * score.moments
        persistence += score.persistence_mae_points * score.moments
    if not total:
        return DenseScore(model_name, 0, None, None, None)
    return DenseScore(
        model_name=model_name,
        moments=total,
        mae_points=mae / total,
        bias_points=bias / total,
        persistence_mae_points=persistence / total,
    )


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
    its lookback clears the climb) — enough, under the old episode-count
    gate, to promote the model that is markedly WORSE on real episodes. An
    episode has to start below the cap and cross it.

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
    answered = 0
    for rows, samples, is_reset in pairs:
        # Extracted once per window, not per row -- the whole point of the
        # binary search below.
        source_times = [sample.source_ts for sample in samples]
        own = [row for row in rows if row.model_name == model_name]
        # A forecast the predictor deliberately withheld -- no live reading
        # to extrapolate from (IDLE), nothing reported at all (NO_DATA), or
        # a just-turned-over cycle with one reading (RESET_PENDING) -- is
        # not a coverage failure, so it isn't counted as a moment at all.
        # Rows written before `status` existed are NULL and stay counted,
        # since we genuinely don't know which they were.
        own = [
            row for row in own if row.status not in ("IDLE", "NO_DATA", "RESET_PENDING")
        ]
        moments += len(own)
        for row in own:
            # A recorded probability with no ETA is a deliberate answer --
            # "the median future survives; the tail risk is N%" -- not a
            # missing one, so it counts as answered even though there is no
            # ETA error to take.
            if row.eta_calendar is None:
                if row.prob_exhaust_before_reset is not None:
                    answered += 1
                continue
            answered += 1
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
        answered=answered,
        episodes=len(errors),
        mae_hours=statistics.fmean(abs(e) for e in errors) if errors else None,
        bias_hours=statistics.fmean(errors) if errors else None,
    )


def used_percent_at(
    samples: list[SampleRow],
    origin_index: int,
    target_ts: float,
    is_reset: ResetPredicate,
) -> float | None:
    """The window's used% at `target_ts`, or None if that moment isn't
    scorable from this cycle.

    Interpolated linearly between the same-cycle samples straddling the
    target. "Newest sample at or before" was exact only for a fixed window
    observed densely; across a sparse gap it returned a level that was
    hours stale in BOTH directions — a rolling window can decay unobserved
    (a 50% origin that rolled to ~5% before the next sample read as still
    50%, rewarding persistence for the gap), and a fixed one can climb.
    Interpolation uses both observations, and a sample landing exactly at
    the target is simply itself.

    Scorable requires the cycle to demonstrably survive to `target_ts`:
    a reset before it, the origin's own declared `resets_at` passing, or
    the data simply ending all make the question "used% at +h in this
    cycle" unanswerable rather than zero. A reset inside the gap that
    straddles the target stays unanswerable too — the boundary's exact
    moment within the gap is unknown, so the level at the target is as
    well.
    """
    origin = samples[origin_index]
    if origin.resets_at is not None and target_ts > origin.resets_at:
        return None
    previous = origin
    for offset in range(origin_index + 1, len(samples)):
        current = samples[offset]
        if is_reset(previous, current):
            return None
        if current.source_ts >= target_ts:
            span = current.source_ts - previous.source_ts
            if span <= 0:
                return current.used_percent
            fraction = (target_ts - previous.source_ts) / span
            return previous.used_percent + fraction * (
                current.used_percent - previous.used_percent
            )
        previous = current
    return None  # data ends before the horizon -- unknown, not zero


def score_dense(
    model_names: list[str], pairs: list[ScoringPair], horizon_h: float
) -> dict[str, DenseScore]:
    """Fixed-horizon used% error per model, on MATCHED moments: a moment
    counts only when EVERY model has a scorable prediction for that same
    made_at. Scoring each model on its own moment set let a model that
    answers only on easy moments beat a default graded on harder ones —
    measured on real stored history as inflating a challenger's apparent
    edge from 11.6% to 13.2%.

    The prediction graded is the model's own `predicted_used_percent`
    (linear: the rate projection; the simulating model: its mean simulated
    level — where the rhythm knowledge lives). Rows from before that
    column existed fall back to used + burn·h, which was exact for linear
    and approximate for the simulating model. This is the metric that
    makes model selection decidable: exhaustion episodes arrive a couple
    per week, dense moments arrive hundreds per day.

    Persistence (predicted = used% now) is computed on the same matched
    moments and carried in every result, so "beats persistence" is exact.
    """
    errors: dict[str, list[float]] = {name: [] for name in model_names}
    persistence_errors: list[float] = []
    for rows, samples, is_reset in pairs:
        source_times = [sample.source_ts for sample in samples]
        by_moment: dict[float, dict[str, float]] = {}
        for row in rows:
            if row.model_name not in errors or row.status != "OK":
                continue
            if row.used_percent >= _EXHAUSTED_PERCENT:
                continue
            prediction = row.predicted_used_percent
            if prediction is None and row.burn_per_hour is not None:
                prediction = min(
                    100.0, max(0.0, row.used_percent + row.burn_per_hour * horizon_h)
                )
            if prediction is None:
                continue
            by_moment.setdefault(row.made_at, {})[row.model_name] = prediction
        for made_at in sorted(by_moment):
            predictions = by_moment[made_at]
            if len(predictions) != len(model_names):
                continue  # not every model answered here — unmatched
            index = _origin_sample_index(source_times, made_at)
            if index is None:
                continue
            truth = used_percent_at(
                samples, index, made_at + horizon_h * 3600.0, is_reset
            )
            if truth is None:
                continue
            for name, prediction in predictions.items():
                errors[name].append(prediction - truth)
            persistence_errors.append(samples[index].used_percent - truth)
    persistence_mae = (
        statistics.fmean(abs(e) for e in persistence_errors)
        if persistence_errors
        else None
    )
    return {
        name: DenseScore(
            model_name=name,
            moments=len(model_errors),
            mae_points=(
                statistics.fmean(abs(e) for e in model_errors) if model_errors else None
            ),
            bias_points=statistics.fmean(model_errors) if model_errors else None,
            persistence_mae_points=persistence_mae if model_errors else None,
        )
        for name, model_errors in errors.items()
    }


def brier_score(outcomes: list[tuple[float, int]]) -> float | None:
    """Mean squared error of probability claims against 0/1 outcomes —
    lower is better, and a model that can't beat always-guessing-the-base-
    rate (compare against `brier_score([(base, o) for _, o in ...])`) has
    an uninformative probability, however plausible it renders."""
    if not outcomes:
        return None
    return statistics.fmean((prob - outcome) ** 2 for prob, outcome in outcomes)


def reliability_buckets(
    outcomes: list[tuple[float, int]],
) -> list[tuple[float, float, int, float]]:
    """(lo, hi, n, realized frequency) per claimed-probability band — the
    table that shows WHERE a probability is miscalibrated, not just that it
    is. Bands chosen to keep single-user sample sizes readable."""
    bands = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]
    table = []
    for lo, hi in bands:
        hits = [outcome for prob, outcome in outcomes if lo <= prob < hi]
        if hits:
            table.append((lo, hi, len(hits), statistics.fmean(hits)))
    return table


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


def better_model(default: str, dense: dict[str, DenseScore]) -> tuple[str, str]:
    """(chosen model, why) — the model `auto` should trust, and a sentence
    saying how that was decided, because a silent switch is worse than no
    switch.

    Decided on the DENSE metric, never on exhaustion episodes (those stay a
    printed diagnostic — a couple per week can't decide anything). A
    challenger displaces the default only by clearing every bar:

    - at least MIN_DENSE_MOMENTS scored moments of its own;
    - at least MIN_RELATIVE_IMPROVEMENT lower MAE than the default;
    - the same margin over PERSISTENCE measured on its own moments — the
      zero-parameter baseline correlated moments can't fake a win against,
      and the anti-overfitting bar: a model that can't beat "nothing
      changes" on one user's data has learned that user's noise, not their
      usage.

    There is deliberately no you-win-by-default branch when the default has
    no dense score: with no comparison possible, the default stands.
    """
    ranked: list[tuple[float, DenseScore]] = []
    for name, score in dense.items():
        if name == default or name == PERSISTENCE or score.mae_points is None:
            continue
        ranked.append((score.mae_points, score))
    # Explicit key: on an exact MAE tie the tuple comparison would fall
    # through to the (orderless) dataclass and raise.
    ranked.sort(key=lambda pair: pair[0])
    if not ranked:
        return default, f"{default}: no challenger has scored dense moments yet"
    winner_mae, winner = ranked[0]

    if winner.moments < MIN_DENSE_MOMENTS:
        return default, (
            f"{default}: not enough dense history to compare models "
            f"({winner.moments} of {MIN_DENSE_MOMENTS} scored moments)"
        )
    baseline = dense.get(default)
    if baseline is None or baseline.mae_points is None:
        return default, (
            f"{default}: no dense score of its own to compare against "
            f"({winner.model_name} has {winner.moments} moments)"
        )
    if baseline.mae_points == 0.0:
        # A zero-error baseline is unbeatable by margin arithmetic (and the
        # division below would crash the daemon at startup). Flat quota is
        # exactly where this happens: both models predict "no change" and
        # both are right.
        return default, f"{default}: already exact on the dense metric"
    vs_default = (baseline.mae_points - winner_mae) / baseline.mae_points
    if vs_default < MIN_RELATIVE_IMPROVEMENT:
        return default, (
            f"{default}: {winner.model_name} is {vs_default:.0%} better on the "
            f"dense metric ({winner_mae:.2f} vs "
            f"{baseline.mae_points:.2f} pts), under the "
            f"{MIN_RELATIVE_IMPROVEMENT:.0%} margin worth switching for"
        )
    persistence_mae = winner.persistence_mae_points
    if persistence_mae is None:
        return default, f"{default}: no persistence baseline on the same moments"
    if persistence_mae == 0.0:
        # "Nothing changes" was exactly right on these moments; no model
        # can clear a margin over zero.
        return default, (
            f"{default}: persistence is already exact on "
            f"{winner.model_name}'s moments — nothing to beat it by"
        )
    vs_persistence = (persistence_mae - winner_mae) / persistence_mae
    if vs_persistence < MIN_RELATIVE_IMPROVEMENT:
        return default, (
            f"{default}: {winner.model_name} beats {default} but not the "
            f"persistence baseline ({winner_mae:.2f} vs "
            f"{persistence_mae:.2f} pts on its own moments) — a model that "
            f"can't beat 'nothing changes' hasn't earned a switch"
        )
    return winner.model_name, (
        f"{winner.model_name}: {vs_default:.0%} lower dense MAE than {default} "
        f"({winner_mae:.2f} vs {baseline.mae_points:.2f} pts) and "
        f"{vs_persistence:.0%} under persistence, over {winner.moments} moments"
    )
