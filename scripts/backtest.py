#!/usr/bin/env python
"""Score the predictors against what actually happened.

The point of persisting history from day one was to make "this model is
better" a measurement rather than an assertion. This replays the stored
`samples` — for each past moment, hiding everything after it, running each
predictor as it would have run then, and comparing its answer to the future
the store already contains.

Four numbers per model, in the order they matter:

- **coverage** — share of moments that produced an ETA at all. A model that
  says nothing is not accurate, it is absent, and this is the metric that
  exposed the original problem (30% for Claude, 0% for Codex).
- **MAE / bias** — hours of error against the real exhaustion time, and
  whether the model runs early or late. Bias is the more actionable of the
  two: a consistently early forecast is a different bug from a noisy one.
- **P90 calibration** — how often the truth actually landed inside the P90
  bound that claims to contain it 90% of the time. Only the simulating
  model reports one.

Usage:
    uv run python scripts/backtest.py
    uv run python scripts/backtest.py --db /tmp/copy.db --stride 5
    uv run python scripts/backtest.py --models linear
"""

from __future__ import annotations

import argparse
import bisect
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenwatchdog.config import DEFAULT_DB_PATH, Config, load_config  # noqa: E402
from tokenwatchdog.models import Provider, Window, WindowKind  # noqa: E402
from tokenwatchdog.predictor import (  # noqa: E402
    _PREDICTORS,
    _reset_predicate,
    _split_into_blocks,
)
from tokenwatchdog.scoring import (  # noqa: E402
    DENSE_HORIZON_H,
    MIN_DENSE_MOMENTS,
    brier_score,
    realized_exhaustion_hours,
    reliability_buckets,
    used_percent_at,
)
from tokenwatchdog.store import SampleRow, Store, TokenEventRow  # noqa: E402


@dataclass
class Score:
    moments: int = 0
    with_eta: int = 0
    errors_h: list[float] | None = None
    p90_checked: int = 0
    p90_contained: int = 0
    dense_errors: list[float] | None = None
    persistence_errors: list[float] | None = None
    risk: list[tuple[float, int]] | None = None  # (claimed prob, 0/1 outcome)

    def __post_init__(self) -> None:
        if self.errors_h is None:
            self.errors_h = []
        if self.dense_errors is None:
            self.dense_errors = []
        if self.persistence_errors is None:
            self.persistence_errors = []
        if self.risk is None:
            self.risk = []


def _truth_hours(samples: list[SampleRow], index: int, is_reset) -> float | None:
    """What actually happened after `samples[index]` — the same definition
    `predictor.select_predictor` grades against, so a change measured here
    can't disagree with the choice made at runtime. The predicate is the
    window's own (rolling windows don't reset on drops)."""
    return realized_exhaustion_hours(samples, index, is_reset)


def _window_at(provider: Provider, kind: WindowKind, sample: SampleRow) -> Window:
    return Window(
        provider=provider,
        kind=kind,
        used_percent=sample.used_percent,
        window_minutes=300 if kind is WindowKind.W5H else 10080,
        resets_at=sample.resets_at,
        source_ts=sample.source_ts,
        is_estimated=sample.is_estimated,
        source_file="backtest",
    )


def _exhausts_before_reset(
    samples: list[SampleRow], index: int, is_reset
) -> int | None:
    """0/1 outcome for the probability claim made at `samples[index]`, or
    None when the data can't settle it. 1 = the window hit 100% in this
    cycle; 0 = the cycle demonstrably ended first (a reset was observed, or
    the origin's own declared resets_at passed within the data); None = the
    data simply ends while the question is still open."""
    if samples[index].used_percent >= 100.0:
        return None  # nothing was being predicted; see realized_exhaustion_hours
    cycle_end = samples[index].resets_at
    for offset in range(index + 1, len(samples)):
        if cycle_end is not None and samples[offset].source_ts > cycle_end:
            return 0
        if is_reset(samples[offset - 1], samples[offset]):
            return 0
        if samples[offset].used_percent >= 100.0:
            return 1
    return None


def _score_pair(
    provider: Provider,
    kind: WindowKind,
    samples: list[SampleRow],
    token_events: list[TokenEventRow],
    cfg: Config,
    model_names: list[str],
    stride: int,
) -> dict[str, Score]:
    scores = {name: Score() for name in model_names}
    is_reset = _reset_predicate(provider, kind)
    horizon_h = DENSE_HORIZON_H[kind]
    # Sliced by bisect, not rebuilt by a linear scan per moment -- the scan
    # was O(moments x events) and dominated the replay's wall clock.
    event_times = [e.ts for e in token_events]
    for index in range(1, len(samples), stride):
        now = samples[index].source_ts
        # Everything the predictor would have had at that moment, and
        # nothing it wouldn't -- the whole exercise is worthless if future
        # data leaks backwards into the inputs.
        history = samples[: index + 1]
        past_events = token_events[: bisect.bisect_right(event_times, now)]
        window = _window_at(provider, kind, samples[index])
        truth_h = _truth_hours(samples, index, is_reset)
        outcome = _exhausts_before_reset(samples, index, is_reset)
        used_at_horizon = used_percent_at(
            samples, index, now + horizon_h * 3600.0, is_reset
        )
        used_now = samples[index].used_percent

        forecasts = {
            name: _PREDICTORS[name].forecast(window, history, past_events, cfg, now)
            for name in model_names
        }
        # Dense scoring is MATCHED: the moment counts only when every model
        # produced a prediction for it — scoring each model on its own
        # moments let one that answers only on easy moments look better
        # than a default graded on harder ones.
        dense_predictions = {
            name: forecast.predicted_used_percent
            for name, forecast in forecasts.items()
            if forecast.status == "OK" and forecast.predicted_used_percent is not None
        }
        dense_scorable = (
            len(dense_predictions) == len(model_names)
            and used_at_horizon is not None
            and used_now < 100.0
        )
        for name in model_names:
            score = scores[name]
            score.moments += 1
            forecast = forecasts[name]
            assert score.dense_errors is not None
            assert score.persistence_errors is not None
            assert score.risk is not None
            if dense_scorable:
                assert used_at_horizon is not None
                score.dense_errors.append(dense_predictions[name] - used_at_horizon)
                score.persistence_errors.append(used_now - used_at_horizon)
            if forecast.prob_exhaust_before_reset is not None and outcome is not None:
                score.risk.append((forecast.prob_exhaust_before_reset, outcome))
            eta = forecast.eta_calendar
            if eta is None:
                continue
            score.with_eta += 1
            if truth_h is None:
                continue
            assert score.errors_h is not None
            score.errors_h.append((eta.timestamp() - now) / 3600.0 - truth_h)
            if forecast.eta_p90 is not None:
                score.p90_checked += 1
                if truth_h <= (forecast.eta_p90.timestamp() - now) / 3600.0:
                    score.p90_contained += 1
    return scores


def _print_scores(label: str, scores: dict[str, Score], horizon_h: float) -> None:
    print(f"\n{label}")
    header = f"  {'model':<12}{'coverage':>10}{'scored':>8}{'MAE h':>9}{'bias h':>9}{'P90 hit':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, score in scores.items():
        if not score.moments:
            continue
        coverage = f"{100.0 * score.with_eta / score.moments:.0f}%"
        errors = score.errors_h or []
        mae = f"{statistics.fmean(abs(e) for e in errors):.1f}" if errors else "—"
        bias = f"{statistics.fmean(errors):+.1f}" if errors else "—"
        p90 = (
            f"{100.0 * score.p90_contained / score.p90_checked:.0f}%"
            if score.p90_checked
            else "—"
        )
        print(f"  {name:<12}{coverage:>10}{len(errors):>8}{mae:>9}{bias:>9}{p90:>10}")

    _print_dense(scores, horizon_h)
    _print_risk(scores)


def _print_dense(scores: dict[str, Score], horizon_h: float) -> None:
    """The decision metric: used% error at this window's own horizon, with
    the persistence baseline on the same moments. This is what `auto`
    switches on — the episode table above stays a diagnostic."""
    rows = [
        (name, score.dense_errors or [])
        for name, score in scores.items()
        if score.dense_errors
    ]
    if not rows:
        return
    print(f"  dense used% at +{horizon_h:g}h ({len(rows[0][1])} moments):")
    for name, errors in rows:
        print(
            f"    {name:<12}MAE {statistics.fmean(abs(e) for e in errors):6.2f} pts"
            f"   bias {statistics.fmean(errors):+6.2f} pts"
        )
    # Persistence is identical for every model on the same moments; print
    # once, from the first row that has it.
    for _, score in scores.items():
        if score.persistence_errors:
            baseline = score.persistence_errors
            print(
                f"    {'persistence':<12}MAE "
                f"{statistics.fmean(abs(e) for e in baseline):6.2f} pts"
                f"   bias {statistics.fmean(baseline):+6.2f} pts"
            )
            break


def _print_risk(scores: dict[str, Score]) -> None:
    """Probability calibration: the Risk column's claims against what
    happened, next to the do-nothing benchmark of always guessing the base
    rate. A Brier above that benchmark means the probability is worse than
    uninformative, however precise it renders."""
    for name, score in scores.items():
        outcomes = score.risk or []
        if not outcomes:
            continue
        brier = brier_score(outcomes)
        base = statistics.fmean(outcome for _, outcome in outcomes)
        base_brier = brier_score([(base, outcome) for _, outcome in outcomes])
        assert brier is not None and base_brier is not None
        print(
            f"  risk ({name}): n={len(outcomes)}, Brier {brier:.3f} "
            f"(always-base-rate {base_brier:.3f}, base rate {base:.2f})"
        )
        for lo, hi, n, realized in reliability_buckets(outcomes):
            print(
                f"    claimed {lo:.1f}-{min(hi, 1.0):.1f}: n={n:<5} "
                f"realized {realized:.2f}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="score every Nth sample (the simulating model is much slower)",
    )
    parser.add_argument(
        "--models",
        default=",".join(_PREDICTORS),
        help=f"comma-separated subset of {','.join(_PREDICTORS)}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Monte Carlo seed (default: 0, so comparisons are reproducible)",
    )
    args = parser.parse_args(argv)
    random.seed(args.seed)

    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = set(model_names) - set(_PREDICTORS)
    if unknown:
        parser.error(f"unknown model(s) {sorted(unknown)}")

    cfg = load_config()
    store = Store(args.db)
    try:
        events_by_provider = {
            provider: store.recent_token_events(provider, 0.0) for provider in Provider
        }
        any_scored = False
        total_dense: dict[str, int] = dict.fromkeys(model_names, 0)
        for provider in Provider:
            for kind in WindowKind:
                samples = store.recent_samples(provider, kind, 0.0)
                if len(samples) < 3:
                    continue
                any_scored = True
                blocks = _split_into_blocks(samples, _reset_predicate(provider, kind))
                scores = _score_pair(
                    provider,
                    kind,
                    samples,
                    events_by_provider[provider],
                    cfg,
                    model_names,
                    max(args.stride, 1),
                )
                for name, score in scores.items():
                    total_dense[name] += len(score.dense_errors or [])
                span_h = (samples[-1].source_ts - samples[0].source_ts) / 3600.0
                _print_scores(
                    f"{provider.value}/{kind.value} — {len(samples)} samples over "
                    f"{span_h:.0f}h, {len(blocks)} block(s), "
                    f"{len(events_by_provider[provider])} token events",
                    scores,
                    DENSE_HORIZON_H[kind],
                )
        if not any_scored:
            print("No window has enough stored history to score yet.")
            return 1
    finally:
        store.close()

    print(
        "\ncoverage = share of moments that produced an ETA · bias < 0 = forecasts "
        "exhaustion earlier than it happened\nscored = moments where the window "
        "actually did exhaust later in the same cycle, so an error is computable"
    )
    _print_verdict(total_dense, max(args.stride, 1))
    return 0


def _print_verdict(dense_by_model: dict[str, int], stride: int) -> None:
    """Whether any of this proves anything yet — judged on the DENSE metric,
    the same one `auto` decides on. Note the replay is strided while the
    runtime gate sees every stored tick, so runtime accumulates moments
    `stride` times faster than this run scored them."""
    best = max(dense_by_model.values(), default=0)
    if best >= MIN_DENSE_MOMENTS:
        print(
            f"\nEnough dense moments ({best}) for a decidable comparison — "
            f'predictor.model = "auto" applies the double bar (beat the default '
            f"AND persistence by 15%) to the full stored history."
        )
        return
    print(
        f"\nNot enough dense moments in this replay to choose between models: "
        f"{best} of the {MIN_DENSE_MOMENTS} needed (stride {stride} scores every "
        f"{stride}th tick).\nDifferences below that bar are as likely to be one "
        f'person\'s luck as a real improvement, so predictor.model = "auto" '
        f"stays on the default."
    )


if __name__ == "__main__":
    sys.exit(main())
