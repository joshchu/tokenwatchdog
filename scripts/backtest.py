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
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenwatchdog.config import DEFAULT_DB_PATH, Config, load_config  # noqa: E402
from tokenwatchdog.models import Provider, Window, WindowKind  # noqa: E402
from tokenwatchdog.predictor import (  # noqa: E402
    _PREDICTORS,
    _is_reset,
    _split_into_blocks,
)
from tokenwatchdog.scoring import (  # noqa: E402
    MIN_EPISODES_TO_GRADUATE,
    realized_exhaustion_hours,
)
from tokenwatchdog.store import SampleRow, Store, TokenEventRow  # noqa: E402


@dataclass
class Score:
    moments: int = 0
    with_eta: int = 0
    errors_h: list[float] | None = None
    p90_checked: int = 0
    p90_contained: int = 0

    def __post_init__(self) -> None:
        if self.errors_h is None:
            self.errors_h = []


def _truth_hours(samples: list[SampleRow], index: int) -> float | None:
    """What actually happened after `samples[index]` — the same definition
    `predictor.select_predictor` grades against, so a change measured here
    can't disagree with the choice made at runtime."""
    return realized_exhaustion_hours(samples, index, _is_reset)


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
    for index in range(1, len(samples), stride):
        now = samples[index].source_ts
        # Everything the predictor would have had at that moment, and
        # nothing it wouldn't -- the whole exercise is worthless if future
        # data leaks backwards into the inputs.
        history = samples[: index + 1]
        past_events = [e for e in token_events if e.ts <= now]
        window = _window_at(provider, kind, samples[index])
        truth_h = _truth_hours(samples, index)

        for name in model_names:
            score = scores[name]
            score.moments += 1
            forecast = _PREDICTORS[name].forecast(
                window, history, past_events, cfg, now
            )
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


def _print_scores(label: str, scores: dict[str, Score]) -> None:
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
    args = parser.parse_args(argv)

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
        total_episodes: dict[str, int] = dict.fromkeys(model_names, 0)
        for provider in Provider:
            for kind in WindowKind:
                samples = store.recent_samples(provider, kind, 0.0)
                if len(samples) < 3:
                    continue
                any_scored = True
                blocks = _split_into_blocks(samples)
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
                    total_episodes[name] += len(score.errors_h or [])
                span_h = (samples[-1].source_ts - samples[0].source_ts) / 3600.0
                _print_scores(
                    f"{provider.value}/{kind.value} — {len(samples)} samples over "
                    f"{span_h:.0f}h, {len(blocks)} block(s), "
                    f"{len(events_by_provider[provider])} token events",
                    scores,
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
    _print_verdict(total_episodes)
    return 0


def _print_verdict(episodes_by_model: dict[str, int]) -> None:
    """Whether any of this proves anything yet.

    Printed because the honest answer is usually "no": scoring needs the
    window to have actually reached 100% later in the same cycle, and a
    weekly window supplies at most a couple of those a week however long
    the tool runs. Without this line it's far too easy to read a
    two-episode MAE difference as a result."""
    best = max(episodes_by_model.values(), default=0)
    if best >= MIN_EPISODES_TO_GRADUATE:
        print(
            f"\nEnough scored episodes ({best}) to compare models — "
            f'predictor.model = "auto" will act on this.'
        )
        return
    print(
        f"\nNot enough scored episodes yet to choose between models: {best} of the "
        f"{MIN_EPISODES_TO_GRADUATE} needed.\nMAE differences below that are as "
        f"likely to be one person's luck as a real improvement, so "
        f'predictor.model = "auto"\nstays on the default. Tuning a constant '
        f"against this sample would be fitting noise."
    )


if __name__ == "__main__":
    sys.exit(main())
