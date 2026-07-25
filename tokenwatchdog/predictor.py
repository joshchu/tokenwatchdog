"""Exhaustion prediction — tiered, swappable, backtestable.

Two models: `linear` (robust trailing-slope fit, the cold-start default —
sane from the very first samples) and `montecarlo` (a learned hour-of-week
burn profile, simulated forward to a P50/P90 exhaustion band once enough
history exists to populate it). Both share resets_at derivation and the
working-hours projector via the same `Predictor` protocol, so a front-end
or alerts.py never has to know which one produced a given Forecast.

Both measure burn from **token throughput** when they can, not from the
slope of the reported percentage. Both providers quantize that percentage to
whole numbers, which is fine for a 5-hour window moving ~20%/h and useless
for a 7-day window moving ~0.6%/h: over any short lookback the integer
simply doesn't change, and the honest slope of an unchanging series is zero.
`token_events` has no such floor — and a calibration against the
authoritative percentage over the whole block converts tokens/h into %/h
without needing to know the provider's real token cap.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, tzinfo
from typing import Protocol

from tokenwatchdog.blocks import block_anchor
from tokenwatchdog.config import (
    Config,
    ConfigError,
    WorkingHoursConfig,
    resolve_timezone,
)
from tokenwatchdog.models import (
    BurnBasis,
    Confidence,
    Forecast,
    ForecastStatus,
    Provider,
    Window,
    WindowKind,
    window_duration_seconds,
)
from tokenwatchdog.scoring import better_model, score_model
from tokenwatchdog.store import SampleRow, Store, TokenEventRow


class Predictor(Protocol):
    name: str

    def forecast(
        self,
        window: Window,
        history: list[SampleRow],
        token_events: list[TokenEventRow],
        cfg: Config,
        now: float,
    ) -> Forecast:
        """`history` is this (provider, kind)'s recent samples, oldest
        first. `token_events` is the same provider's per-request token
        usage, oldest first — the high-resolution burn signal both models
        prefer over differencing `history`'s quantized percentages."""
        ...


class LinearPredictor:
    name = "linear"

    def forecast(
        self,
        window: Window,
        history: list[SampleRow],
        token_events: list[TokenEventRow],
        cfg: Config,
        now: float,
    ) -> Forecast:
        tz = resolve_timezone(cfg)
        block = _current_block(history)
        resets_at = window.resets_at
        if resets_at is None:
            resets_at = _derive_resets_at(window, block, token_events, now)
            if resets_at is not None:
                # Thread the derived value back onto the window so it rides
                # along on Forecast.window — alerts.py re-arms on resets_at
                # advancing, which never happens if this stays None forever
                # (as it would for every Claude reading straight from the
                # provider).
                window = replace(window, resets_at=resets_at)

        if _is_rate_stale(window, cfg, now):
            return _status_only_forecast(
                window,
                self.name,
                status="IDLE",
                n_samples=len(block.samples),
                now=now,
                resets_at=resets_at,
            )

        if block.block_started_at is not None and len(block.samples) < 2:
            return _status_only_forecast(
                window,
                self.name,
                status="RESET_PENDING",
                n_samples=len(block.samples),
                now=now,
                resets_at=resets_at,
            )

        if len(block.samples) < 2:
            return _status_only_forecast(
                window,
                self.name,
                status="OK",
                n_samples=len(block.samples),
                now=now,
                resets_at=resets_at,
            )

        fit_samples = _slope_fit_samples(block.samples, cfg, window.kind, now)
        percent_burn = _robust_slope_per_hour(fit_samples)
        token_burn = _token_burn_per_hour(block.samples, token_events, cfg, window, now)

        if token_burn is not None and (
            token_burn.burn_per_hour > 0 or percent_burn <= 0
        ):
            burn_per_hour = token_burn.burn_per_hour
            basis: BurnBasis = "tokens"
            n_samples = token_burn.n_events
        else:
            # Either nothing calibrated, or the token log claims zero burn
            # while the authoritative percentage says otherwise. The latter
            # is real: an account-wide percentage also counts usage that
            # never lands in this machine's token log (Claude Desktop and
            # agent mode burn the same quota without writing a CLI
            # transcript). Reporting a confident zero there would be worse
            # than a coarse slope, so the coarse slope wins that tie.
            burn_per_hour = percent_burn
            basis = "percent"
            n_samples = len(fit_samples)

        return _project_forecast(
            window=window,
            burn_per_hour=burn_per_hour,
            resets_at=resets_at,
            n_samples=n_samples,
            cfg=cfg,
            now=now,
            tz=tz,
            model_name=self.name,
            burn_basis=basis,
        )


class MonteCarloPredictor:
    """Learns a burn-rate distribution per hour-of-week bucket from every
    stored past block (not just the live one), then simulates many random
    futures — sampling a burn from the appropriate bucket hour by hour —
    to get a distribution of exhaustion times instead of one point
    estimate. Needs real history to say anything a flat average couldn't;
    with none, it degrades to reporting the plain mean observed burn via
    the same `_project_forecast` linear uses."""

    name = "montecarlo"

    def forecast(
        self,
        window: Window,
        history: list[SampleRow],
        token_events: list[TokenEventRow],
        cfg: Config,
        now: float,
    ) -> Forecast:
        tz = resolve_timezone(cfg)
        block = _current_block(history)
        resets_at = window.resets_at
        if resets_at is None:
            resets_at = _derive_resets_at(window, block, token_events, now)
            if resets_at is not None:
                window = replace(window, resets_at=resets_at)

        if _is_rate_stale(window, cfg, now):
            return _status_only_forecast(
                window,
                self.name,
                status="IDLE",
                n_samples=len(block.samples),
                now=now,
                resets_at=resets_at,
            )
        if block.block_started_at is not None and len(block.samples) < 2:
            return _status_only_forecast(
                window,
                self.name,
                status="RESET_PENDING",
                n_samples=len(block.samples),
                now=now,
                resets_at=resets_at,
            )

        percent_per_token = _percent_per_token(block.samples, token_events)
        raw_buckets = _build_burn_buckets(
            history, token_events, percent_per_token, tz, now
        )
        basis: BurnBasis = "tokens" if percent_per_token else "percent"
        n_observations = sum(len(entries) for entries in raw_buckets.values())
        if n_observations == 0:
            return _status_only_forecast(
                window,
                self.name,
                status="OK",
                n_samples=len(block.samples),
                now=now,
                resets_at=resets_at,
            )

        halflife_days = cfg.predictor.bucket_decay_halflife_days
        weighted = {
            bucket: _weighted_pool(entries, now, halflife_days)
            for bucket, entries in raw_buckets.items()
        }
        all_values = [v for values, _ in weighted.values() for v in values]
        all_weights = [w for _, weights in weighted.values() for w in weights]
        # Decay-weighted, matching how every draw below is weighted -- a
        # plain mean here would quietly disagree with the simulation it's
        # reported alongside.
        mean_burn = _weighted_mean(all_values, all_weights)
        # Cumulative weights, built ONCE per bucket. random.choices()
        # recomputes them on every single call when handed raw weights,
        # which is O(bucket size) per draw against ~mc_runs * horizon draws
        # per tick -- measured at 2.4us/draw for a 100-entry bucket versus
        # 0.47us with cum_weights, and 21us versus 0.57us at 1000.
        buckets = {b: _to_cumulative(pool) for b, pool in weighted.items()}
        fallback_pool = _to_cumulative((all_values, all_weights))

        now_dt = datetime.fromtimestamp(now, tz=tz)
        time_to_reset_h = (resets_at - now) / 3600.0 if resets_at is not None else None
        # Exhaustion is a WITHIN-THIS-CYCLE question: the window refills at
        # its own reset, so a simulated future that runs past it isn't "on
        # pace to exhaust," it's a new cycle starting fresh. Simulations
        # therefore run only up to the reset when it's known — never
        # beyond it — so a low-but-nonzero burn with a reset coming soon
        # correctly reports "not at risk" instead of an ETA far past the
        # point where the quota will have already refilled. Without a
        # known reset (cold start), fall back to a 2-week cap so we're not
        # simulating indefinitely.
        horizon_h = time_to_reset_h if time_to_reset_h is not None else 24.0 * 14

        mc_runs = max(cfg.predictor.mc_runs, 1)
        # Every run contributes exactly one outcome, exhausted or not — a
        # run that never reaches 100% is censored AT the horizon, not
        # dropped. Percentiles below are computed over all `mc_runs`
        # outcomes: dropping non-exhausting runs first would make "P50"
        # the median of whichever minority actually exhausted, which is
        # a materially earlier (and wrong) number whenever most simulated
        # futures don't exhaust within the horizon at all.
        # `is None` rather than `or`: an already-exhausted window simulates to
        # 0.0 hours, which is falsy, so `or horizon_h` rewrote every run as
        # censored and reported a 0% chance of exhausting on a window that is
        # ALREADY exhausted -- rendered as "Risk 0%" on a red 100% row.
        simulated = (
            _simulate_exhaustion_hours(
                window.used_percent, now, buckets, fallback_pool, tz, horizon_h
            )
            for _ in range(mc_runs)
        )
        outcomes = [horizon_h if hours is None else hours for hours in simulated]
        outcomes.sort()
        # Because the horizon IS the reset (when known), a non-censored
        # outcome already means "exhausted before the reset" -- there's
        # nothing past the horizon to have compared against.
        prob_before_reset = (
            sum(1 for h in outcomes if h < horizon_h) / mc_runs
            if time_to_reset_h is not None
            else None
        )

        p50_h = _percentile_within_horizon(outcomes, 0.5, horizon_h)
        p90_h = _percentile_within_horizon(outcomes, 0.9, horizon_h)
        confidence = _confidence(
            n_observations,
            coverage=_horizon_bucket_coverage(now, horizon_h, buckets, tz),
        )
        if p50_h is None:
            # More than half the simulated futures never exhausted within the
            # horizon, so there is no point ETA to report -- and, critically,
            # none is synthesized. Falling back to `remaining / mean_burn`
            # here (as this did) contradicts the simulation that just ran:
            # mean_burn is an average over the whole week including the idle
            # hours, so for burn concentrated OUTSIDE the hours before the
            # reset it yields a confident ETA and exhausts_before_reset=True
            # at a simulated 0% risk -- a "🔥 burning" row the model itself
            # disagrees with, and a prediction that then gets graded as
            # though the model had made it.
            #
            # The rate and the probability are still reported. This is
            # precisely where the probability carries the most information:
            # "the median future is fine, but one in five isn't" is the whole
            # reason to simulate rather than extrapolate.
            return Forecast(
                window=window,
                status="OK",
                model_name=self.name,
                burn_per_hour=mean_burn,
                time_to_reset_h=time_to_reset_h,
                eta_calendar=None,
                eta_workhours=None,
                eta_p50=None,
                eta_p90=None,
                prob_exhaust_before_reset=prob_before_reset,
                confidence=confidence,
                exhausts_before_reset=False,
                n_samples=n_observations,
                burn_basis=basis,
            )

        horizon = _eta_horizon(window, resets_at, now_dt, tz)
        eta_p50, eta_p90 = _cap_at_horizon(
            now_dt + timedelta(hours=p50_h),
            now_dt + timedelta(hours=p90_h) if p90_h is not None else None,
            horizon,
        )
        eta_workhours, _ = _cap_at_horizon(
            project_workhours_exhaustion(now_dt, p50_h, cfg.working_hours),
            None,
            horizon,
        )

        return Forecast(
            window=window,
            status="OK",
            model_name=self.name,
            burn_per_hour=mean_burn,
            time_to_reset_h=time_to_reset_h,
            eta_calendar=eta_p50,  # point estimate = the median simulated outcome
            eta_workhours=eta_workhours,
            eta_p50=eta_p50,
            eta_p90=eta_p90,
            prob_exhaust_before_reset=prob_before_reset,
            confidence=confidence,
            # The simulation horizon IS the reset, so any uncensored p50 is
            # already before it -- see the horizon_h comment above.
            exhausts_before_reset=eta_p50 is not None and time_to_reset_h is not None,
            n_samples=n_observations,
            burn_basis=basis,
        )


_PREDICTORS: dict[str, Predictor] = {
    "linear": LinearPredictor(),
    "montecarlo": MonteCarloPredictor(),
}


_DEFAULT_MODEL = "linear"


def all_predictor_names() -> list[str]:
    return list(_PREDICTORS)


def predictor_named(name: str) -> Predictor:
    return _PREDICTORS[name]


def select_predictor(cfg: Config, store: Store | None = None) -> tuple[Predictor, str]:
    """(the authoritative predictor, why it was chosen).

    `auto` grades each model against this machine's own stored forecasts and
    picks the one that measured best — a real comparison on real usage, not
    a history-volume threshold standing in for one. It keeps the default
    unless a challenger clears both bars in scoring.py, because with a
    single user's history the difference between "better model" and "lucky
    fit" is exactly what a small sample can't show. The reason string is
    returned rather than swallowed: a silently switched model is worse than
    an unswitched one.
    """
    if cfg.predictor.model != "auto":
        predictor = _PREDICTORS.get(cfg.predictor.model)
        if predictor is None:
            raise ConfigError(
                f"predictor.model={cfg.predictor.model!r} isn't implemented yet "
                f"(available: {sorted(_PREDICTORS)}) — seasonal_profile/holtwinters "
                f"are a planned future addition, not built yet"
            )
        return predictor, f"{cfg.predictor.model}: set explicitly in config"
    if store is None:
        return _PREDICTORS[_DEFAULT_MODEL], f"{_DEFAULT_MODEL}: no history to grade"
    chosen, reason = _grade_stored_models(cfg, store)
    return _PREDICTORS[chosen], reason


def _grade_stored_models(cfg: Config, store: Store) -> tuple[str, str]:
    """Grade every model against this machine's own stored forecasts."""
    del cfg  # retention already bounds what's in the table
    pairs = []
    for provider in Provider:
        for kind in WindowKind:
            samples = store.recent_samples(provider, kind, 0.0)
            rows = store.recent_forecasts(provider, kind, 0.0)
            if len(samples) >= 3 and rows:
                pairs.append((rows, samples))
    if not pairs:
        return _DEFAULT_MODEL, f"{_DEFAULT_MODEL}: no stored forecasts to grade yet"
    scores = {name: score_model(name, pairs, _is_reset) for name in _PREDICTORS}
    return better_model(_DEFAULT_MODEL, scores)


# -- shared preprocessing: the "clean burn series" ---------------------------


@dataclass(frozen=True)
class _BlockView:
    samples: list[SampleRow]  # current-block samples only, oldest first
    block_started_at: float | None  # known ONLY if a reset was observed


_RESET_DROP_PERCENT = 5.0
_RESET_NEAR_ZERO_PERCENT = 10.0


def _is_reset(prev: SampleRow, curr: SampleRow) -> bool:
    """A drop has to clear `_RESET_DROP_PERCENT`, not just be negative —
    and for `is_estimated` samples, land near zero too.

    Claude's token-compute samples are a trailing rolling-window sum (see
    providers/claude.py Source B), so used_percent drifts down on its own
    as old events age out of the window even with no real reset, and a
    single old burst aging out can produce a large drop, not just a small
    drift. A magnitude threshold alone still false-positives on that. A
    true reset always lands near zero; a big rolling-window swing only
    coincidentally would, so requiring both cuts most false positives
    without needing to model the rolling window itself. Non-estimated
    sources (Codex, Claude Desktop) report an authoritative percentage —
    any real drop there is trustworthy on its own.
    """
    if curr.resets_at is not None and prev.resets_at is not None:
        if curr.resets_at > prev.resets_at + 1.0:
            return True
    dropped = curr.used_percent < prev.used_percent - _RESET_DROP_PERCENT
    if not dropped:
        return False
    if curr.is_estimated:
        return curr.used_percent < _RESET_NEAR_ZERO_PERCENT
    return True


def _split_into_blocks(history: list[SampleRow]) -> list[list[SampleRow]]:
    """`history` cut at every detected reset — oldest block first. Shared by
    `_current_block` (which only wants the last one) and the montecarlo
    bucket builder (which wants burn observations from every past block,
    not just the live one)."""
    if not history:
        return []
    blocks: list[list[SampleRow]] = [[history[0]]]
    for i in range(1, len(history)):
        if _is_reset(history[i - 1], history[i]):
            blocks.append([])
        blocks[-1].append(history[i])
    return blocks


def _current_block(history: list[SampleRow]) -> _BlockView:
    """The tail of `history` since the most recently observed reset. If no
    reset appears in `history` at all, the whole slice is "the current
    block" but its true start is unknown — that's a fact, not a bug; report
    it as such rather than guessing when the block began."""
    blocks = _split_into_blocks(history)
    if not blocks:
        return _BlockView(samples=[], block_started_at=None)
    block = blocks[-1]
    started_at = block[0].source_ts if len(blocks) > 1 else None
    return _BlockView(samples=block, block_started_at=started_at)


# -- token-velocity burn: the high-resolution signal -------------------------

# How far the authoritative percentage must have moved across the block
# before its ratio to token count is worth trusting. Both providers report
# whole numbers (measured: 901/901 Claude samples, 17/17 Codex), so a single
# step carries +/-50% error; three steps brings that under ~17%.
_MIN_CALIBRATION_PERCENT = 3.0


@dataclass(frozen=True)
class _TokenBurn:
    burn_per_hour: float
    n_events: int  # token events behind the rate — the evidence, for confidence


def _percent_per_token(
    block_samples: list[SampleRow], token_events: list[TokenEventRow]
) -> float | None:
    """How much of this window's quota one token consumes, measured rather
    than assumed — the authoritative percentage's movement across the whole
    block divided by the tokens spent over the same span.

    This is what lets a token rate become a %/h rate without knowing the
    provider's real token cap (which Codex never publishes and Claude only
    estimates). It also self-corrects for tokens the local log can't see:
    if some fraction of account-wide usage never writes a transcript here,
    the ratio absorbs it as long as the mix stays roughly steady.

    None when the block hasn't moved enough to calibrate against, or ends
    saturated (a clipped percentage understates the real movement).
    """
    if len(block_samples) < 2:
        return None
    first, last = block_samples[0], block_samples[-1]
    if last.used_percent >= 100.0:
        return None
    delta_percent = last.used_percent - first.used_percent
    if delta_percent < _MIN_CALIBRATION_PERCENT:
        return None
    tokens = sum(
        e.total_tokens
        for e in token_events
        if first.source_ts <= e.ts <= last.source_ts
    )
    if tokens <= 0:
        return None
    return delta_percent / tokens


def _token_burn_per_hour(
    block_samples: list[SampleRow],
    token_events: list[TokenEventRow],
    cfg: Config,
    window: Window,
    now: float,
) -> _TokenBurn | None:
    """Recent token throughput expressed as %/h. None when there's no
    calibration to convert it with.

    The rate divides by the *full* lookback, not by the span of the events
    actually seen: one burst two minutes ago is a burst, not a sustained
    rate, and dividing by its own two-minute span would report it as one.
    """
    percent_per_token = _percent_per_token(block_samples, token_events)
    if percent_per_token is None:
        return None
    lookback_hours = _lookback_minutes(cfg, window.kind) / 60.0
    if lookback_hours <= 0:
        return None
    recent = [e for e in token_events if e.ts >= now - lookback_hours * 3600]
    tokens = sum(e.total_tokens for e in recent)
    return _TokenBurn(
        burn_per_hour=percent_per_token * tokens / lookback_hours,
        n_events=len(recent),
    )


def tokens_burned_past_quota(
    window: Window, history: list[SampleRow], token_events: list[TokenEventRow]
) -> int | None:
    """Tokens actually consumed since `used_percent` pinned at its 100%
    ceiling this cycle -- Codex's own reported percentage cannot go past
    100, so `burn_per_hour` reads a flat 0.00 once it's there even while
    real usage (and real cost) continues. None until `window` has actually
    hit 100%; once it has, sums every token event observed since the
    earliest sample in the live streak that's still >= 100%, so tokens
    burned while climbing TOWARD the cap aren't counted as burned PAST it.
    None (not 0) when there's no token-event history to sum -- a provider
    that doesn't ingest per-event token counts (Claude Desktop) has
    nothing to report here, and 0 would misleadingly claim it does."""
    if window.used_percent < 100.0:
        return None
    boundary_ts = _saturation_started_at(history)
    if boundary_ts is None or not token_events:
        return None
    return sum(e.total_tokens for e in token_events if e.ts >= boundary_ts)


def _overage_rates(
    window: Window, cfg: Config
) -> tuple[float, float, float, float] | None:
    """(input, output, cache_write, cache_read) $/million rates for this
    window's provider, or None for a provider with no pricing config at
    all. Codex's cache fields are always 0 in token_events (see
    providers/codex.py), so its cache rates are pinned at 0.0 here rather
    than exposed as config -- there's nothing for them to ever multiply."""
    if window.provider is Provider.CODEX:
        return (
            cfg.codex.input_price_per_million_usd,
            cfg.codex.output_price_per_million_usd,
            0.0,
            0.0,
        )
    if window.provider is Provider.CLAUDE:
        return (
            cfg.claude.input_price_per_million_usd,
            cfg.claude.output_price_per_million_usd,
            cfg.claude.cache_write_price_per_million_usd,
            cfg.claude.cache_read_price_per_million_usd,
        )
    return None


def overage_cost_usd(
    window: Window,
    history: list[SampleRow],
    token_events: list[TokenEventRow],
    cfg: Config,
) -> float | None:
    """Dollar estimate for tokens burned since `window` pinned at 100%,
    priced from this provider's configured rates (see _overage_rates).
    None (never 0.0) unless a price is actually configured AND there's
    overage token data to price -- $0.00 would read as "you owe nothing,"
    a materially different claim from "unpriced."

    Claude prices cache writes/reads on their own rates, separate from
    input/output -- Anthropic bills them on distinct tiers (reads usually
    well below base input), unlike Codex where cache tokens are already
    folded into input/output counts (see providers/codex.py). Reusing a
    single two-rate formula for both would silently misprice whichever
    one doesn't fit it."""
    rates = _overage_rates(window, cfg)
    if rates is None or not any(rate > 0.0 for rate in rates):
        return None
    if window.used_percent < 100.0:
        return None
    boundary_ts = _saturation_started_at(history)
    if boundary_ts is None:
        return None
    relevant = [e for e in token_events if e.ts >= boundary_ts]
    if not relevant:
        return None
    input_rate, output_rate, cache_write_rate, cache_read_rate = rates
    input_tokens = sum(e.input_tokens for e in relevant)
    output_tokens = sum(e.output_tokens for e in relevant)
    cache_write_tokens = sum(e.cache_creation for e in relevant)
    cache_read_tokens = sum(e.cache_read for e in relevant)
    return (
        input_tokens * input_rate
        + output_tokens * output_rate
        + cache_write_tokens * cache_write_rate
        + cache_read_tokens * cache_read_rate
    ) / 1_000_000.0


def _saturation_started_at(history: list[SampleRow]) -> float | None:
    """The source_ts of the earliest sample, walking back from the newest,
    in an unbroken run of used_percent >= 100 -- i.e. since quota was
    first hit in the CURRENT cycle. A real reset always drops used_percent
    well below 100 (see _is_reset), so this can never walk back across a
    reset boundary without hitting a < 100 sample first; no block-splitting
    needed."""
    boundary_ts: float | None = None
    for sample in reversed(history):
        if sample.used_percent < 100.0:
            break
        boundary_ts = sample.source_ts
    return boundary_ts


def _is_rate_stale(window: Window, cfg: Config, now: float) -> bool:
    """Whether this reading is too old for its *rate* to mean anything.

    Per-window, because the two windows live on different timescales: a
    5-hour window's burn rate is meaningless minutes after the last
    request, while a 7-day window's sustained pace is not invalidated by a
    lunch break. One shared threshold had to be wrong for one of them.

    This says nothing about the *level*. `used_percent` is a durable fact
    for the rest of the cycle — quota does not un-consume while you are
    away — which is why alerts.py gates the threshold alert on
    `level_still_in_cycle` instead of on this.
    """
    stale_minutes = (
        cfg.thresholds.stale_after_minutes_w5h
        if window.kind is WindowKind.W5H
        else cfg.thresholds.stale_after_minutes_weekly
    )
    return (now - window.source_ts) > stale_minutes * 60


def _derive_resets_at(
    window: Window,
    block: _BlockView,
    token_events: list[TokenEventRow],
    now: float,
) -> float | None:
    """Only Claude ever needs this: Codex always supplies `resets_at`
    itself, straight from real account state; neither Claude Desktop nor
    token-compute do.

    Two derivations, best first:

    1. **Deterministic anchor** (5-hour window only). The 5-hour window is
       a fixed block anchored at the first request after an idle gap, so
       `blocks.block_anchor` computes the reset outright from the token log
       — available immediately, with no reset ever having to be caught in
       the act. The 7-day window has no comparably real anchor rule, so it
       does not get this path rather than getting a guessed one.
    2. **Last observed reset + one duration.** Both windows are
       fixed-duration cycles that re-anchor at each real reset, so the next
       one follows from the last one we actually saw. If no reset appears
       in history at all, the true anchor predates our data and is
       genuinely unknown — report that rather than guessing a calendar day.

    If a projection from (2) has already elapsed — a real reset happened
    but its used_percent drop was too small to clear `_is_reset`'s
    threshold, which a lightly used window can do — advance by however many
    whole cycles were missed. Same fixed-cycle assumption, carried through
    cycles we didn't directly witness, rather than reporting a reset in the
    past.
    """
    duration = window_duration_seconds(window.kind)
    if window.kind is WindowKind.W5H and token_events:
        # Authoritative when there's an activity log to read: an anchor gives
        # the real reset, and no anchor means the block has already expired
        # -- the window is empty and the next one doesn't exist until the
        # next request. Falling through to (2) there would advance a stale
        # anchor into the future and invent a reset for a block that isn't
        # running, which is how a 26-hour-old reading of a 5-hour window came
        # to look like live state (see alerts.level_still_in_cycle).
        anchor = block_anchor([e.ts for e in token_events], duration, now)
        return anchor + duration if anchor is not None else None
    if block.block_started_at is None:
        return None
    resets_at = block.block_started_at + duration
    if resets_at <= now:
        missed_cycles = int((now - resets_at) // duration) + 1
        resets_at += missed_cycles * duration
    return resets_at


def _lookback_minutes(cfg: Config, kind: WindowKind) -> float:
    return (
        cfg.burn.lookback_w5h_minutes
        if kind is WindowKind.W5H
        else cfg.burn.lookback_weekly_minutes
    )


def _slope_fit_samples(
    block_samples: list[SampleRow], cfg: Config, kind: WindowKind, now: float
) -> list[SampleRow]:
    """The short recency-weighted tail the slope is actually fit against —
    deliberately narrower than the full block so an old burst doesn't bias
    'right now'. Widens to the last two block samples if the configured
    lookback is too tight to have caught anything."""
    cutoff = now - _lookback_minutes(cfg, kind) * 60
    tail = [s for s in block_samples if s.source_ts >= cutoff]
    if len(tail) >= 2:
        return tail
    return block_samples[-2:] if len(block_samples) >= 2 else block_samples


def _robust_slope_per_hour(samples: list[SampleRow]) -> float:
    """Median of fixed-lag-k slopes (%/h), k = max(1, n // 4) — NOT the
    median of all C(n,2) pairwise slopes (Theil-Sen), which this replaced.

    used_percent only ever moves up within a block (a real quota counter
    never partially un-consumes), so a "burst outlier" here can only be a
    PERMANENT step from that point forward, never a reverting point
    anomaly. That matters: a step after position i corrupts i*(n-i) of
    the C(n,2) all-pairs slopes, which is >50% (past the median's
    breakdown point) whenever the step lands near the middle of the
    window — measured concretely as producing an estimate *worse* than a
    naive endpoint-to-endpoint slope. A step can corrupt at most k of the
    (n-k) fixed-lag-k slopes regardless of WHERE it falls, bounding
    contamination well under 50% for any position — while the wider
    baseline (k samples apart, not 1) keeps each slope less noise-amplified
    than raw consecutive deltas would be against Claude's is_estimated
    source, which can legitimately wobble within a block (see _is_reset)."""
    n = len(samples)
    if n < 2:
        return 0.0
    lag = max(1, n // 4)
    slopes: list[float] = []
    for i in range(n - lag):
        dt_hours = (samples[i + lag].source_ts - samples[i].source_ts) / 3600.0
        if dt_hours <= 0:
            continue
        slopes.append(
            (samples[i + lag].used_percent - samples[i].used_percent) / dt_hours
        )
    if not slopes:
        return 0.0
    slopes.sort()
    mid = len(slopes) // 2
    if len(slopes) % 2 == 1:
        return slopes[mid]
    return (slopes[mid - 1] + slopes[mid]) / 2.0


# -- montecarlo: hour-of-week burn buckets + simulation ----------------------

_HOUR_OF_WEEK_BUCKETS = 168  # 24 * 7

# A pair of samples more than this far apart produces too coarse an average
# to attribute to a single hour-of-week bucket meaningfully — e.g. "idle for
# three days, then one burst" would otherwise teach the model a tiny,
# misleading rate smeared across a bucket that was actually idle the whole
# time. Observations that far apart are dropped rather than misattributed.
_MAX_BUCKET_GAP_HOURS = 3.0

WeightedPool = tuple[list[float], list[float]]  # (values, weights) for random.choices


def _hour_of_week(ts: float, tz: tzinfo) -> int:
    dt = datetime.fromtimestamp(ts, tz=tz)
    return dt.weekday() * 24 + dt.hour


def _build_burn_buckets(
    history: list[SampleRow],
    token_events: list[TokenEventRow],
    percent_per_token: float | None,
    tz: tzinfo,
    now: float,
) -> dict[int, list[tuple[float, float]]]:
    """(observation_ts, burn_%/h) bucketed by hour-of-week — the profile the
    simulation samples its futures from. Prefers the token series; falls
    back to differencing percentages when there's no calibration."""
    if percent_per_token is not None and percent_per_token > 0 and token_events:
        return _token_burn_buckets(token_events, percent_per_token, tz, now)
    return _percent_burn_buckets(history, tz)


def _token_burn_buckets(
    token_events: list[TokenEventRow],
    percent_per_token: float,
    tz: tzinfo,
    now: float,
) -> dict[int, list[tuple[float, float]]]:
    """One observation per clock hour across the whole token history.

    The important part is that an hour with **no** token events yields a
    real zero rather than no observation at all. That distinction is what
    the profile is for: an hour you were asleep did not burn quota, and
    recording it as missing means the simulation fills it from the
    all-hours average instead — i.e. assumes you burn quota overnight at
    your daytime rate, which is exactly the flat-24/7 assumption the
    hour-of-week model exists to replace.

    (Differencing percentages can't produce those zeros for the sparse
    sources: `source_ts` only advances when a request happens, so overnight
    shows up as one wide gap that `_MAX_BUCKET_GAP_HOURS` then discards.)
    """
    tokens_by_hour: dict[int, int] = {}
    for event in token_events:
        slot = int(event.ts // 3600)
        tokens_by_hour[slot] = tokens_by_hour.get(slot, 0) + event.total_tokens
    first_slot = int(token_events[0].ts // 3600)
    # Stop before the hour in progress: it is only partly elapsed, so its
    # token count would read as a full hour's throughput and understate.
    last_slot = int(now // 3600) - 1
    buckets: dict[int, list[tuple[float, float]]] = {}
    for slot in range(first_slot, last_slot + 1):
        ts = slot * 3600.0
        burn = percent_per_token * tokens_by_hour.get(slot, 0)
        buckets.setdefault(_hour_of_week(ts, tz), []).append((ts, burn))
    return buckets


def _percent_burn_buckets(
    history: list[SampleRow], tz: tzinfo
) -> dict[int, list[tuple[float, float]]]:
    """Fallback profile: consecutive same-block sample pairs across EVERY
    observed block, not just the live one. Coarser than the token path
    (whole-number percentages, and long idle gaps get dropped rather than
    counted as zero), but it needs no calibration."""
    buckets: dict[int, list[tuple[float, float]]] = {}
    for block in _split_into_blocks(history):
        for i in range(1, len(block)):
            prev, curr = block[i - 1], block[i]
            dt_hours = (curr.source_ts - prev.source_ts) / 3600.0
            if dt_hours <= 0 or dt_hours > _MAX_BUCKET_GAP_HOURS:
                continue
            burn = (curr.used_percent - prev.used_percent) / dt_hours
            bucket = _hour_of_week(prev.source_ts, tz)
            buckets.setdefault(bucket, []).append((prev.source_ts, burn))
    return buckets


def _weighted_pool(
    entries: list[tuple[float, float]], now: float, halflife_days: float
) -> WeightedPool:
    """(values, weights) per `predictor.bucket_decay_halflife_days` — an
    observation from one halflife ago carries half the weight of one from
    today, so a rhythm change (e.g. a vacation, a new project) ages out
    instead of permanently anchoring the profile to old behavior."""
    values = [burn for _, burn in entries]
    if halflife_days <= 0:
        return values, [1.0] * len(values)
    halflife_seconds = halflife_days * 24 * 3600
    weights = [0.5 ** ((now - ts) / halflife_seconds) for ts, _ in entries]
    return values, weights


def _to_cumulative(pool: WeightedPool) -> WeightedPool:
    """(values, cumulative weights) — what `random.choices(cum_weights=...)`
    wants, so it can binary-search instead of re-accumulating per draw."""
    values, weights = pool
    return values, list(itertools.accumulate(weights))


def _weighted_mean(values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0:
        return sum(values) / len(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights, strict=True)) / total_weight


def _horizon_bucket_coverage(
    now: float, horizon_hours: float, buckets: dict[int, WeightedPool], tz: tzinfo
) -> float:
    """Fraction of the hours the simulation is about to walk through whose
    own hour-of-week bucket has real history.

    Raw observation count alone overstates confidence badly: hundreds of
    observations concentrated in a few days still leaves most of a week
    unlearned, and every uncovered hour is silently drawn from the
    all-hours pool instead. This is what `_confidence` needs in order not
    to call that "high"."""
    hours = max(int(horizon_hours), 1)
    covered = sum(
        1
        for i in range(hours)
        if buckets.get(_hour_of_week(now + i * 3600.0, tz), ([], []))[0]
    )
    return covered / hours


def _simulate_exhaustion_hours(
    used_percent: float,
    start_ts: float,
    buckets: dict[int, WeightedPool],
    fallback_pool: WeightedPool,
    tz: tzinfo,
    horizon_hours: float,
) -> float | None:
    """One simulated future: walk forward hour by hour from `start_ts`,
    each hour resampling a burn from that hour-of-week's empirical bucket
    (falling back to the all-buckets pool if that specific hour has no
    history yet), accumulating usage until it would hit 100%. Returns None
    if it doesn't exhaust within `horizon_hours` — "didn't run out," not
    an error; the caller treats that as right-censored AT the horizon
    rather than discarding the run.

    Both pools carry CUMULATIVE weights (see `_to_cumulative`) — this is
    the hot loop, `mc_runs * horizon_hours` draws per tick."""
    remaining = 100.0 - used_percent
    if remaining <= 0:
        return 0.0
    elapsed = 0.0
    cursor_ts = start_ts
    while elapsed < horizon_hours:
        bucket = _hour_of_week(cursor_ts, tz)
        values, cum_weights = buckets.get(bucket, ([], []))
        if not values:
            values, cum_weights = fallback_pool
        burn = max(random.choices(values, cum_weights=cum_weights, k=1)[0], 0.0)
        if burn >= remaining:
            # Interpolate inside the hour it runs out in rather than
            # rounding up to the whole hour. Without this the simulation
            # can't express any exhaustion sooner than 60 minutes away --
            # so for a window resetting in half an hour every outcome
            # landed past the horizon and got censored, and the P50/P90
            # band went silent exactly when it was most urgent.
            return elapsed + remaining / burn
        remaining -= burn
        elapsed += 1.0
        cursor_ts += 3600.0
    return None


def _percentile_within_horizon(
    sorted_values: list[float], quantile: float, horizon_hours: float
) -> float | None:
    """The `quantile`-th value of an ALREADY-SORTED list that mixes real
    exhaustion hours with `horizon_hours`-censored non-exhausting runs.
    Returns None if that percentile lands on a censored run — i.e. fewer
    than `quantile` of all simulated futures exhausted within the horizon
    at all, so there is no meaningful point estimate to report, not just
    an imprecise one."""
    if not sorted_values:
        return None
    index = min(int(quantile * len(sorted_values)), len(sorted_values) - 1)
    value = sorted_values[index]
    return None if value >= horizon_hours else value


def _confidence(n_samples: int, coverage: float | None = None) -> Confidence:
    """How much the burn estimate should be trusted, from the volume of
    evidence behind it and — for the hour-of-week models — how much of the
    period being projected through that evidence actually covers.

    Coverage can only ever lower the rating, never raise it: simulating 168
    hours off a profile that knows half of them is not a high-confidence
    forecast no matter how many observations those known hours hold."""
    if n_samples >= 10:
        rating: Confidence = "high"
    elif n_samples >= 3:
        rating = "medium"
    else:
        rating = "low"
    if coverage is None:
        return rating
    if coverage < 0.5:
        return "low"
    if coverage < 0.8 and rating == "high":
        return "medium"
    return rating


def _eta_horizon(
    window: Window, resets_at: float | None, now_dt: datetime, tz: tzinfo
) -> datetime:
    """The moment past which an ETA stops meaning anything.

    Normally the next reset. When that's unknown, the window's own duration
    still bounds it: a 5-hour window refills at least every 5 hours, so an
    exhaustion date beyond that is impossible regardless of what the
    arithmetic says. Backtesting turned up a 5-hour window being handed a
    3311-hour ETA — 4.5 months out — purely because no reset had been
    derived yet, which is a precise-looking number for an event that cannot
    occur. This is a bound implied by what the window is, not a guess about
    when it resets."""
    if resets_at is not None:
        return datetime.fromtimestamp(resets_at, tz=tz)
    return now_dt + timedelta(seconds=window_duration_seconds(window.kind))


def _cap_at_horizon(
    primary: datetime | None, secondary: datetime | None, horizon: datetime
) -> tuple[datetime | None, datetime | None]:
    """Blank out whichever of these ETAs lands at or after `horizon`.

    Applied to each ETA on its own, deliberately. Deciding it once from the
    24/7 projection and applying the answer to both is what let the
    working-hours ETA — which is always the later of the two, often by days
    — sit in the table pointing past a reset the code had already decided
    not to show ETAs past. An ETA after the reset is not an exhaustion
    time: the window refills first, so continued accumulation to 100% is
    hypothetical."""
    return (
        None if primary is not None and primary >= horizon else primary,
        None if secondary is not None and secondary >= horizon else secondary,
    )


def _status_only_forecast(
    window: Window,
    model_name: str,
    *,
    status: ForecastStatus,
    n_samples: int,
    now: float,
    resets_at: float | None,
) -> Forecast:
    return Forecast(
        window=window,
        status=status,
        model_name=model_name,
        burn_per_hour=0.0,
        time_to_reset_h=(resets_at - now) / 3600.0 if resets_at is not None else None,
        eta_calendar=None,
        eta_workhours=None,
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=None,
        confidence=None,  # IDLE/RESET_PENDING/too-few-samples: nothing forecast
        exhausts_before_reset=False,
        n_samples=n_samples,
    )


def _project_forecast(
    *,
    window: Window,
    burn_per_hour: float,
    resets_at: float | None,
    n_samples: int,
    cfg: Config,
    now: float,
    tz: tzinfo,
    model_name: str,
    burn_basis: BurnBasis | None = None,
    confidence: Confidence | None = None,
    prob_exhaust_before_reset: float | None = None,
) -> Forecast:
    now_dt = datetime.fromtimestamp(now, tz=tz)
    time_to_reset_h = (resets_at - now) / 3600.0 if resets_at is not None else None

    if burn_per_hour <= 0:
        # Flat/negative burn means there's no exhaustion trajectory at all —
        # not "unknown," "none." Confidence rates how much a burn-rate
        # estimate should be trusted; with nothing being forecast, there's
        # nothing for it to rate, so it's None rather than a number that
        # reads as confidence in an ETA that isn't shown.
        return Forecast(
            window=window,
            status="OK",
            model_name=model_name,
            burn_per_hour=burn_per_hour,
            time_to_reset_h=time_to_reset_h,
            eta_calendar=None,
            eta_workhours=None,
            eta_p50=None,
            eta_p90=None,
            prob_exhaust_before_reset=prob_exhaust_before_reset,
            confidence=None,
            exhausts_before_reset=False,
            n_samples=n_samples,
            burn_basis=burn_basis,
        )

    remaining_percent = max(0.0, 100.0 - window.used_percent)
    hours_to_exhaust = remaining_percent / burn_per_hour
    try:
        eta_calendar, eta_workhours = _cap_at_horizon(
            now_dt + timedelta(hours=hours_to_exhaust),
            project_workhours_exhaustion(now_dt, hours_to_exhaust, cfg.working_hours),
            _eta_horizon(window, resets_at, now_dt, tz),
        )
    except (OverflowError, RuntimeError):
        # A burn rate close enough to zero (but not exactly zero -- e.g.
        # floating-point noise off two nearly-identical readings) turns
        # a tiny remaining-percent into an exhaustion horizon of
        # literally centuries. Calendar arithmetic that far out either
        # overflows datetime's range outright, or blows the
        # working-hours projector's iteration guard (config is already
        # validated at load time, so that guard can only fire here for
        # an input too large to converge, never a real misconfiguration).
        # Either way there's no meaningful ETA to report at that
        # remove -- None is honest, not a crash.
        eta_calendar = None
        eta_workhours = None

    return Forecast(
        window=window,
        status="OK",
        model_name=model_name,
        burn_per_hour=burn_per_hour,
        time_to_reset_h=time_to_reset_h,
        eta_calendar=eta_calendar,
        eta_workhours=eta_workhours,
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=prob_exhaust_before_reset,
        confidence=confidence or _confidence(n_samples),
        # The wall-clock question, on purpose (see Forecast.exhausts_before_
        # reset): surviving the cap above means the 24/7 projection lands
        # before the reset.
        exhausts_before_reset=eta_calendar is not None and time_to_reset_h is not None,
        n_samples=n_samples,
        burn_basis=burn_basis,
    )


# -- working-hours calendar projection ---------------------------------------


def project_workhours_exhaustion(
    now: datetime, burn_hours_needed: float, wh: WorkingHoursConfig
) -> datetime:
    """Advance the clock by `burn_hours_needed` hours of *burn time*,
    counting only configured working intervals — a burn-hours budget of 20
    starting Friday 3pm lands the following Tuesday, skipping the weekend.

    If working-hours tracking is disabled, this degrades to a plain 24/7
    projection (making eta_workhours == eta_calendar, a sensible default)."""
    if not wh.enabled or burn_hours_needed <= 0:
        return now + timedelta(hours=max(burn_hours_needed, 0.0))

    cursor = now
    remaining = burn_hours_needed
    guard = 0
    while remaining > 1e-9:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("working-hours projection did not converge")
        if _in_working_interval(cursor, wh):
            interval_end = _working_interval_end(cursor, wh)
            available_h = (interval_end - cursor).total_seconds() / 3600.0
            take = min(available_h, remaining)
            cursor = cursor + timedelta(hours=take)
            remaining -= take
            if remaining > 1e-9:
                cursor = _start_of_next_working_interval(cursor, wh)
        else:
            cursor = _start_of_next_working_interval(cursor, wh)
    return cursor


def _in_working_interval(dt: datetime, wh: WorkingHoursConfig) -> bool:
    return (
        dt.weekday() in wh.day_numbers()
        and wh.start_time() <= dt.time() < wh.end_time()
    )


def _working_interval_end(dt: datetime, wh: WorkingHoursConfig) -> datetime:
    return datetime.combine(dt.date(), wh.end_time(), tzinfo=dt.tzinfo)


def _start_of_next_working_interval(dt: datetime, wh: WorkingHoursConfig) -> datetime:
    if dt.weekday() in wh.day_numbers() and dt.time() < wh.start_time():
        return datetime.combine(dt.date(), wh.start_time(), tzinfo=dt.tzinfo)
    next_date = dt.date() + timedelta(days=1)
    for _ in range(8):
        if next_date.weekday() in wh.day_numbers():
            return datetime.combine(next_date, wh.start_time(), tzinfo=dt.tzinfo)
        next_date += timedelta(days=1)
    raise ConfigError("working_hours.days must be non-empty when enabled")
