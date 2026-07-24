"""Exhaustion prediction — tiered, swappable, backtestable.

Two models: `linear` (robust trailing-slope fit, the cold-start default —
sane from the very first samples) and `montecarlo` (a learned hour-of-week
burn profile, simulated forward to a P50/P90 exhaustion band once enough
history exists to populate it). Both share resets_at derivation and the
working-hours projector via the same `Predictor` protocol, so a front-end
or alerts.py never has to know which one produced a given Forecast.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, tzinfo
from typing import Protocol

from tokenwatchdog.config import (
    Config,
    ConfigError,
    WorkingHoursConfig,
    resolve_timezone,
)
from tokenwatchdog.models import (
    Confidence,
    Forecast,
    ForecastStatus,
    Window,
    WindowKind,
)
from tokenwatchdog.store import SampleRow, TokenEventRow

_FIVE_HOURS_SECONDS = 5 * 3600


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
        first. `token_events` is Claude's finer-grained signal — unused by
        both current models, present so a future token-native model
        doesn't need an interface change to get it."""
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
        del token_events  # unused by linear; kept for v1.1 models' sake

        tz = resolve_timezone(cfg)
        block = _current_block(history)
        resets_at = window.resets_at
        if resets_at is None:
            resets_at = _derive_resets_at(window, block, now)
            if resets_at is not None:
                # Thread the derived value back onto the window so it rides
                # along on Forecast.window — alerts.py re-arms on resets_at
                # advancing, which never happens if this stays None forever
                # (as it would for every Claude reading straight from the
                # provider).
                window = replace(window, resets_at=resets_at)

        is_idle = (now - window.source_ts) > cfg.thresholds.stale_after_minutes * 60
        if is_idle:
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
        burn_per_hour = _theil_sen_slope_per_hour(fit_samples)
        return _project_forecast(
            window=window,
            burn_per_hour=burn_per_hour,
            resets_at=resets_at,
            n_samples=len(fit_samples),
            cfg=cfg,
            now=now,
            tz=tz,
            model_name=self.name,
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
        del token_events  # bucketing works off used_percent samples, not raw tokens

        tz = resolve_timezone(cfg)
        block = _current_block(history)
        resets_at = window.resets_at
        if resets_at is None:
            resets_at = _derive_resets_at(window, block, now)
            if resets_at is not None:
                window = replace(window, resets_at=resets_at)

        is_idle = (now - window.source_ts) > cfg.thresholds.stale_after_minutes * 60
        if is_idle:
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

        raw_buckets = _build_burn_buckets(history, tz)
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
        buckets = {
            bucket: _weighted_pool(entries, now, halflife_days)
            for bucket, entries in raw_buckets.items()
        }
        fallback_pool = (
            [v for values, _ in buckets.values() for v in values],
            [w for _, weights in buckets.values() for w in weights],
        )
        mean_burn = sum(fallback_pool[0]) / len(fallback_pool[0])

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
        outcomes = [
            _simulate_exhaustion_hours(
                window.used_percent, now, buckets, fallback_pool, tz, horizon_h
            )
            or horizon_h
            for _ in range(mc_runs)
        ]
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
        if p50_h is None:
            # More than half the simulated futures never exhausted within
            # the horizon — there's no meaningful point ETA, so this is
            # the same "OK, no ETA" shape linear reports for burn <= 0.
            return _project_forecast(
                window=window,
                burn_per_hour=mean_burn,
                resets_at=resets_at,
                n_samples=n_observations,
                cfg=cfg,
                now=now,
                tz=tz,
                model_name=self.name,
            )

        eta_p50 = now_dt + timedelta(hours=p50_h)
        eta_p90 = now_dt + timedelta(hours=p90_h) if p90_h is not None else None
        exhausts_before_reset = (
            False if time_to_reset_h is None else p50_h < time_to_reset_h
        )

        return Forecast(
            window=window,
            status="OK",
            model_name=self.name,
            burn_per_hour=mean_burn,
            time_to_reset_h=time_to_reset_h,
            eta_calendar=eta_p50,  # point estimate = the median simulated outcome
            eta_workhours=project_workhours_exhaustion(
                now_dt, p50_h, cfg.working_hours
            ),
            eta_p50=eta_p50,
            eta_p90=eta_p90,
            prob_exhaust_before_reset=prob_before_reset,
            confidence=_confidence(n_observations),
            exhausts_before_reset=exhausts_before_reset,
            n_samples=n_observations,
        )


_PREDICTORS: dict[str, Predictor] = {
    "linear": LinearPredictor(),
    "montecarlo": MonteCarloPredictor(),
}


def select_predictor(cfg: Config) -> Predictor:
    # "auto" doesn't yet graduate to montecarlo based on history volume —
    # that graduation logic (and the seasonal_profile tier it was designed
    # to sit alongside) isn't built. Pick montecarlo explicitly via
    # predictor.model if you want it.
    if cfg.predictor.model == "auto":
        return _PREDICTORS["linear"]
    predictor = _PREDICTORS.get(cfg.predictor.model)
    if predictor is None:
        raise ConfigError(
            f"predictor.model={cfg.predictor.model!r} isn't implemented yet "
            f"(available: {sorted(_PREDICTORS)}) — seasonal_profile/holtwinters "
            f"are a planned future addition, not built yet"
        )
    return predictor


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


_SEVEN_DAYS_SECONDS = 7 * 24 * 3600


def _derive_resets_at(window: Window, block: _BlockView, now: float) -> float | None:
    """Only Claude ever needs this: Codex always supplies `resets_at`
    itself, straight from real account state; neither Claude Desktop nor
    token-compute do.

    Both Claude windows are fixed-duration cycles (5h / 7d) that re-anchor
    at each actual reset, so the next one is simply the last OBSERVED
    reset plus that duration — derived from this account's own real
    behavior, never assumed against a calendar. If no reset has been
    observed yet (block_started_at is None), the true anchor predates our
    history and is genuinely unknown — report that honestly rather than
    guessing a calendar day.

    If that next-cycle projection has already elapsed — a real reset
    happened since but its used_percent drop was too small to clear
    `_is_reset`'s threshold (a lightly used window can do this) — advance
    by however many whole cycles were missed. It's the same fixed-cycle
    assumption already being made, just carried forward through cycles we
    didn't directly see a boundary for, rather than reporting a reset
    that's already in the past.
    """
    if block.block_started_at is None:
        return None
    duration = (
        _FIVE_HOURS_SECONDS if window.kind is WindowKind.W5H else _SEVEN_DAYS_SECONDS
    )
    resets_at = block.block_started_at + duration
    if resets_at <= now:
        missed_cycles = int((now - resets_at) // duration) + 1
        resets_at += missed_cycles * duration
    return resets_at


def _slope_fit_samples(
    block_samples: list[SampleRow], cfg: Config, kind: WindowKind, now: float
) -> list[SampleRow]:
    """The short recency-weighted tail the slope is actually fit against —
    deliberately narrower than the full block so an old burst doesn't bias
    'right now'. Widens to the last two block samples if the configured
    lookback is too tight to have caught anything."""
    lookback_minutes = (
        cfg.burn.lookback_w5h_minutes
        if kind is WindowKind.W5H
        else cfg.burn.lookback_weekly_minutes
    )
    cutoff = now - lookback_minutes * 60
    tail = [s for s in block_samples if s.source_ts >= cutoff]
    if len(tail) >= 2:
        return tail
    return block_samples[-2:] if len(block_samples) >= 2 else block_samples


def _theil_sen_slope_per_hour(samples: list[SampleRow]) -> float:
    """Median of all pairwise slopes (%/h) — robust to a single burst
    outlier in a way a least-squares fit isn't."""
    slopes: list[float] = []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            dt_hours = (samples[j].source_ts - samples[i].source_ts) / 3600.0
            if dt_hours <= 0:
                continue
            slopes.append(
                (samples[j].used_percent - samples[i].used_percent) / dt_hours
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
    history: list[SampleRow], tz: tzinfo
) -> dict[int, list[tuple[float, float]]]:
    """(source_ts, burn_%/h) observations bucketed by hour-of-week, built
    from consecutive same-block sample pairs across EVERY observed block,
    not just the live one — this is what lets the model learn "nothing
    happens on weekends" instead of assuming a flat 24/7 rate. Keeping the
    timestamp alongside each observation is what lets `_weighted_pool`
    weigh recent weeks more than old ones."""
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
    """(values, weights) for `random.choices`, per
    `predictor.bucket_decay_halflife_days` — an observation from one
    halflife ago carries half the weight of one from today, so a rhythm
    change (e.g. a vacation, a new project) ages out instead of
    permanently anchoring the profile to old behavior."""
    values = [burn for _, burn in entries]
    if halflife_days <= 0:
        return values, [1.0] * len(values)
    halflife_seconds = halflife_days * 24 * 3600
    weights = [0.5 ** ((now - ts) / halflife_seconds) for ts, _ in entries]
    return values, weights


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
    rather than discarding the run."""
    remaining = 100.0 - used_percent
    if remaining <= 0:
        return 0.0
    elapsed = 0.0
    cursor_ts = start_ts
    while elapsed < horizon_hours:
        bucket = _hour_of_week(cursor_ts, tz)
        values, weights = buckets.get(bucket, ([], []))
        if not values:
            values, weights = fallback_pool
        burn = max(random.choices(values, weights=weights, k=1)[0], 0.0)
        remaining -= burn
        elapsed += 1.0
        cursor_ts += 3600.0
        if remaining <= 0:
            return elapsed
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


def _confidence(n_samples: int) -> Confidence:
    if n_samples >= 10:
        return "high"
    if n_samples >= 3:
        return "medium"
    return "low"


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
            prob_exhaust_before_reset=None,
            confidence=None,
            exhausts_before_reset=False,
            n_samples=n_samples,
        )

    remaining_percent = max(0.0, 100.0 - window.used_percent)
    hours_to_exhaust = remaining_percent / burn_per_hour
    exhausts_before_reset = (
        False if time_to_reset_h is None else hours_to_exhaust < time_to_reset_h
    )

    # A linear projection that lands AFTER a known reset isn't a real ETA —
    # the window refills before usage ever gets there, so "exhaustion"
    # under continued accumulation is hypothetical, not a risk. Report no
    # ETA rather than a number that reads as "you'll run out then" when
    # you won't. (The reset time itself is already shown separately.)
    if time_to_reset_h is not None and not exhausts_before_reset:
        eta_calendar = None
        eta_workhours = None
    else:
        eta_calendar = now_dt + timedelta(hours=hours_to_exhaust)
        eta_workhours = project_workhours_exhaustion(
            now_dt, hours_to_exhaust, cfg.working_hours
        )

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
        prob_exhaust_before_reset=None,
        confidence=_confidence(n_samples),
        exhausts_before_reset=exhausts_before_reset,
        n_samples=n_samples,
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
