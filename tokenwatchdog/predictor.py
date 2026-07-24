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
from datetime import datetime, time, timedelta, tzinfo
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
            resets_at = _derive_resets_at(window, block, now, tz)
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
            resets_at = _derive_resets_at(window, block, now, tz)
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

        buckets = _build_burn_buckets(history, tz)
        all_burns = [v for values in buckets.values() for v in values]
        if not all_burns:
            return _status_only_forecast(
                window,
                self.name,
                status="OK",
                n_samples=len(block.samples),
                now=now,
                resets_at=resets_at,
            )

        now_dt = datetime.fromtimestamp(now, tz=tz)
        time_to_reset_h = (resets_at - now) / 3600.0 if resets_at is not None else None
        # Simulations run at most 2 weeks out, or 3x the time to reset if
        # that's longer — no point simulating a year to find a P90 that's
        # actually "never, at this rate."
        horizon_h = (
            max(time_to_reset_h * 3, 24.0 * 14) if time_to_reset_h else 24.0 * 14
        )

        mc_runs = max(cfg.predictor.mc_runs, 1)
        exhaustion_hours = [
            hours
            for hours in (
                _simulate_exhaustion_hours(
                    window.used_percent, now, buckets, all_burns, tz, horizon_h
                )
                for _ in range(mc_runs)
            )
            if hours is not None
        ]
        mean_burn = sum(all_burns) / len(all_burns)

        if not exhaustion_hours:
            # Every simulation ran out the clock without reaching 100% —
            # burn is too low/negative on average to say anything sharper
            # than the plain mean, so fall back to the same projection
            # linear uses.
            return _project_forecast(
                window=window,
                burn_per_hour=mean_burn,
                resets_at=resets_at,
                n_samples=len(all_burns),
                cfg=cfg,
                now=now,
                tz=tz,
                model_name=self.name,
            )

        exhaustion_hours.sort()
        p50_h = _percentile(exhaustion_hours, 0.5)
        p90_h = _percentile(exhaustion_hours, 0.9)
        prob_before_reset = (
            sum(1 for h in exhaustion_hours if h < time_to_reset_h) / mc_runs
            if time_to_reset_h is not None
            else None
        )

        eta_p50 = now_dt + timedelta(hours=p50_h)
        eta_p90 = now_dt + timedelta(hours=p90_h)
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
            confidence=_confidence(len(all_burns)),
            exhausts_before_reset=exhausts_before_reset,
            n_samples=len(all_burns),
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


def _is_reset(prev: SampleRow, curr: SampleRow) -> bool:
    """A drop has to clear `_RESET_DROP_PERCENT`, not just be negative:
    Claude's token-compute samples are a trailing rolling-window sum (see
    providers/claude.py Source B), so used_percent drifts down on its own
    as old events age out of the window even with no real reset. A true
    reset is a cliff (toward 0%); aging-out is a slow drift. This is an
    imperfect heuristic, not a real block-boundary signal for that source.
    """
    return curr.used_percent < prev.used_percent - _RESET_DROP_PERCENT or (
        prev.resets_at is not None
        and curr.resets_at is not None
        and curr.resets_at > prev.resets_at + 1.0
    )


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


def _derive_resets_at(
    window: Window, block: _BlockView, now: float, tz: tzinfo
) -> float | None:
    """Only Claude ever needs this: Codex always supplies `resets_at`
    itself, but neither Claude Desktop nor token-compute do — deriving a
    reset time from the window's own definition is this layer's job, not
    the provider's."""
    if window.kind is WindowKind.W5H:
        if block.block_started_at is not None:
            return block.block_started_at + _FIVE_HOURS_SECONDS
        return None  # this block's start predates our observation window
    # WEEKLY: fixed calendar anchor, next Monday 00:00 local time — an
    # unvalidated approximation. Worth cross-checking against Claude Code's
    # own /usage bars once there's enough real usage to compare against.
    now_dt = datetime.fromtimestamp(now, tz=tz)
    days_ahead = (7 - now_dt.weekday()) % 7 or 7
    anchor_date = now_dt.date() + timedelta(days=days_ahead)
    anchor = datetime.combine(anchor_date, time(0, 0), tzinfo=tz)
    return anchor.timestamp()


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


def _hour_of_week(ts: float, tz: tzinfo) -> int:
    dt = datetime.fromtimestamp(ts, tz=tz)
    return dt.weekday() * 24 + dt.hour


def _build_burn_buckets(history: list[SampleRow], tz: tzinfo) -> dict[int, list[float]]:
    """Empirical burn-%/h observations bucketed by hour-of-week, built from
    consecutive same-block sample pairs across EVERY observed block, not
    just the live one — this is what lets the model learn "nothing happens
    on weekends" instead of assuming a flat 24/7 rate. Each bucket's list
    is the raw observations, not a summary statistic, so the simulator can
    resample from the real empirical distribution rather than a Gaussian
    assumption that bursty usage doesn't actually follow."""
    buckets: dict[int, list[float]] = {}
    for block in _split_into_blocks(history):
        for i in range(1, len(block)):
            prev, curr = block[i - 1], block[i]
            dt_hours = (curr.source_ts - prev.source_ts) / 3600.0
            if dt_hours <= 0:
                continue
            burn = (curr.used_percent - prev.used_percent) / dt_hours
            bucket = _hour_of_week(prev.source_ts, tz)
            buckets.setdefault(bucket, []).append(burn)
    return buckets


def _simulate_exhaustion_hours(
    used_percent: float,
    start_ts: float,
    buckets: dict[int, list[float]],
    fallback_values: list[float],
    tz: tzinfo,
    horizon_hours: float,
) -> float | None:
    """One simulated future: walk forward hour by hour from `start_ts`,
    each hour resampling a burn from that hour-of-week's empirical bucket
    (falling back to the all-buckets pool if that specific hour has no
    history yet), accumulating usage until it would hit 100%. Returns None
    if it doesn't exhaust within `horizon_hours` — "didn't run out," not
    an error."""
    remaining = 100.0 - used_percent
    if remaining <= 0:
        return 0.0
    elapsed = 0.0
    cursor_ts = start_ts
    while elapsed < horizon_hours:
        bucket = _hour_of_week(cursor_ts, tz)
        values = buckets.get(bucket) or fallback_values
        burn = max(random.choice(values), 0.0)
        remaining -= burn
        elapsed += 1.0
        cursor_ts += 3600.0
        if remaining <= 0:
            return elapsed
    return None


def _percentile(sorted_values: list[float], quantile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(quantile * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


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
        confidence="low",
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
            confidence=_confidence(n_samples),
            exhausts_before_reset=False,
            n_samples=n_samples,
        )

    remaining_percent = max(0.0, 100.0 - window.used_percent)
    hours_to_exhaust = remaining_percent / burn_per_hour
    eta_calendar = now_dt + timedelta(hours=hours_to_exhaust)
    eta_workhours = project_workhours_exhaustion(
        now_dt, hours_to_exhaust, cfg.working_hours
    )
    exhausts_before_reset = (
        False if time_to_reset_h is None else hours_to_exhaust < time_to_reset_h
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
