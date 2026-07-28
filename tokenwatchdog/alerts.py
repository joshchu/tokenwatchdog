"""Alerting — 90% threshold + burn-too-fast, with a persisted re-arm state
machine so each condition fires at most once per window period.
"""

from __future__ import annotations

from tokenwatchdog.config import Config
from tokenwatchdog.models import (
    Alert,
    AlertKind,
    Forecast,
    Provider,
    Window,
    WindowKind,
    level_still_in_cycle,
)
from tokenwatchdog.store import AlertStateRow, Store


def alert_key(provider: Provider, kind: WindowKind, alert_kind: AlertKind) -> str:
    return f"{provider.value}:{kind.value}:{alert_kind}"


def evaluate(forecast: Forecast, cfg: Config, store: Store, now: float) -> list[Alert]:
    """Check both alert kinds for one Forecast. Persists any state change
    (fire, or re-arm) as it goes — called once per watched window per tick.

    NO_DATA is skipped outright — there is no reading. IDLE is skipped only
    when the level can't be vouched for either (see `level_still_in_cycle`);
    when it can, the threshold alert is still allowed through on the
    strength of the level alone, while the burn alert stays suppressed by
    its own `status == "OK"` requirement, because a burn rate measured from
    a stale reading really is untrustworthy.

    A daemon started fresh against a 3-day-old 95% reading therefore still
    stays quiet — not because the reading is old, but because a window whose
    `resets_at` has since passed is no longer at 95%."""
    if forecast.status == "NO_DATA":
        return []
    if forecast.status == "IDLE" and not level_still_in_cycle(forecast.window, now):
        return []
    fired = [
        alert
        for alert in (
            _evaluate_threshold(forecast, cfg, store, now),
            _evaluate_burn(forecast, cfg, store, now),
        )
        if alert is not None
    ]
    return fired


def _evaluate_threshold(
    forecast: Forecast, cfg: Config, store: Store, now: float
) -> Alert | None:
    window = forecast.window
    condition = window.used_percent >= cfg.thresholds.warn_percent
    message = (
        f"{window.provider.value} {window.kind.value}: {window.used_percent:.0f}% "
        f"used (warns at {cfg.thresholds.warn_percent:.0f}%)"
    )
    return _evaluate(
        forecast,
        cfg,
        store,
        now,
        alert_kind="threshold",
        condition=condition,
        message=message,
    )


def _evaluate_burn(
    forecast: Forecast, cfg: Config, store: Store, now: float
) -> Alert | None:
    """Fires only when exhaustion is actually imminent — projected within
    `burn_alert_within_hours` (default 1h) — not merely "sometime before
    the window resets," which could be days off and isn't urgent. Gating
    on distance-from-reset instead of real urgency is what made this fire
    for burns that were technically on pace but nowhere near soon."""
    window = forecast.window
    hours_to_exhaust: float | None = None
    if (
        forecast.status == "OK"
        and forecast.exhausts_before_reset
        and window.used_percent >= cfg.thresholds.burn_min_percent
        and forecast.eta_calendar is not None
    ):
        hours_to_exhaust = (forecast.eta_calendar.timestamp() - now) / 3600.0
    condition = (
        hours_to_exhaust is not None
        and hours_to_exhaust <= cfg.thresholds.burn_alert_within_hours
    )

    minutes_to_exhaust = max(hours_to_exhaust or 0.0, 0.0) * 60.0
    message = (
        f"{window.provider.value} {window.kind.value} is burning too fast: "
        f"{window.used_percent:.0f}% used, projected to exhaust in "
        f"~{minutes_to_exhaust:.0f} min (before its next reset)"
    )
    return _evaluate(
        forecast,
        cfg,
        store,
        now,
        alert_kind="burn",
        condition=condition,
        message=message,
    )


def _evaluate(
    forecast: Forecast,
    cfg: Config,
    store: Store,
    now: float,
    *,
    alert_kind: AlertKind,
    condition: bool,
    message: str,
) -> Alert | None:
    window = forecast.window
    key = alert_key(window.provider, window.kind, alert_kind)
    state = store.get_alert_state(key)
    armed = _rearm_if_needed(key, state, window, cfg, store, alert_kind=alert_kind)

    if not (armed and condition):
        return None

    store.set_alert_state(
        key, armed=False, last_fired_at=now, reset_epoch_at_fire=window.resets_at
    )
    return Alert(
        key=key,
        provider=window.provider,
        kind=window.kind,
        alert_kind=alert_kind,
        message=message,
        fired_at=now,
    )


def _rearm_if_needed(
    key: str,
    state: AlertStateRow | None,
    window: Window,
    cfg: Config,
    store: Store,
    *,
    alert_kind: AlertKind,
) -> bool:
    """Whether the alert is armed right now, re-arming (and persisting the
    re-arm) first if warranted.

    Re-arm triggers: `resets_at` has advanced since the last fire — robust
    even while idle, since it's a fact about the world, not something we
    have to catch changing in real time. The threshold alert additionally
    re-arms on a hysteresis drop (used_percent falls back below
    `warn_percent - threshold_hysteresis`), so the same block can re-warn if
    usage dips and climbs again.
    """
    if state is None or state.armed:
        return True
    reset_advanced = (
        window.resets_at is not None
        and state.reset_epoch_at_fire is not None
        and window.resets_at > state.reset_epoch_at_fire + 1.0
    )
    hysteresis_cleared = alert_kind == "threshold" and (
        window.used_percent
        < cfg.thresholds.warn_percent - cfg.thresholds.threshold_hysteresis
    )
    if not (reset_advanced or hysteresis_cleared):
        return False
    store.set_alert_state(
        key,
        armed=True,
        last_fired_at=state.last_fired_at,
        reset_epoch_at_fire=state.reset_epoch_at_fire,
    )
    return True
