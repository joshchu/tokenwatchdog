"""Core data model shared by every TokenWatchDog subsystem.

Provider-agnostic: nothing here knows how Codex or Claude data gets
collected — see providers/.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Literal


class Provider(StrEnum):
    CODEX = "codex"
    CLAUDE = "claude"


class WindowKind(StrEnum):
    W5H = "w5h"
    WEEKLY = "weekly"


_W5H_SECONDS = 5 * 3600
_WEEKLY_SECONDS = 7 * 24 * 3600


def window_duration_seconds(kind: WindowKind) -> float:
    """How long one cycle of this window lasts. A plain fact about the
    window, which is why it lives here rather than in the predictor: both
    the predictor (bounding an ETA) and alerts (deciding whether a reading's
    cycle is still the live one) need it."""
    return _W5H_SECONDS if kind is WindowKind.W5H else _WEEKLY_SECONDS


ForecastStatus = Literal["OK", "IDLE", "NO_DATA", "RESET_PENDING"]
Confidence = Literal["low", "medium", "high"]
AlertKind = Literal["threshold", "burn"]

# Which signal a burn rate was measured from. "tokens" is the good case:
# real per-request token throughput, scaled into %/h by a ratio calibrated
# against the authoritative percentage. "percent" is the fallback — the slope
# of the reported percentage itself, which both providers quantize to whole
# numbers and which therefore cannot resolve a slow-moving weekly window.
# Recorded per forecast so scripts/backtest.py can score the two separately.
BurnBasis = Literal["tokens", "percent"]


@dataclass(frozen=True)
class Window:
    """One quota snapshot for a single (provider, window kind)."""

    provider: Provider
    kind: WindowKind
    used_percent: float
    window_minutes: int
    resets_at: float | None
    source_ts: float
    is_estimated: bool
    source_file: str


def level_still_in_cycle(window: Window, now: float) -> bool:
    """Whether `window.used_percent` still describes the live cycle.

    Staleness is a fact about a *rate*, not a *level*: quota does not
    un-consume while you're away, so a reading of 100% taken six hours ago
    against a window that resets in four days is still exactly 100% now.
    Requires a known `resets_at` — without one we can't establish that no
    reset has happened since.

    `now < resets_at` alone is not enough because a derived reset can advance
    by whole cycles after its original boundary. The reading must also fall
    inside the cycle that ends at `resets_at`.
    """
    if window.resets_at is None or now >= window.resets_at:
        return False
    cycle_started_at = window.resets_at - window_duration_seconds(window.kind)
    return window.source_ts >= cycle_started_at


@dataclass(frozen=True)
class Forecast:
    """A predictor's exhaustion estimate for one Window, as of `now`."""

    window: Window
    status: ForecastStatus
    model_name: str
    burn_per_hour: float
    time_to_reset_h: float | None
    eta_calendar: datetime | None
    eta_workhours: datetime | None
    eta_p50: datetime | None
    eta_p90: datetime | None
    prob_exhaust_before_reset: float | None
    confidence: Confidence | None  # None when there's no exhaustion trajectory at all
    # Whether the *wall-clock* projection lands before this window's next
    # reset — deliberately the 24/7 question, not the working-hours one:
    # this is what the burn alert means by urgent, and usage that is
    # happening right now doesn't pause because it's 9pm. eta_workhours
    # answers the separate planning question and is shown alongside.
    exhausts_before_reset: bool
    n_samples: int
    burn_basis: BurnBasis | None = None  # None when nothing was measured
    # None until used_percent has actually pinned at 100 this cycle -- see
    # predictor.tokens_burned_past_quota for why burn_per_hour alone goes
    # blind right at the one moment it matters most.
    tokens_burned_past_quota: int | None = None
    # None unless a per-token price is configured (see predictor.
    # overage_cost_usd) -- distinct from 0.0, which would claim "you owe
    # nothing" rather than "unpriced."
    cost_burned_past_quota_usd: float | None = None


@dataclass(frozen=True)
class RetainedPrediction:
    """A saved percentile forecast shown when the current idle forecast
    cannot safely recompute one.

    This is deliberately separate from Forecast: it is presentation context,
    not a new model result, and must never drive alerts or model scoring.
    """

    provider: Provider
    kind: WindowKind
    made_at: float
    used_percent: float
    eta_p50: datetime
    eta_p90: datetime | None


@dataclass(frozen=True)
class Alert:
    """One fired notification for a (provider, window kind, alert kind)."""

    key: str
    provider: Provider
    kind: WindowKind
    alert_kind: AlertKind
    message: str
    fired_at: float


@dataclass(frozen=True)
class MonitorState:
    """Return value of Engine.tick() — the one snapshot every front-end renders."""

    now: float
    windows: tuple[Window, ...]
    # The authoritative forecast per watched window — one per (provider,
    # kind), from the model `predictor.model` selected. Alerts fire off
    # these, and only these.
    forecasts: tuple[Forecast, ...]
    alerts: tuple[Alert, ...]
    # Every model's forecast for every watched window, including the
    # authoritative ones above. Two models answer genuinely different
    # questions — "at the pace of the last few hours" versus "given how you
    # use this across a week" — so the dashboard shows both, and their
    # disagreement is itself information. Also what makes each model's
    # accuracy measurable against the same moments (see scoring.py).
    all_forecasts: tuple[Forecast, ...] = ()
    # The latest still-valid persisted Monte Carlo result for an idle window
    # whose current forecast cannot produce a percentile. Kept out of
    # all_forecasts so alerts and scoring never mistake a retained display
    # value for a fresh model output.
    retained_predictions: tuple[RetainedPrediction, ...] = ()

    def forecast_from(self, model_name: str, window: Window) -> Forecast | None:
        for forecast in self.all_forecasts:
            if (
                forecast.model_name == model_name
                and forecast.window.provider is window.provider
                and forecast.window.kind is window.kind
            ):
                return forecast
        return None

    def retained_prediction_for(self, window: Window) -> RetainedPrediction | None:
        for prediction in self.retained_predictions:
            if (
                prediction.provider is window.provider
                and prediction.kind is window.kind
            ):
                return prediction
        return None
