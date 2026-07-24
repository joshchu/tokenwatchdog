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


ForecastStatus = Literal["OK", "IDLE", "NO_DATA", "RESET_PENDING"]
Confidence = Literal["low", "medium", "high"]
AlertKind = Literal["threshold", "burn"]


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
    exhausts_before_reset: bool
    n_samples: int
    # None until used_percent has actually pinned at 100 this cycle -- see
    # predictor.tokens_burned_past_quota for why burn_per_hour alone goes
    # blind right at the one moment it matters most.
    tokens_burned_past_quota: int | None = None
    # None unless a per-token price is configured (see predictor.
    # overage_cost_usd) -- distinct from 0.0, which would claim "you owe
    # nothing" rather than "unpriced."
    cost_burned_past_quota_usd: float | None = None


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
    forecasts: tuple[Forecast, ...]
    alerts: tuple[Alert, ...]
