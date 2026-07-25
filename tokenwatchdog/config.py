"""TOML configuration for TokenWatchDog.

Config lives at ~/.tokenwatchdog/config.toml. A missing file is created with
these defaults on first run.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from datetime import time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

DEFAULT_CONFIG_DIR = Path.home() / ".tokenwatchdog"
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.toml"
DEFAULT_DB_PATH = DEFAULT_CONFIG_DIR / "history.db"

_DAY_NUMBERS = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

_VALID_PREDICTOR_MODELS = {
    "auto",
    "linear",
    "seasonal_profile",
    "montecarlo",
    "holtwinters",
}
_VALID_CLAUDE_SOURCES = {"auto", "desktop", "tokens"}
_VALID_LIMIT_MODES = {"p90", "plan"}
_VALID_WINDOW_KINDS = {"weekly", "w5h"}


class ConfigError(ValueError):
    """Raised when config.toml has an invalid value. Fail loud, not silently."""


@dataclass(frozen=True)
class ProvidersConfig:
    codex: bool = True
    claude: bool = True


@dataclass(frozen=True)
class CodexConfig:
    home: str = ""  # "" -> $CODEX_HOME or ~/.codex
    # 0.0 = not configured -> no $ estimate shown, tokens-past-quota shows
    # instead. Codex's own API never reports a price -- these only ever
    # reflect what you configure yourself, matching your actual plan's
    # overage/credit rate. Output pricing (which includes reasoning
    # tokens) is usually several times input pricing.
    input_price_per_million_usd: float = 0.0
    output_price_per_million_usd: float = 0.0


@dataclass(frozen=True)
class ClaudeConfig:
    source: str = "auto"  # "auto" (desktop->tokens) | "desktop" | "tokens"
    config_dir: str = ""  # "" -> $CLAUDE_CONFIG_DIR or ~/.claude
    limit_mode: str = "p90"  # "p90" | "plan"
    plan_limits_tokens: dict[str, int] = field(
        default_factory=lambda: {"default_claude_max_5x": 88_000}
    )
    # 0.0 = not configured -> no $ estimate shown, tokens-past-quota shows
    # instead. Four rates, not two: Anthropic prices cache writes and
    # cache reads on their own distinct tiers (usually well below base
    # input for reads), unlike Codex where those are already folded into
    # input/output token counts (see providers/codex.py) -- reusing
    # Codex's two-rate formula here would silently misprice real,
    # separately-billed usage.
    input_price_per_million_usd: float = 0.0
    output_price_per_million_usd: float = 0.0
    cache_write_price_per_million_usd: float = 0.0
    cache_read_price_per_million_usd: float = 0.0


@dataclass(frozen=True)
class WindowsConfig:
    watch: list[str] = field(default_factory=lambda: ["weekly", "w5h"])


@dataclass(frozen=True)
class ThresholdsConfig:
    warn_percent: float = 90.0
    burn_min_percent: float = 25.0
    # The burn alert fires only when projected to actually exhaust the
    # window within this many hours -- not merely "sometime before the
    # window resets," which could be days away and isn't urgent.
    burn_alert_within_hours: float = 1.0
    # How old a reading may be before its BURN RATE is treated as stale.
    # Per-window because the two live on different timescales: minutes after
    # your last request a 5-hour window's rate really is meaningless,
    # while a 7-day window's sustained pace survives a lunch break. One
    # shared number had to be wrong for one of them, and at 10 minutes it
    # was wrong for the weekly window -- which is the one the exhaustion
    # forecast is actually about. Neither affects the used-% level, which
    # stays true for the rest of the cycle regardless (see alerts.py).
    stale_after_minutes_w5h: float = 10.0
    stale_after_minutes_weekly: float = 180.0
    threshold_hysteresis: float = 10.0


@dataclass(frozen=True)
class BurnConfig:
    # Both of these double as the token-throughput averaging window (where
    # they mean "sustained pace" rather than "typing right now") and as the
    # fallback percent-slope lookback. Both were widened on measurement --
    # see scripts/backtest.py, scored on one-step-ahead rate error against
    # 750+ real moments per window:
    #
    #   weekly 60m -> 360m: rate MAE 1.52 -> 1.36, stdev 2.67 -> 1.80, and
    #     readings reporting a flat zero 47% -> 15%. A whole-number weekly
    #     percentage needs hours of span to move at all (~0.6%/h evenly
    #     spread means a 60-minute window sees a change of 0 or 1 and
    #     nothing between). 720m was no better than noise and costs twice
    #     the responsiveness.
    #   w5h 15m -> 60m: MAE is FLAT here (8.8-9.0, inside noise), but bias
    #     +1.92 -> +0.57 -- 15 minutes is about one request, so the estimate
    #     was systematically over-reading the rate by ~2%/h -- and flat-zero
    #     readings 59% -> 52%. Chosen on bias, not on accuracy.
    lookback_weekly_minutes: float = 360.0
    lookback_w5h_minutes: float = 60.0


@dataclass(frozen=True)
class PredictorConfig:
    model: str = "auto"
    history_retention_weeks: int = 8
    mc_runs: int = 2000
    bucket_decay_halflife_days: int = 21


@dataclass(frozen=True)
class WorkingHoursConfig:
    enabled: bool = True
    start: str = "09:00"
    end: str = "17:00"
    days: list[str] = field(default_factory=lambda: ["Mon", "Tue", "Wed", "Thu", "Fri"])

    def start_time(self) -> dt_time:
        return _parse_hhmm(self.start)

    def end_time(self) -> dt_time:
        return _parse_hhmm(self.end)

    def day_numbers(self) -> frozenset[int]:
        return frozenset(_DAY_NUMBERS[d] for d in self.days)


@dataclass(frozen=True)
class NotificationsConfig:
    enabled: bool = True
    sound: str = "bark"  # "bark" speaks "Woof! Woof!"; or any macOS system sound name
    notifier: str = ""  # "" -> auto (terminal-notifier then osascript)


@dataclass(frozen=True)
class UiConfig:
    mascot: bool = True


@dataclass(frozen=True)
class Config:
    poll_interval_seconds: float = 60.0
    timezone: str = ""  # "" -> system local
    providers: ProvidersConfig = field(default_factory=ProvidersConfig)
    codex: CodexConfig = field(default_factory=CodexConfig)
    claude: ClaudeConfig = field(default_factory=ClaudeConfig)
    windows: WindowsConfig = field(default_factory=WindowsConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    burn: BurnConfig = field(default_factory=BurnConfig)
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    working_hours: WorkingHoursConfig = field(default_factory=WorkingHoursConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    ui: UiConfig = field(default_factory=UiConfig)


_SECTION_FIELDS: dict[str, type] = {
    "providers": ProvidersConfig,
    "codex": CodexConfig,
    "claude": ClaudeConfig,
    "windows": WindowsConfig,
    "thresholds": ThresholdsConfig,
    "burn": BurnConfig,
    "predictor": PredictorConfig,
    "working_hours": WorkingHoursConfig,
    "notifications": NotificationsConfig,
    "ui": UiConfig,
}


def load_config(path: Path | None = None) -> Config:
    """Load config.toml, writing defaults on first run. Validates before return."""
    path = path or DEFAULT_CONFIG_PATH
    if path.exists():
        text = path.read_text()
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = render_default_toml()
        path.write_text(text)
    cfg = _build_config(tomllib.loads(text))
    _validate(cfg)
    return cfg


def _build_config(raw: dict) -> Config:
    top_level_names = {"poll_interval_seconds", "timezone"}
    unknown_top = set(raw) - top_level_names - set(_SECTION_FIELDS)
    if unknown_top:
        raise ConfigError(f"config.toml: unknown key(s) {sorted(unknown_top)}")
    top_level = {name: raw[name] for name in top_level_names if name in raw}
    sections = {
        name: _build_section(cls, raw.get(name, {}))
        for name, cls in _SECTION_FIELDS.items()
    }
    return Config(**top_level, **sections)


_RENAMED_KEYS = {
    # old key -> new key. Grows (never shrinks) as fields get renamed, so
    # anyone upgrading gets a fix, not a bare crash on their existing
    # config.toml.
    "burn_margin_hours": "burn_alert_within_hours",
    "stale_after_minutes": "stale_after_minutes_w5h / stale_after_minutes_weekly",
    # Removed rather than renamed: both promised behavior that was never
    # built (there is no seasonal_profile tier for min_history_days to
    # graduate into, and the P50/P90 band is now always shown when the
    # montecarlo model produces one). Named here so an existing config.toml
    # says that instead of failing with a bare "unknown key".
    "min_history_days": "removed (predictor.model selects the model directly)",
    "report_uncertainty": "removed (the montecarlo band is always shown)",
}


def _build_section(cls: type, data: dict):
    valid_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - valid_names
    if unknown:
        parts = [
            f"{key!r} → {_RENAMED_KEYS[key]}" if key in _RENAMED_KEYS else repr(key)
            for key in sorted(unknown)
        ]
        raise ConfigError(
            f"{cls.__name__}: unknown key(s) in config.toml: {', '.join(parts)} "
            f"— update or remove them"
        )
    return cls(**data)


def _parse_hhmm(value: str) -> dt_time:
    try:
        hour_str, minute_str = value.split(":", 1)
        return dt_time(hour=int(hour_str), minute=int(minute_str))
    except (ValueError, AttributeError) as exc:
        raise ConfigError(f"working_hours: invalid HH:MM value {value!r}") from exc


def _validate(cfg: Config) -> None:
    if cfg.timezone:
        try:
            ZoneInfo(cfg.timezone)
        except ZoneInfoNotFoundError as exc:
            raise ConfigError(
                f"timezone must be a valid IANA name, got {cfg.timezone!r}"
            ) from exc
    if not 0.0 <= cfg.thresholds.warn_percent <= 100.0:
        raise ConfigError("thresholds.warn_percent must be within [0, 100]")
    if not 0.0 <= cfg.thresholds.burn_min_percent <= 100.0:
        raise ConfigError("thresholds.burn_min_percent must be within [0, 100]")
    if cfg.codex.input_price_per_million_usd < 0.0:
        raise ConfigError("codex.input_price_per_million_usd must not be negative")
    if cfg.codex.output_price_per_million_usd < 0.0:
        raise ConfigError("codex.output_price_per_million_usd must not be negative")
    for field_name in (
        "input_price_per_million_usd",
        "output_price_per_million_usd",
        "cache_write_price_per_million_usd",
        "cache_read_price_per_million_usd",
    ):
        if getattr(cfg.claude, field_name) < 0.0:
            raise ConfigError(f"claude.{field_name} must not be negative")
    if cfg.predictor.model not in _VALID_PREDICTOR_MODELS:
        raise ConfigError(
            f"predictor.model must be one of {sorted(_VALID_PREDICTOR_MODELS)}, "
            f"got {cfg.predictor.model!r}"
        )
    if cfg.claude.source not in _VALID_CLAUDE_SOURCES:
        raise ConfigError(
            f"claude.source must be one of {sorted(_VALID_CLAUDE_SOURCES)}, "
            f"got {cfg.claude.source!r}"
        )
    if cfg.claude.limit_mode not in _VALID_LIMIT_MODES:
        raise ConfigError(
            f"claude.limit_mode must be one of {sorted(_VALID_LIMIT_MODES)}, "
            f"got {cfg.claude.limit_mode!r}"
        )
    unknown_kinds = set(cfg.windows.watch) - _VALID_WINDOW_KINDS
    if unknown_kinds:
        raise ConfigError(f"windows.watch: unknown kind(s) {sorted(unknown_kinds)}")
    unknown_days = set(cfg.working_hours.days) - set(_DAY_NUMBERS)
    if unknown_days:
        raise ConfigError(f"working_hours.days: unknown day(s) {sorted(unknown_days)}")
    if cfg.working_hours.enabled and not cfg.working_hours.days:
        raise ConfigError("working_hours.days must be non-empty when enabled")
    start_time = _parse_hhmm(cfg.working_hours.start)
    end_time = _parse_hhmm(cfg.working_hours.end)
    if cfg.working_hours.enabled and start_time >= end_time:
        raise ConfigError(
            "working_hours.start must be earlier than working_hours.end "
            f"(got {cfg.working_hours.start!r} >= {cfg.working_hours.end!r}) — "
            "an empty or reversed interval would spin forever projecting "
            "an exhaustion date"
        )


def resolve_timezone(cfg: Config) -> tzinfo:
    """cfg.timezone if set, else the system's IANA zone, else a fixed offset."""
    if cfg.timezone:
        return ZoneInfo(cfg.timezone)
    local_name = _local_iana_name()
    if local_name:
        try:
            return ZoneInfo(local_name)
        except ZoneInfoNotFoundError:
            pass
    fallback = datetime.now().astimezone().tzinfo
    assert fallback is not None  # astimezone() on a naive datetime always attaches one
    return fallback


def _local_iana_name() -> str | None:
    etc_localtime = Path("/etc/localtime")
    if not etc_localtime.is_symlink():
        return None
    target = os.readlink(etc_localtime)
    marker = "zoneinfo/"
    if marker not in target:
        return None
    return target.split(marker, 1)[1]


def render_default_toml() -> str:
    return _DEFAULT_TOML


_DEFAULT_TOML = """\
# TokenWatchDog configuration. Delete any key to fall back to its built-in
# default (see tokenwatchdog/config.py for the full set of dataclasses).

poll_interval_seconds = 60
timezone = ""                        # "" -> system local; on Windows, set an IANA
                                     # name (e.g. "America/New_York") for DST

[providers]
codex = true
claude = true

[codex]
home = ""                            # "" -> $CODEX_HOME or ~/.codex
input_price_per_million_usd = 0.0    # set to your plan's real overage/credit rate
output_price_per_million_usd = 0.0   # to show $ burned past quota instead of just tokens

[claude]
source = "auto"                      # "auto" (desktop->tokens) | "desktop" | "tokens"
config_dir = ""                      # "" -> $CLAUDE_CONFIG_DIR or ~/.claude
limit_mode = "p90"                   # "p90" | "plan"
input_price_per_million_usd = 0.0        # set to your plan's real overage/credit
output_price_per_million_usd = 0.0       # rate to show $ burned past quota instead
cache_write_price_per_million_usd = 0.0  # of just tokens -- Anthropic prices cache
cache_read_price_per_million_usd = 0.0   # writes/reads separately from input/output

[claude.plan_limits_tokens]          # used when limit_mode = "plan"
default_claude_max_5x = 88000

[windows]
watch = ["weekly", "w5h"]

[thresholds]
warn_percent = 90.0
burn_min_percent = 25.0
burn_alert_within_hours = 1.0        # alert only if projected to exhaust this soon
stale_after_minutes_w5h = 10         # how old a reading may be before its BURN RATE
stale_after_minutes_weekly = 180     # is stale (the used-% level never goes stale)
threshold_hysteresis = 10.0

[burn]
lookback_weekly_minutes = 360        # doubles as the token-throughput averaging
lookback_w5h_minutes = 60            # window; both widened on measured rate error

[predictor]
model = "auto"                       # auto | linear | seasonal_profile | montecarlo | holtwinters
history_retention_weeks = 8
mc_runs = 2000
bucket_decay_halflife_days = 21

[working_hours]
enabled = true
start = "09:00"
end = "17:00"
days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

[notifications]
enabled = true
sound = "bark"                       # "bark" speaks "Woof! Woof!" via `say`;
                                      # or any macOS system sound name (e.g. "Submarine")
notifier = ""                        # "" -> auto (terminal-notifier then osascript)

[ui]
mascot = true
"""
