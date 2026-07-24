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


@dataclass(frozen=True)
class ClaudeConfig:
    source: str = "auto"  # "auto" (desktop->tokens) | "desktop" | "tokens"
    config_dir: str = ""  # "" -> $CLAUDE_CONFIG_DIR or ~/.claude
    limit_mode: str = "p90"  # "p90" | "plan"
    plan_limits_tokens: dict[str, int] = field(
        default_factory=lambda: {"default_claude_max_5x": 88_000}
    )


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
    stale_after_minutes: float = 10.0
    threshold_hysteresis: float = 10.0


@dataclass(frozen=True)
class BurnConfig:
    lookback_weekly_minutes: float = 60.0
    lookback_w5h_minutes: float = 15.0


@dataclass(frozen=True)
class PredictorConfig:
    model: str = "auto"
    min_history_days: int = 7
    report_uncertainty: bool = True
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
}


def _build_section(cls: type, data: dict):
    valid_names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - valid_names
    if unknown:
        parts = [
            f"{key!r} was renamed to {_RENAMED_KEYS[key]!r}"
            if key in _RENAMED_KEYS
            else repr(key)
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
    if not 0.0 <= cfg.thresholds.warn_percent <= 100.0:
        raise ConfigError("thresholds.warn_percent must be within [0, 100]")
    if not 0.0 <= cfg.thresholds.burn_min_percent <= 100.0:
        raise ConfigError("thresholds.burn_min_percent must be within [0, 100]")
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
timezone = ""                        # "" -> system local

[providers]
codex = true
claude = true

[codex]
home = ""                            # "" -> $CODEX_HOME or ~/.codex

[claude]
source = "auto"                      # "auto" (desktop->tokens) | "desktop" | "tokens"
config_dir = ""                      # "" -> $CLAUDE_CONFIG_DIR or ~/.claude
limit_mode = "p90"                   # "p90" | "plan"

[claude.plan_limits_tokens]          # used when limit_mode = "plan"
default_claude_max_5x = 88000

[windows]
watch = ["weekly", "w5h"]

[thresholds]
warn_percent = 90.0
burn_min_percent = 25.0
burn_alert_within_hours = 1.0        # alert only if projected to exhaust this soon
stale_after_minutes = 10
threshold_hysteresis = 10.0

[burn]
lookback_weekly_minutes = 60
lookback_w5h_minutes = 15

[predictor]
model = "auto"                       # auto | linear | seasonal_profile | montecarlo | holtwinters
min_history_days = 7
report_uncertainty = true
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
