"""Claude CLI usage source — authoritative quota by asking Claude Code itself.

``claude -p "/usage" --output-format text`` is a local slash command: no LLM
call, no quota spent, finishes in under a second, and prints the same
server-side state that powers the interactive ``/usage`` screen::

    Current session: 88% used · resets Jul 29 at 5pm (America/New_York)
    Current week (all models): 5% used · resets Aug 5 at 1pm (America/New_York)

Compared to the other Claude sources this is the only one that reports the
*server's* numbers without this tool touching credentials at all — the
``claude`` binary owns its own token storage and refresh. The trade-offs it
carries, handled here:

- **Reset precision is one minute.** The CLI rounds the reset instant to the
  nearest minute before printing (Anthropic's actual reset timestamps sit a
  fraction of a second off the hour, e.g. ``:59:59.95``, which is why the
  Claude app — which truncates instead — can display one minute earlier).
  Still exact enough that the countdown is right to well under a minute,
  where sample-derived estimation had a 2–3 minute floor.
- **Each spawn hits Anthropic's usage endpoint**, which rate-limits
  aggressively. Spawns are capped at one per 180s — ticks in between reuse
  the last parse with its original source_ts — and any failed spawn doubles
  the wait, up to 30 minutes.
- **Failure never blocks the tick.** A missing binary, timeout, non-zero
  exit, or unrecognized output serves the cached windows while they are
  younger than 15 minutes, else nothing, so the desktop/token sources take
  over.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tokenwatchdog.config import Config
from tokenwatchdog.models import Provider, Window, WindowKind

_SPAWN_INTERVAL_S = 180.0  # the underlying endpoint 429s when polled faster
_BACKOFF_MAX_S = 1800.0
_CACHE_TTL_S = 900.0  # after this, stop shadowing the desktop/token sources
_SPAWN_TIMEOUT_S = 90.0

_SOURCE_LABEL = "claude -p /usage"

# "Current session: 88% used · resets Jul 29 at 5pm (America/New_York)"
# The reset clause is optional (a window with no usage may not print one),
# and the minutes part is optional (the CLI omits ":00").
_USAGE_LINE = re.compile(
    r"^Current (?P<label>session|week \(all models\)):\s+"
    r"(?P<percent>\d+(?:\.\d+)?)% used"
    r"(?:\s+·\s+resets\s+(?P<when>[^(]+?)\s+\((?P<tz>[^)]+)\))?\s*$"
)
_RESET_PHRASE = re.compile(
    r"^(?:(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+at\s+)?"
    r"(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?(?P<ampm>am|pm)$"
)
_MONTHS = {
    name: number
    for number, name in enumerate(
        "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), start=1
    )
}
_LABEL_KINDS = {
    "session": (WindowKind.W5H, 300),
    "week (all models)": (WindowKind.WEEKLY, 10080),
}

# stdout text, or None when the spawn itself failed
_SpawnFn = Callable[[Config], str | None]


class CliUsageSource:
    """One instance per ClaudeProvider — holds the throttle/cache state that
    makes a 60s poll loop compatible with a 180s-floor spawn cadence."""

    def __init__(self, *, spawn: _SpawnFn | None = None) -> None:
        self._spawn = spawn if spawn is not None else _spawn_claude_usage
        self._cache: list[Window] = []
        self._cache_at: float = 0.0
        self._next_spawn_at: float = 0.0
        self._backoff_s: float = _SPAWN_INTERVAL_S

    def read(self, cfg: Config, now: float) -> list[Window]:
        if now < self._next_spawn_at:
            return self._cached(now)
        output = self._spawn(cfg)
        windows = parse_usage_output(output, now) if output is not None else []
        if windows:
            self._cache = windows
            self._cache_at = now
            self._backoff_s = _SPAWN_INTERVAL_S
            self._next_spawn_at = now + _SPAWN_INTERVAL_S
            return list(windows)
        self._backoff_s = min(self._backoff_s * 2.0, _BACKOFF_MAX_S)
        self._next_spawn_at = now + self._backoff_s
        return self._cached(now)

    def _cached(self, now: float) -> list[Window]:
        if self._cache and now - self._cache_at < _CACHE_TTL_S:
            return list(self._cache)
        return []


def parse_usage_output(output: str, now: float) -> list[Window]:
    windows: list[Window] = []
    for line in output.splitlines():
        match = _USAGE_LINE.match(line.strip())
        if match is None:
            continue
        kind, minutes = _LABEL_KINDS[match.group("label")]
        windows.append(
            Window(
                provider=Provider.CLAUDE,
                kind=kind,
                used_percent=float(match.group("percent")),
                window_minutes=minutes,
                resets_at=_parse_reset(match.group("when"), match.group("tz"), now),
                source_ts=now,
                is_estimated=False,
                source_file=_SOURCE_LABEL,
            )
        )
    return windows


def _parse_reset(when: str | None, tz_name: str | None, now: float) -> float | None:
    """'Aug 5 at 1pm' + 'America/New_York' -> epoch seconds, else None.

    The CLI never prints a year, so pick the one that lands the reset in
    the only range a live window's reset can be in: (now - a rounding
    minute, now + 8 days]. A date more than ~2 days in the past belongs to
    the year ahead (the Dec -> Jan edge).
    """
    if when is None or tz_name is None:
        return None
    phrase = _RESET_PHRASE.match(when.strip())
    if phrase is None:
        return None
    try:
        tz = ZoneInfo(tz_name.strip())
    except (ZoneInfoNotFoundError, ValueError):
        return None
    hour = int(phrase.group("hour")) % 12
    if phrase.group("ampm") == "pm":
        hour += 12
    minute = int(phrase.group("minute") or 0)
    now_dt = datetime.fromtimestamp(now, tz=tz)
    if phrase.group("month") is None:
        month, day = now_dt.month, now_dt.day
    else:
        month, day = _MONTHS[phrase.group("month")], int(phrase.group("day"))
    try:
        candidate = now_dt.replace(
            month=month, day=day, hour=hour, minute=minute, second=0, microsecond=0
        )
    except ValueError:
        return None
    if (now_dt - candidate).days >= 2:
        candidate = candidate.replace(year=candidate.year + 1)
    return candidate.timestamp()


def _spawn_claude_usage(cfg: Config) -> str | None:
    binary = cfg.claude.cli_path or shutil.which("claude")
    if not binary:
        return None
    try:
        result = subprocess.run(
            [binary, "-p", "/usage", "--output-format", "text"],
            capture_output=True,
            text=True,
            timeout=_SPAWN_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout
