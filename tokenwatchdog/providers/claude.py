"""Claude Code / Claude Desktop quota reader.

Three sources, in decreasing order of fidelity:

- **Source C — the claude CLI** (`providers/claude_cli.py`): the
  authoritative reading — spawns ``claude -p "/usage"`` and parses the
  server-reported percent and reset time for both windows. No credential
  handling anywhere in this tool: the ``claude`` binary owns its own auth.
  Throttled and cached because each spawn hits Anthropic's rate-limited
  usage endpoint; see that module for the details.
- **Source A — Claude Desktop** (`plan-usage-history.json`): a ready-made
  `fh` (five-hour %) / `sd` (weekly %) snapshot. No `resets_at` at all —
  deriving a reset time from the sample series is the predictor's job, not
  this module's.
- **Source B — token-compute**: sum per-message token usage from
  `~/.claude/projects/**/*.jsonl` and divide by an estimated limit, since
  the CLI never persists a percentage. The 5-hour window sums its own
  fixed block (see `blocks.block_anchor`) and can therefore report a real
  `resets_at`; the 7-day window has no comparable anchor rule and stays a
  trailing sum with an unknown reset. Every parsed usage
  line is durably upserted into `store.token_events` (dedup keep-LAST,
  ccusage bug #888) — that's what lets `_read_tokens` avoid re-scanning
  hundreds of MB of transcripts on every ~1-minute poll: only files touched
  within the retention lookback get re-read, and the running total is a SQL
  sum over what's already stored.

`cfg.claude.source`: "cli", "desktop", or "tokens" pins one source; "auto"
prefers the CLI reading, falls back to Desktop, then to tokens. In auto
mode a live CLI reading is returned *appended after* the Desktop
history: the engine keeps the last window per kind for the live view (CLI
wins) while still backfilling every retained Desktop sample into the store —
the predictor's burn profile learns from the denser history either way.
Token events are ingested into the store unconditionally — storage-first, so
a future predictor has real history to learn from even if another source
happened to be the live one most of the time.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenwatchdog.blocks import block_anchor
from tokenwatchdog.config import Config
from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.providers.claude_cli import CliUsageSource
from tokenwatchdog.store import Store, TokenEventRow
from tokenwatchdog.timeutil import parse_iso_to_epoch

_W5H_SECONDS = 5 * 3600
_WEEKLY_SECONDS = 7 * 24 * 3600
_W5H_WINDOW_MINUTES = 300
_WEEKLY_WINDOW_MINUTES = 10080

_DEFAULT_TIER = "default_claude_max_5x"
_FALLBACK_5H_TOKEN_LIMIT = 88_000  # community-estimated tokens/5h for the Max5x tier

# p90 limit_mode: don't trust a percentile estimate until this much history
# exists; below it, fall back to the static community number (5h) or to
# NO_DATA (weekly — there's no community number to fall back to, and
# guessing one would violate "never fabricate a reading").
_MIN_HISTORY_FOR_P90_5H_SECONDS = 2 * 3600
_MIN_HISTORY_FOR_P90_WEEKLY_SECONDS = 24 * 3600
_P90_SAFETY_FACTOR = 0.9  # treat the largest historical burst as ~90% of the true cap


class ClaudeProvider:
    name = "claude"

    def __init__(self, cli: CliUsageSource | None = None) -> None:
        # One long-lived source per provider: it owns the spawn throttle
        # that reconciles a 60s poll loop with the CLI's 180s floor.
        self._cli = cli if cli is not None else CliUsageSource()

    def read(self, cfg: Config, store: Store) -> list[Window]:
        now = time.time()
        _ingest_token_events(cfg, store, now)

        if cfg.claude.source == "tokens":
            return _read_tokens(cfg, store, now)
        exact_windows = (
            self._cli.read(cfg, now) if cfg.claude.source in ("auto", "cli") else []
        )
        if cfg.claude.source == "cli":
            return exact_windows

        desktop_windows = _read_desktop(cfg)
        if cfg.claude.source == "desktop":
            return desktop_windows
        # auto: Desktop history first so it backfills the store, the exact
        # reading last so it wins the live view (the engine keeps the last
        # window per kind).
        if exact_windows:
            return desktop_windows + exact_windows
        return desktop_windows if desktop_windows else _read_tokens(cfg, store, now)


# -- Source A: Claude Desktop --------------------------------------------


def _desktop_history_path() -> Path:
    return (
        Path.home()
        / "Library"
        / "Application Support"
        / "Claude"
        / "plan-usage-history.json"
    )


def _read_desktop(cfg: Config) -> list[Window]:
    """Every retained sample, oldest first — not just the latest.

    Desktop writes this file on its own every ~5 minutes and keeps a few
    days of history regardless of whether TokenWatchDog is running. If we
    only ever read the newest point, a period with the poll loop off would
    be a silent gap even though Desktop's own file already has the data
    to fill it: the engine inserts every returned Window (idempotently —
    already-seen ones are a no-op) and only the last one for a given
    window kind is used for the live forecast, so returning the full
    history here is both a correct backfill and a no-op on the samples
    already stored. Measured cost of re-inserting ~5 days of retained
    history (2880 rows) once already stored: ~17ms — negligible against
    a ~60s poll interval, so this isn't worth bounding further.
    """
    path = _desktop_history_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    samples = raw.get("samples")
    if not isinstance(samples, list):
        return []
    dated = sorted(
        (
            s
            for s in samples
            if isinstance(s, dict) and isinstance(s.get("t"), (int, float))
        ),
        key=lambda s: s["t"],
    )
    if not dated:
        return []
    # Desktop can retain samples from more than one Claude org if the
    # account has switched between them — each org has its own
    # independent quota, and the store has no org dimension, so mixing
    # them would look like an arbitrary usage jump or a spurious reset.
    # Only the org the LATEST sample belongs to is relevant to "current"
    # usage; older samples from a different org aren't this account's
    # history.
    current_org = dated[-1].get("org")
    windows: list[Window] = []
    for entry in dated:
        if entry.get("org") != current_org:
            continue
        usage = entry.get("u")
        if not isinstance(usage, dict):
            continue
        source_ts = entry["t"] / 1000.0
        fh, sd = usage.get("fh"), usage.get("sd")
        if isinstance(fh, (int, float)):
            windows.append(
                Window(
                    provider=Provider.CLAUDE,
                    kind=WindowKind.W5H,
                    used_percent=float(fh),
                    window_minutes=_W5H_WINDOW_MINUTES,
                    resets_at=None,
                    source_ts=source_ts,
                    is_estimated=False,
                    source_file=str(path),
                )
            )
        if isinstance(sd, (int, float)):
            windows.append(
                Window(
                    provider=Provider.CLAUDE,
                    kind=WindowKind.WEEKLY,
                    used_percent=float(sd),
                    window_minutes=_WEEKLY_WINDOW_MINUTES,
                    resets_at=None,
                    source_ts=source_ts,
                    is_estimated=False,
                    source_file=str(path),
                )
            )
    return windows


# -- Source B: token-compute ---------------------------------------------


def _claude_config_dir(cfg: Config) -> Path:
    if cfg.claude.config_dir:
        return Path(cfg.claude.config_dir).expanduser()
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".claude"


def _lookback_seconds(cfg: Config) -> float:
    weeks = max(cfg.predictor.history_retention_weeks, 1)
    return weeks * 7 * 24 * 3600


def _ingest_token_events(cfg: Config, store: Store, now: float) -> None:
    """Upsert newly-observed assistant usage lines into token_events.

    Only files modified since the last pass are read. The bound used to be
    the retention lookback alone, which skips almost nothing — practically
    every transcript is inside 8 weeks — so each ~1-minute poll re-parsed
    ~40k lines and re-upserted ~48k token events that were already stored,
    measured at 10.4s of a 13.5s tick. On a first run there is no cursor and
    the retention window is scanned in full, which is the backfill.
    """
    projects_dir = _claude_config_dir(cfg) / "projects"
    if not projects_dir.is_dir():
        return  # nothing scanned, so no cursor is recorded either
    source_key = f"{Provider.CLAUDE.value}:{projects_dir}"
    cursor = store.get_ingest_cursor(source_key)
    since_mtime = cursor if cursor is not None else now - _lookback_seconds(cfg)
    for path in projects_dir.glob("**/*.jsonl"):
        try:
            if path.stat().st_mtime <= since_mtime:
                continue
        except OSError:
            continue
        for line in _iter_assistant_usage_lines(path):
            event = _token_event_from_line(line)
            if event is None:
                continue
            store.upsert_token_event(provider=Provider.CLAUDE, **event)
    store.set_ingest_cursor(source_key, now)


def _iter_assistant_usage_lines(path: Path) -> Iterator[dict[str, Any]]:
    try:
        with path.open("r", errors="replace") as fh:
            for raw_line in fh:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    line = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(line, dict):
                    continue
                message = line.get("message")
                if not isinstance(message, dict) or message.get("role") != "assistant":
                    continue
                if not isinstance(message.get("usage"), dict):
                    continue
                yield line
    except OSError:
        return


def _token_event_from_line(line: dict[str, Any]) -> dict[str, Any] | None:
    message = line["message"]
    usage = message["usage"]
    ts = parse_iso_to_epoch(line.get("timestamp"))
    request_id = line.get("requestId")
    message_id = message.get("id")
    model = message.get("model")
    if (
        ts is None
        or not isinstance(request_id, str)
        or not isinstance(message_id, str)
        or not isinstance(model, str)
    ):
        return None
    return {
        "request_id": request_id,
        "message_id": message_id,
        "ts": ts,
        "model": model,
        "input_tokens": _usage_int(usage, "input_tokens"),
        "output_tokens": _usage_int(usage, "output_tokens"),
        "cache_creation": _usage_int(usage, "cache_creation_input_tokens"),
        "cache_read": _usage_int(usage, "cache_read_input_tokens"),
    }


def _usage_int(usage: dict[str, Any], key: str) -> int:
    value = usage.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _read_tokens(cfg: Config, store: Store, now: float) -> list[Window]:
    all_events = store.recent_token_events(
        Provider.CLAUDE, now - _lookback_seconds(cfg)
    )
    if not all_events:
        return []
    latest_ts = all_events[-1].ts
    tier = _read_rate_limit_tier()
    source_file = str(_claude_config_dir(cfg) / "projects")
    # The 5-hour window is a fixed block anchored at the first request after
    # an idle gap, not a rolling 5-hour sum -- so its usage is the block's
    # own tokens, and its reset is the block's own expiry. Summing a
    # trailing 5 hours instead kept pre-reset tokens in the total across a
    # boundary, which both over-reported usage and smeared the drop that
    # `predictor._is_reset` needs to see out over hours, so the reset was
    # usually never detected at all.
    w5h_anchor = block_anchor([e.ts for e in all_events], _W5H_SECONDS, now)

    windows: list[Window] = []
    for kind, window_minutes, window_seconds in (
        (WindowKind.W5H, _W5H_WINDOW_MINUTES, _W5H_SECONDS),
        (WindowKind.WEEKLY, _WEEKLY_WINDOW_MINUTES, _WEEKLY_SECONDS),
    ):
        # The weekly window has no equally real anchor rule (a 7-day gap in
        # activity is not what starts a new weekly cycle), so it stays a
        # trailing sum rather than getting an invented anchor.
        if kind is WindowKind.W5H:
            window_start = w5h_anchor if w5h_anchor is not None else now
            resets_at = w5h_anchor + _W5H_SECONDS if w5h_anchor is not None else None
        else:
            window_start = now - window_seconds
            resets_at = None
        limit = _estimate_limit_tokens(
            cfg, all_events, kind, window_seconds, tier, now, window_start
        )
        if limit is None:
            continue  # not enough history to estimate responsibly -> NO_DATA
        events_in_window = [e for e in all_events if e.ts >= window_start]
        windows.append(
            Window(
                provider=Provider.CLAUDE,
                kind=kind,
                used_percent=_usage_percent(events_in_window, limit),
                window_minutes=window_minutes,
                resets_at=resets_at,
                source_ts=latest_ts,
                is_estimated=True,
                source_file=source_file,
            )
        )
    return windows


def _usage_percent(events: list[TokenEventRow], limit_tokens: int) -> float:
    if limit_tokens <= 0:
        return 0.0
    total = sum(e.total_tokens for e in events)
    return min(100.0, 100.0 * total / limit_tokens)


def _read_rate_limit_tier() -> str:
    """Reads the top-level plan-tier hint only — never the org/seat identity
    alongside it, and never any token/auth value from this file."""
    path = Path.home() / ".claude.json"
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return _DEFAULT_TIER
    oauth = raw.get("oauthAccount")
    if not isinstance(oauth, dict):
        return _DEFAULT_TIER
    tier = oauth.get("userRateLimitTier")
    return tier if isinstance(tier, str) else _DEFAULT_TIER


def _estimate_limit_tokens(
    cfg: Config,
    events: list[TokenEventRow],
    kind: WindowKind,
    window_seconds: float,
    tier: str,
    now: float,
    window_start: float,
) -> int | None:
    is_5h = kind is WindowKind.W5H
    if cfg.claude.limit_mode == "plan":
        configured = cfg.claude.plan_limits_tokens.get(tier)
        if configured is not None:
            return int(configured)
        return _FALLBACK_5H_TOKEN_LIMIT if is_5h else None

    # Only look at windows that have fully COMPLETED before the current one
    # started. Including the live window here would make it one of the
    # candidates the max is taken over, so the live total could never
    # exceed limit * _P90_SAFETY_FACTOR — a self-referential ceiling that
    # silently caps every p90 reading at 90%, no matter how much is used.
    #
    # Cut at the live window's own start, not at `now - window_seconds`:
    # the latter creeps forward every tick, so the denominator moved
    # continuously and used_percent drifted for reasons that had nothing to
    # do with usage — phantom burn, and phantom drops for `_is_reset` to
    # misread. Anchored this way it only changes when a window actually
    # turns over, which is when it should.
    historical = [e for e in events if e.ts < window_start]
    span = historical[-1].ts - historical[0].ts if historical else 0.0
    min_history = (
        _MIN_HISTORY_FOR_P90_5H_SECONDS
        if is_5h
        else _MIN_HISTORY_FOR_P90_WEEKLY_SECONDS
    )
    if span < min_history:
        return _FALLBACK_5H_TOKEN_LIMIT if is_5h else None

    observed_max = _sliding_window_max_tokens(historical, window_seconds)
    if observed_max <= 0:
        return _FALLBACK_5H_TOKEN_LIMIT if is_5h else None
    return int(observed_max / _P90_SAFETY_FACTOR)


def _sliding_window_max_tokens(
    events: list[TokenEventRow], window_seconds: float
) -> int:
    """Largest token total found in any window_seconds-wide span of the
    (time-ordered) events — a cold-start P90-ish proxy for 'the biggest
    burst you were ever allowed,' used as a token-limit estimate until a
    real rate-limit event teaches us the truth."""
    if not events:
        return 0
    totals = [e.total_tokens for e in events]
    left = 0
    running = 0
    best = 0
    for right in range(len(events)):
        running += totals[right]
        while events[right].ts - events[left].ts > window_seconds:
            running -= totals[left]
            left += 1
        best = max(best, running)
    return best
