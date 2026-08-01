"""Codex CLI quota reader — parses local rollout JSONL session logs.

Three load-bearing rules, all learned the hard way from real rollout files:

- Classify each rate-limit block by `window_minutes`, NEVER by the
  primary/secondary key name — which window carries which key has already
  flipped once across Codex builds.
- Prefer the absolute `resets_at` (epoch seconds); fall back to computing it
  from `resets_in_seconds` + the event's own timestamp for older builds.
- Never trust "newest file, last line" alone: sessions interleave, and a
  session resumed after sitting idle can re-emit its last-known rate limits
  under a brand-new timestamp. Recent candidates are compared and the
  snapshot announcing the latest `resets_at` wins (see `_supersedes`).

Rollouts also lag the account whenever usage happens on a surface that
doesn't write local sessions (Codex web/cloud, another machine), so the
freshest reading comes from `providers/codex_app_server.py` — asking the
codex binary itself — and competes with the rollout snapshots under the
same rule.

Malformed/missing data returns an empty list rather than a guessed value —
never fabricate a reading.

Every usage-bearing `token_count` event also carries
`payload.info.last_token_usage` — the real per-step token delta for that turn,
independent of `used_percent` (which clamps at 100 and goes blind to further
usage). Codex can repeat that delta on a later status-only snapshot; the
unchanged cumulative total distinguishes and removes those duplicates.
The remaining events are ingested into `store.token_events` the same way
providers/claude.py ingests its own token log, so
`predictor.tokens_burned_past_quota` has real data once a window saturates.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from tokenwatchdog.config import Config
from tokenwatchdog.models import Provider, Window, WindowKind
from tokenwatchdog.providers.codex_app_server import AppServerUsageSource
from tokenwatchdog.store import Store
from tokenwatchdog.timeutil import parse_iso_to_epoch

_W5H_MIN_MINUTES = 250
_W5H_MAX_MINUTES = 350
_WEEKLY_MIN_MINUTES = 9000

# A brand-new session file can be newest-by-mtime yet have no token_count
# event logged to it yet (nothing has happened in it), and the file with the
# newest mtime is not necessarily the one holding the newest account state
# (rate limits are account-wide, not per-session). Every candidate's last
# snapshot competes; this only bounds how far back the scan reaches.
_MAX_CANDIDATE_FILES = 5

# Two snapshots whose resets_at differ by no more than this are treated as
# announcing the same cycle state: a rolling window's resets_at is a sliding
# projection that jitters by seconds between requests, while a genuine
# window rotation moves it forward by days.
_RESETS_TIE_TOLERANCE_S = 60.0

# Bump this when ingestion semantics change in a way that requires existing
# rollout files to be revisited. V2 removes repeated cumulative snapshots that
# V1 stored as if each one were new usage.
_INGEST_VERSION = 2


class CodexProvider:
    name = "codex"

    def __init__(self, app_server: AppServerUsageSource | None = None) -> None:
        # One long-lived app-server source per provider: it owns the spawn
        # throttle that reconciles a 60s poll loop with a 180s spawn floor.
        self._app_server = (
            app_server if app_server is not None else AppServerUsageSource()
        )
        # path -> (mtime, last rate-limits event). Comparing candidates means
        # parsing several whole rollout files, and long-lived session files
        # reach many MB; re-parsing unchanged ones every ~60s tick is the
        # same measured mistake the ingest cursor below already fixed.
        self._snapshot_cache: dict[Path, tuple[float, dict[str, Any] | None]] = {}

    def read(self, cfg: Config, store: Store) -> list[Window]:
        now = time.time()
        home = _codex_home(cfg)
        _ingest_token_events(cfg, home, store, now)
        best: dict[WindowKind, Window] = {}

        def compete(window: Window) -> None:
            incumbent = best.get(window.kind)
            if incumbent is None or _supersedes(window, incumbent):
                best[window.kind] = window

        # The live app-server reading is just another candidate under the
        # same supersedes rule: rollouts lag the account whenever usage
        # happens on another surface (web/cloud/another machine), while a
        # rollout line a local turn wrote seconds ago is fresher than a
        # throttled reading from up to 180s back — per-window recency and
        # announced-reset arbitration sort both cases out.
        for window in self._app_server.read(cfg, now):
            compete(window)
        candidates = _rollout_files_newest_first(home)[:_MAX_CANDIDATE_FILES]
        for path in candidates:
            event = self._last_snapshot(path)
            if event is None:
                continue
            for window in _windows_from_event(event, str(path)):
                compete(window)
        kept = set(candidates)
        self._snapshot_cache = {
            path: entry for path, entry in self._snapshot_cache.items() if path in kept
        }
        return [best[kind] for kind in sorted(best, key=lambda kind: kind.value)]

    def _last_snapshot(self, path: Path) -> dict[str, Any] | None:
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        cached = self._snapshot_cache.get(path)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        event = _find_last_token_count_event(path)
        self._snapshot_cache[path] = (mtime, event)
        return event


def _supersedes(challenger: Window, incumbent: Window) -> bool:
    """Pick between two sessions' readings of the same window.

    Line recency alone cannot arbitrate: a session resumed after sitting
    idle re-emits its LAST-KNOWN rate limits under a brand-new timestamp
    (observed live: an overnight session's compaction task echoed the
    previous quota cycle's 99%-used/old-resets_at an hour after a fresh
    session had already reported the rotated window at 5%). The announced
    reset time can arbitrate, because it only ever moves forward — sliding
    forward on a rolling window, jumping forward by days when the window
    rotates — so the snapshot announcing the later reset is the newer
    account state no matter when its line was written. Within the jitter
    tolerance, recency breaks the tie.
    """
    if challenger.resets_at is not None and incumbent.resets_at is not None:
        if abs(challenger.resets_at - incumbent.resets_at) > _RESETS_TIE_TOLERANCE_S:
            return challenger.resets_at > incumbent.resets_at
    elif (challenger.resets_at is None) != (incumbent.resets_at is None):
        # A snapshot that knows its reset outranks one that doesn't.
        return incumbent.resets_at is None
    return challenger.source_ts > incumbent.source_ts


def _codex_home(cfg: Config) -> Path:
    if cfg.codex.home:
        return Path(cfg.codex.home).expanduser()
    env = os.environ.get("CODEX_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".codex"


def _rollout_files_newest_first(home: Path) -> list[Path]:
    if not home.is_dir():
        return []
    candidates = list(home.glob("sessions/**/rollout-*.jsonl"))
    candidates += list(home.glob("archived_sessions/**/rollout-*.jsonl"))
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates


def _windows_from_event(event: dict[str, Any], source_file: str) -> list[Window]:
    event_epoch = parse_iso_to_epoch(event.get("timestamp"))
    if event_epoch is None:
        return []
    rate_limits = event["payload"]["rate_limits"]
    windows: list[Window] = []
    for key in ("primary", "secondary"):
        window = _window_from_block(rate_limits.get(key), event_epoch, source_file)
        if window is not None:
            windows.append(window)
    return windows


def _classify(window_minutes: int) -> WindowKind | None:
    if _W5H_MIN_MINUTES <= window_minutes <= _W5H_MAX_MINUTES:
        return WindowKind.W5H
    if window_minutes >= _WEEKLY_MIN_MINUTES:
        return WindowKind.WEEKLY
    return None


def _iter_token_count_events(path: Path) -> Iterator[dict[str, Any]]:
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
                payload = line.get("payload")
                if isinstance(payload, dict) and payload.get("type") == "token_count":
                    yield line
    except OSError:
        return


def _find_last_token_count_event(path: Path) -> dict[str, Any] | None:
    last: dict[str, Any] | None = None
    for line in _iter_token_count_events(path):
        if isinstance(line["payload"].get("rate_limits"), dict):
            last = line
    return last


def _lookback_seconds(cfg: Config) -> float:
    weeks = max(cfg.predictor.history_retention_weeks, 1)
    return weeks * 7 * 24 * 3600


def _ingest_token_events(cfg: Config, home: Path, store: Store, now: float) -> None:
    """Upserts every usage-bearing token_count event's real per-step delta
    (`payload.info.last_token_usage`) into store.token_events -- the
    signal `used_percent` alone can't provide once it clamps at 100%.
    Dedup key is the session file + the event's own timestamp (Codex has
    no message/request id of its own), so re-scanning an already-seen
    line is a harmless no-op re-upsert of identical values -- mirrors
    providers/claude.py's own token-log ingestion, including its cursor:
    only rollouts modified since the last pass are read, since re-parsing
    unchanged ones cost 2.2s per tick to learn nothing. First run has no
    cursor and scans the retention window as a backfill."""
    if not home.is_dir():
        # No cursor is recorded for a directory that wasn't there. Recording
        # one anyway would mark a retention backfill as "done" against
        # nothing, so sessions appearing later with honest older mtimes -- a
        # restore, a sync, an install-then-import -- would never be read.
        return
    source_key = f"{Provider.CODEX.value}:v{_INGEST_VERSION}:{home}"
    cursor = store.get_ingest_cursor(source_key)
    since_mtime = cursor if cursor is not None else now - _lookback_seconds(cfg)
    for path in _rollout_files_newest_first(home):
        try:
            if path.stat().st_mtime <= since_mtime:
                continue
        except OSError:
            continue
        previous_total: int | None = None
        for line in _iter_token_count_events(path):
            cumulative_total = _cumulative_total_tokens(line)
            if cumulative_total is not None and cumulative_total == previous_total:
                # Codex sometimes emits the same total_token_usage snapshot
                # again under a new timestamp. last_token_usage is repeated on
                # that line too, but no new tokens were consumed: the
                # cumulative total did not move. V1 ingested these timestamps
                # as separate deltas, so delete the exact old identity as part
                # of this versioned rescan as well as skipping it now.
                timestamp = line.get("timestamp")
                if isinstance(timestamp, str):
                    store.delete_token_event(
                        provider=Provider.CODEX,
                        request_id=path.stem,
                        message_id=timestamp,
                    )
                continue
            if cumulative_total is not None:
                previous_total = cumulative_total
            event = _token_event_from_line(line, path)
            if event is None:
                continue
            store.upsert_token_event(provider=Provider.CODEX, **event)
    store.set_ingest_cursor(source_key, now)


def _cumulative_total_tokens(line: dict[str, Any]) -> int | None:
    """Codex's per-session running total, used only to recognize repeats."""
    info = line["payload"].get("info")
    total_usage = info.get("total_token_usage") if isinstance(info, dict) else None
    total = total_usage.get("total_tokens") if isinstance(total_usage, dict) else None
    if not isinstance(total, (int, float)):
        return None
    return int(total)


def _token_event_from_line(line: dict[str, Any], path: Path) -> dict[str, Any] | None:
    ts = parse_iso_to_epoch(line.get("timestamp"))
    info = line["payload"].get("info")
    last_usage = info.get("last_token_usage") if isinstance(info, dict) else None
    if ts is None or not isinstance(last_usage, dict):
        return None
    input_tokens = last_usage.get("input_tokens")
    output_tokens = last_usage.get("output_tokens")
    if not isinstance(input_tokens, (int, float)) or not isinstance(
        output_tokens, (int, float)
    ):
        return None
    return {
        "request_id": path.stem,
        "message_id": line["timestamp"],
        "ts": ts,
        "model": "codex",
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        # Codex's cache breakdown (cached_input_tokens, cache_write_
        # input_tokens) is already folded INTO input/output above, unlike
        # Claude's, where cache reads/writes are separate additive
        # dimensions -- setting these to 0 avoids double-counting rather
        # than reusing Claude's schema-shape for a different accounting
        # model.
        "cache_creation": 0,
        "cache_read": 0,
    }


def _resets_at(block: dict[str, Any], event_epoch: float | None) -> float | None:
    resets_at = block.get("resets_at")
    if isinstance(resets_at, (int, float)):
        return float(resets_at)
    resets_in = block.get("resets_in_seconds")
    if isinstance(resets_in, (int, float)) and event_epoch is not None:
        return event_epoch + float(resets_in)
    return None


def _window_from_block(
    block: Any, event_epoch: float, source_file: str
) -> Window | None:
    if not isinstance(block, dict):
        return None
    window_minutes = block.get("window_minutes")
    used_percent = block.get("used_percent")
    if not isinstance(window_minutes, (int, float)) or not isinstance(
        used_percent, (int, float)
    ):
        return None
    kind = _classify(int(window_minutes))
    if kind is None:
        return None
    return Window(
        provider=Provider.CODEX,
        kind=kind,
        used_percent=float(used_percent),
        window_minutes=int(window_minutes),
        resets_at=_resets_at(block, event_epoch),
        source_ts=event_epoch,
        is_estimated=False,
        source_file=source_file,
    )
