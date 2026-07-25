"""Codex CLI quota reader — parses local rollout JSONL session logs.

Two load-bearing rules, both learned the hard way from real rollout files:

- Classify each rate-limit block by `window_minutes`, NEVER by the
  primary/secondary key name — which window carries which key has already
  flipped once across Codex builds.
- Prefer the absolute `resets_at` (epoch seconds); fall back to computing it
  from `resets_in_seconds` + the event's own timestamp for older builds.

Malformed/missing data returns an empty list rather than a guessed value —
never fabricate a reading.

Every `token_count` event also carries `payload.info.last_token_usage` — the
real per-step token delta for that turn, independent of `used_percent`
(which clamps at 100 and goes blind to further usage). Ingested into
`store.token_events` the same way providers/claude.py ingests its own token
log, so `predictor.tokens_burned_past_quota` has real data once a window
saturates.
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
from tokenwatchdog.store import Store
from tokenwatchdog.timeutil import parse_iso_to_epoch

_W5H_MIN_MINUTES = 250
_W5H_MAX_MINUTES = 350
_WEEKLY_MIN_MINUTES = 9000

# A brand-new session file can be newest-by-mtime yet have no token_count
# event logged to it yet (nothing has happened in it). Fall back through a
# few more recent files rather than reporting no data while an older,
# still-valid reading (rate limits are account-wide, not per-session) sits
# right there.
_MAX_CANDIDATE_FILES = 5


class CodexProvider:
    name = "codex"

    def read(self, cfg: Config, store: Store) -> list[Window]:
        home = _codex_home(cfg)
        _ingest_token_events(cfg, home, store, time.time())
        candidates = _rollout_files_newest_first(home)[:_MAX_CANDIDATE_FILES]
        for path in candidates:
            event = _find_last_token_count_event(path)
            if event is None:
                continue
            windows = _windows_from_event(event, str(path))
            if windows:
                return windows
        return []


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
    """Upserts every token_count event's real per-step token delta
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
    source_key = f"{Provider.CODEX.value}:{home}"
    cursor = store.get_ingest_cursor(source_key)
    since_mtime = cursor if cursor is not None else now - _lookback_seconds(cfg)
    for path in _rollout_files_newest_first(home):
        try:
            if path.stat().st_mtime <= since_mtime:
                continue
        except OSError:
            continue
        for line in _iter_token_count_events(path):
            event = _token_event_from_line(line, path)
            if event is None:
                continue
            store.upsert_token_event(provider=Provider.CODEX, **event)
    store.set_ingest_cursor(source_key, now)


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
