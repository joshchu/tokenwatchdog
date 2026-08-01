"""Codex app-server usage source — authoritative quota by asking codex itself.

``codex app-server`` is the JSON-RPC-over-stdio service behind the Codex
desktop app's own usage display. One short-lived spawn answers
``account/rateLimits/read`` with the SERVER's current numbers — no LLM call,
no quota spent, and no credential handling here: the codex binary owns its
own auth. It is fresher than any rollout file, because rollouts only record
a snapshot when a local turn completes — usage burned on other surfaces
(Codex web/cloud, another machine) is invisible to them until the next
local turn. Observed live: rollouts said 5% two hours after the account had
moved to 7%.

The exchange (JSONL frames, one JSON-RPC message per line)::

    -> {"id": 1, "method": "initialize", "params": {"clientInfo": {...}}}
    <- {"id": 1, "result": {...}}
    -> {"method": "initialized"}
    -> {"id": 2, "method": "account/rateLimits/read", "params": {}}
    <- {"id": 2, "result": {"rateLimits": {"primary": {"usedPercent": 7,
         "windowDurationMins": 10080, "resetsAt": 1786190296}, ...}}}

Unsolicited server notifications can interleave; anything that isn't the
awaited response id is skipped. Windows are classified by
``windowDurationMins`` exactly like the rollout reader — never by key name.

Operational rules (mirrors providers/claude_cli.py):

- spawns are capped at one per 180s; ticks in between reuse the last
  reading with its original source_ts, and every reading competes with the
  rollout snapshots under the same supersedes rule, so a genuinely fresher
  local rollout line still wins between spawns;
- a failed spawn doubles the wait, up to 30 minutes;
- cached readings stop shadowing the rollout fallback after 15 minutes;
- the subprocess is always terminated — a hung server cannot leak.
"""

from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable
from typing import IO, Any

from tokenwatchdog.config import Config
from tokenwatchdog.models import Provider, Window, WindowKind

_SPAWN_INTERVAL_S = 180.0
_BACKOFF_MAX_S = 1800.0
_CACHE_TTL_S = 900.0
_REPLY_TIMEOUT_S = 20.0

_SOURCE_LABEL = "codex app-server"

_W5H_MIN_MINUTES = 250
_W5H_MAX_MINUTES = 350
_WEEKLY_MIN_MINUTES = 9000

# The parsed `result` object of account/rateLimits/read, or None on any miss.
_QueryFn = Callable[[Config], "dict[str, Any] | None"]


class AppServerUsageSource:
    """One instance per CodexProvider — holds the throttle/cache state that
    makes a 60s poll loop compatible with a 180s spawn floor."""

    def __init__(self, *, query: _QueryFn | None = None) -> None:
        self._query = query if query is not None else _spawn_and_query
        self._cache: list[Window] = []
        self._cache_at: float = 0.0
        self._next_spawn_at: float = 0.0
        self._backoff_s: float = _SPAWN_INTERVAL_S

    def read(self, cfg: Config, now: float) -> list[Window]:
        if now < self._next_spawn_at:
            return self._cached(now)
        result = self._query(cfg)
        windows = windows_from_result(result, now) if result is not None else []
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


def windows_from_result(result: dict[str, Any], now: float) -> list[Window]:
    snapshot = result.get("rateLimits")
    if not isinstance(snapshot, dict):
        return []
    windows: list[Window] = []
    for key in ("primary", "secondary"):
        entry = snapshot.get(key)
        if not isinstance(entry, dict):
            continue
        used_percent = entry.get("usedPercent")
        window_minutes = entry.get("windowDurationMins")
        if not isinstance(used_percent, (int, float)) or not isinstance(
            window_minutes, (int, float)
        ):
            continue
        kind = _classify(int(window_minutes))
        if kind is None:
            continue
        resets_at = entry.get("resetsAt")
        windows.append(
            Window(
                provider=Provider.CODEX,
                kind=kind,
                used_percent=float(used_percent),
                window_minutes=int(window_minutes),
                resets_at=float(resets_at)
                if isinstance(resets_at, (int, float))
                else None,
                source_ts=now,
                is_estimated=False,
                source_file=_SOURCE_LABEL,
            )
        )
    return windows


def _classify(window_minutes: int) -> WindowKind | None:
    if _W5H_MIN_MINUTES <= window_minutes <= _W5H_MAX_MINUTES:
        return WindowKind.W5H
    if window_minutes >= _WEEKLY_MIN_MINUTES:
        return WindowKind.WEEKLY
    return None


# -- default (real) implementation of the query seam -----------------------


def _spawn_and_query(cfg: Config) -> dict[str, Any] | None:
    binary = cfg.codex.cli_path or shutil.which("codex")
    if not binary:
        return None
    try:
        process = subprocess.Popen(
            [binary, "app-server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        assert process.stdin is not None and process.stdout is not None
        replies = _reader_queue(process.stdout)
        _send(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "clientInfo": {
                        "name": "tokenwatchdog",
                        "title": "TokenWatchDog",
                        "version": "0.1.0",
                    }
                },
            },
        )
        if _await_reply(replies, want_id=1) is None:
            return None
        _send(process.stdin, {"jsonrpc": "2.0", "method": "initialized"})
        _send(
            process.stdin,
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "account/rateLimits/read",
                "params": {},
            },
        )
        reply = _await_reply(replies, want_id=2)
        if reply is None or not isinstance(reply.get("result"), dict):
            return None
        return reply["result"]
    except (OSError, ValueError):
        return None
    finally:
        process.kill()


def _send(stdin: IO[str], message: dict[str, Any]) -> None:
    stdin.write(json.dumps(message) + "\n")
    stdin.flush()


def _reader_queue(stdout: IO[str]) -> queue.Queue[str]:
    lines: queue.Queue[str] = queue.Queue()

    def pump() -> None:
        for line in stdout:
            lines.put(line)

    threading.Thread(target=pump, daemon=True).start()
    return lines


def _await_reply(
    lines: queue.Queue[str], *, want_id: int, timeout: float = _REPLY_TIMEOUT_S
) -> dict[str, Any] | None:
    """Next JSON-RPC message with the awaited id; skips notifications."""
    import time

    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        try:
            raw = lines.get(timeout=remaining)
        except queue.Empty:
            return None
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("id") == want_id:
            return message
