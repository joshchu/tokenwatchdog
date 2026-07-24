"""Codex rollout-JSONL reader tests.

Fixtures are built inline rather than as separate files under
tests/fixtures/ — each test is then readable standalone without
cross-referencing a fixture file.
"""

from __future__ import annotations

import json
import os
import time as time_mod

from tokenwatchdog.config import load_config
from tokenwatchdog.models import Provider, WindowKind
from tokenwatchdog.providers.codex import CodexProvider
from tokenwatchdog.timeutil import parse_iso_to_epoch


def _write_rollout(codex_home, lines, *, day="15"):
    session_dir = codex_home / "sessions" / "2026" / "07" / day
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "rollout-test.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    return path


def _cfg_for(tmp_path, codex_home):
    config_path = tmp_path / "config.toml"
    config_path.write_text(f'[codex]\nhome = "{codex_home}"\n')
    return load_config(config_path)


def _token_count_event(ts, rate_limits, last_token_usage=None):
    info = {"last_token_usage": last_token_usage} if last_token_usage else {}
    return {
        "timestamp": ts,
        "type": "event_msg",
        "payload": {"type": "token_count", "info": info, "rate_limits": rate_limits},
    }


def test_current_weekly_only_build(tmp_path, store):
    codex_home = tmp_path / "codex_home"
    _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-07-24T01:00:00.000Z",
                {
                    "primary": {
                        "used_percent": 87.5,
                        "window_minutes": 10080,
                        "resets_at": 1785260936,
                    },
                    "secondary": None,
                    "plan_type": "team",
                },
            )
        ],
    )
    cfg = _cfg_for(tmp_path, codex_home)
    windows = CodexProvider().read(cfg, store)
    assert len(windows) == 1
    window = windows[0]
    assert window.provider is Provider.CODEX
    assert window.kind is WindowKind.WEEKLY
    assert window.used_percent == 87.5
    assert window.resets_at == 1785260936.0
    assert window.is_estimated is False


def test_legacy_5h_plus_weekly_build(tmp_path, store):
    codex_home = tmp_path / "codex_home"
    _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-06-01T01:00:00.000Z",
                {
                    "primary": {
                        "used_percent": 10.0,
                        "window_minutes": 300,
                        "resets_at": 1000.0,
                    },
                    "secondary": {
                        "used_percent": 20.0,
                        "window_minutes": 10080,
                        "resets_at": 2000.0,
                    },
                },
            )
        ],
    )
    cfg = _cfg_for(tmp_path, codex_home)
    windows = {w.kind: w for w in CodexProvider().read(cfg, store)}
    assert windows[WindowKind.W5H].used_percent == 10.0
    assert windows[WindowKind.WEEKLY].used_percent == 20.0


def test_synthetic_resets_in_seconds_variant(tmp_path, store):
    codex_home = tmp_path / "codex_home"
    event_ts = "2026-07-24T00:00:00.000Z"
    _write_rollout(
        codex_home,
        [
            _token_count_event(
                event_ts,
                {
                    "primary": {
                        "used_percent": 5.0,
                        "window_minutes": 300,
                        "resets_in_seconds": 3600,
                    },
                    "secondary": None,
                },
            )
        ],
    )
    cfg = _cfg_for(tmp_path, codex_home)
    windows = CodexProvider().read(cfg, store)
    assert len(windows) == 1
    expected = parse_iso_to_epoch(event_ts) + 3600
    assert windows[0].resets_at == expected


def test_malformed_lines_are_skipped(tmp_path, store):
    codex_home = tmp_path / "codex_home"
    session_dir = codex_home / "sessions" / "2026" / "07" / "15"
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "rollout-test.jsonl"
    good_event = _token_count_event(
        "2026-07-24T00:00:00.000Z",
        {
            "primary": {
                "used_percent": 42.0,
                "window_minutes": 10080,
                "resets_at": 999.0,
            },
            "secondary": None,
        },
    )
    lines = [
        "not json at all",
        json.dumps(
            {
                "timestamp": "2026-07-24T00:00:00.000Z",
                "payload": {"type": "token_count", "rate_limits": None},
            }
        ),
        json.dumps(
            {"timestamp": "2026-07-24T00:00:00.000Z", "payload": {"type": "other"}}
        ),
        json.dumps(good_event),
    ]
    path.write_text("\n".join(lines) + "\n")
    cfg = _cfg_for(tmp_path, codex_home)
    windows = CodexProvider().read(cfg, store)
    assert len(windows) == 1
    assert windows[0].used_percent == 42.0


def test_no_rollout_files_returns_empty(tmp_path, store):
    cfg = _cfg_for(tmp_path, tmp_path / "empty_codex_home")
    assert CodexProvider().read(cfg, store) == []


def test_ingests_token_events_from_last_token_usage(tmp_path, store):
    """Regression: `used_percent` clamps at 100 and goes blind past it, but
    the same line's `info.last_token_usage` is the real per-step delta --
    already computed by Codex itself, verified against a live session to
    match total_token_usage's own running delta exactly."""
    codex_home = tmp_path / "codex_home"
    _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-07-24T19:40:04.014Z",
                {
                    "primary": {
                        "used_percent": 100.0,
                        "window_minutes": 10080,
                        "resets_at": 1785260936,
                    },
                    "secondary": None,
                },
                last_token_usage={
                    "input_tokens": 128544,
                    "cached_input_tokens": 126720,
                    "cache_write_input_tokens": 0,
                    "output_tokens": 115,
                    "reasoning_output_tokens": 0,
                    "total_tokens": 128659,
                },
            )
        ],
    )
    cfg = _cfg_for(tmp_path, codex_home)
    CodexProvider().read(cfg, store)

    events = store.recent_token_events(Provider.CODEX, 0.0)
    assert len(events) == 1
    assert events[0].input_tokens == 128544
    assert events[0].output_tokens == 115
    # Codex's cache breakdown is already folded into input/output above --
    # these stay 0 rather than double-counting a subset of input_tokens.
    assert events[0].cache_creation == 0
    assert events[0].cache_read == 0
    assert events[0].ts == parse_iso_to_epoch("2026-07-24T19:40:04.014Z")


def test_ingests_every_token_count_event_not_just_the_last(tmp_path, store):
    codex_home = tmp_path / "codex_home"
    rate_limits = {
        "primary": {"used_percent": 100.0, "window_minutes": 10080, "resets_at": 1.0},
        "secondary": None,
    }
    _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-07-24T19:40:00.000Z",
                rate_limits,
                last_token_usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            ),
            _token_count_event(
                "2026-07-24T19:40:04.000Z",
                rate_limits,
                last_token_usage={
                    "input_tokens": 20,
                    "output_tokens": 8,
                    "total_tokens": 28,
                },
            ),
        ],
    )
    cfg = _cfg_for(tmp_path, codex_home)
    CodexProvider().read(cfg, store)

    events = store.recent_token_events(Provider.CODEX, 0.0)
    assert {e.input_tokens for e in events} == {10, 20}


def test_token_event_ingestion_is_idempotent_on_rescan(tmp_path, store):
    """The same session file gets rescanned every poll tick while it's
    still within the retention lookback -- re-upserting an already-seen
    line must not double-count it."""
    codex_home = tmp_path / "codex_home"
    _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-07-24T19:40:00.000Z",
                {
                    "primary": {
                        "used_percent": 50.0,
                        "window_minutes": 10080,
                        "resets_at": 1.0,
                    },
                    "secondary": None,
                },
                last_token_usage={
                    "input_tokens": 10,
                    "output_tokens": 5,
                    "total_tokens": 15,
                },
            )
        ],
    )
    cfg = _cfg_for(tmp_path, codex_home)
    CodexProvider().read(cfg, store)
    CodexProvider().read(cfg, store)  # a second poll tick, same file unchanged

    assert len(store.recent_token_events(Provider.CODEX, 0.0)) == 1


def test_missing_last_token_usage_is_skipped_not_crashed(tmp_path, store):
    codex_home = tmp_path / "codex_home"
    _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-07-24T19:40:00.000Z",
                {
                    "primary": {
                        "used_percent": 50.0,
                        "window_minutes": 10080,
                        "resets_at": 1.0,
                    },
                    "secondary": None,
                },
            )  # no last_token_usage at all
        ],
    )
    cfg = _cfg_for(tmp_path, codex_home)
    windows = CodexProvider().read(cfg, store)
    assert len(windows) == 1  # window detection is unaffected
    assert store.recent_token_events(Provider.CODEX, 0.0) == []


def test_newest_by_mtime_is_selected(tmp_path, store):
    codex_home = tmp_path / "codex_home"
    old_path = _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-07-01T00:00:00.000Z",
                {
                    "primary": {
                        "used_percent": 1.0,
                        "window_minutes": 10080,
                        "resets_at": 1.0,
                    },
                    "secondary": None,
                },
            )
        ],
        day="01",
    )
    new_path = _write_rollout(
        codex_home,
        [
            _token_count_event(
                "2026-07-24T00:00:00.000Z",
                {
                    "primary": {
                        "used_percent": 99.0,
                        "window_minutes": 10080,
                        "resets_at": 2.0,
                    },
                    "secondary": None,
                },
            )
        ],
        day="24",
    )
    now = time_mod.time()
    os.utime(old_path, (now - 1000, now - 1000))
    os.utime(new_path, (now, now))

    cfg = _cfg_for(tmp_path, codex_home)
    windows = CodexProvider().read(cfg, store)
    assert len(windows) == 1
    assert windows[0].used_percent == 99.0
