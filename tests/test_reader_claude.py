"""Claude Desktop + token-compute reader tests.

Block-anchoring / resets_at derivation is the predictor's job, not the
provider's, so those cases are covered by test_predictor.py instead of here.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import pytest

from tokenwatchdog.config import load_config
from tokenwatchdog.models import Provider, WindowKind
from tokenwatchdog.predictor import tokens_burned_past_quota
from tokenwatchdog.providers import claude as claude_provider
from tokenwatchdog.store import SampleRow


@pytest.fixture
def sandbox_home(tmp_path, monkeypatch):
    """Redirects Path.home() so tests never touch the real ~/.claude* data."""
    monkeypatch.setattr(claude_provider.Path, "home", lambda: tmp_path)
    return tmp_path


def _recent_iso(seconds_ago: float) -> str:
    """An ISO-8601 timestamp `seconds_ago` before real wall-clock now.

    The provider reads real time.time() internally (by design — it's a live
    monitor, not a pure function), so fixtures must be recent relative to
    the actual clock, not a hardcoded calendar date — otherwise a token
    event can silently fall outside the 5h/7d lookback window depending on
    what day the suite happens to run.
    """
    dt = datetime.fromtimestamp(time.time() - seconds_ago, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _assistant_line(
    *,
    ts,
    request_id,
    message_id,
    output_tokens=0,
    input_tokens=1,
    cache_creation=0,
    cache_read=0,
    model="claude-fable-5",
):
    return {
        "timestamp": ts,
        "requestId": request_id,
        "message": {
            "id": message_id,
            "role": "assistant",
            "model": model,
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cache_creation_input_tokens": cache_creation,
                "cache_read_input_tokens": cache_read,
            },
        },
    }


def _write_config(tmp_path, text):
    config_path = tmp_path / "config.toml"
    config_path.write_text(text)
    return load_config(config_path)


def test_desktop_source_maps_fh_sd_to_w5h_and_weekly(sandbox_home, tmp_path, store):
    desktop_dir = sandbox_home / "Library" / "Application Support" / "Claude"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "plan-usage-history.json").write_text(
        json.dumps(
            {
                "version": 2,
                "samples": [
                    {"t": 1_000_000, "org": "org-1", "u": {"fh": 40, "sd": 12}},
                    {"t": 2_000_000, "org": "org-1", "u": {"fh": 55, "sd": 15}},
                ],
            }
        )
    )
    cfg = _write_config(tmp_path, '[claude]\nsource = "desktop"\n')

    windows = {w.kind: w for w in claude_provider.ClaudeProvider().read(cfg, store)}
    assert windows[WindowKind.W5H].used_percent == 55.0
    assert windows[WindowKind.WEEKLY].used_percent == 15.0
    assert windows[WindowKind.W5H].source_ts == 2000.0
    assert windows[WindowKind.W5H].is_estimated is False


def test_desktop_source_backfills_every_retained_sample(sandbox_home, tmp_path, store):
    """Regression: Desktop retains ~5 days of history on its own regardless
    of whether TokenWatchDog is running. Only ever reading the latest
    point would silently drop that history during any downtime, even
    though Desktop's own file already has it -- read() must return every
    retained sample (engine.py persists each one; only the last is used
    for the live forecast) so downtime isn't a gap for this source."""
    desktop_dir = sandbox_home / "Library" / "Application Support" / "Claude"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "plan-usage-history.json").write_text(
        json.dumps(
            {
                "version": 2,
                "samples": [
                    {"t": 1_000_000, "org": "org-1", "u": {"fh": 40, "sd": 12}},
                    {"t": 2_000_000, "org": "org-1", "u": {"fh": 55, "sd": 15}},
                    {"t": 3_000_000, "org": "org-1", "u": {"fh": 70, "sd": 18}},
                ],
            }
        )
    )
    cfg = _write_config(tmp_path, '[claude]\nsource = "desktop"\n')

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    w5h_percents = sorted(w.used_percent for w in windows if w.kind is WindowKind.W5H)
    assert w5h_percents == [40.0, 55.0, 70.0]

    for window in windows:
        store.insert_sample(
            captured_at=window.source_ts,
            provider=window.provider,
            window_kind=window.kind,
            source_ts=window.source_ts,
            used_percent=window.used_percent,
            window_minutes=window.window_minutes,
            resets_at=window.resets_at,
            is_estimated=window.is_estimated,
            source_file=window.source_file,
        )
    stored = store.recent_samples(Provider.CLAUDE, WindowKind.W5H, since_ts=0.0)
    assert [s.used_percent for s in stored] == [40.0, 55.0, 70.0]


def test_desktop_backfill_excludes_samples_from_a_different_org(
    sandbox_home, tmp_path, store
):
    """Regression: Desktop can retain samples from more than one Claude org
    if the account switched between them. Each org has its own independent
    quota, and the store has no org dimension, so backfilling a prior org's
    samples alongside the current one would look like an arbitrary usage
    jump or a spurious reset in the current org's history."""
    desktop_dir = sandbox_home / "Library" / "Application Support" / "Claude"
    desktop_dir.mkdir(parents=True)
    (desktop_dir / "plan-usage-history.json").write_text(
        json.dumps(
            {
                "version": 2,
                "samples": [
                    {"t": 1_000_000, "org": "org-old", "u": {"fh": 90, "sd": 80}},
                    {"t": 2_000_000, "org": "org-old", "u": {"fh": 95, "sd": 85}},
                    {"t": 3_000_000, "org": "org-new", "u": {"fh": 5, "sd": 3}},
                    {"t": 4_000_000, "org": "org-new", "u": {"fh": 10, "sd": 6}},
                ],
            }
        )
    )
    cfg = _write_config(tmp_path, '[claude]\nsource = "desktop"\n')

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    w5h_percents = sorted(w.used_percent for w in windows if w.kind is WindowKind.W5H)
    assert w5h_percents == [5.0, 10.0]  # only org-new, never org-old's 90/95


def test_desktop_source_missing_file_returns_empty(sandbox_home, tmp_path, store):
    cfg = _write_config(tmp_path, '[claude]\nsource = "desktop"\n')
    assert claude_provider.ClaudeProvider().read(cfg, store) == []


def test_token_compute_dedup_keeps_last_and_largest(sandbox_home, tmp_path, store):
    projects_dir = sandbox_home / ".claude" / "projects" / "proj1"
    projects_dir.mkdir(parents=True)
    partial = _assistant_line(
        ts=_recent_iso(120),
        request_id="req_1",
        message_id="msg_1",
        output_tokens=10,
    )
    final = _assistant_line(
        ts=_recent_iso(115),
        request_id="req_1",
        message_id="msg_1",
        output_tokens=500,
    )
    (projects_dir / "session1.jsonl").write_text(
        json.dumps(partial) + "\n" + json.dumps(final) + "\n"
    )
    cfg = _write_config(tmp_path, '[claude]\nsource = "tokens"\nlimit_mode = "plan"\n')

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    events = store.recent_token_events(Provider.CLAUDE, since_ts=0.0)
    assert len(events) == 1
    assert events[0].output_tokens == 500
    assert any(w.kind is WindowKind.W5H for w in windows)


def test_subagent_and_main_transcript_same_id_not_double_counted(
    sandbox_home, tmp_path, store
):
    proj_dir = sandbox_home / ".claude" / "projects" / "proj1"
    subagent_dir = proj_dir / "subagents"
    subagent_dir.mkdir(parents=True)
    line = _assistant_line(
        ts=_recent_iso(120),
        request_id="req_shared",
        message_id="msg_shared",
        output_tokens=100,
    )
    (proj_dir / "session1.jsonl").write_text(json.dumps(line) + "\n")
    (subagent_dir / "sub1.jsonl").write_text(json.dumps(line) + "\n")
    cfg = _write_config(tmp_path, '[claude]\nsource = "tokens"\nlimit_mode = "plan"\n')

    claude_provider.ClaudeProvider().read(cfg, store)
    events = store.recent_token_events(Provider.CLAUDE, since_ts=0.0)
    assert len(events) == 1
    assert events[0].output_tokens == 100


def test_plan_limit_mode_uses_configured_tokens(sandbox_home, tmp_path, store):
    proj_dir = sandbox_home / ".claude" / "projects" / "proj1"
    proj_dir.mkdir(parents=True)
    line = _assistant_line(
        ts=_recent_iso(120),
        request_id="req_1",
        message_id="msg_1",
        output_tokens=1000,
    )
    (proj_dir / "session1.jsonl").write_text(json.dumps(line) + "\n")
    cfg = _write_config(
        tmp_path,
        '[claude]\nsource = "tokens"\nlimit_mode = "plan"\n\n'
        "[claude.plan_limits_tokens]\ndefault_claude_max_5x = 2000\n",
    )

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    w5h = next(w for w in windows if w.kind is WindowKind.W5H)
    # input(1) + output(1000) + cache(0) + cache(0) = 1001 tokens / 2000 limit
    assert w5h.used_percent == pytest.approx(1001 / 2000 * 100)
    assert w5h.is_estimated is True


def test_rate_limit_tier_from_claude_json_selects_plan_limit(
    sandbox_home, tmp_path, store
):
    (sandbox_home / ".claude.json").write_text(
        json.dumps({"oauthAccount": {"userRateLimitTier": "custom_tier"}})
    )
    proj_dir = sandbox_home / ".claude" / "projects" / "proj1"
    proj_dir.mkdir(parents=True)
    line = _assistant_line(
        ts=_recent_iso(120),
        request_id="req_1",
        message_id="msg_1",
        output_tokens=100,
    )
    (proj_dir / "session1.jsonl").write_text(json.dumps(line) + "\n")
    cfg = _write_config(
        tmp_path,
        '[claude]\nsource = "tokens"\nlimit_mode = "plan"\n\n'
        "[claude.plan_limits_tokens]\ncustom_tier = 500\n",
    )

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    w5h = next(w for w in windows if w.kind is WindowKind.W5H)
    assert w5h.used_percent == pytest.approx(101 / 500 * 100)


def test_auto_source_falls_back_to_tokens_when_desktop_missing(
    sandbox_home, tmp_path, store
):
    proj_dir = sandbox_home / ".claude" / "projects" / "proj1"
    proj_dir.mkdir(parents=True)
    line = _assistant_line(
        ts=_recent_iso(120),
        request_id="req_1",
        message_id="msg_1",
        output_tokens=100,
    )
    (proj_dir / "session1.jsonl").write_text(json.dumps(line) + "\n")
    cfg = _write_config(tmp_path, '[claude]\nsource = "auto"\nlimit_mode = "plan"\n')

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    assert any(w.kind is WindowKind.W5H and w.is_estimated for w in windows)


def test_weekly_omitted_under_p90_without_enough_history(sandbox_home, tmp_path, store):
    proj_dir = sandbox_home / ".claude" / "projects" / "proj1"
    proj_dir.mkdir(parents=True)
    line = _assistant_line(
        ts=_recent_iso(120),
        request_id="req_1",
        message_id="msg_1",
        output_tokens=100,
    )
    (proj_dir / "session1.jsonl").write_text(json.dumps(line) + "\n")
    cfg = _write_config(tmp_path, '[claude]\nsource = "tokens"\nlimit_mode = "p90"\n')

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    kinds = {w.kind for w in windows}
    assert WindowKind.W5H in kinds  # 5h always has a fallback estimate
    assert WindowKind.WEEKLY not in kinds  # no fabricated weekly denominator


def test_tokens_burned_past_quota_sees_claudes_own_ingested_events(
    sandbox_home, tmp_path, store
):
    """End-to-end: providers/claude.py's own real token-log ingestion
    (not a synthetic TokenEventRow) combined with predictor.
    tokens_burned_past_quota -- proves the two pieces actually fit
    together, not just that each is independently correct. Ingestion
    runs unconditionally regardless of cfg.claude.source (see read()),
    so this holds even though "tokens" is the live source here."""
    proj_dir = sandbox_home / ".claude" / "projects" / "proj1"
    proj_dir.mkdir(parents=True)
    line = _assistant_line(
        ts=_recent_iso(120),
        request_id="req_1",
        message_id="msg_1",
        input_tokens=1,
        output_tokens=999,  # input(1) + output(999) = 1000 == the configured limit
    )
    (proj_dir / "session1.jsonl").write_text(json.dumps(line) + "\n")
    cfg = _write_config(
        tmp_path,
        '[claude]\nsource = "tokens"\nlimit_mode = "plan"\n\n'
        "[claude.plan_limits_tokens]\ndefault_claude_max_5x = 1000\n",
    )

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    w5h = next(w for w in windows if w.kind is WindowKind.W5H)
    assert w5h.used_percent == pytest.approx(100.0)

    history = [
        SampleRow(
            captured_at=w5h.source_ts,
            source_ts=w5h.source_ts,
            used_percent=w5h.used_percent,
            resets_at=w5h.resets_at,
            is_estimated=w5h.is_estimated,
        )
    ]
    events = store.recent_token_events(Provider.CLAUDE, since_ts=0.0)
    assert tokens_burned_past_quota(w5h, history, events) == 1000


def test_p90_limit_estimate_excludes_the_live_window(sandbox_home, tmp_path, store):
    """Regression: the p90 estimate must be built from windows that have
    already ENDED. If the live window's own total were one of the
    candidates the max is taken over, it could never exceed
    observed_max — capping every p90 reading at the safety factor (90%)
    no matter how much is actually used."""
    proj_dir = sandbox_home / ".claude" / "projects" / "proj1"
    proj_dir.mkdir(parents=True)
    lines = [
        # Two historical events, 2h apart, both outside the live 5h window —
        # together they set a known observed_max of ~1000 tokens.
        _assistant_line(
            ts=_recent_iso(8 * 3600),
            request_id="req_hist_1",
            message_id="msg_hist_1",
            output_tokens=500,
        ),
        _assistant_line(
            ts=_recent_iso(6 * 3600),
            request_id="req_hist_2",
            message_id="msg_hist_2",
            output_tokens=500,
        ),
        # A live-window burst that massively exceeds that historical max.
        _assistant_line(
            ts=_recent_iso(60),
            request_id="req_live",
            message_id="msg_live",
            output_tokens=50_000,
        ),
    ]
    (proj_dir / "session1.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + "\n"
    )
    cfg = _write_config(tmp_path, '[claude]\nsource = "tokens"\nlimit_mode = "p90"\n')

    windows = claude_provider.ClaudeProvider().read(cfg, store)
    w5h = next(w for w in windows if w.kind is WindowKind.W5H)
    assert w5h.used_percent > 90.0


def test_unchanged_transcripts_are_not_re_read_every_poll(tmp_path, store):
    """Regression, with a number: ingestion used to be bounded only by the
    retention window, which excludes almost nothing — practically every
    transcript is inside 8 weeks. So each ~1-minute poll re-parsed ~40k lines
    and re-upserted ~48k already-stored token events, measured at 10.4s of a
    13.5s tick. A cursor over file mtimes takes that to ~0.4s.

    Counts file opens rather than elapsed time, so it can't go flaky on a
    loaded machine."""
    projects = tmp_path / "projects" / "proj"
    projects.mkdir(parents=True)
    transcript = projects / "session.jsonl"

    def line(ts, request_id, message_id, input_tokens):
        return json.dumps(
            _assistant_line(
                ts=ts,
                request_id=request_id,
                message_id=message_id,
                input_tokens=input_tokens,
            )
        )

    transcript.write_text(
        line("2026-07-24T10:00:00.000Z", "req_1", "msg_1", 100) + "\n"
    )
    os.utime(transcript, (1_000_050.0, 1_000_050.0))
    cfg = _write_config(tmp_path, f'[claude]\nconfig_dir = "{tmp_path}"\n')

    real_open = Path.open

    def ingest_counting_opens(now):
        """Transcript opens performed by one ingest pass. Patched only around
        the call, so the test's own file writing isn't counted."""
        opens = 0

        def counting_open(self, *args, **kwargs):
            nonlocal opens
            if self.suffix == ".jsonl":
                opens += 1
            return real_open(self, *args, **kwargs)

        with mock.patch.object(Path, "open", counting_open):
            claude_provider._ingest_token_events(cfg, store, now)
        return opens

    backfill = ingest_counting_opens(1_000_100.0)
    unchanged = ingest_counting_opens(1_000_200.0)  # nothing touched on disk
    transcript.write_text(
        transcript.read_text()
        + line("2026-07-24T11:00:00.000Z", "req_2", "msg_2", 200)
        + "\n"
    )
    os.utime(transcript, (1_000_250.0, 1_000_250.0))
    appended = ingest_counting_opens(1_000_300.0)

    assert backfill == 1  # first run reads it
    assert unchanged == 0  # untouched -> never opened
    assert appended == 1  # modified -> read again
    # And the appended event really did land, so skipping isn't losing data.
    assert {
        e.input_tokens for e in store.recent_token_events(Provider.CLAUDE, 0.0)
    } == {
        100,
        200,
    }
