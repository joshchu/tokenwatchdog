"""CLI usage source tests — output parsing plus the spawn throttle, backoff,
and cache-TTL state machine that reconciles a 60s poll loop with a 180s
spawn floor."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from tokenwatchdog.models import Provider, WindowKind
from tokenwatchdog.providers.claude_cli import CliUsageSource, parse_usage_output

_ET = ZoneInfo("America/New_York")
_NOW = datetime(2026, 7, 29, 14, 0, tzinfo=_ET).timestamp()

# Verbatim shape of `claude -p "/usage" --output-format text`.
_OUTPUT = (
    "You are currently using your subscription to power your Claude Code usage\n"
    "\n"
    "Current session: 88% used · resets Jul 29 at 5pm (America/New_York)\n"
    "Current week (all models): 5% used · resets Aug 5 at 1pm (America/New_York)\n"
)


def _epoch(*args: int) -> float:
    return datetime(*args, tzinfo=_ET).timestamp()


class _CountingSpawn:
    """Fake spawn: replays `outputs` (None = failed spawn), counting calls."""

    def __init__(self, outputs):
        self.calls = 0
        self._outputs = outputs

    def __call__(self, cfg):
        self.calls += 1
        return self._outputs[min(self.calls - 1, len(self._outputs) - 1)]


def test_parses_both_windows_with_exact_reset_epochs():
    windows = {w.kind: w for w in parse_usage_output(_OUTPUT, _NOW)}
    w5h, weekly = windows[WindowKind.W5H], windows[WindowKind.WEEKLY]
    assert w5h.used_percent == 88.0
    assert w5h.window_minutes == 300
    assert w5h.resets_at == _epoch(2026, 7, 29, 17, 0)
    assert weekly.used_percent == 5.0
    assert weekly.window_minutes == 10080
    assert weekly.resets_at == _epoch(2026, 8, 5, 13, 0)
    for window in windows.values():
        assert window.provider is Provider.CLAUDE
        assert window.source_ts == _NOW
        assert window.is_estimated is False


def test_parses_explicit_minutes_and_noon_midnight():
    output = (
        "Current session: 1% used · resets Jul 30 at 12:30am (America/New_York)\n"
        "Current week (all models): 2% used · resets Jul 30 at 12pm (America/New_York)\n"
    )
    windows = {w.kind: w for w in parse_usage_output(output, _NOW)}
    assert windows[WindowKind.W5H].resets_at == _epoch(2026, 7, 30, 0, 30)
    assert windows[WindowKind.WEEKLY].resets_at == _epoch(2026, 7, 30, 12, 0)


def test_missing_reset_clause_still_reports_the_percent():
    windows = parse_usage_output("Current session: 0% used\n", _NOW)
    assert len(windows) == 1
    assert windows[0].used_percent == 0.0
    assert windows[0].resets_at is None


def test_unrecognized_timezone_drops_only_the_reset():
    output = "Current session: 3% used · resets Jul 29 at 5pm (Mars/Olympus)\n"
    windows = parse_usage_output(output, _NOW)
    assert len(windows) == 1
    assert windows[0].resets_at is None


def test_year_rollover_lands_january_reset_in_the_next_year():
    december = datetime(2026, 12, 30, 9, 0, tzinfo=_ET).timestamp()
    output = (
        "Current week (all models): 9% used · resets Jan 5 at 1pm (America/New_York)\n"
    )
    (window,) = parse_usage_output(output, december)
    assert window.resets_at == _epoch(2027, 1, 5, 13, 0)


def test_unrecognized_output_parses_to_nothing():
    assert parse_usage_output("Please run /login\n", _NOW) == []


def test_per_model_week_lines_are_ignored():
    output = _OUTPUT + (
        "Current week (Opus): 40% used · resets Aug 5 at 1pm (America/New_York)\n"
    )
    kinds = [w.kind for w in parse_usage_output(output, _NOW)]
    assert kinds == [WindowKind.W5H, WindowKind.WEEKLY]


def test_spawns_at_most_every_180s_and_serves_the_cache_between(cfg):
    spawn = _CountingSpawn([_OUTPUT])
    source = CliUsageSource(spawn=spawn)
    first = source.read(cfg, _NOW)
    assert len(first) == 2
    again = source.read(cfg, _NOW + 60.0)
    assert spawn.calls == 1
    assert again == first  # cached windows keep their original source_ts
    source.read(cfg, _NOW + 181.0)
    assert spawn.calls == 2


def test_failed_spawns_back_off_exponentially(cfg):
    spawn = _CountingSpawn([None])
    source = CliUsageSource(spawn=spawn)
    assert source.read(cfg, _NOW) == []
    assert source.read(cfg, _NOW + 200.0) == []  # inside the doubled 360s wait
    assert spawn.calls == 1
    source.read(cfg, _NOW + 400.0)
    assert spawn.calls == 2


def test_stale_cache_stops_shadowing_the_fallback_sources(cfg):
    spawn = _CountingSpawn([_OUTPUT, None, None, None])
    source = CliUsageSource(spawn=spawn)
    assert source.read(cfg, _NOW)
    assert source.read(cfg, _NOW + 200.0)  # spawn fails; cache is 200s old
    assert source.read(cfg, _NOW + 1000.0) == []  # cache past its 900s TTL
