"""Codex app-server usage source tests — response parsing plus the spawn
throttle, backoff, and cache-TTL state machine (same contract as the Claude
CLI source's)."""

from __future__ import annotations

from tokenwatchdog.models import Provider, WindowKind
from tokenwatchdog.providers.codex_app_server import (
    AppServerUsageSource,
    windows_from_result,
)

_NOW = 1_786_000_000.0

# Verbatim shape of a live account/rateLimits/read result.
_RESULT = {
    "rateLimits": {
        "limitId": "codex",
        "limitName": None,
        "primary": {
            "usedPercent": 7,
            "windowDurationMins": 10080,
            "resetsAt": 1_786_190_296,
        },
        "secondary": None,
        "planType": "team",
        "rateLimitReachedType": None,
    },
}


class _CountingQuery:
    """Fake query: replays `results` (None = failed spawn), counting calls."""

    def __init__(self, results):
        self.calls = 0
        self._results = results

    def __call__(self, cfg):
        self.calls += 1
        return self._results[min(self.calls - 1, len(self._results) - 1)]


def test_parses_the_weekly_window_from_a_live_result_shape():
    (window,) = windows_from_result(_RESULT, _NOW)
    assert window.provider is Provider.CODEX
    assert window.kind is WindowKind.WEEKLY
    assert window.used_percent == 7.0
    assert window.window_minutes == 10080
    assert window.resets_at == 1_786_190_296.0
    assert window.source_ts == _NOW
    assert window.is_estimated is False


def test_classifies_a_secondary_5h_window_and_skips_unknown_minutes():
    result = {
        "rateLimits": {
            "primary": {"usedPercent": 7, "windowDurationMins": 10080},
            "secondary": {"usedPercent": 12, "windowDurationMins": 300},
        }
    }
    kinds = {w.kind: w for w in windows_from_result(result, _NOW)}
    assert kinds[WindowKind.W5H].used_percent == 12.0
    assert kinds[WindowKind.WEEKLY].resets_at is None  # absent field -> unknown

    odd = {"rateLimits": {"primary": {"usedPercent": 3, "windowDurationMins": 60}}}
    assert windows_from_result(odd, _NOW) == []


def test_malformed_results_parse_to_nothing():
    assert windows_from_result({}, _NOW) == []
    assert windows_from_result({"rateLimits": "exhausted"}, _NOW) == []
    assert (
        windows_from_result({"rateLimits": {"primary": {"usedPercent": "7"}}}, _NOW)
        == []
    )


def test_spawns_at_most_every_180s_and_serves_the_cache_between(cfg):
    query = _CountingQuery([_RESULT])
    source = AppServerUsageSource(query=query)
    first = source.read(cfg, _NOW)
    assert len(first) == 1
    again = source.read(cfg, _NOW + 60.0)
    assert query.calls == 1
    assert again == first  # cached windows keep their original source_ts
    source.read(cfg, _NOW + 181.0)
    assert query.calls == 2


def test_failed_queries_back_off_exponentially(cfg):
    query = _CountingQuery([None])
    source = AppServerUsageSource(query=query)
    assert source.read(cfg, _NOW) == []
    assert source.read(cfg, _NOW + 200.0) == []  # inside the doubled 360s wait
    assert query.calls == 1
    source.read(cfg, _NOW + 400.0)
    assert query.calls == 2


def test_stale_cache_stops_shadowing_the_rollout_fallback(cfg):
    query = _CountingQuery([_RESULT, None, None, None])
    source = AppServerUsageSource(query=query)
    assert source.read(cfg, _NOW)
    assert source.read(cfg, _NOW + 200.0)  # query fails; cache is 200s old
    assert source.read(cfg, _NOW + 1000.0) == []  # cache past its 900s TTL
