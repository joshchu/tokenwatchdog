"""CLI rendering smoke tests — just confirm _render doesn't crash and the
alert log only appears when there's something to show."""

from __future__ import annotations

import dataclasses
import os
import pty
import sys
import time
import tty
from datetime import datetime, timezone

from rich.console import Console

from tokenwatchdog.cli import _render, _row_style, _spacebar_pressed
from tokenwatchdog.models import (
    Alert,
    Forecast,
    MonitorState,
    Provider,
    Window,
    WindowKind,
)


def _window(used_percent=50.0, resets_at=None):
    return Window(
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        used_percent=used_percent,
        window_minutes=10080,
        resets_at=resets_at,
        source_ts=1000.0,
        is_estimated=False,
        source_file="test",
    )


def _forecast(window):
    return Forecast(
        window=window,
        status="OK",
        model_name="linear",
        burn_per_hour=1.0,
        time_to_reset_h=None,
        eta_calendar=None,
        eta_workhours=None,
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=None,
        confidence="high",
        exhausts_before_reset=False,
        n_samples=5,
    )


def _render_to_text(renderable) -> str:
    console = Console(width=120, record=True)
    console.print(renderable)
    return console.export_text()


def test_render_without_alerts_has_no_alert_panel(cfg):
    window = _window()
    state = MonitorState(
        now=1000.0, windows=(window,), forecasts=(_forecast(window),), alerts=()
    )
    text = _render_to_text(_render(state, cfg, []))
    assert "Recent alerts" not in text


def test_render_with_alerts_shows_the_alert_panel_and_message(cfg):
    window = _window(used_percent=95.0)
    alert = Alert(
        key="codex:weekly:threshold",
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        alert_kind="threshold",
        message="codex weekly: 95% used (warns at 90%)",
        fired_at=1000.0,
    )
    state = MonitorState(
        now=1000.0, windows=(window,), forecasts=(_forecast(window),), alerts=(alert,)
    )
    text = _render_to_text(_render(state, cfg, [alert]))
    assert "Recent alerts" in text
    assert "codex weekly: 95% used" in text


def test_row_style_is_red_over_the_warn_threshold(cfg):
    forecast = _forecast(_window(used_percent=95.0))
    assert _row_style(forecast, cfg, now=1000.0) == "red"


def test_row_style_is_yellow_when_burning_but_not_yet_urgent(cfg):
    forecast = dataclasses.replace(
        _forecast(_window(used_percent=50.0)),
        exhausts_before_reset=True,
        eta_calendar=datetime.fromtimestamp(1000.0 + 10 * 3600, tz=timezone.utc),
    )
    assert _row_style(forecast, cfg, now=1000.0) == "yellow"


def test_row_style_is_red_when_burn_is_imminent(cfg):
    forecast = dataclasses.replace(
        _forecast(_window(used_percent=50.0)),
        exhausts_before_reset=True,
        eta_calendar=datetime.fromtimestamp(1000.0 + 0.1 * 3600, tz=timezone.utc),
    )
    assert _row_style(forecast, cfg, now=1000.0) == "red"


def test_row_style_is_none_when_not_exhausting(cfg):
    forecast = _forecast(_window(used_percent=50.0))
    assert _row_style(forecast, cfg, now=1000.0) is None


def test_row_style_is_none_for_stale_idle_even_over_threshold(cfg):
    """Regression: alerts.py already treats IDLE the same as NO_DATA -- a
    stale snapshot it doesn't trust enough to alert on. Highlighting an
    idle 95% reading red would flag data that's already been flagged as
    unreliable."""
    forecast = dataclasses.replace(_forecast(_window(used_percent=95.0)), status="IDLE")
    assert _row_style(forecast, cfg, now=1000.0) is None


def test_row_style_is_red_at_full_exhaustion_even_when_idle(cfg):
    """Regression: 100% used is the literal last-known reading, not a
    predicted trend -- it stays true even once the window has gone stale,
    unlike the 95%-idle case above where the number could plausibly have
    already dropped. A window that hit its limit and then stopped being
    polled must not read as merely "idle" on the dashboard."""
    forecast = dataclasses.replace(
        _forecast(_window(used_percent=100.0)), status="IDLE"
    )
    assert _row_style(forecast, cfg, now=1000.0) == "red"


def test_row_shows_tokens_burned_past_quota_when_set(cfg):
    forecast = dataclasses.replace(
        _forecast(_window(used_percent=100.0)), tokens_burned_past_quota=530_000
    )
    state = MonitorState(
        now=1000.0, windows=(forecast.window,), forecasts=(forecast,), alerts=()
    )
    text = _render_to_text(_render(state, cfg, []))
    assert "530K tok" in text


def test_row_shows_dollar_cost_over_token_count_when_both_are_set(cfg):
    """A priced $ estimate is more informative than the raw token count it
    was computed from -- show one or the other, not both."""
    forecast = dataclasses.replace(
        _forecast(_window(used_percent=100.0)),
        tokens_burned_past_quota=530_000,
        cost_burned_past_quota_usd=12.34,
    )
    state = MonitorState(
        now=1000.0, windows=(forecast.window,), forecasts=(forecast,), alerts=()
    )
    text = _render_to_text(_render(state, cfg, []))
    assert "$12.34" in text
    assert "530K tok" not in text


def test_spacebar_pressed_falls_back_to_plain_sleep_when_not_a_tty(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    class _NotATty:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _NotATty())
    assert _spacebar_pressed(42.0) is False
    assert sleeps == [42.0]


def test_spacebar_pressed_returns_true_on_a_real_keypress(monkeypatch):
    """A real pty, not a mock -- proves the select/read wiring actually
    reacts to a keypress instead of just asserting the fallback branch.
    cbreak is set on the slave BEFORE writing the key: writing first (then
    switching modes) leaves the byte sitting in the canonical-mode line
    buffer, which is not guaranteed to become visible to select/read after
    the mode switch and made this flaky."""
    master_fd, slave_fd = pty.openpty()
    tty.setcbreak(slave_fd)
    stdin_fake = os.fdopen(slave_fd, "r")
    try:
        monkeypatch.setattr(sys, "stdin", stdin_fake)
        os.write(master_fd, b" ")
        start = time.monotonic()
        assert _spacebar_pressed(5.0) is True
        assert time.monotonic() - start < 2.0
    finally:
        stdin_fake.close()
        os.close(master_fd)


def test_spacebar_pressed_returns_false_when_the_interval_elapses(monkeypatch):
    master_fd, slave_fd = pty.openpty()
    tty.setcbreak(slave_fd)
    stdin_fake = os.fdopen(slave_fd, "r")
    try:
        monkeypatch.setattr(sys, "stdin", stdin_fake)
        assert _spacebar_pressed(0.05) is False
    finally:
        stdin_fake.close()
        os.close(master_fd)


def test_render_shows_a_refreshing_indicator_in_the_header(cfg):
    window = _window()
    state = MonitorState(
        now=1000.0, windows=(window,), forecasts=(_forecast(window),), alerts=()
    )
    text = _render_to_text(_render(state, cfg, [], refreshing=True))
    assert "refreshing" in text
