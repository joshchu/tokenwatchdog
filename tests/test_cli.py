"""CLI rendering smoke tests — just confirm _render doesn't crash and the
alert log only appears when there's something to show."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

from rich.console import Console

from tokenwatchdog.cli import _render, _row_style
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
