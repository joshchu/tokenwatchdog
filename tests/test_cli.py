"""CLI rendering smoke tests — just confirm _render doesn't crash and the
alert log only appears when there's something to show."""

from __future__ import annotations

import dataclasses
import os
import sys
import time
from datetime import datetime, timezone

import pytest
from rich.console import Console

import tokenwatchdog.cli as cli_module
from tokenwatchdog.cli import _render, _row, _row_style, _spacebar_pressed
from tokenwatchdog.models import (
    Alert,
    Forecast,
    MonitorState,
    Provider,
    RetainedPrediction,
    Window,
    WindowKind,
)

try:
    import pty
    import tty
except ImportError:  # Windows: these tests exercise the POSIX input path only
    pty = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]


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


def _render_to_text(renderable, *, width=120) -> str:
    console = Console(width=width, record=True)
    console.print(renderable)
    return console.export_text()


def test_reset_data_clears_store_without_loading_config(monkeypatch, capsys):
    calls = []

    class _FakeStore:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return None

        def reset_history(self, reset_at):
            calls.append(reset_at)
            return 12

    monkeypatch.setattr(cli_module, "Store", _FakeStore)
    monkeypatch.setattr(cli_module.time, "time", lambda: 1234.5)
    monkeypatch.setattr(
        cli_module,
        "load_config",
        lambda: pytest.fail("--reset-data must not load config or start the engine"),
    )

    assert cli_module.main(["--reset-data"]) == 0
    assert calls == [1234.5]
    output = capsys.readouterr().out
    assert "removed 12 saved data row(s)" in output
    assert "pre-reset history will not be relearned" in output


def test_reset_data_cannot_be_combined_with_a_monitor_mode():
    with pytest.raises(SystemExit):
        cli_module._parse_args(["--reset-data", "--once"])


def test_render_without_alerts_has_no_alert_panel(cfg):
    window = _window()
    state = MonitorState(
        now=1000.0, windows=(window,), forecasts=(_forecast(window),), alerts=()
    )
    text = _render_to_text(_render(state, cfg, []))
    assert "Recent alerts" not in text


def test_dashboard_uses_a_high_contrast_accent(cfg):
    window = _window()
    state = MonitorState(
        now=1000.0, windows=(window,), forecasts=(_forecast(window),), alerts=()
    )

    rendered = _render(state, cfg, [])
    panel = rendered.renderables[0]

    assert isinstance(panel.title, cli_module.Text)
    assert panel.title.style == "bold bright_cyan"
    assert panel.border_style == "bright_cyan"


def test_eta_headers_name_the_rate_and_prediction_questions(cfg):
    window = _window()
    state = MonitorState(
        now=1000.0, windows=(window,), forecasts=(_forecast(window),), alerts=()
    )
    text = _render_to_text(_render(state, cfg, []), width=200)

    assert "ETA on burn %/h" in text
    assert "Predicted ETA" in text
    assert "P50 → P90" in text
    assert "ETA trend" not in text
    assert "ETA rhythm" not in text


def test_burn_eta_stays_linear_when_predicted_model_is_authoritative(cfg):
    window = _window()
    linear = dataclasses.replace(
        _forecast(window),
        burn_per_hour=2.0,
        eta_calendar=datetime(2026, 7, 27, 10, tzinfo=timezone.utc),
    )
    predicted = dataclasses.replace(
        _forecast(window),
        model_name="montecarlo",
        burn_per_hour=9.0,
        eta_calendar=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p50=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p90=datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    )
    state = MonitorState(
        now=1000.0,
        windows=(window,),
        forecasts=(predicted,),
        alerts=(),
        all_forecasts=(linear, predicted),
    )

    text = _render_to_text(_render(state, cfg, []), width=200)
    assert "+2.00" in text
    assert "+9.00" not in text

    row = _row(
        predicted,
        timezone.utc,
        burn_rate=linear,
        predicted=predicted,
    )
    assert row[5] == "Mon 10:00"
    assert row[6] == "Tue 12:00 → Wed 12:00"


def test_predicted_eta_remains_visible_when_the_recent_rate_is_idle():
    window = _window()
    idle_burn_rate = dataclasses.replace(
        _forecast(window),
        status="IDLE",
        burn_per_hour=0.0,
        eta_calendar=None,
        eta_workhours=None,
        confidence=None,
    )
    idle_prediction = dataclasses.replace(
        _forecast(window),
        status="IDLE",
        model_name="montecarlo",
        eta_calendar=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p50=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p90=datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    )

    row = _row(
        idle_burn_rate,
        timezone.utc,
        burn_rate=idle_burn_rate,
        predicted=idle_prediction,
    )

    assert row[3] == "idle"
    assert row[4] == "—"
    assert row[5] == "—"
    assert row[6] == "Tue 12:00 → Wed 12:00"


def test_single_predicted_percentile_is_labeled_p50():
    window = _window()
    predicted = dataclasses.replace(
        _forecast(window),
        model_name="montecarlo",
        eta_calendar=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p50=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p90=None,
    )

    row = _row(
        predicted,
        timezone.utc,
        burn_rate=_forecast(window),
        predicted=predicted,
    )

    assert row[6] == "P50 Tue 12:00"


def test_retained_prediction_is_labeled_saved(cfg):
    window = _window()
    idle = dataclasses.replace(
        _forecast(window),
        status="IDLE",
        model_name="montecarlo",
        eta_calendar=None,
        eta_p50=None,
        eta_p90=None,
    )
    retained = RetainedPrediction(
        provider=window.provider,
        kind=window.kind,
        made_at=900.0,
        used_percent=window.used_percent,
        eta_p50=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p90=datetime(2026, 7, 29, 12, tzinfo=timezone.utc),
    )

    row = _row(
        idle,
        timezone.utc,
        burn_rate=idle,
        predicted=idle,
        retained=retained,
    )

    assert row[6] == "saved Tue 12:00 → Wed 12:00"

    state = MonitorState(
        now=1000.0,
        windows=(window,),
        forecasts=(idle,),
        alerts=(),
        all_forecasts=(idle,),
        retained_predictions=(retained,),
    )
    text = _render_to_text(_render(state, cfg, []), width=200)
    assert "saved " in text


def test_a_censored_band_with_known_risk_renders_safe(cfg):
    """The simulation ran and most futures survive to the reset: that is an
    answer, not an absence. Before this, 4,513 consecutive OK weekly ticks
    rendered "—" while every one was a genuine (and correct) "you won't run
    out"."""
    window = _window()
    predicted = dataclasses.replace(
        _forecast(window),
        model_name="montecarlo",
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=0.12,
        confidence="high",  # the simulation actually ran
    )

    row = _row(
        predicted, timezone.utc, burn_rate=_forecast(window), predicted=predicted
    )

    assert row[6] == "safe (risk 12%)"


def test_safe_needs_a_simulation_that_actually_ran(cfg):
    """A status-only result (no confidence, no probability) is a genuine
    "no answer" and must stay a dash — safe is earned by a run, not
    assumed from silence."""
    window = _window()
    predicted = dataclasses.replace(
        _forecast(window),
        model_name="montecarlo",
        eta_p50=None,
        prob_exhaust_before_reset=None,
        confidence=None,
    )

    row = _row(
        predicted, timezone.utc, burn_rate=_forecast(window), predicted=predicted
    )

    assert row[6] == "—"


def test_a_fresh_safe_answer_beats_a_retained_band(cfg):
    """A fresh censored result must win over an older saved band — the
    saved band describes a moment the simulation has since re-answered."""
    window = _window()
    predicted = dataclasses.replace(
        _forecast(window),
        model_name="montecarlo",
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=0.05,
        confidence="medium",
    )
    retained = RetainedPrediction(
        provider=window.provider,
        kind=window.kind,
        made_at=900.0,
        used_percent=window.used_percent,
        eta_p50=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
        eta_p90=None,
    )

    row = _row(
        predicted,
        timezone.utc,
        burn_rate=_forecast(window),
        predicted=predicted,
        retained=retained,
    )

    assert row[6] == "safe (risk 5%)"


def test_working_hours_fallback_is_labeled():
    window = _window()
    burn_rate = dataclasses.replace(
        _forecast(window),
        eta_workhours=datetime(2026, 7, 28, 12, tzinfo=timezone.utc),
    )

    row = _row(
        burn_rate,
        timezone.utc,
        burn_rate=burn_rate,
    )

    assert row[6] == "hours Tue 12:00"


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
    """Overage rides in the Used % cell, attached to the 100% it qualifies,
    rather than in a column of its own that is blank on every other row."""
    forecast = dataclasses.replace(
        _forecast(_window(used_percent=100.0)), tokens_burned_past_quota=530_000
    )
    state = MonitorState(
        now=1000.0, windows=(forecast.window,), forecasts=(forecast,), alerts=()
    )
    text = _render_to_text(_render(state, cfg, []))
    assert "100% (+530K)" in text


def test_a_window_below_the_cap_shows_a_bare_percentage(cfg):
    """No empty parentheses on the ordinary row — overage is meaningful for
    exactly one value of this cell."""
    forecast = _forecast(_window(used_percent=62.0))
    state = MonitorState(
        now=1000.0, windows=(forecast.window,), forecasts=(forecast,), alerts=()
    )
    text = _render_to_text(_render(state, cfg, []))
    assert "62%" in text
    assert "(+" not in text


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
    assert "100% (+$12.34)" in text
    assert "530K" not in text


def test_spacebar_pressed_falls_back_to_plain_sleep_when_not_a_tty(monkeypatch):
    sleeps = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(cli_module, "msvcrt", None)

    class _NotATty:
        def isatty(self):
            return False

    monkeypatch.setattr(sys, "stdin", _NotATty())
    assert _spacebar_pressed(42.0) is False
    assert sleeps == [42.0]


def test_spacebar_pressed_uses_the_windows_console_api(monkeypatch):
    class _WindowsTty:
        @staticmethod
        def isatty():
            return True

    class _FakeMsvcrt:
        @staticmethod
        def kbhit():
            return True

        @staticmethod
        def getwch():
            return " "

    monkeypatch.setattr(cli_module, "msvcrt", _FakeMsvcrt())
    monkeypatch.setattr(sys, "stdin", _WindowsTty())

    assert _spacebar_pressed(42.0) is True


@pytest.mark.skipif(pty is None or tty is None, reason="POSIX pty API is unavailable")
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


@pytest.mark.skipif(pty is None or tty is None, reason="POSIX pty API is unavailable")
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
