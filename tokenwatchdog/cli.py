"""CLI front-end — a rich.live.Live terminal dashboard over Engine.tick().

`--headless` runs the identical loop with no rendering, for running as a
background agent — alerts fire from the engine either way, never from this
module.
"""

from __future__ import annotations

import argparse
import contextlib
import select
import sys
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime, tzinfo

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from tokenwatchdog.config import Config, load_config, resolve_timezone
from tokenwatchdog.engine import Engine
from tokenwatchdog.models import Alert, Forecast, MonitorState

try:
    import termios
    import tty
except ImportError:  # Windows has neither -- fall back to a plain sleep
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

_CALM_EMOJI = "🐶"
_ALERT_EMOJI = "🐕"
_BURN_EMOJI = "🔥"

# How many past alerts stay visible in the terminal after they fire — an OS
# notification (and the bark) is easy to miss or forget the reason for, so
# the dashboard keeps a short, visible trail of what actually fired and why.
_ALERT_LOG_SIZE = 10


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config()
    with Engine(cfg=cfg) as engine:
        if args.once:
            state = engine.tick()
            alert_log: deque[Alert] = deque(state.alerts, maxlen=_ALERT_LOG_SIZE)
            Console().print(_render(state, cfg, alert_log))
            return 0
        if args.headless:
            return _run_headless(engine, cfg)
        return _run_live(engine, cfg)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tokenwatchdog")
    parser.add_argument(
        "--once", action="store_true", help="tick once, print, and exit"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the poll loop with no display — alerts only (LaunchAgent mode)",
    )
    return parser.parse_args(argv)


def _run_live(engine: Engine, cfg: Config) -> int:
    console = Console()
    console.clear()  # start at the top of the terminal, not wherever the shell left it
    console.print("[dim]Press space to refresh now, Ctrl-C to quit.[/dim]")
    alert_log: deque[Alert] = deque(maxlen=_ALERT_LOG_SIZE)
    try:
        with (
            _cbreak_stdin(),
            Live(console=console, screen=False, refresh_per_second=4) as live,
        ):
            while True:
                state = engine.tick()
                alert_log.extend(state.alerts)
                live.update(_render(state, cfg, alert_log), refresh=True)
                if _spacebar_pressed(cfg.poll_interval_seconds):
                    # Immediate feedback that the keypress registered -- the
                    # next tick can still take a moment (reading Codex/Claude
                    # logs), so without this the display just looks frozen.
                    live.update(
                        _render(state, cfg, alert_log, refreshing=True), refresh=True
                    )
    except KeyboardInterrupt:
        pass
    return 0


@contextlib.contextmanager
def _cbreak_stdin():
    """Puts the terminal into cbreak mode (no line-buffering, no echo) for
    the whole live session. Toggling this on/off every poll cycle left a
    window where the terminal was back in cooked/echo-on mode in between --
    a keypress landing in that window got echoed straight to the screen by
    the tty driver itself (bypassing Rich entirely), desyncing Live's
    line-count bookkeeping and leaving a stray duplicate border line
    behind on the next redraw. Falls back to a no-op when stdin isn't an
    interactive TTY or on a platform without termios/tty (Windows)."""
    if termios is None or tty is None or not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    tty.setcbreak(fd)  # keeps ISIG, so Ctrl-C still raises KeyboardInterrupt
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def _spacebar_pressed(seconds: float) -> bool:
    """Waits up to `seconds`, returning True if a spacebar press woke it
    early (so the caller can trigger an immediate refresh) or False once
    the full interval elapses. Assumes stdin is already in cbreak mode
    (see _cbreak_stdin) — falls back to a plain sleep, always returning
    False, when stdin isn't an interactive TTY or termios/tty aren't
    available."""
    if termios is None or tty is None or not sys.stdin.isatty():
        time.sleep(seconds)
        return False
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        ready, _, _ = select.select([sys.stdin], [], [], remaining)
        if not ready:
            return False
        if sys.stdin.read(1) == " ":
            return True


def _run_headless(engine: Engine, cfg: Config) -> int:
    try:
        while True:
            engine.tick()
            time.sleep(cfg.poll_interval_seconds)
    except KeyboardInterrupt:
        pass
    return 0


def _render(
    state: MonitorState,
    cfg: Config,
    alert_log: Sequence[Alert],
    *,
    refreshing: bool = False,
) -> Group:
    tz = resolve_timezone(cfg)
    # All wall-clock columns below are already tz-converted (see _row/_fmt_dt)
    # — label them with the actual abbreviation so a local time is never
    # mistaken for UTC at a glance, matching the "updated ... EDT" header.
    tz_label = datetime.fromtimestamp(state.now, tz=tz).strftime("%Z")
    table = Table(expand=True)
    for column in (
        "Provider",
        "Window",
        "Used %",
        "Status",
        "Burn %/h",
        f"ETA (calendar, {tz_label})",
        f"ETA (working hrs, {tz_label})",
        f"Resets ({tz_label})",
        "Conf.",
    ):
        table.add_column(column)

    for forecast in sorted(
        state.forecasts, key=lambda f: (f.window.provider.value, f.window.kind.value)
    ):
        table.add_row(*_row(forecast, tz), style=_row_style(forecast, cfg, state.now))

    mascot = _mascot_glyph(state.forecasts, cfg)
    updated = datetime.fromtimestamp(state.now, tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    status_suffix = " — refreshing…" if refreshing else ""
    panel = Panel(
        table,
        title=f"{mascot} TokenWatchDog — updated {updated}{status_suffix}",
        subtitle=(
            "* = estimated (token-based limit is a guess, not an official cap)"
            " · red = alert · yellow = trending toward exhaustion"
        ),
        border_style="blue",
    )
    if not alert_log:
        return Group(panel)
    return Group(panel, _render_alert_log(alert_log, tz))


def _render_alert_log(alert_log: Sequence[Alert], tz: tzinfo) -> Panel:
    """What actually fired, and why — a notification banner or a bark is
    easy to miss or forget the reason for once it's gone; this stays on
    screen so "why did it just bark" always has an answer close by."""
    table = Table(expand=True, box=None, show_header=True)
    table.add_column("Fired", width=8)
    table.add_column("Alert")
    for alert in reversed(list(alert_log)):
        fired = datetime.fromtimestamp(alert.fired_at, tz=tz).strftime("%H:%M:%S")
        label = f"{alert.provider.value}/{alert.kind.value} · {alert.alert_kind}"
        table.add_row(fired, f"[bold]{label}[/bold] — {alert.message}")
    return Panel(table, title="🔔 Recent alerts", border_style="yellow")


def _row(forecast: Forecast, tz: tzinfo) -> tuple[str, ...]:
    window = forecast.window
    used = f"{window.used_percent:.0f}%" + ("*" if window.is_estimated else "")
    burn = f"{forecast.burn_per_hour:+.2f}" if forecast.status == "OK" else "—"
    resets_dt = (
        datetime.fromtimestamp(window.resets_at, tz=tz) if window.resets_at else None
    )
    return (
        window.provider.value,
        window.kind.value,
        used,
        _status_label(forecast),
        burn,
        _fmt_dt(forecast.eta_calendar, tz),
        _fmt_dt(forecast.eta_workhours, tz),
        _fmt_dt(resets_dt, tz),
        forecast.confidence or "—",
    )


def _status_label(forecast: Forecast) -> str:
    if forecast.status == "OK" and forecast.exhausts_before_reset:
        return f"{_BURN_EMOJI} burning"
    if forecast.status == "OK":
        return "ok"
    return forecast.status.lower()


def _fmt_dt(dt: datetime | None, tz: tzinfo) -> str:
    if dt is None:
        return "—"
    return dt.astimezone(tz).strftime("%a %H:%M")


def _row_style(forecast: Forecast, cfg: Config, now: float) -> str | None:
    """Red mirrors the exact condition alerts.py fires on — current state,
    not the armed/fired history, so the highlight doesn't vanish the tick
    after a one-shot notification already fired. Yellow is the weaker "on
    pace to exhaust before reset" signal the Status column already labels
    "burning," for a burn that isn't urgent enough to be red yet.

    NO_DATA/IDLE are excluded exactly like alerts.evaluate() excludes them —
    a stale reading is data the predictor didn't trust enough to alert on,
    so painting it red/yellow would flag a number that's already been
    flagged as unreliable."""
    if forecast.status in ("NO_DATA", "IDLE"):
        return None
    window = forecast.window
    if window.used_percent >= cfg.thresholds.warn_percent:
        return "red"
    if forecast.status != "OK" or not forecast.exhausts_before_reset:
        return None
    if (
        window.used_percent >= cfg.thresholds.burn_min_percent
        and forecast.eta_calendar is not None
        and (forecast.eta_calendar.timestamp() - now) / 3600.0
        <= cfg.thresholds.burn_alert_within_hours
    ):
        return "red"
    return "yellow"


def _mascot_glyph(forecasts: tuple[Forecast, ...], cfg: Config) -> str:
    if any(f.status == "OK" and f.exhausts_before_reset for f in forecasts):
        return _BURN_EMOJI
    if any(f.window.used_percent >= cfg.thresholds.warn_percent for f in forecasts):
        return _ALERT_EMOJI
    return _CALM_EMOJI


if __name__ == "__main__":
    sys.exit(main())
