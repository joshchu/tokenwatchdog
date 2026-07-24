"""CLI front-end — a rich.live.Live terminal dashboard over Engine.tick().

`--headless` runs the identical loop with no rendering, for running as a
background agent — alerts fire from the engine either way, never from this
module.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, tzinfo

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from tokenwatchdog.config import Config, load_config, resolve_timezone
from tokenwatchdog.engine import Engine
from tokenwatchdog.models import Forecast, MonitorState

_CALM_EMOJI = "🐶"
_ALERT_EMOJI = "🐕"
_BURN_EMOJI = "🔥"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = load_config()
    with Engine(cfg=cfg) as engine:
        if args.once:
            state = engine.tick()
            Console().print(_render(state, cfg))
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
    try:
        with Live(console=console, screen=False, refresh_per_second=1) as live:
            while True:
                state = engine.tick()
                live.update(_render(state, cfg))
                time.sleep(cfg.poll_interval_seconds)
    except KeyboardInterrupt:
        pass
    return 0


def _run_headless(engine: Engine, cfg: Config) -> int:
    try:
        while True:
            engine.tick()
            time.sleep(cfg.poll_interval_seconds)
    except KeyboardInterrupt:
        pass
    return 0


def _render(state: MonitorState, cfg: Config) -> Panel:
    tz = resolve_timezone(cfg)
    table = Table(expand=True)
    for column in (
        "Provider",
        "Window",
        "Used %",
        "Status",
        "Burn %/h",
        "ETA (calendar)",
        "ETA (working hrs)",
        "Resets",
        "Conf.",
    ):
        table.add_column(column)

    for forecast in sorted(
        state.forecasts, key=lambda f: (f.window.provider.value, f.window.kind.value)
    ):
        table.add_row(*_row(forecast, tz))

    mascot = _mascot_glyph(state.forecasts)
    updated = datetime.fromtimestamp(state.now, tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    return Panel(
        table,
        title=f"{mascot} TokenWatchDog — updated {updated}",
        subtitle="* = estimated (token-based limit is a guess, not an official cap)",
        border_style="blue",
    )


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
        forecast.confidence,
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


def _mascot_glyph(forecasts: tuple[Forecast, ...]) -> str:
    if any(f.status == "OK" and f.exhausts_before_reset for f in forecasts):
        return _BURN_EMOJI
    if any(f.window.used_percent >= 90.0 for f in forecasts):
        return _ALERT_EMOJI
    return _CALM_EMOJI


if __name__ == "__main__":
    sys.exit(main())
