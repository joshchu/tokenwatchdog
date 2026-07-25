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

from tokenwatchdog.alerts import level_still_in_cycle
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

# Which model answers the "rhythm" question (see _fmt_rhythm). The other
# column is whichever model is authoritative, answering the trend question.
_RHYTHM_MODEL = "montecarlo"


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
    # Only the simulating model produces a probability; without it the column
    # would be an always-empty stripe of width, which on a narrow terminal
    # costs every other column real characters.
    show_risk = any(
        f.prob_exhaust_before_reset is not None for f in state.all_forecasts
    )
    table = Table(expand=True)
    for column in (
        "Provider",
        "Window",
        "Used %",
        "Past cap",
        "Status",
        "Burn %/h",
        f"ETA trend ({tz_label})",
        f"ETA rhythm ({tz_label})",
        f"Resets ({tz_label})",
        *(("Risk",) if show_risk else ()),
        "Conf.",
    ):
        table.add_column(column)

    for forecast in sorted(
        state.forecasts, key=lambda f: (f.window.provider.value, f.window.kind.value)
    ):
        table.add_row(
            *_row(
                forecast,
                tz,
                rhythm=state.forecast_from(_RHYTHM_MODEL, forecast.window),
                show_risk=show_risk,
            ),
            style=_row_style(forecast, cfg, state.now),
        )

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


def _row(
    forecast: Forecast,
    tz: tzinfo,
    *,
    rhythm: Forecast | None = None,
    show_risk: bool = False,
) -> tuple[str, ...]:
    window = forecast.window
    used = f"{window.used_percent:.0f}%" + ("*" if window.is_estimated else "")
    burn = _fmt_burn(forecast)
    resets_dt = (
        datetime.fromtimestamp(window.resets_at, tz=tz) if window.resets_at else None
    )
    return (
        window.provider.value,
        window.kind.value,
        used,
        _fmt_overage(forecast),
        _status_label(forecast),
        burn,
        _fmt_dt(forecast.eta_calendar, tz),
        _fmt_rhythm(forecast, rhythm, tz),
        _fmt_dt(resets_dt, tz),
        *((_fmt_risk(rhythm or forecast),) if show_risk else ()),
        forecast.confidence or "—",
    )


def _fmt_burn(forecast: Forecast) -> str:
    """The rate, marked `~` when it came from the fallback percent slope
    rather than real token throughput — that's the case where the source is
    a whole-number percentage too coarse to resolve a slow window, so the
    number deserves to look approximate."""
    if forecast.status != "OK":
        return "—"
    suffix = "~" if forecast.burn_basis == "percent" else ""
    return f"{forecast.burn_per_hour:+.2f}{suffix}"


def _fmt_rhythm(forecast: Forecast, rhythm: Forecast | None, tz: tzinfo) -> str:
    """ "Given how you actually use this across a week, when do you run out."

    Two ways to answer that, and the better one wins when it's available:
    the simulated hour-of-week profile LEARNED from your token history
    (shown as a P50 → P90 band, because a band is the honest shape for
    bursty usage — a bare "Thu 15:00" implies precision the data doesn't
    have), or, while that profile is still thin, the working hours you
    DECLARED in config. The presence of a band is what distinguishes them.

    This is deliberately a different question from the trend column beside
    it, which asks "at the pace of the last few hours, projected around the
    clock." Both were already being computed; only one was ever shown."""
    if rhythm is not None and rhythm.eta_p50 is not None:
        if rhythm.eta_p90 is None:
            return _fmt_dt(rhythm.eta_p50, tz)
        return f"{_fmt_dt(rhythm.eta_p50, tz)} → {_fmt_dt(rhythm.eta_p90, tz)}"
    return _fmt_dt(forecast.eta_workhours, tz)


def _fmt_risk(forecast: Forecast) -> str:
    """Probability of exhausting before the reset, when a model computes one.
    Distinct from the ETA: a 40% chance of running out is worth seeing even
    though the median future doesn't."""
    if forecast.prob_exhaust_before_reset is None:
        return "—"
    return f"{forecast.prob_exhaust_before_reset * 100:.0f}%"


def _fmt_overage(forecast: Forecast) -> str:
    """The most informative thing we know about usage since the window
    pinned at 100% -- a priced $ estimate when configured (see
    predictor.overage_cost_usd), else the raw token count, else nothing.
    used_percent alone reads as "0 burn" here (see _row for why); this is
    what actually moves once it does."""
    if forecast.cost_burned_past_quota_usd is not None:
        return f"${forecast.cost_burned_past_quota_usd:,.2f}"
    if forecast.tokens_burned_past_quota is not None:
        return _fmt_tokens(forecast.tokens_burned_past_quota)
    return "—"


def _fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M tok"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K tok"
    return f"{n} tok"


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

    Full exhaustion is checked first and overrides everything below: 100%
    used is the literal last-known reading, not a predicted trend.

    The IDLE handling tracks alerts.evaluate() deliberately, including its
    level-vs-rate split — a stale *rate* is untrustworthy, but a stale
    *level* is just a level, so an idle window still paints red for being
    over the warn threshold as long as its cycle hasn't turned over."""
    window = forecast.window
    if window.used_percent >= 100.0:
        return "red"
    if forecast.status == "NO_DATA":
        return None
    if forecast.status == "IDLE" and not level_still_in_cycle(window, now):
        return None
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
