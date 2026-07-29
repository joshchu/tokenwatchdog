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
from rich.text import Text

from tokenwatchdog.config import Config, load_config, resolve_timezone
from tokenwatchdog.engine import Engine
from tokenwatchdog.models import (
    Alert,
    Forecast,
    MonitorState,
    RetainedPrediction,
    level_still_in_cycle,
)
from tokenwatchdog.store import Store

try:
    import termios
    import tty
except ImportError:  # Windows has neither -- fall back to a plain sleep
    termios = None  # type: ignore[assignment]
    tty = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # Windows-only console input API
    msvcrt = None  # type: ignore[assignment]

_CALM_EMOJI = "🐶"
_ALERT_EMOJI = "🐕"
_BURN_EMOJI = "🔥"

# How many past alerts stay visible in the terminal after they fire — an OS
# notification (and the bark) is easy to miss or forget the reason for, so
# the dashboard keeps a short, visible trail of what actually fired and why.
_ALERT_LOG_SIZE = 10

# Keep the two display questions stable even when `auto` changes which model
# is authoritative for alerts: one is always the direct ETA implied by the
# displayed burn rate, the other is always the learned-history prediction.
_BURN_RATE_ETA_MODEL = "linear"
_PREDICTED_ETA_MODEL = "montecarlo"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.reset_data:
        with Store() as store:
            removed = store.reset_history(time.time())
        Console().print(
            f"Reset complete: removed {removed} saved data row(s). "
            "Configuration and provider logs were preserved; their pre-reset "
            "history will not be relearned. Restart any running monitor."
        )
        return 0

    cfg = load_config()
    with Engine(cfg=cfg) as engine:
        if args.once:
            state = engine.tick()
            alert_log: deque[Alert] = deque(state.alerts, maxlen=_ALERT_LOG_SIZE)
            Console().print(
                _render(
                    state,
                    cfg,
                    alert_log,
                    model_choice_reason=engine.model_choice_reason,
                )
            )
            return 0
        if args.headless:
            return _run_headless(engine, cfg)
        return _run_live(engine, cfg)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tokenwatchdog")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="tick once, print, and exit")
    mode.add_argument(
        "--headless",
        action="store_true",
        help="run the poll loop with no display — alerts only (background/service mode)",
    )
    mode.add_argument(
        "--reset-data",
        action="store_true",
        help=(
            "delete learned history and saved runtime state, prevent old "
            "provider logs from refilling it, and exit"
        ),
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
                reason = engine.model_choice_reason
                live.update(
                    _render(state, cfg, alert_log, model_choice_reason=reason),
                    refresh=True,
                )
                if _spacebar_pressed(cfg.poll_interval_seconds):
                    # Immediate feedback that the keypress registered -- the
                    # next tick can still take a moment (reading Codex/Claude
                    # logs), so without this the display just looks frozen.
                    live.update(
                        _render(
                            state,
                            cfg,
                            alert_log,
                            refreshing=True,
                            model_choice_reason=reason,
                        ),
                        refresh=True,
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
    False, when stdin isn't an interactive TTY. POSIX uses select() against a
    cbreak terminal; Windows polls the stdlib msvcrt console API."""
    if msvcrt is not None and sys.stdin.isatty():
        deadline = time.monotonic() + seconds
        while True:
            # The stdlib stubs expose these only when type-checking on win32;
            # on macOS/Linux mypy sees the importable module shape without its
            # platform-gated members.
            if msvcrt.kbhit():  # type: ignore[attr-defined]
                key = msvcrt.getwch()  # type: ignore[attr-defined]
                if key == " ":
                    return True
                if key == "\x03":
                    raise KeyboardInterrupt
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            # msvcrt has no timed wait. A short sleep keeps this responsive
            # without turning the poll interval into a busy loop.
            time.sleep(min(remaining, 0.05))
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
    model_choice_reason: str = "",
) -> Group:
    tz = resolve_timezone(cfg)
    # All wall-clock values below are already tz-converted (see _row/_fmt_dt).
    # The panel's "updated ... EDT" clock establishes their timezone, and the
    # reset header repeats it; putting it on both ETA headers too costs scarce
    # table width without adding another clock context.
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
        "Status",
        "Burn %/h",
        "ETA on burn %/h",
        "Predicted ETA (P50 → P90)",
        f"Resets ({tz_label})",
        *(("Risk",) if show_risk else ()),
        "Conf.",
    ):
        table.add_column(column)

    for forecast in sorted(
        state.forecasts, key=lambda f: (f.window.provider.value, f.window.kind.value)
    ):
        burn_rate = (
            state.forecast_from(_BURN_RATE_ETA_MODEL, forecast.window) or forecast
        )
        table.add_row(
            *_row(
                forecast,
                tz,
                burn_rate=burn_rate,
                predicted=state.forecast_from(_PREDICTED_ETA_MODEL, forecast.window),
                retained=state.retained_prediction_for(forecast.window),
                show_risk=show_risk,
            ),
            style=_row_style(forecast, cfg, state.now),
        )

    mascot = _mascot_glyph(state.forecasts, cfg)
    # Which model is authoritative and why -- shown because `auto` can change
    # it from what config literally says, and an unexplained switch is worse
    # than no switch.
    model_note = f"model {model_choice_reason}" if model_choice_reason else ""
    updated = datetime.fromtimestamp(state.now, tz=tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    status_suffix = " — refreshing…" if refreshing else ""
    title = f"{mascot} TokenWatchDog — updated {updated}{status_suffix}"
    panel = Panel(
        table,
        title=Text(title, style="bold bright_cyan"),
        subtitle=(
            "* = estimated (token-based limit is a guess, not an official cap)"
            " · red = alert · yellow = trending toward exhaustion"
            " · safe = most simulated futures survive to the reset"
            + (f" · {model_note}" if model_note else "")
        ),
        border_style="bright_cyan",
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
    burn_rate: Forecast,
    predicted: Forecast | None = None,
    retained: RetainedPrediction | None = None,
    show_risk: bool = False,
) -> tuple[str, ...]:
    window = forecast.window
    burn = _fmt_burn(burn_rate)
    resets_dt = (
        datetime.fromtimestamp(window.resets_at, tz=tz) if window.resets_at else None
    )
    return (
        window.provider.value,
        window.kind.value,
        _fmt_used(forecast),
        _status_label(forecast),
        burn,
        _fmt_dt(burn_rate.eta_calendar, tz),
        _fmt_predicted_eta(burn_rate, predicted, retained, tz),
        _fmt_dt(resets_dt, tz),
        *((_fmt_risk(predicted or forecast),) if show_risk else ()),
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


def _fmt_predicted_eta(
    burn_rate: Forecast,
    predicted: Forecast | None,
    retained: RetainedPrediction | None,
    tz: tzinfo,
) -> str:
    """ "Given how you actually use this across a week, when do you run out."

    Three ways to answer that, and the better one wins when it's available:
    the simulated hour-of-week profile LEARNED from your token history
    (shown as a P50 → P90 band, because a band is the honest shape for
    bursty usage — a bare "Thu 15:00" implies precision the data doesn't
    have), its latest compatible saved result when stale cycle metadata stops
    a fresh simulation, or, while that profile is still thin, the working
    hours you DECLARED in config. Labels distinguish the single-value cases.

    This is deliberately a different question from the burn-rate ETA beside
    it, which asks "at the measured rate, projected around the clock."

    A censored fresh result — the simulation ran and MOST futures survive
    to the reset, with the tail risk known — renders as "safe (risk N%)"
    rather than "—": the model answered, and silence was indistinguishable
    from having nothing to say. Measured before this existed: 4,513
    consecutive OK ticks on the weekly window rendered "—" while every one
    of them was a genuine (and correct) "you won't run out"."""
    if predicted is not None and predicted.eta_p50 is not None:
        if predicted.eta_p90 is None:
            return f"P50 {_fmt_dt(predicted.eta_p50, tz)}"
        return f"{_fmt_dt(predicted.eta_p50, tz)} → {_fmt_dt(predicted.eta_p90, tz)}"
    if (
        predicted is not None
        and predicted.confidence is not None  # the simulation actually ran
        and predicted.prob_exhaust_before_reset is not None
    ):
        return f"safe (risk {predicted.prob_exhaust_before_reset * 100:.0f}%)"
    if retained is not None:
        if retained.eta_p90 is None:
            return f"saved P50 {_fmt_dt(retained.eta_p50, tz)}"
        return (
            f"saved {_fmt_dt(retained.eta_p50, tz)} → {_fmt_dt(retained.eta_p90, tz)}"
        )
    workhours = _fmt_dt(burn_rate.eta_workhours, tz)
    return f"hours {workhours}" if workhours != "—" else workhours


def _fmt_risk(forecast: Forecast) -> str:
    """Probability of exhausting before the reset, when a model computes one.
    Distinct from the ETA: a 40% chance of running out is worth seeing even
    though the median future doesn't."""
    if forecast.prob_exhaust_before_reset is None:
        return "—"
    return f"{forecast.prob_exhaust_before_reset * 100:.0f}%"


def _fmt_used(forecast: Forecast) -> str:
    """Used percent, with what's been spent PAST the cap in parentheses once
    the percentage pins at 100 -- e.g. `100% (+7.5M)`.

    Overage lives here rather than in a column of its own because it is
    meaningful for exactly one value of this cell: below 100% it is always
    blank, so a dedicated column spent the whole table's width to say nothing
    on almost every row. Attached to the number it qualifies, it reads as
    "100%, and 7.5M tokens beyond that."

    A priced estimate wins over a raw count when rates are configured (see
    predictor.overage_cost_usd); with neither, the percentage stands alone."""
    used = f"{forecast.window.used_percent:.0f}%"
    if forecast.window.is_estimated:
        used += "*"
    if forecast.cost_burned_past_quota_usd is not None:
        return f"{used} (+${forecast.cost_burned_past_quota_usd:,.2f})"
    if forecast.tokens_burned_past_quota is not None:
        return f"{used} (+{_fmt_tokens(forecast.tokens_burned_past_quota)})"
    return used


def _fmt_tokens(n: int) -> str:
    """Bare magnitude — the `(+…)` it renders inside supplies the unit, and a
    `$` prefix is what distinguishes a priced overage from a token count."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


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
