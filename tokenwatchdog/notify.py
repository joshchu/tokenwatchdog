"""Notification delivery — terminal-notifier, falling back to osascript.

A notifier failure must never kill the polling loop: alert de-dup state is
already persisted by the time this is called, so a delivery failure here
just means one missed OS notification, not a retry storm or a crashed
engine.
"""

from __future__ import annotations

import shutil
import subprocess

try:
    import winsound
except ImportError:  # not on Windows
    winsound = None  # type: ignore[assignment]

from tokenwatchdog.config import Config
from tokenwatchdog.models import Alert

_TIMEOUT_SECONDS = 5

# macOS ships no actual dog-bark system sound (no audio asset is bundled or
# fetched here — an unlicensed one has no business in this repo), so the
# "bark" sound is delivered as speech via `say` instead of a named system
# sound. Case-insensitive so "Bark"/"BARK" in config also work.
_BARK_SOUND_NAME = "bark"
_BARK_PHRASE = "Woof! Woof!"


def notify(alert: Alert, cfg: Config) -> None:
    if not cfg.notifications.enabled:
        return
    title = f"TokenWatchDog — {alert.provider.value}"
    subtitle = f"{alert.kind.value} · {alert.alert_kind}"
    try:
        _deliver(title, subtitle, alert.message, cfg)
    except (OSError, subprocess.SubprocessError):
        pass


def _deliver(title: str, subtitle: str, message: str, cfg: Config) -> None:
    sound = cfg.notifications.sound
    use_bark = sound.strip().lower() == _BARK_SOUND_NAME

    notifier = cfg.notifications.notifier or shutil.which("terminal-notifier")
    if notifier:
        args = [
            notifier,
            "-title",
            title,
            "-subtitle",
            subtitle,
            "-message",
            message,
            "-group",
            "tokenwatchdog",
        ]
        if sound and not use_bark:
            args += ["-sound", sound]
        subprocess.run(args, check=False, timeout=_TIMEOUT_SECONDS)
    else:
        osascript = shutil.which("osascript")
        if osascript is not None:
            script = (
                f"display notification {_applescript_str(message)} "
                f"with title {_applescript_str(title)} "
                f"subtitle {_applescript_str(subtitle)}"
            )
            subprocess.run(
                [osascript, "-e", script], check=False, timeout=_TIMEOUT_SECONDS
            )

    if use_bark:
        _bark()


def _bark() -> None:
    """Audible cue for the "bark" sound setting. macOS: a spoken woof via
    `say`. Windows has no `say` equivalent and no bundled bark audio to
    ship, so it rings the system exclamation sound via the stdlib
    `winsound` module instead -- not a literal bark, but still an audible,
    unmistakable "something needs your attention.\""""
    say = shutil.which("say")
    if say is not None:
        subprocess.run([say, _BARK_PHRASE], check=False, timeout=_TIMEOUT_SECONDS)
    elif winsound is not None:
        # typeshed's winsound stub only declares these two under win32, so
        # mypy running on macOS/Linux sees a module with neither -- a
        # static-analysis gap, not a real portability issue (the try/except
        # ImportError above already covers this module not existing here).
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)  # type: ignore[attr-defined]


def _applescript_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
