"""Notification delivery tests — mocked subprocess/shutil, no real system
notifications or audio playback are ever triggered."""

from __future__ import annotations

import dataclasses
import types

import pytest

import tokenwatchdog.notify as notify_module
from tokenwatchdog.models import Alert, Provider, WindowKind
from tokenwatchdog.notify import notify


def _alert(**overrides):
    kwargs = dict(
        key="codex:weekly:threshold",
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        alert_kind="threshold",
        message="test message",
        fired_at=0.0,
    )
    kwargs.update(overrides)
    return Alert(**kwargs)


def _with_sound(cfg, sound):
    return dataclasses.replace(
        cfg, notifications=dataclasses.replace(cfg.notifications, sound=sound)
    )


@pytest.fixture
def fake_which(monkeypatch):
    """shutil.which resolves any binary name to a fake path."""
    monkeypatch.setattr(notify_module.shutil, "which", lambda name: f"/usr/bin/{name}")


@pytest.fixture
def calls(monkeypatch):
    recorded: list[list[str]] = []

    def fake_run(args, **kwargs):
        recorded.append(args)
        return None

    monkeypatch.setattr(notify_module.subprocess, "run", fake_run)
    return recorded


def test_bark_sound_calls_say_and_omits_notifier_sound_flag(cfg, fake_which, calls):
    notify(_alert(), _with_sound(cfg, "bark"))
    notifier_call = next(c for c in calls if c[0] == "/usr/bin/terminal-notifier")
    assert "-sound" not in notifier_call
    say_call = next(c for c in calls if c[0] == "/usr/bin/say")
    assert say_call == ["/usr/bin/say", "Woof! Woof!"]


def test_bark_is_case_insensitive(cfg, fake_which, calls):
    notify(_alert(), _with_sound(cfg, "BARK"))
    assert any(c[0] == "/usr/bin/say" for c in calls)


def test_regular_sound_name_passed_through_no_bark(cfg, fake_which, calls):
    notify(_alert(), _with_sound(cfg, "Submarine"))
    notifier_call = next(c for c in calls if c[0] == "/usr/bin/terminal-notifier")
    assert "-sound" in notifier_call
    assert "Submarine" in notifier_call
    assert not any(c[0] == "/usr/bin/say" for c in calls)


def test_disabled_notifications_send_nothing(cfg, fake_which, calls):
    disabled = dataclasses.replace(
        cfg, notifications=dataclasses.replace(cfg.notifications, enabled=False)
    )
    notify(_alert(), disabled)
    assert calls == []


def test_notifier_failure_is_swallowed(cfg, fake_which, monkeypatch):
    def raise_oserror(args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(notify_module.subprocess, "run", raise_oserror)
    notify(_alert(), cfg)  # must not raise


def _no_say(name):
    return None if name == "say" else f"/usr/bin/{name}"


def test_bark_falls_back_to_a_windows_beep_when_say_is_unavailable(
    cfg, monkeypatch, calls
):
    """Regression: on Windows there's no `say` and no bundled bark audio to
    ship, so the audible cue for the "bark" setting should still play --
    via the stdlib winsound module -- instead of silently doing nothing."""
    monkeypatch.setattr(notify_module.shutil, "which", _no_say)
    beeps = []
    fake_winsound = types.SimpleNamespace(
        MessageBeep=lambda beep_type: beeps.append(beep_type),
        MB_ICONEXCLAMATION="MB_ICONEXCLAMATION",
    )
    monkeypatch.setattr(notify_module, "winsound", fake_winsound)
    notify(_alert(), _with_sound(cfg, "bark"))
    assert beeps == ["MB_ICONEXCLAMATION"]
    assert not any(c[0] == "/usr/bin/say" for c in calls)


def test_bark_does_nothing_when_neither_say_nor_winsound_is_available(
    cfg, monkeypatch, calls
):
    monkeypatch.setattr(notify_module.shutil, "which", _no_say)
    monkeypatch.setattr(notify_module, "winsound", None)
    notify(_alert(), _with_sound(cfg, "bark"))  # must not raise
    assert not any(c[0] == "/usr/bin/say" for c in calls)
