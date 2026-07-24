"""Alert de-dup/re-arm state machine tests."""

from __future__ import annotations

from datetime import datetime, timezone

from tokenwatchdog.alerts import evaluate
from tokenwatchdog.models import Forecast, Provider, Window, WindowKind
from tokenwatchdog.store import Store


def _dt(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _forecast(
    used_percent,
    *,
    resets_at=100_000.0,
    status="OK",
    exhausts=False,
    eta_calendar=None,
    time_to_reset_h=None,
):
    window = Window(
        provider=Provider.CODEX,
        kind=WindowKind.WEEKLY,
        used_percent=used_percent,
        window_minutes=10080,
        resets_at=resets_at,
        source_ts=0.0,
        is_estimated=False,
        source_file="test",
    )
    return Forecast(
        window=window,
        status=status,
        model_name="linear",
        burn_per_hour=1.0,
        time_to_reset_h=time_to_reset_h,
        eta_calendar=eta_calendar,
        eta_workhours=None,
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=None,
        confidence="high",
        exhausts_before_reset=exhausts,
        n_samples=5,
    )


def test_threshold_fires_once_then_suppressed(cfg, store):
    now = 1000.0
    assert evaluate(_forecast(89.0), cfg, store, now) == []
    fired = evaluate(_forecast(91.0), cfg, store, now)
    assert len(fired) == 1
    assert fired[0].alert_kind == "threshold"
    assert evaluate(_forecast(92.0), cfg, store, now + 60) == []


def test_threshold_rearms_when_resets_at_advances(cfg, store):
    now = 1000.0
    evaluate(_forecast(91.0, resets_at=100_000.0), cfg, store, now)  # fires, disarms
    assert evaluate(_forecast(91.0, resets_at=100_000.0), cfg, store, now + 60) == []
    fired_again = evaluate(_forecast(91.0, resets_at=200_000.0), cfg, store, now + 120)
    assert len(fired_again) == 1


def test_threshold_rearms_on_hysteresis_drop(cfg, store):
    now = 1000.0
    evaluate(_forecast(91.0), cfg, store, now)  # fires, disarms
    below_hysteresis = 90.0 - cfg.thresholds.threshold_hysteresis - 1.0
    evaluate(_forecast(below_hysteresis), cfg, store, now + 60)  # re-arms, no fire
    fired = evaluate(_forecast(91.0), cfg, store, now + 120)
    assert len(fired) == 1


def test_burn_alert_fires_when_exhaustion_is_imminent(cfg, store):
    now = 1000.0
    # Exhausts in 30 minutes -- clearly within the default 1h alert window.
    forecast = _forecast(
        30.0, exhausts=True, eta_calendar=_dt(now + 1800), time_to_reset_h=5.0
    )
    fired = evaluate(forecast, cfg, store, now)
    assert any(a.alert_kind == "burn" for a in fired)


def test_burn_alert_does_not_fire_below_min_percent(cfg, store):
    now = 1000.0
    forecast = _forecast(
        10.0, exhausts=True, eta_calendar=_dt(now + 1800), time_to_reset_h=5.0
    )
    fired = evaluate(forecast, cfg, store, now)
    assert not any(a.alert_kind == "burn" for a in fired)


def test_burn_alert_does_not_fire_when_exhaustion_is_not_imminent(cfg, store):
    """Regression: technically "on pace to exhaust before the reset" isn't
    enough on its own -- if that's still hours away, it isn't urgent, and
    alerting on it is exactly the "spurious" noise this gate exists to
    prevent."""
    now = 1000.0
    forecast = _forecast(
        30.0,
        exhausts=True,
        eta_calendar=_dt(now + 3 * 3600),  # 3h out -- beyond the 1h default
        time_to_reset_h=5.0,
    )
    fired = evaluate(forecast, cfg, store, now)
    assert not any(a.alert_kind == "burn" for a in fired)


def test_burn_alert_rearms_when_resets_at_advances(cfg, store):
    """Regression: this only works if the predictor actually threads a
    derived resets_at onto Forecast.window (Claude never gets one straight
    from a provider) — otherwise reset_epoch_at_fire stays NULL forever and
    a Claude burn alert can fire at most once for the database's lifetime."""
    now = 1000.0
    first = _forecast(
        30.0,
        resets_at=100_000.0,
        exhausts=True,
        eta_calendar=_dt(now + 3600),
        time_to_reset_h=5.0,
    )
    fired = evaluate(first, cfg, store, now)
    assert any(a.alert_kind == "burn" for a in fired)

    still_disarmed = evaluate(first, cfg, store, now + 60)
    assert not any(a.alert_kind == "burn" for a in still_disarmed)

    second = _forecast(
        30.0,
        resets_at=200_000.0,
        exhausts=True,
        eta_calendar=_dt(now + 3600),
        time_to_reset_h=5.0,
    )
    fired_again = evaluate(second, cfg, store, now + 120)
    assert any(a.alert_kind == "burn" for a in fired_again)


def test_idle_threshold_still_fires_while_the_level_is_in_cycle(cfg, store):
    """Staleness invalidates a burn RATE, not a used-% LEVEL. A window that
    read 95% an hour ago and doesn't reset for another day is still at 95%
    now -- quota doesn't un-consume while you're away, so staying silent
    here was suppressing a true alert about a real number."""
    forecast = _forecast(95.0, status="IDLE", resets_at=100_000.0)
    fired = evaluate(forecast, cfg, store, 1000.0)
    assert [a.alert_kind for a in fired] == ["threshold"]


def test_idle_burn_alert_never_fires(cfg, store):
    """The other half of the split: a rate measured from a stale reading is
    exactly what shouldn't be trusted, so the burn alert stays suppressed
    even though the threshold alert now gets through."""
    forecast = _forecast(
        95.0,
        status="IDLE",
        exhausts=True,
        eta_calendar=_dt(1800.0),
        resets_at=100_000.0,
    )
    fired = evaluate(forecast, cfg, store, 1000.0)
    assert all(a.alert_kind != "burn" for a in fired)


def test_idle_forecast_past_its_reset_never_fires(cfg, store):
    """The case the blanket IDLE skip was really protecting: a daemon
    started fresh against an old 95% reading whose window has since turned
    over. Without a resets_at still in the future there's no basis for
    claiming the level survived, so it stays quiet."""
    stale = _forecast(95.0, status="IDLE", resets_at=500.0)
    assert evaluate(stale, cfg, store, 1000.0) == []
    unknown_reset = _forecast(95.0, status="IDLE", resets_at=None)
    assert evaluate(unknown_reset, cfg, store, 1000.0) == []


def test_no_data_forecast_never_fires(cfg, store):
    forecast = _forecast(95.0, status="NO_DATA")
    assert evaluate(forecast, cfg, store, 1000.0) == []


def test_alert_state_survives_reopen(tmp_path, cfg):
    store = Store(tmp_path / "history.db")
    now = 1000.0
    fired = evaluate(_forecast(91.0), cfg, store, now)
    assert len(fired) == 1
    store.close()

    reopened = Store(tmp_path / "history.db")
    fired_again = evaluate(_forecast(91.0), cfg, reopened, now + 60)
    assert fired_again == []
    reopened.close()
