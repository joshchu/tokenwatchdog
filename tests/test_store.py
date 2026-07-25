"""Store persistence tests: idempotent inserts, dedup-keep-last token
events, alert_state round trip, and survival across a reopen."""

from __future__ import annotations

from tokenwatchdog.models import Forecast, Provider, Window, WindowKind
from tokenwatchdog.store import Store


def _sample_kwargs(**overrides):
    kwargs = dict(
        captured_at=1000.0,
        provider=Provider.CODEX,
        window_kind=WindowKind.WEEKLY,
        source_ts=500.0,
        used_percent=42.0,
        window_minutes=10080,
        resets_at=2000.0,
        is_estimated=False,
        source_file="rollout.jsonl",
    )
    kwargs.update(overrides)
    return kwargs


def _forecast(source_ts: float = 500.0) -> Forecast:
    return Forecast(
        window=Window(
            provider=Provider.CODEX,
            kind=WindowKind.WEEKLY,
            used_percent=42.0,
            window_minutes=10080,
            resets_at=2000.0,
            source_ts=source_ts,
            is_estimated=False,
            source_file="rollout.jsonl",
        ),
        status="OK",
        model_name="linear",
        burn_per_hour=1.0,
        time_to_reset_h=58.0,
        eta_calendar=None,
        eta_workhours=None,
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=None,
        confidence="high",
        exhausts_before_reset=False,
        n_samples=5,
    )


def test_duplicate_sample_insert_is_a_noop(store):
    assert store.insert_sample(**_sample_kwargs()) is True
    assert store.insert_sample(**_sample_kwargs()) is False
    rows = store.recent_samples(Provider.CODEX, WindowKind.WEEKLY, since_ts=0.0)
    assert len(rows) == 1


def test_recent_samples_respects_since_ts_and_order(store):
    for ts in (100.0, 300.0, 200.0):
        store.insert_sample(**_sample_kwargs(source_ts=ts, captured_at=ts))
    rows = store.recent_samples(Provider.CODEX, WindowKind.WEEKLY, since_ts=150.0)
    assert [r.source_ts for r in rows] == [200.0, 300.0]


def test_store_survives_reopen(tmp_path):
    db_path = tmp_path / "history.db"
    store = Store(db_path)
    store.insert_sample(**_sample_kwargs())
    store.close()

    reopened = Store(db_path)
    rows = reopened.recent_samples(Provider.CODEX, WindowKind.WEEKLY, since_ts=0.0)
    assert len(rows) == 1
    reopened.close()


def test_token_event_upsert_keeps_last(store):
    store.upsert_token_event(
        provider=Provider.CLAUDE,
        request_id="req_1",
        message_id="msg_1",
        ts=10.0,
        model="claude-x",
        input_tokens=1,
        output_tokens=1,
        cache_creation=0,
        cache_read=0,
    )
    store.upsert_token_event(
        provider=Provider.CLAUDE,
        request_id="req_1",
        message_id="msg_1",
        ts=11.0,
        model="claude-x",
        input_tokens=5,
        output_tokens=500,
        cache_creation=10,
        cache_read=20,
    )
    events = store.recent_token_events(Provider.CLAUDE, since_ts=0.0)
    assert len(events) == 1
    assert events[0].output_tokens == 500


def test_alert_state_round_trip(store):
    assert store.get_alert_state("codex:weekly:threshold") is None
    store.set_alert_state(
        "codex:weekly:threshold",
        armed=False,
        last_fired_at=123.0,
        reset_epoch_at_fire=456.0,
    )
    state = store.get_alert_state("codex:weekly:threshold")
    assert state is not None
    assert state.armed is False
    assert state.last_fired_at == 123.0
    assert state.reset_epoch_at_fire == 456.0


def test_prune_older_than_removes_old_rows(store):
    store.insert_sample(**_sample_kwargs(source_ts=100.0, captured_at=100.0))
    store.insert_sample(
        **_sample_kwargs(source_ts=900.0, captured_at=900.0, resets_at=3000.0)
    )
    store.prune_older_than(500.0)
    rows = store.recent_samples(Provider.CODEX, WindowKind.WEEKLY, since_ts=0.0)
    assert [r.source_ts for r in rows] == [900.0]


def test_reset_history_clears_saved_data_and_durably_rejects_old_logs(tmp_path):
    db_path = tmp_path / "history.db"
    store = Store(db_path)
    store.insert_sample(**_sample_kwargs())
    store.upsert_token_event(
        provider=Provider.CODEX,
        request_id="req_before",
        message_id="msg_before",
        ts=600.0,
        model="gpt-5",
        input_tokens=100,
        output_tokens=20,
        cache_creation=0,
        cache_read=0,
    )
    store.insert_forecast(made_at=700.0, forecast=_forecast())
    store.set_alert_state(
        "codex:weekly:threshold",
        armed=False,
        last_fired_at=800.0,
        reset_epoch_at_fire=2000.0,
    )
    store.set_ingest_cursor("codex:/old/home", 900.0)

    assert store.reset_history(reset_at=1000.0) == 5
    store.close()

    # Reopening proves the floor is persistent, not merely cached on the
    # connection that performed the reset.
    reopened = Store(db_path)
    assert reopened.recent_samples(Provider.CODEX, WindowKind.WEEKLY, 0.0) == []
    assert reopened.recent_token_events(Provider.CODEX, 0.0) == []
    assert reopened.recent_forecasts(Provider.CODEX, WindowKind.WEEKLY, 0.0) == []
    assert reopened.get_alert_state("codex:weekly:threshold") is None
    assert reopened.get_ingest_cursor("codex:/old/home") is None

    # A provider rescan cannot silently refill training with pre-reset logs.
    assert (
        reopened.insert_sample(**_sample_kwargs(source_ts=999.0, captured_at=1100.0))
        is False
    )
    reopened.upsert_token_event(
        provider=Provider.CODEX,
        request_id="req_old_rescan",
        message_id="msg_old_rescan",
        ts=999.0,
        model="gpt-5",
        input_tokens=100,
        output_tokens=20,
        cache_creation=0,
        cache_read=0,
    )
    assert reopened.recent_token_events(Provider.CODEX, 0.0) == []

    # The boundary itself and all later usage start the new training history.
    assert reopened.insert_sample(**_sample_kwargs(source_ts=1000.0)) is True
    reopened.upsert_token_event(
        provider=Provider.CODEX,
        request_id="req_after",
        message_id="msg_after",
        ts=1001.0,
        model="gpt-5",
        input_tokens=100,
        output_tokens=20,
        cache_creation=0,
        cache_read=0,
    )
    assert len(reopened.recent_samples(Provider.CODEX, WindowKind.WEEKLY, 0.0)) == 1
    assert len(reopened.recent_token_events(Provider.CODEX, 0.0)) == 1
    reopened.close()
