"""Store persistence tests: idempotent inserts, dedup-keep-last token
events, alert_state round trip, and survival across a reopen."""

from __future__ import annotations

from tokenwatchdog.models import Provider, WindowKind
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
