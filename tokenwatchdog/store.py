"""SQLite persistence — the durable history every predictor learns from.

`samples` is the source of truth for burn-rate lookback; `token_events` is a
finer-grained Claude signal for future predictors; `alert_state` survives
restarts so alerts don't re-fire; every tick's `forecasts` are recorded so a
later backtest can compare models.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from tokenwatchdog.config import DEFAULT_DB_PATH
from tokenwatchdog.models import Forecast, Provider, WindowKind

_SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
  id INTEGER PRIMARY KEY,
  captured_at    REAL NOT NULL,
  provider       TEXT NOT NULL,
  window_kind    TEXT NOT NULL,
  source_ts      REAL NOT NULL,
  used_percent   REAL NOT NULL,
  window_minutes INTEGER NOT NULL,
  resets_at      REAL,
  is_estimated   INTEGER NOT NULL DEFAULT 0,
  source_file    TEXT NOT NULL,
  -- idempotent re-insert of a static line. resets_at is deliberately NOT
  -- part of this key: it's nullable, and SQLite never treats two NULLs as
  -- equal in a UNIQUE constraint, which would defeat dedup entirely for
  -- any provider that doesn't supply resets_at (Claude, currently always).
  UNIQUE(provider, window_kind, source_ts)
);
CREATE INDEX IF NOT EXISTS idx_samples_key_ts ON samples(provider, window_kind, source_ts);

CREATE TABLE IF NOT EXISTS token_events (
  provider       TEXT NOT NULL,
  request_id     TEXT NOT NULL,
  message_id     TEXT NOT NULL,
  ts             REAL NOT NULL,
  model          TEXT NOT NULL,
  input_tokens   INTEGER NOT NULL,
  output_tokens  INTEGER NOT NULL,
  cache_creation INTEGER NOT NULL,
  cache_read     INTEGER NOT NULL,
  PRIMARY KEY (provider, request_id, message_id)
);
CREATE INDEX IF NOT EXISTS idx_token_ts ON token_events(provider, ts);

CREATE TABLE IF NOT EXISTS alert_state (
  alert_key           TEXT PRIMARY KEY,
  armed               INTEGER NOT NULL DEFAULT 1,
  last_fired_at       REAL,
  reset_epoch_at_fire REAL
);

CREATE TABLE IF NOT EXISTS forecasts (
  id INTEGER PRIMARY KEY,
  made_at      REAL NOT NULL,
  provider     TEXT NOT NULL,
  window_kind  TEXT NOT NULL,
  model_name   TEXT NOT NULL,
  used_percent REAL NOT NULL,
  eta_calendar REAL,
  eta_p50      REAL,
  eta_p90      REAL
);
"""


@dataclass(frozen=True)
class SampleRow:
    captured_at: float
    source_ts: float
    used_percent: float
    resets_at: float | None
    is_estimated: bool


@dataclass(frozen=True)
class TokenEventRow:
    ts: float
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation: int
    cache_read: int


@dataclass(frozen=True)
class AlertStateRow:
    armed: bool
    last_fired_at: float | None
    reset_epoch_at_fire: float | None


class Store:
    """Thin wrapper over the history.db connection. Not thread-safe by
    design — one Engine, one thread, one connection; there's no need for
    more on a single machine polling once a minute."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        if self._path.parent != Path():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- samples --------------------------------------------------------

    def insert_sample(
        self,
        *,
        captured_at: float,
        provider: Provider,
        window_kind: WindowKind,
        source_ts: float,
        used_percent: float,
        window_minutes: int,
        resets_at: float | None,
        is_estimated: bool,
        source_file: str,
    ) -> bool:
        """INSERT OR IGNORE — a static (unchanged) reading is a silent no-op
        because (provider, window_kind, source_ts, resets_at) is UNIQUE.
        Returns True if a new row was actually written."""
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO samples "
            "(captured_at, provider, window_kind, source_ts, used_percent, "
            " window_minutes, resets_at, is_estimated, source_file) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                captured_at,
                provider.value,
                window_kind.value,
                source_ts,
                used_percent,
                window_minutes,
                resets_at,
                int(is_estimated),
                source_file,
            ),
        )
        return cur.rowcount == 1

    def recent_samples(
        self, provider: Provider, kind: WindowKind, since_ts: float
    ) -> list[SampleRow]:
        """Samples at or after `since_ts`, oldest first — the lookback window
        a predictor fits its burn rate against."""
        rows = self._conn.execute(
            "SELECT captured_at, source_ts, used_percent, resets_at, is_estimated "
            "FROM samples WHERE provider = ? AND window_kind = ? AND source_ts >= ? "
            "ORDER BY source_ts ASC",
            (provider.value, kind.value, since_ts),
        ).fetchall()
        return [
            SampleRow(
                captured_at=row["captured_at"],
                source_ts=row["source_ts"],
                used_percent=row["used_percent"],
                resets_at=row["resets_at"],
                is_estimated=bool(row["is_estimated"]),
            )
            for row in rows
        ]

    # -- token_events -----------------------------------------------------

    def upsert_token_event(
        self,
        *,
        provider: Provider,
        request_id: str,
        message_id: str,
        ts: float,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cache_creation: int,
        cache_read: int,
    ) -> None:
        """Unconditional overwrite on conflict. Correct ONLY because callers
        feed rows in file order (oldest line first) — the final call for a
        given id is therefore the physically-last (largest, final) usage
        record for that message, matching ccusage bug #888's fix: keep LAST,
        not first."""
        self._conn.execute(
            "INSERT INTO token_events "
            "(provider, request_id, message_id, ts, model, "
            " input_tokens, output_tokens, cache_creation, cache_read) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(provider, request_id, message_id) DO UPDATE SET "
            "  ts=excluded.ts, model=excluded.model, "
            "  input_tokens=excluded.input_tokens, "
            "  output_tokens=excluded.output_tokens, "
            "  cache_creation=excluded.cache_creation, "
            "  cache_read=excluded.cache_read",
            (
                provider.value,
                request_id,
                message_id,
                ts,
                model,
                input_tokens,
                output_tokens,
                cache_creation,
                cache_read,
            ),
        )

    def recent_token_events(
        self, provider: Provider, since_ts: float
    ) -> list[TokenEventRow]:
        rows = self._conn.execute(
            "SELECT ts, model, input_tokens, output_tokens, cache_creation, cache_read "
            "FROM token_events WHERE provider = ? AND ts >= ? ORDER BY ts ASC",
            (provider.value, since_ts),
        ).fetchall()
        return [
            TokenEventRow(
                ts=row["ts"],
                model=row["model"],
                input_tokens=row["input_tokens"],
                output_tokens=row["output_tokens"],
                cache_creation=row["cache_creation"],
                cache_read=row["cache_read"],
            )
            for row in rows
        ]

    # -- alert_state --------------------------------------------------------

    def get_alert_state(self, alert_key: str) -> AlertStateRow | None:
        row = self._conn.execute(
            "SELECT armed, last_fired_at, reset_epoch_at_fire "
            "FROM alert_state WHERE alert_key = ?",
            (alert_key,),
        ).fetchone()
        if row is None:
            return None
        return AlertStateRow(
            armed=bool(row["armed"]),
            last_fired_at=row["last_fired_at"],
            reset_epoch_at_fire=row["reset_epoch_at_fire"],
        )

    def set_alert_state(
        self,
        alert_key: str,
        *,
        armed: bool,
        last_fired_at: float | None,
        reset_epoch_at_fire: float | None,
    ) -> None:
        self._conn.execute(
            "INSERT INTO alert_state (alert_key, armed, last_fired_at, reset_epoch_at_fire) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(alert_key) DO UPDATE SET "
            "  armed=excluded.armed, last_fired_at=excluded.last_fired_at, "
            "  reset_epoch_at_fire=excluded.reset_epoch_at_fire",
            (alert_key, int(armed), last_fired_at, reset_epoch_at_fire),
        )

    # -- forecasts --------------------------------------------------------

    def insert_forecast(self, *, made_at: float, forecast: Forecast) -> None:
        window = forecast.window
        self._conn.execute(
            "INSERT INTO forecasts "
            "(made_at, provider, window_kind, model_name, used_percent, "
            " eta_calendar, eta_p50, eta_p90) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                made_at,
                window.provider.value,
                window.kind.value,
                forecast.model_name,
                window.used_percent,
                _to_epoch(forecast.eta_calendar),
                _to_epoch(forecast.eta_p50),
                _to_epoch(forecast.eta_p90),
            ),
        )

    # -- retention --------------------------------------------------------

    def prune_older_than(self, cutoff_ts: float) -> None:
        """Drop samples/token_events/forecasts older than `cutoff_ts`. Call
        once at startup, not every tick — this is a maintenance sweep, not a
        hot path, and there's no measured need to run it more often."""
        self._conn.execute("DELETE FROM samples WHERE source_ts < ?", (cutoff_ts,))
        self._conn.execute("DELETE FROM token_events WHERE ts < ?", (cutoff_ts,))
        self._conn.execute("DELETE FROM forecasts WHERE made_at < ?", (cutoff_ts,))


def _to_epoch(dt: object) -> float | None:
    if dt is None:
        return None
    return dt.timestamp()  # type: ignore[attr-defined]
