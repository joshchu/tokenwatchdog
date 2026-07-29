"""SQLite persistence — the durable history every predictor learns from.

`samples` is the source of truth for burn-rate lookback; `token_events` is a
finer-grained Claude signal for future predictors; `alert_state` survives
restarts so alerts don't re-fire; every tick's `forecasts` are recorded so a
later backtest can compare models. `app_state` preserves the cutoff for an
explicit reset so provider logs cannot silently reconstruct forgotten data.
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

-- Durable application metadata that must survive a history reset. In
-- particular, providers can reconstruct old usage from their own logs, so
-- deleting rows alone would just refill the database on the next poll.
CREATE TABLE IF NOT EXISTS app_state (
  key   TEXT PRIMARY KEY,
  value REAL NOT NULL
);

-- How far each provider's log ingestion has already got, so a poll only
-- reads files that changed since the last one. Without it every tick
-- re-parsed every transcript inside the retention window and re-upserted
-- every token event in them -- measured at 12.6s of a 13.5s tick, on files
-- that had not changed.
CREATE TABLE IF NOT EXISTS ingest_cursor (
  source          TEXT PRIMARY KEY,  -- provider whose logs these are
  scanned_through REAL NOT NULL      -- files modified at/before this are done
);

CREATE TABLE IF NOT EXISTS alert_state (
  alert_key           TEXT PRIMARY KEY,
  armed               INTEGER NOT NULL DEFAULT 1,
  last_fired_at       REAL,
  reset_epoch_at_fire REAL
);

-- Every tick's forecast from every model, so a past prediction can be graded
-- against what actually happened. scoring.py (which decides what
-- predictor.model = "auto" trusts) reads made_at, model_name, eta_calendar
-- and status -- status being what separates an ETA the predictor correctly
-- withheld from one that was silently broken. The idle dashboard also reads
-- used_percent plus eta_p50/eta_p90 from the latest compatible non-idle row,
-- but only as labeled display context. The other columns remain a record for
-- inspecting a past run by hand. scripts/backtest.py deliberately re-runs
-- predictors instead of reading these, since its job is to score code that
-- hasn't shipped yet.
CREATE TABLE IF NOT EXISTS forecasts (
  id INTEGER PRIMARY KEY,
  made_at        REAL NOT NULL,
  provider       TEXT NOT NULL,
  window_kind    TEXT NOT NULL,
  model_name     TEXT NOT NULL,
  used_percent   REAL NOT NULL,
  eta_calendar   REAL,
  eta_p50        REAL,
  eta_p90        REAL,
  eta_workhours  REAL,
  status         TEXT,
  burn_per_hour  REAL,
  burn_basis     TEXT,
  time_to_reset_h REAL,
  exhausts_before_reset INTEGER
);
CREATE INDEX IF NOT EXISTS idx_forecasts_key_made
ON forecasts(provider, window_kind, model_name, made_at DESC);

-- Keep a user-requested reset durable across provider rescans. These live in
-- SQLite rather than provider-specific readers so every present and future
-- ingestion path observes the same boundary.
CREATE TRIGGER IF NOT EXISTS reject_samples_before_history_floor
BEFORE INSERT ON samples
WHEN NEW.source_ts < COALESCE(
  (SELECT value FROM app_state WHERE key = 'history_floor'),
  0
)
BEGIN
  SELECT RAISE(IGNORE);
END;

CREATE TRIGGER IF NOT EXISTS reject_token_events_before_history_floor
BEFORE INSERT ON token_events
WHEN NEW.ts < COALESCE(
  (SELECT value FROM app_state WHERE key = 'history_floor'),
  0
)
BEGIN
  SELECT RAISE(IGNORE);
END;
"""

# Columns added to `forecasts` after the table first shipped. CREATE TABLE
# IF NOT EXISTS silently leaves an existing table alone, so an upgrade needs
# these applied explicitly or every write below fails on the old schema.
_FORECAST_COLUMNS_ADDED_LATER = {
    "eta_workhours": "REAL",
    "status": "TEXT",
    "burn_per_hour": "REAL",
    "burn_basis": "TEXT",
    "time_to_reset_h": "REAL",
    "exhausts_before_reset": "INTEGER",
    # A recorded probability is a deliberate answer even when the median
    # future never exhausts (no ETA) -- scoring counts such a row as
    # answered rather than as a coverage miss, and the backtest's risk
    # calibration grades the probabilities themselves.
    "prob_exhaust_before_reset": "REAL",
    # The model's own expected used% at its kind's dense horizon -- what
    # model selection grades (see models.Forecast.predicted_used_percent).
    "predicted_used_percent": "REAL",
}


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

    @property
    def total_tokens(self) -> int:
        """All four buckets summed — what actually consumes quota.

        Safe to add for both providers despite their different accounting
        models: Claude's cache fields are separate additive dimensions, and
        Codex's are already folded into input/output so it stores 0 in them
        (see providers/codex.py)."""
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation
            + self.cache_read
        )


@dataclass(frozen=True)
class ForecastRow:
    """A past forecast, read back to grade it (see scoring.py). Only the
    columns scoring actually reads — the table records more per forecast, but
    a field here with no reader is just something else to keep true.

    `used_percent` and `burn_per_hour` feed the dense fixed-horizon metric
    (predicted used% at +h = used + burn·h); `prob_exhaust_before_reset`
    marks a deliberate no-ETA answer as answered. The latter two are NULL on
    rows written before their columns existed — genuinely not recorded."""

    made_at: float
    model_name: str
    eta_calendar: float | None
    status: str | None
    used_percent: float
    burn_per_hour: float | None
    prob_exhaust_before_reset: float | None
    predicted_used_percent: float | None


@dataclass(frozen=True)
class ForecastSnapshotRow:
    """The last non-idle model result used as a possible display fallback.

    Unlike ForecastRow this exposes the saved percentile and usage level.
    The engine still decides whether the row belongs to the current window
    and is recent enough to show.
    """

    made_at: float
    used_percent: float
    eta_p50: float | None
    eta_p90: float | None
    status: str


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
        self._add_missing_forecast_columns()

    def _add_missing_forecast_columns(self) -> None:
        """Bring an existing history.db up to the current `forecasts` shape.

        Additive only, and each column is nullable — rows written before a
        column existed keep NULL there, which is exactly right: the value
        genuinely wasn't recorded, and backfilling a guess would corrupt the
        very history a backtest reads."""
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(forecasts)").fetchall()
        }
        for column, sql_type in _FORECAST_COLUMNS_ADDED_LATER.items():
            if column not in existing:
                self._conn.execute(
                    f"ALTER TABLE forecasts ADD COLUMN {column} {sql_type}"
                )

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
        because (provider, window_kind, source_ts) is UNIQUE. Returns True
        if a new row was actually written."""
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

    def delete_token_event(
        self, *, provider: Provider, request_id: str, message_id: str
    ) -> None:
        """Remove one provider event by its durable source identity.

        This is intentionally narrow rather than a bulk-delete API. A provider
        may discover on a later, versioned rescan that a previously ingested
        source line was only a repeated cumulative snapshot; deleting that
        exact line repairs the history without disturbing any real events.
        """
        self._conn.execute(
            "DELETE FROM token_events "
            "WHERE provider = ? AND request_id = ? AND message_id = ?",
            (provider.value, request_id, message_id),
        )

    # -- ingest_cursor ------------------------------------------------------

    def get_ingest_cursor(self, source_key: str) -> float | None:
        """Modification time through which this log directory is already
        ingested, or None for one never scanned (where the caller should fall
        back to the whole retention window as a backfill).

        Keyed by provider AND directory. Keyed by provider alone, pointing
        `claude.config_dir` / `codex.home` at a different directory —
        a second profile, a moved home — would skip everything already in it
        forever, since its files' honest mtimes predate a cursor built from
        the old directory."""
        row = self._conn.execute(
            "SELECT scanned_through FROM ingest_cursor WHERE source = ?",
            (source_key,),
        ).fetchone()
        return row["scanned_through"] if row else None

    def set_ingest_cursor(self, source_key: str, scanned_through: float) -> None:
        """Record the moment a scan STARTED, not when it finished — a file
        appended to mid-scan then has an mtime past the cursor and gets
        re-read next time, instead of being silently skipped."""
        self._conn.execute(
            "INSERT INTO ingest_cursor (source, scanned_through) VALUES (?, ?) "
            "ON CONFLICT(source) DO UPDATE SET scanned_through=excluded.scanned_through",
            (source_key, scanned_through),
        )

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
            " eta_calendar, eta_p50, eta_p90, eta_workhours, status, "
            " burn_per_hour, burn_basis, time_to_reset_h, exhausts_before_reset, "
            " prob_exhaust_before_reset, predicted_used_percent) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                made_at,
                window.provider.value,
                window.kind.value,
                forecast.model_name,
                window.used_percent,
                _to_epoch(forecast.eta_calendar),
                _to_epoch(forecast.eta_p50),
                _to_epoch(forecast.eta_p90),
                _to_epoch(forecast.eta_workhours),
                forecast.status,
                forecast.burn_per_hour,
                forecast.burn_basis,
                forecast.time_to_reset_h,
                int(forecast.exhausts_before_reset),
                forecast.prob_exhaust_before_reset,
                forecast.predicted_used_percent,
            ),
        )

    def recent_forecasts(
        self, provider: Provider, kind: WindowKind, since_ts: float
    ) -> list[ForecastRow]:
        """Past forecasts for one window, oldest first — what scoring.py
        grades. Rows written before the schema gained `status` have NULL
        there; that's a genuine "not recorded," not a value to invent."""
        rows = self._conn.execute(
            "SELECT made_at, model_name, eta_calendar, status, used_percent, "
            "  burn_per_hour, prob_exhaust_before_reset, predicted_used_percent "
            "FROM forecasts WHERE provider = ? AND window_kind = ? AND made_at >= ? "
            "ORDER BY made_at ASC",
            (provider.value, kind.value, since_ts),
        ).fetchall()
        return [
            ForecastRow(
                made_at=row["made_at"],
                model_name=row["model_name"],
                eta_calendar=row["eta_calendar"],
                status=row["status"],
                used_percent=row["used_percent"],
                burn_per_hour=row["burn_per_hour"],
                prob_exhaust_before_reset=row["prob_exhaust_before_reset"],
                predicted_used_percent=row["predicted_used_percent"],
            )
            for row in rows
        ]

    def latest_non_idle_forecast(
        self,
        provider: Provider,
        kind: WindowKind,
        model_name: str,
        *,
        at_or_before: float,
    ) -> ForecastSnapshotRow | None:
        """Latest result backed by a non-stale reading, including barriers.

        RESET_PENDING is intentionally returned rather than skipped. If a
        reset was observed after an older valid percentile, that newer row
        must prevent the old cycle's prediction from resurfacing.
        """
        row = self._conn.execute(
            "SELECT made_at, used_percent, eta_p50, eta_p90, status "
            "FROM forecasts "
            "WHERE provider = ? AND window_kind = ? AND model_name = ? "
            "  AND made_at <= ? AND status <> 'IDLE' "
            "ORDER BY made_at DESC, id DESC LIMIT 1",
            (provider.value, kind.value, model_name, at_or_before),
        ).fetchone()
        if row is None:
            return None
        return ForecastSnapshotRow(
            made_at=row["made_at"],
            used_percent=row["used_percent"],
            eta_p50=row["eta_p50"],
            eta_p90=row["eta_p90"],
            status=row["status"],
        )

    # -- retention --------------------------------------------------------

    def prune_older_than(self, cutoff_ts: float) -> None:
        """Drop samples/token_events/forecasts older than `cutoff_ts`. Call
        once at startup, not every tick — this is a maintenance sweep, not a
        hot path, and there's no measured need to run it more often."""
        self._conn.execute("DELETE FROM samples WHERE source_ts < ?", (cutoff_ts,))
        self._conn.execute("DELETE FROM token_events WHERE ts < ?", (cutoff_ts,))
        self._conn.execute("DELETE FROM forecasts WHERE made_at < ?", (cutoff_ts,))

    def reset_history(self, reset_at: float) -> int:
        """Forget learned/runtime data and reject provider history before now.

        Codex and Claude keep their own append-only logs. Clearing our tables
        without a durable floor would therefore appear to work, then silently
        restore the old training history on the next provider scan.

        Configuration is file-backed and intentionally outside this reset.
        Returns the number of application rows removed.
        """
        tables = (
            "samples",
            "token_events",
            "forecasts",
            "alert_state",
            "ingest_cursor",
        )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            removed = sum(
                self._conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in tables
            )
            for table in tables:
                self._conn.execute(f"DELETE FROM {table}")
            self._conn.execute(
                "INSERT INTO app_state (key, value) VALUES ('history_floor', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (reset_at,),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return removed


def _to_epoch(dt: object) -> float | None:
    if dt is None:
        return None
    return dt.timestamp()  # type: ignore[attr-defined]
