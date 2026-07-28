"""The shared core every front-end wraps: Engine.tick(now) -> MonitorState.

read-all-providers -> store -> predict -> alert -> notify, one pass per
tick. Every front-end (rich TUI, `--headless` daemon, a future menu-bar app)
is a thin shell over this.
"""

from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone

from tokenwatchdog.alerts import evaluate as evaluate_alerts
from tokenwatchdog.config import Config, load_config
from tokenwatchdog.models import (
    Alert,
    Forecast,
    MonitorState,
    Provider,
    RetainedPrediction,
    Window,
    WindowKind,
    window_duration_seconds,
)
from tokenwatchdog.notify import notify
from tokenwatchdog.predictor import (
    all_predictor_names,
    overage_cost_usd,
    predictor_named,
    select_predictor,
    tokens_burned_past_quota,
)
from tokenwatchdog.providers.base import QuotaProvider
from tokenwatchdog.providers.claude import ClaudeProvider
from tokenwatchdog.providers.codex import CodexProvider
from tokenwatchdog.store import Store


class Engine:
    def __init__(
        self,
        cfg: Config | None = None,
        store: Store | None = None,
        providers: list[QuotaProvider] | None = None,
    ) -> None:
        self.cfg = cfg or load_config()
        self.store = store or Store()
        self._providers = (
            providers if providers is not None else self._build_providers()
        )
        self._predictor, self.model_choice_reason = select_predictor(
            self.cfg, self.store
        )
        self._pruned = False

    def _build_providers(self) -> list[QuotaProvider]:
        providers: list[QuotaProvider] = []
        if self.cfg.providers.codex:
            providers.append(CodexProvider())
        if self.cfg.providers.claude:
            providers.append(ClaudeProvider())
        return providers

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def tick(self, now: float | None = None) -> MonitorState:
        now = now if now is not None else time.time()
        self._prune_once(now)

        windows_by_key: dict[tuple[Provider, WindowKind], Window] = {}
        for provider in self._providers:
            for window in provider.read(self.cfg, self.store):
                windows_by_key[(window.provider, window.kind)] = window
                self.store.insert_sample(
                    captured_at=now,
                    provider=window.provider,
                    window_kind=window.kind,
                    source_ts=window.source_ts,
                    used_percent=window.used_percent,
                    window_minutes=window.window_minutes,
                    resets_at=window.resets_at,
                    is_estimated=window.is_estimated,
                    source_file=window.source_file,
                )

        forecasts: list[Forecast] = []
        all_forecasts: list[Forecast] = []
        retained_predictions: list[RetainedPrediction] = []
        alerts: list[Alert] = []
        for provider_enum, kind in self._watched_pairs():
            matched_window = windows_by_key.get((provider_enum, kind))
            if matched_window is None:
                forecast = _no_data_forecast(provider_enum, kind, now)
                all_forecasts.append(forecast)
            else:
                # Every model, every tick. They answer different questions
                # (see MonitorState.all_forecasts), the dashboard shows both,
                # and storing both is what lets scoring.py compare them on
                # identical moments later. Costs a few hundred ms against a
                # ~60s poll interval.
                produced = {
                    name: self._forecast_for(
                        name, provider_enum, kind, matched_window, now
                    )
                    for name in all_predictor_names()
                }
                for candidate in produced.values():
                    self.store.insert_forecast(made_at=now, forecast=candidate)
                all_forecasts.extend(produced.values())
                retained = self._retained_prediction_for(
                    produced.get("montecarlo"), now
                )
                if retained is not None:
                    retained_predictions.append(retained)
                forecast = produced[self._predictor.name]
            forecasts.append(forecast)

            fired = evaluate_alerts(forecast, self.cfg, self.store, now)
            for alert in fired:
                notify(alert, self.cfg)
            alerts.extend(fired)

        return MonitorState(
            now=now,
            windows=tuple(windows_by_key.values()),
            forecasts=tuple(forecasts),
            alerts=tuple(alerts),
            all_forecasts=tuple(all_forecasts),
            retained_predictions=tuple(retained_predictions),
        )

    def _retained_prediction_for(
        self, current: Forecast | None, now: float
    ) -> RetainedPrediction | None:
        """Recover a still-useful saved band for an otherwise blank idle row.

        The latest non-idle result is a barrier as well as a candidate: a
        newer RESET_PENDING or OK-without-percentiles row invalidates any
        older band. The saved usage level must exactly match the level still
        displayed, and no snapshot survives longer than one quota window.
        """
        if (
            current is None
            or current.status != "IDLE"
            or current.eta_p50 is not None
            # A confidence rating means Monte Carlo actually ran. Its blank
            # P50 is then a real censored result ("most futures survive"),
            # which must win over an older saved band. The fallback is only
            # for the status-only path where stale cycle state prevented a
            # fresh simulation.
            or current.confidence is not None
        ):
            return None

        window = current.window
        saved = self.store.latest_non_idle_forecast(
            window.provider,
            window.kind,
            current.model_name,
            at_or_before=now,
        )
        if (
            saved is None
            or saved.status != "OK"
            or saved.eta_p50 is None
            or saved.used_percent != window.used_percent
        ):
            return None

        duration = window_duration_seconds(window.kind)
        if saved.made_at < now - duration or saved.eta_p50 <= now:
            return None
        if window.resets_at is not None:
            cycle_started_at = window.resets_at - duration
            if now >= window.resets_at or saved.made_at < cycle_started_at:
                return None

        return RetainedPrediction(
            provider=window.provider,
            kind=window.kind,
            made_at=saved.made_at,
            used_percent=saved.used_percent,
            eta_p50=datetime.fromtimestamp(saved.eta_p50, tz=timezone.utc),
            eta_p90=(
                datetime.fromtimestamp(saved.eta_p90, tz=timezone.utc)
                if saved.eta_p90 is not None
                else None
            ),
        )

    def _forecast_for(
        self,
        model_name: str,
        provider_enum: Provider,
        kind: WindowKind,
        window: Window,
        now: float,
    ) -> Forecast:
        # The full retention window, not just one reset cycle: linear only
        # ever looks at a short recent tail regardless, but montecarlo needs
        # many past blocks to populate its hour-of-week buckets. Retained
        # history is already bounded to a few thousand rows, so querying
        # all of it every tick is still cheap.
        weeks = max(self.cfg.predictor.history_retention_weeks, 1)
        lookback_seconds = weeks * 7 * 24 * 3600
        history = self.store.recent_samples(provider_enum, kind, now - lookback_seconds)
        # Unconditional: Codex now ingests its own per-event token counts too
        # (providers/codex.py), so this is real data for both providers, not
        # just Claude -- empty and harmless for any provider that doesn't.
        token_events = self.store.recent_token_events(
            provider_enum, now - lookback_seconds
        )
        forecast = predictor_named(model_name).forecast(
            window, history, token_events, self.cfg, now
        )
        burned = tokens_burned_past_quota(window, history, token_events)
        cost = overage_cost_usd(window, history, token_events, self.cfg)
        if burned is None and cost is None:
            return forecast
        return dataclasses.replace(
            forecast,
            tokens_burned_past_quota=burned,
            cost_burned_past_quota_usd=cost,
        )

    def _watched_pairs(self) -> list[tuple[Provider, WindowKind]]:
        kinds = [WindowKind(k) for k in self.cfg.windows.watch]
        pairs: list[tuple[Provider, WindowKind]] = []
        for provider_enum, enabled in (
            (Provider.CODEX, self.cfg.providers.codex),
            (Provider.CLAUDE, self.cfg.providers.claude),
        ):
            if enabled:
                pairs.extend((provider_enum, kind) for kind in kinds)
        return pairs

    def _prune_once(self, now: float) -> None:
        """Maintenance sweep, not a hot path — once per process start is
        plenty; there's no measured need to run it more often."""
        if self._pruned:
            return
        weeks = max(self.cfg.predictor.history_retention_weeks, 1)
        self.store.prune_older_than(now - weeks * 7 * 24 * 3600)
        self._pruned = True


def _no_data_forecast(provider: Provider, kind: WindowKind, now: float) -> Forecast:
    """No provider returned this (provider, kind) this tick — e.g. Codex's
    5h window when a build isn't emitting it. The placeholder Window exists
    only to satisfy Forecast's shape; it is never written to the store, so
    it can never be mistaken for a real reading."""
    placeholder = Window(
        provider=provider,
        kind=kind,
        used_percent=0.0,
        window_minutes=300 if kind is WindowKind.W5H else 10080,
        resets_at=None,
        source_ts=now,
        is_estimated=True,
        source_file="",
    )
    return Forecast(
        window=placeholder,
        status="NO_DATA",
        model_name="none",
        burn_per_hour=0.0,
        time_to_reset_h=None,
        eta_calendar=None,
        eta_workhours=None,
        eta_p50=None,
        eta_p90=None,
        prob_exhaust_before_reset=None,
        confidence=None,  # nothing was forecast at all -- nothing to rate
        exhausts_before_reset=False,
        n_samples=0,
    )
