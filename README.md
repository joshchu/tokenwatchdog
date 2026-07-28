<p align="center">
  <img src="tokenwatchdog/assets/watchdog.svg" width="160" height="160" alt="TokenWatchDog mascot">
</p>

<h1 align="center">TokenWatchDog</h1>

<p align="center">
  A local watchdog for <b>Codex CLI</b> and <b>Claude Code</b> usage quota —
  polls your 5-hour and weekly limits, alerts before you hit them, and
  predicts when each window will actually run out.
</p>

<p align="center">
  <img src="tokenwatchdog/assets/screenshot.svg" alt="TokenWatchDog terminal dashboard">
</p>

---

## Why

Codex and Claude Code both cap usage on a 5-hour and a weekly window, and
it's easy to blow through either one mid-task. TokenWatchDog polls your
local usage data every minute and answers two questions existing usage
trackers don't:

- **When will I actually run out?** Not "you're at 62%" — an ETA, computed
  from your own recent burn rate, capped sensibly at the window's reset. That
  rate comes from real token throughput, not from the slope of the reported
  percentage: both providers round that percentage to whole numbers, and a
  weekly window burned evenly moves ~0.6% an hour, so the integer doesn't
  change often enough to forecast from at all.
- **When will I run out, given I only work certain hours?** The weekly
  projection can count only your configured working hours instead of
  assuming you burn quota 24/7 — so "exhausts Thursday" means Thursday
  *during work*, not Thursday at 3am.

It also alerts at 90% usage and separately when quota is being burned fast
enough that it will run out before the window naturally resets.

## Features

- Polls Codex CLI and Claude Code quota (5-hour + weekly) on a configurable
  interval (default ~1 minute)
- Burn rate measured from per-request token throughput, converted to %/h by a
  ratio calibrated against the provider's own reported percentage — so it
  needs no knowledge of an unpublished token cap, and falls back to the
  percentage's own slope (marked `~` in the dashboard) when there's nothing
  to calibrate against
- Two exhaustion predictors: a robust (outlier-resistant) burn-rate fit by
  default, or an opt-in Monte Carlo model that learns your hour-of-week usage
  rhythm from token history and simulates a P50/P90 exhaustion band plus a
  probability of running out before the reset, instead of one point guess
- **Two ETAs side by side, because they answer different questions.**
  **ETA on burn %/h** projects the displayed recent burn rate around the clock;
  **Predicted ETA (P50 → P90)** spreads the same budget over how you actually
  use it across a week — learned from your token history as a percentile
  range, or from your configured working hours while that profile is thin.
  Both models run every tick, so their disagreement is visible rather than
  hidden behind a config switch
- Each ETA is capped independently against the next reset — an ETA past the
  reset describes an event that can't happen, and no window is ever projected
  to take longer to exhaust than its own duration
- `predictor.model = "auto"` grades every model against **your** stored
  forecasts and keeps the default unless a challenger wins by a real margin
  across enough scored episodes; it reports which it chose and why, and
  declines rather than switching on a sample too small to mean anything
- Staleness applies to the burn *rate*, not the used-% *level*: quota doesn't
  un-consume while you're away, so a 100%-used reading still alerts once its
  cycle is confirmed live. The burn-rate ETA disappears when that rate is
  stale, while the learned prediction may continue from an in-cycle level
- `scripts/backtest.py` replays your stored history to score each predictor
  (ETA coverage, mean error, bias, P90 calibration), so "which model is
  better" is a measurement rather than an argument
- 90%-usage alert and a separate "burning too fast" alert, both native
  notifications with a cute spoken "Woof! Woof!" by default (any system
  sound name works too, or turn notifications off entirely with
  `notifications.enabled = false` in `config.toml`), each de-duplicated so
  you get at most one ping per window
- Persists history locally (SQLite) so the Monte Carlo model has real data
  to learn from, from the very first run
- Terminal dashboard UI — rows highlight red for an active alert, at 100%
  used, or yellow for usage trending toward exhaustion — with a `--headless`
  mode for running as a background agent
- Works with either provider alone or both at once; gracefully shows "no
  data" instead of guessing when a provider doesn't expose a window
- Once either provider's own reported percentage pins at 100%, burn rate
  alone goes blind — so the usage cell keeps counting real tokens spent past
  that point (from the same session/token logs) in parentheses, as
  `100% (+7.5M)`, and shows a `$` estimate instead of a raw token count
  once you set your plan's real
  overage rate: `codex.input_price_per_million_usd` /
  `output_price_per_million_usd` for Codex, or the same plus
  `cache_write_price_per_million_usd` / `cache_read_price_per_million_usd`
  under `[claude]` (Anthropic prices cache tokens on their own tiers). All
  default to `0`, i.e. off — this tool never guesses a price for you

## Quickstart

Requires Python 3.13+.

```bash
git clone https://github.com/joshchu/tokenwatchdog.git
cd tokenwatchdog
uv sync
uv run python -m tokenwatchdog
```

That opens a live terminal dashboard — press space to refresh immediately
instead of waiting out the poll interval, Ctrl-C to quit. Useful variants:

```bash
uv run python -m tokenwatchdog --once       # one reading, then exit
uv run python -m tokenwatchdog --headless   # poll loop, alerts only, no UI
uv run python -m tokenwatchdog --reset-data # clear learned/saved data, then exit
```

Config lives at `~/.tokenwatchdog/config.toml` (created with sane defaults on
first run) — poll interval, working hours, alert thresholds, which
providers/windows to watch, and notification settings all live there.
History is stored in `~/.tokenwatchdog/history.db`. Use `--reset-data` to
delete usage samples, token events, forecasts, alert state, and ingestion
cursors while preserving configuration and the original Codex/Claude logs.
The reset time is saved so rescanning those logs cannot silently restore
pre-reset training history; only usage at or after the reset is learned.
Stop any already-running TokenWatchDog process before resetting, then restart
it so its in-memory model choice is refreshed too. Do not delete `history.db`
manually: that would also delete the reset boundary and allow old logs to be
imported again.

## The metrics

### Dashboard columns

One row per quota — each provider's 5-hour and 7-day windows are independent
caps and are predicted separately. All wall-clock columns are shown in
`timezone` from your config, with the abbreviation in the header so a local
time is never misread as UTC.

| Column | What it means |
|---|---|
| **Provider** / **Window** | `codex` or `claude`, and which cap: `w5h` (5-hour) or `weekly` (7-day). |
| **Used %** | How much of the window is gone. Whole numbers, because that's the resolution both providers report. A trailing `*` means the number is not a percentage the provider stated: either it was computed from token counts against an unpublished limit (Claude's `tokens` source), or the row is `no_data` and the `0%` is a placeholder standing in for a reading that doesn't exist. Once the percentage pins at 100, what you've spent *past* the cap follows in parentheses — `100% (+7.5M)` for 7.5M tokens beyond it, or `100% (+$12.34)` once your plan's overage rates are configured. A percentage can't climb past 100, so that figure is the only thing still moving at the moment spending stops being free. It's absent for a source with no per-request token history (Claude Desktop), since there'd be nothing to sum. |
| **Status** | `ok` — live, and not on pace to exhaust. `🔥 burning` — on pace to hit 100% before the reset. `idle` — the newest reading is older than this window's staleness threshold (10 min for 5-hour, 3 h for weekly, both configurable), so no *rate* can be measured; the **Used %** level still counts. `reset_pending` — the window just turned over and there's only one reading in the new cycle. `no_data` — the provider isn't reporting this window at all (today, Codex's 5-hour). |
| **Burn %/h** | Percent of *this window's* quota consumed per hour at the currently measured rate — not tokens per hour, and not a share of anything else, so it divides straight into the percent remaining. Normally measured from real token throughput and converted by the percent-per-token calibration; a trailing `~` means it fell back to the slope of the reported percentage itself, which is quantized to whole numbers and therefore coarse. `—` on any row that isn't live (`idle`, `no_data`, `reset_pending`), because a rate is the one thing a stale reading can't support. |
| **ETA on burn %/h** | When the window runs out if the displayed recent burn rate continues around the clock. It is the direct counterpart to **Burn %/h**: percent remaining divided by that rate, rendered as a weekday and time. |
| **Predicted ETA (P50 → P90)** | When it runs out *given how you actually use this across a week*. A value such as `Sat 08:53 → Mon 19:46` is a range: the left timestamp is the median simulated exhaustion time (P50), and the right is the later 90th-percentile time (P90). While the hour-of-week profile is still thin, a lone timestamp is the working-hours projection from your configured `working_hours`, not a percentile band. Unlike the burn-rate ETA, the learned range can remain while the row is `idle`, provided the last-used level can still be proven to belong to the current quota cycle. |
| **Resets** | When this window's quota refills. Reported directly by Codex; for Claude, computed from the 5-hour block anchor, or `—` for the weekly window until a reset is actually observed. |
| **Risk** | Probability of exhausting before that reset — the share of simulated futures that reach 100% in time. Worth seeing separately from the ETA: a 40% chance of running out matters even when the median future doesn't. Only present while a simulating model is running, and the column is hidden entirely when no model produces one. |
| **Conf.** | How much evidence is behind the estimate: `high` at 10+ observations, `medium` at 3+, else `low`. The hour-of-week models additionally downgrade it by how much of the period being projected actually has data (below 50% coverage is `low` regardless); coverage can only lower a rating, never raise it. Under the default `linear` there's no profile to cover, so what you see is the observation count alone. |

Both ETA columns render as weekday + time and are capped at the window's own
reset — or at the window's duration when no reset is known yet — so an ETA is
strictly less than a week out and the weekday is unambiguous (assuming a
provider-reported reset really is the next one; nothing re-clamps it).

A blank ETA is a real answer rather than a missing one. **ETA on burn %/h**
is blank on an `idle`, `no_data`, or `reset_pending` row because there is no
live rate to extend. **Predicted ETA** may remain on an `idle` row; it is blank
when the last-used level cannot be tied to the current cycle, the history is
too thin, or most simulated futures survive until the reset.

Row colors track the alert conditions closely: **red** for something that
fired or would fire (over the warn threshold, or burning with an imminent
ETA), **yellow** for the weaker "on pace to exhaust before reset." One
deliberate divergence: a 100% reading always paints red, even when it's stale
enough that alerting has stopped vouching for the cycle — being at your cap is
worth seeing regardless. The dog in the title agrees — 🐶 calm, 🐕 at the warn
threshold, 🔥 burning.

### Backtest scores

`scripts/backtest.py` prints one block per window (with how many samples,
hours, cycles, and token events it had to work with) and one row per model:

| Score | What it means |
|---|---|
| **coverage** | Share of replayed moments where the model produced an ETA at all. Low isn't automatically bad — declining to guess from a signal too coarse to forecast is correct — but a model that's usually silent can't be graded on much. |
| **scored** | Of those, how many belong to a cycle that reached 100% later, so an error is computable. Real exhaustions are the scarce resource — a weekly window supplies at most a couple a week — but be careful: this counts forecast *moments*, and many moments can belong to one exhaustion. See the caveat below. |
| **MAE h** | Mean absolute error of the ETA, in hours. |
| **bias h** | Mean *signed* error. Positive means the ETAs ran late (predicting exhaustion after it happened); negative means early and alarmist. Tracked separately because the two directions aren't equally costly, and because a change can meaningfully cut bias while leaving MAE inside the noise. |
| **P90 hit** | Calibration of the band: how often the truth actually landed at or before the model's P90. A well-calibrated P90 sits near 90% — much higher means the band is wider than it needs to be, much lower means it's overconfident. Blank for models that don't produce a band. |

An episode has to *start* below the cap and cross it. A moment already at
100% is not one: every model trivially answers "exhausted now," so scoring it
measured the gap to the next stored sample — the ~5-minute cadence of the
underlying data — and called that forecast error. That inflated MAE into
something flattering (0.1h without forecasting anything), made `P90 hit`
read 0% for arithmetic rather than calibration reasons (a P90 of 0.0h cannot
contain a positive truth), and, worst, let a saturated stretch mint one
scorable episode per tick for whichever model still answers at the cap —
about 120 in two hours for the simulating model against 9 for the default,
enough to clear the graduation bar and promote the *worse* model. Excluding
at-cap origins fixes all three.

Two things to still keep in mind when reading the table:

- **`scored` counts forecast moments, not distinct events.** Many correlated
  forecasts made while climbing toward one exhaustion each count separately —
  on this history, 14 episodes point at just 2 real crossings. So 30 is a
  floor on evidence, not on independent events.
- **`P90 hit` is conditional**, on an episode both producing a band and then
  exhausting, so it isn't a full unconditional calibration check. It prints
  `—` while no genuine episode has carried an uncensored P90.

Expect the honest numbers to be sparse and to stay that way: on ~6.5 days of
real history the count is 14 for the default model and 3 for the simulating
one, so `auto` reports "not enough" and stays put. That is the design working.
The footer says so explicitly, but treat its verdict as describing the replay
only — it and the runtime `auto` gate share a definition but not their inputs
(replayed forecasts versus stored ones).

## How it works

TokenWatchDog reads quota data **only from local files already on your
machine** — Codex's session logs and Claude Code's/Claude Desktop's local
usage data. It never calls any API and never transmits anything. Those
session logs do contain your conversation history, but TokenWatchDog only
ever extracts numeric usage fields (token counts, timestamps, model names)
from each line — the message content itself is never logged, stored, or
transmitted anywhere.

Burn rate is measured from token throughput wherever possible. Neither
provider publishes its real token cap, so instead of guessing one, the tool
measures how much of the window one token actually consumed — the reported
percentage's movement across the current cycle, divided by the tokens spent
over that same span — and uses that ratio to turn recent tokens/hour into
%/h. It refuses to calibrate from a cycle that hasn't moved at least a few
points, or one that ends pinned at 100%, and falls back to the percentage's
own slope in those cases.

Two models run each tick. **ETA on burn %/h** comes from a robust
(outlier-resistant) fit of the recent rate, projected forward to when usage
would hit 100% — capped at the window's own reset, or at its duration when no
reset time has been derived yet. **Predicted ETA** buckets historical burn by
hour-of-week and simulates thousands of random futures to produce a P50/P90
band and a probability of exhausting before the reset. Because that bucketing
is built from token history, an hour with no requests contributes a real zero
rather than no data at all — which is what lets it learn that you don't burn
quota overnight, instead of filling those hours in from your daytime average.
While the profile is still thin, **Predicted ETA** falls back to the working
hours you declared in config: the burn budget run through only those intervals
instead of treating every hour as billable. Once learned, the prediction keeps
advancing through an idle period while the last-used percentage is known to
remain in the current cycle. It does not revive an expired or unknown-cycle
reading, and its `idle` status continues to suppress burn-rate alerts.

`predictor.model` picks which of the two is *authoritative* — the one alerts
fire from. `"auto"` decides that by grading both against your own stored
forecasts, and stays on the burn-rate model unless the other wins by a real
margin over enough scored episodes. It says which and why rather than
switching silently. That choice does not swap the dashboard columns: the
burn-rate ETA remains beside **Burn %/h**, and the learned prediction remains
under **Predicted ETA**.

The "burning too fast" alert uses the authoritative model's ETA. With the
default burn-rate model that is **ETA on burn %/h**; if the learned model
eventually wins enough scored episodes, it becomes **Predicted ETA**.

## Status

The core (both providers, both predictors, alerting, terminal dashboard,
measured model selection) is implemented and tested. A packaged standalone
binary and a menu-bar front-end are natural next steps but aren't built yet.

Model selection is wired up but has not yet fired on real data: grading needs
windows that actually reached 100%, and a weekly window supplies at most a
couple of those a week, so `auto` currently reports "not enough scored
history" and stays on the burn-rate model. That's the intended behavior, not
a gap — run `scripts/backtest.py` to see where your own history stands.

## Limitations

- **Codex's usage *percentage* can't be backfilled.** Its session logs carry
  only the current rate-limit snapshot, not a time series, so a stretch with
  the tool off leaves a real gap in the percentage history — a property of
  what Codex exposes, not a bug. Its per-request *token* counts are a time
  series and are backfilled, which is why burn rate survives downtime better
  than the percentage does.
- **Claude history can be backfilled.** Both Claude sources keep their own
  history independent of whether TokenWatchDog is running — token-compute
  re-scans Claude Code's own transcripts (bounded by
  `predictor.history_retention_weeks`), and Desktop mode reads everything
  `plan-usage-history.json` still retains (~5 days). Either way, a period
  with the tool off isn't a gap once it's running again.
- **Only tested on macOS so far.** The core polling/prediction logic is plain
  Python, Windows installs receive the IANA timezone data needed for
  DST-aware projections, and the spacebar refresh has a Windows `msvcrt`
  implementation. Set `timezone` to an explicit IANA name such as
  `"America/New_York"` there: the stdlib can apply that zone once installed,
  but it cannot reliably translate the Windows system-zone name to IANA on
  its own, so the empty/system-local fallback is only the current fixed
  offset. The audible "woof" cue falls back to a plain system beep. However,
  no real Windows runner has verified those paths yet, and Windows still has
  no native visual notification banner: the current banner backends
  (`terminal-notifier` / `osascript`) are macOS-only. Claude Desktop discovery
  is also hard-coded to its macOS `~/Library/Application Support/...` path;
  Windows therefore falls back to Claude Code transcripts rather than reading
  the Desktop usage file. A Windows CI job plus a Task Scheduler/service smoke
  test should be required before calling the platform supported.
- **Claude's 5-hour reset is computed; its weekly reset still has to be
  observed.** Claude reports no reset time at all. The 5-hour window is a
  fixed block anchored at your first request after an idle gap, so its reset
  follows from the token log immediately. The weekly window has no equivalent
  rule — a 7-day gap in activity isn't what starts a new weekly cycle — so it
  waits for an actual reset to appear in your history, and until one does the
  countdown is honestly unknown rather than guessed.
- **The weekly token-based percentage is a trailing 7-day sum**, not a true
  fixed cycle, for the same reason: there's no anchor to align it to. It's an
  estimate (flagged `*` in the dashboard), and the tail of it drifts as old
  usage ages out.
- **Only the CLI's own token log is visible.** Claude Desktop and agent mode
  burn the same account quota without writing a transcript here. The
  percent-per-token calibration absorbs a steady share of that, and the tool
  falls back to the reported percentage's own slope whenever the percentage
  is moving while the local log is silent. An abrupt shift in that mix skews
  it either way, though, and over-reading is the more consequential
  direction: if the calibration cycle's usage was mostly invisible, the ratio
  is inflated by that share, and a later CLI-only burst gets multiplied by it
  — an overstated rate, which is the direction that fires the burn alert.
  The learned prediction is skewed the other way, recording an hour burned
  only via Desktop as a genuine zero.

## Development

```bash
uv sync --all-extras --all-groups
uv run pytest          # test suite
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy tokenwatchdog
```

To check whether a prediction change actually helped, score the models
against your own stored history rather than eyeballing the dashboard:

```bash
uv run python scripts/backtest.py --stride 4
```

It replays every stored sample, hides everything after it, runs each
predictor as it would have run at that moment, and compares the answer to
what the store already knows happened next. Monte Carlo replay uses a fixed
seed by default so the same code and history produce the same comparison
(`--seed` overrides it). The scores it prints are
described under [The metrics](#backtest-scores).

## License

[MIT](LICENSE)
