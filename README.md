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

Codex and Claude Code both cap usage on a 5-hour and a weekly rolling window,
and it's easy to blow through either one mid-task. TokenWatchDog polls your
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
- Working-hours-aware calendar projection alongside the plain 24/7 ETA, each
  capped independently against the next reset — an ETA past the reset
  describes an event that can't happen, and no window is ever projected to
  take longer to exhaust than its own duration
- Staleness applies to the burn *rate*, not the used-% *level*: quota doesn't
  un-consume while you're away, so a 100%-used reading still alerts once its
  cycle is confirmed live, while nothing extrapolates from a stale rate
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
  alone goes blind — the "Past cap" column keeps tracking real tokens spent
  past that point (from the same session/token logs), and shows a `$`
  estimate instead of a raw token count once you set your plan's real
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
```

Config lives at `~/.tokenwatchdog/config.toml` (created with sane defaults on
first run) — poll interval, working hours, alert thresholds, which
providers/windows to watch, and notification settings all live there.
History is stored in `~/.tokenwatchdog/history.db`; delete it to reset.

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

By default, exhaustion is predicted from a robust (outlier-resistant) fit of
that rate, projected forward to when usage would hit 100% — capped at the
window's own reset, or at its duration when no reset time has been derived
yet. Set `predictor.model = "montecarlo"` in config to switch to the learned
model instead: it buckets historical burn by hour-of-week and simulates
thousands of random futures to produce a P50/P90 band and a probability of
exhausting before the reset, rather than one linear guess. Because that
bucketing is built from token history, an hour with no requests contributes a
real zero rather than no data at all — which is what lets it learn that you
don't burn quota overnight, instead of filling those hours in from your
daytime average.

The working-hours projection (either model) runs the burn budget through only
your configured working intervals instead of treating every hour as billable.
The "burning too fast" alert deliberately uses the 24/7 projection, not that
one: usage happening right now doesn't pause because it's 9pm.

## Status

The core (both providers, both predictors, alerting, terminal dashboard) is
implemented and tested. A packaged standalone binary, a menu-bar front-end,
and automatic graduation from the default model to Monte Carlo once enough
history accumulates (today it's an explicit config choice) are natural next
steps but aren't built yet.

## Limitations

- **Codex history requires the tool to be running.** Codex's own session
  logs only ever contain the *current* snapshot, not a time series — there's
  nothing to backfill from once the tool starts back up. If TokenWatchDog
  isn't running, that stretch of Codex usage is genuinely unrecoverable; this
  isn't a bug, it's a property of what Codex exposes locally.
- **Claude history can be backfilled.** Both Claude sources keep their own
  history independent of whether TokenWatchDog is running — token-compute
  re-scans Claude Code's own transcripts (bounded by
  `predictor.history_retention_weeks`), and Desktop mode reads everything
  `plan-usage-history.json` still retains (~5 days). Either way, a period
  with the tool off isn't a gap once it's running again.
- **Only tested on macOS so far.** The visual notification banner
  (`terminal-notifier` / `osascript`) and the spacebar-refresh keybinding
  are POSIX/macOS-specific. The audible "woof" cue falls back to a plain
  system beep on Windows instead of doing nothing, but that fallback is
  untested on a real Windows machine. The core polling/prediction logic is
  plain Python and *should* run elsewhere, but it hasn't been run or
  verified on Linux or Windows.
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
  is moving while the local log is silent — but an abrupt shift in that mix
  will briefly under-read.

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
what the store already knows happened next.

## License

[MIT](LICENSE)
