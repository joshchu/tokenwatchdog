<p align="center">
  <img src="tokenwatchdog/assets/watchdog.svg" width="160" height="160" alt="TokenWatchDog mascot">
</p>

<h1 align="center">TokenWatchDog</h1>

<p align="center">
  A local watchdog for <b>Codex CLI</b> and <b>Claude Code</b> usage quota —
  polls your 5-hour and weekly limits, alerts before you hit them, and
  predicts when each window will actually run out.
</p>

---

## Why

Codex and Claude Code both cap usage on a 5-hour and a weekly rolling window,
and it's easy to blow through either one mid-task. TokenWatchDog polls your
local usage data every minute and answers two questions existing usage
trackers don't:

- **When will I actually run out?** Not "you're at 62%" — an ETA, computed
  from your own recent burn rate, capped sensibly at the window's reset.
- **When will I run out, given I only work certain hours?** The weekly
  projection can count only your configured working hours instead of
  assuming you burn quota 24/7 — so "exhausts Thursday" means Thursday
  *during work*, not Thursday at 3am.

It also alerts at 90% usage and separately when quota is being burned fast
enough that it will run out before the window naturally resets.

## Features

- Polls Codex CLI and Claude Code quota (5-hour + weekly) on a configurable
  interval (default ~1 minute)
- Two exhaustion predictors: a robust (outlier-resistant) burn-rate fit by
  default, or an opt-in Monte Carlo model that learns your hour-of-week usage
  rhythm and simulates a P50/P90 exhaustion band instead of one point guess
- Working-hours-aware calendar projection alongside the plain 24/7 ETA
- 90%-usage alert and a separate "burning too fast" alert, both native macOS
  notifications (default sound: a spoken "Woof! Woof!" — any system sound
  name works too), each de-duplicated so you get at most one ping per window
- Persists history locally (SQLite) so the Monte Carlo model has real data
  to learn from, from the very first run
- Terminal dashboard UI, with a `--headless` mode for running as a background
  agent
- Works with either provider alone or both at once; gracefully shows "no
  data" instead of guessing when a provider doesn't expose a window

## Quickstart

Requires Python 3.13+.

```bash
git clone <this-repo-url>
cd tokenwatchdog
uv sync
uv run python -m tokenwatchdog
```

That opens a live terminal dashboard. Useful variants:

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

By default, exhaustion is predicted from a robust (outlier-resistant) fit of
your own recent burn rate, projected forward to when usage would hit 100% —
capped sensibly at the window's own reset. Set `predictor.model = "montecarlo"`
in config to switch to the learned model instead: it buckets historical burn
rate by hour-of-week and simulates thousands of random futures to produce a
P50/P90 exhaustion band and a probability of exhausting before the window
resets, rather than one linear guess. It needs real history to say anything
sharper than the default, which is exactly why history is persisted from the
first run. The working-hours projection (either model) runs the burn budget
through only your configured working intervals instead of treating every
hour as billable.

## Status

The core (both providers, both predictors, alerting, terminal dashboard) is
implemented and tested. A packaged standalone binary, a menu-bar front-end,
and automatic graduation from the default model to Monte Carlo once enough
history accumulates (today it's an explicit config choice) are natural next
steps but aren't built yet.

## Development

```bash
uv sync --all-extras --all-groups
uv run pytest          # test suite
uv run ruff check .    # lint
uv run ruff format .   # format
uv run mypy tokenwatchdog
```

## License

[MIT](LICENSE)
