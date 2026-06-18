---
name: gather-options
description: >-
  Use when you need a FREE per-name options / implied-volatility-surface signal for a ticker (e.g.
  NVDA) — put/call ratio (volume + OI), 25-delta IV skew, IV level (iv30), unusual-options activity.
  Reuse options_flow_signal in src/ophir/agent/options_flow.py, which reads CBOE's free, no-auth
  delayed-quotes JSON (full chain with greeks). NOTE: there is NO free option history, so this is a
  LIVE-loop signal, not backtestable. Reach for it to fetch / refresh / debug / extend the options dim.
---

# Gather options / IV surface (CBOE, per-name)

ophir's only volatility input is the market-wide VIX term structure; this adds a **per-name** read
from CBOE's free, no-auth delayed-quotes JSON — one file per ticker with the spot, a ready 30-day IV
(`iv30`), and every contract's `iv` / `open_interest` / `volume` / greeks.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** CBOE, free + no auth (browser UA) — `https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json`
  (~2.5–6k contracts; `data.iv30` plus per-contract `iv`, `delta`, `open_interest`, `volume`).
- Reuse (do not re-fetch by hand):
  - [agent/options_flow.py:133](src/ophir/agent/options_flow.py:133) — `options_flow_signal(symbol, *, as_of=None)`.
  - [agent/options_flow.py](src/ophir/agent/options_flow.py) — `_load_chain`, `_skew_25d`, `_OPT_RE`.
  - Reuses [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) `last_closed_session`.

## Fields
- `put_call_ratio_volume` / `put_call_ratio_oi` — positioning (Pan-Poteshman: low P/C → outperformance).
- `iv_skew_25d` — front-expiry 25-delta put IV minus call IV (positive = downside fear;
  Xing-Zhang-Zhao: steep skew underperforms).
- `iv30` — CBOE 30-day implied vol level; `spot`; `volume_oi_ratio` — an activity read.
- plus `symbol`, `asof`, `n_contracts`, `source`.

## Point-in-time, fail-safe & caching
- **No free option history exists**, so this is a **LIVE-loop signal, not backtestable.** Each session's
  snapshot is cached under `<DATA_DIR>/cboe/<date>/`; a *past* `as_of` with no cached snapshot returns a
  neutral dict (it never live-fetches a chain and dates it to the past). `iv_rank`/`iv_percentile` are
  deliberately omitted until self-collected history accrues.
- A missing snapshot, an unknown symbol, or any error → a **neutral dict** with a `note`, never raises.

## How to use / extend
- `from ophir.agent.options_flow import options_flow_signal; options_flow_signal("NVDA")`.
- Feeds `ResearchBrief.options` via `gather_options`; surfaced in the debate/manager prompts.
- **Extend:** add per-name IV term-structure slope, an options-implied earnings move (pair with the
  `gather-events` date), or a volume-vs-OI z-score once enough daily snapshots are collected.
