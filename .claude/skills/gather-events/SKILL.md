---
name: gather-events
description: >-
  Use when you need a FREE earnings-calendar event gate for a ticker (e.g. AAPL) — days to the next
  earnings print and days since the last — so the manager/risk gate can avoid holding into a binary
  earnings event or lean into the post-earnings drift window. Reuse earnings_signal in
  src/ophir/agent/events.py, which reads yfinance's calendar (free, no auth). NOTE: the earnings
  schedule is forward-looking and revised, so this is a LIVE gate, not a look-ahead-free backtest
  feature. Reach for it to fetch / refresh / debug / extend the events dimension.
---

# Gather earnings-calendar event gate

The single highest-leverage event read for a daily trader: how many sessions until the next earnings
report. Holding into a binary earnings print is a distinct risk the price/vol features do not capture,
and the post-earnings window is where the documented drift (PEAD) lives.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** Yahoo Finance, free + no auth — `yf.Ticker(symbol).calendar`.
- Reuse (do not re-fetch by hand):
  - [agent/events.py:71](src/ophir/agent/events.py:71) — `earnings_signal(symbol, *, as_of=None)`.
  - [agent/events.py](src/ophir/agent/events.py) — `_fetch_calendar(symbol)` (the network call; monkeypatch in tests),
    `_extract_dates` (handles yfinance's dict- and legacy-frame shapes), `_to_date`.
  - Reuses [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) `last_closed_session`.

## Fields
- `next_earnings_date` / `days_to_next_earnings` — the upcoming print (the event-risk gate).
- `last_earnings_date` / `days_since_last_earnings` — the recent print (the PEAD window).
- plus `symbol`, `asof`, `source`.

## Fail-safe & point-in-time
- Conditioned on the last completed session ≤ `as_of`.
- **Caveat:** the earnings *schedule* is forward-looking and gets revised, so this is a **live trading
  gate, not a look-ahead-free backtest feature** — for backtests, source the historically-known
  schedule instead.
- Any network/parse error or a ticker with no scheduled date → a **neutral dict** with a `note`, never raises.

## How to use / extend
- `from ophir.agent.events import earnings_signal; earnings_signal("AAPL")`.
- It feeds `ResearchBrief.events` via `gather_events` and surfaces `days_to_next_earnings` in the manager
  dossier so the risk gate can avoid (or down-size) a name reporting imminently.
- **Extend:** add the earnings surprise / SUE (yfinance `get_earnings_dates`) for a point-in-time PEAD
  feature gated on the report date, or analyst estimate revisions.
