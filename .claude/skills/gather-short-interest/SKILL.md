---
name: gather-short-interest
description: >-
  Use when you need a ticker's FINRA biweekly consolidated short interest (e.g. AAPL) — settled
  short shares, days-to-cover, short-interest change %, or average daily volume — as a free
  positioning signal. Reuse short_interest_signal in src/ophir/agent/short_interest.py, which queries
  FINRA's free, no-auth consolidated short-interest dataset and gates to the latest settlement on or
  before as_of. This is the standing short POSITION (twice a month), additive to the dark-pool
  intraday short VOLUME — reach for it to fetch / refresh / debug / extend the short-interest dimension.
---

# Gather consolidated short interest (FINRA, biweekly)

The standing short *position*, settled twice a month, complements the intraday off-exchange short
*volume* the `gather-dark-pool` signal reads — they measure different things, so this is additive,
not duplicative. Source is FINRA's free, no-auth consolidated short-interest dataset.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** FINRA, free + no auth — `POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest`
  with a JSON body filtering `symbolCode` and a `settlementDate` range. Returns JSON rows
  (`currentShortPositionQuantity`, `previousShortPositionQuantity`, `daysToCoverQuantity`,
  `changePercent`, `averageDailyVolumeQuantity`, `settlementDate`). Send a browser `User-Agent`.
- Reuse (do not re-fetch by hand):
  - [agent/short_interest.py:112](src/ophir/agent/short_interest.py:112) — `short_interest_signal(symbol, *, as_of=None)`.
  - [agent/short_interest.py](src/ophir/agent/short_interest.py) — `_query(symbol, start, end)` (POST + per-symbol cache).
  - Reuses [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) `last_closed_session` (as-of, no look-ahead).

## Fields
- `short_interest` — settled short shares (the standing position).
- `prev_short_interest` — the prior settlement's short shares.
- `days_to_cover` — short interest / average daily volume.
- `si_change_pct` — % change vs the prior settlement.
- `avg_daily_volume`, `settlement_date`, plus `symbol`, `asof`, `n_settlements`, `source`.

## Fail-safe, point-in-time & caching
- Gates on the most recent `settlementDate <= as_of`, so it never reveals a settlement that was not
  yet public (no look-ahead).
- HTTP error, empty result, or unknown symbol → a **neutral dict** (`n_settlements=0`, values `None`,
  a `note`), never an exception.
- Each `(symbol, end-date)` query is cached under `<DATA_DIR>/finra/shortinterest/`.

## How to use / extend
- `from ophir.agent.short_interest import short_interest_signal; short_interest_signal("AAPL")`.
- It feeds `ResearchBrief.short_interest` via `gather_short_interest`
  ([agent/research.py:203](src/ophir/agent/research.py:203)) and is surfaced in the debate/manager prompts.
- **Extend:** add short % of float (divide by shares outstanding) or the SEC fails-to-deliver file as
  a sibling squeeze-risk flag.
