---
name: gather-dark-pool
description: >-
  Use when you need a FREE dark-pool / off-exchange activity signal for a ticker (e.g. AAPL) —
  off-exchange volume, dark participation %, off-exchange short ratio, or an anomaly z-score — to
  approximate Unusual Whales without paying. Reuse dark_pool_signal in src/ophir/agent/darkpool.py,
  which reads FINRA's free, no-auth daily short-sale file (off-exchange TRF/ADF volume). Reach for
  it to fetch / refresh / debug / extend the dark-pool dimension. It is DAILY AGGREGATE volume, not
  individual real-time prints.
---

# Gather dark-pool (off-exchange) activity

A free, official proxy for Unusual Whales' dark-pool feed. FINRA's daily short-sale-volume file
reports the **off-exchange** (dark pool + internalizer) volume per symbol, which yields a daily
dark-pool participation and short-pressure signal. It is daily *aggregate* volume, **not** the
individual real-time prints UW shows — no free source has the per-print firehose.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** FINRA, free + no auth — `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt`.
  Pipe-delimited header `Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market`; `TotalVolume`
  is the off-exchange (TRF/ADF/ORF-reported) volume. Posted ~6pm ET the same trade day; one ~few-MB
  file covers the whole market. **Send a browser `User-Agent`** — the default Python-urllib agent gets
  a `403`.
- Reuse (do not re-fetch by hand):
  - [agent/darkpool.py:128](src/ophir/agent/darkpool.py:128) — `dark_pool_signal(symbol, *, as_of=None, lookback=20)`: the public signal.
  - [agent/darkpool.py:86](src/ophir/agent/darkpool.py:86) — `_fetch_daily(date)` (download + parse, cached);
    [agent/darkpool.py:92](src/ophir/agent/darkpool.py:92) — `_consolidated_volume` (OHLC volume for the %).
  - Reuses [agent/market_calendar.py](src/ophir/agent/market_calendar.py) `recent_sessions` / `last_closed_session`
    (as-of, no look-ahead) and [agent/feed.py:49](src/ophir/agent/feed.py:49) `load_daily_ohlcv` (consolidated volume).

## Fields
- `off_exchange_volume` — latest-session off-exchange shares (the dark-pool proxy).
- `off_exchange_pct` — `off_exchange_volume / consolidated volume` (dark participation rate); `None`
  if the OHLC parquet lacks that exact session.
- `off_exchange_short_ratio` — off-exchange short / total volume.
- `off_exchange_short_exempt_ratio` — short-exempt / total volume (free-lunch field: the
  `ShortExemptVolume` column is in the same file, parsed but previously unemitted).
- `off_exchange_short_ratio_zscore` — the per-day off-exchange short ratio's latest-vs-baseline
  anomaly; `None` with fewer than 3 days.
- `off_exchange_vol_zscore` — latest vs the trailing `lookback` baseline (an anomaly flag); `None`
  with fewer than 3 days.
- plus `symbol`, `asof`, `lookback`, `n_days`, `source`.

## Fail-safe & caching
- A missing/late file, weekend, unknown symbol, or a `403` → a **neutral dict** (`n_days=0`, values
  `None`, a `note`), never an exception.
- Each day's file is cached under `<DATA_DIR>/finra/regsho/CNMSshvol{date}.txt`, so a session
  downloads once and serves every ticker; non-data responses (404 pages) are not cached.

## How to use / extend
- `from ophir.agent.darkpool import dark_pool_signal; dark_pool_signal("AAPL")`.
- For a participation `%`, ingest the ticker's OHLC first (skill `gather-ohlc`) so the same session's
  consolidated volume is present; the FINRA file can lead yfinance by a day, in which case `pct` is
  omitted (not wrong).
- **Wired in:** `dark_pool_signal` feeds `ResearchBrief.flow` via `gather_flow`
  ([agent/research.py](src/ophir/agent/research.py)) and is surfaced (with the other advisory
  signals) in the bull/bear debate and manager prompts. The sibling positioning signal is
  `gather-short-interest` (FINRA biweekly consolidated short interest — standing position vs this
  intraday volume).
- **Extend:** add FINRA ATS *weekly* per-ATS detail (which dark pool; 2–4-week lag; free FINRA API)
  as a sibling fetcher.
