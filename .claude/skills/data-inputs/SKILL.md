---
name: data-inputs
description: >-
  Use when you need to know what data the ophir trading agent consumes, where a given
  input comes from, or how to add / refresh / debug a data source. Indexes every input
  (OHLC bars, fundamentals, news, technicals, stock splits, dark pool, short interest,
  congressional trades, SEC insider/events, options/IV surface, news sentiment, public
  attention, macro/VIX/credit regime, earnings events, S&P 500 universe, NYSE calendar,
  Alpaca account) to its dedicated gather-* skill and to the full catalog in
  docs/data-inputs.md. Reach for it first when working with ophir's market data, then
  jump to the specific gather skill.
---

# Ophir data inputs (index)

The ophir agent's daily run is assembled from a handful of data sources. This skill is the
map: each input below links to a dedicated `gather-*` skill with the exact existing function
to reuse, its fields, fail-safe behavior, and how to extend it. The full prose catalog —
with every field, file:line, and fail-safe — is [docs/data-inputs.md](docs/data-inputs.md).

## Pipeline at a glance

ingest **OHLC** → engineer the **13 features** → model **forecast** → **research**
(fundamentals + news + technicals) → bull/bear **debate** → **manager** pick → **risk gate**
→ reconcile vs the **Alpaca** account → orders. The **NYSE calendar** gates *when* it runs and
*which* close it conditions on.

## Inputs → gather skills

| Input | Source | Skill |
| --- | --- | --- |
| Daily OHLC bars (+ SPY benchmark) | Yahoo Finance — yfinance `.history` | `gather-ohlc` |
| Fundamentals | Yahoo Finance — yfinance `.info` | `gather-fundamentals` |
| News headlines | Yahoo Finance — yfinance `.news` | `gather-news` |
| Technicals (derived; incl. 12-1 momentum, RS-vs-SPY, ATR%) | ingested OHLC + model forecast | `gather-technicals` |
| Stock-split history | Yahoo Finance — yfinance `.splits` | `gather-splits` |
| Dark-pool / off-exchange activity | FINRA daily short-sale file (free, no-auth) | `gather-dark-pool` |
| Consolidated short interest (biweekly) | FINRA consolidated short-interest API (free, no-auth) | `gather-short-interest` |
| Congressional trades (House PTRs) | US House Clerk yearly ZIP + PDFs (free, no-auth) | `gather-congress` |
| Corporate insider + events (Form 4 / 8-K / 13D-G) | SEC EDGAR submissions JSON + Form 4 XML (free, no-auth) | `gather-insider` |
| Options / IV surface (per-name) | CBOE delayed-quotes JSON (free, no-auth; live-only) | `gather-options` |
| News sentiment (FinBERT, derived) | local FinBERT over existing headlines (no new fetch) | `gather-sentiment` |
| Public attention (pageviews + Reddit) | Wikimedia pageviews + ApeWisdom (free, no-auth) | `gather-attention` |
| Macro / regime (market-wide) | `^VIX`/`^VIX3M` + FRED credit/conditions/curve + OPEX gate | `gather-macro` |
| Earnings-calendar event gate | Yahoo Finance — yfinance `.calendar` | `gather-events` |
| S&P 500 constituents | Wikipedia — `pd.read_html` | `gather-sp500-universe` |
| NYSE trading calendar | pandas-market-calendars (`NYSE`) | `gather-market-calendar` |
| Alpaca paper account + positions | alpaca-py (paper, hardcoded) | `gather-account` |

## Inputs without a gather skill

- **13 engineered features** (`ticker.extract_features`), the **per-ticker forecast**
  (`predict.predict_ticker`), and the **trained checkpoint** (`predict.load_predictor`) are
  computed or loaded, not externally fetched — documented in [docs/data-inputs.md](docs/data-inputs.md).
- The **watchlist** (`ophir-bot/watchlist.txt`) is an operational file; the local **Ollama**
  `gpt-oss:20b` is a service dependency, not data.
- **MASSIVE** (`register.get_massive_client`) is dead code — defined, never called; safe to remove.

## Adding a new data source

1. Write the fetcher in the existing fail-safe style — catch errors, return an empty/neutral
   value, never invent figures.
2. Thread it into the right layer — typically a new `ResearchBrief` advisory field collected by
   `advisory_signals()` and surfaced in the debate / manager prompts (see `flow`, `short_interest`,
   `macro`, `events`, `congress` for the established pattern). Note new model *features* need a
   retrain (the 13-feature input width is hardcoded), so default to the agent/LLM layer.
3. Add a `gather-<source>` skill mirroring the existing ones, then add a row to the table above
   and an entry in [docs/data-inputs.md](docs/data-inputs.md).
