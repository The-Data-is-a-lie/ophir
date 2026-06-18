---
name: gather-insider
description: >-
  Use when you need a FREE SEC EDGAR corporate-insider + event signal for a ticker (e.g. AAPL) —
  Form 4 open-market insider buy/sell (cluster-buy, signed dollar flow with the sell leg), plus 8-K
  material-event recency and a 13D/13G activist-stake flag. Reuse insider_signal in
  src/ophir/agent/edgar.py, which reads SEC EDGAR's free, no-auth submissions JSON + raw Form 4 XML
  and gates strictly on the filingDate. The legal-insider analog to congress, with stronger evidence.
  Reach for it to fetch / refresh / debug / extend the insider dimension.
---

# Gather SEC EDGAR insider + events (Form 4 / 8-K / 13D-G)

Issuer insiders (CEO/CFO/directors) must file Form 4 within ~2 days of trading their own stock. SEC
EDGAR is free and no-auth (a UA with a contact email is required). One per-CIK submissions JSON
indexes every recent filing, so Form 4, 8-K, and 13D/13G all come from a SINGLE fetch; the Form 4
trade detail is clean per-filing XML. This is the legal-insider analog to `gather-congress`, with
stronger academic support and actual disclosed prices.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** SEC EDGAR, free + no auth (UA-with-email required, ≤10 req/s).
  - ticker→CIK: `https://www.sec.gov/files/company_tickers.json`.
  - per-CIK filings index: `https://data.sec.gov/submissions/CIK##########.json` (form, filingDate,
    accessionNumber, primaryDocument, 8-K items).
  - raw Form 4 XML: the archive dir with the XSL render-prefix stripped from `primaryDocument`.
- Reuse (do not re-fetch by hand):
  - [agent/edgar.py:196](src/ophir/agent/edgar.py:196) — `insider_signal(symbol, *, as_of=None, lookback_days=90)`.
  - [agent/edgar.py](src/ophir/agent/edgar.py) — `_cik_map`, `_submissions`, `_parse_form4`.
  - Reuses [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) `last_closed_session`.

## Fields
- `insider_cluster_buyers` / `insider_cluster_flag` — distinct insiders disclosing an open-market BUY
  (code `P`) in the trailing 30 days (flag at ≥2).
- `insider_net_dollar_flow_90d` — buy minus sell **actual** dollars (shares × price) over 90d;
  `insider_sell_dollar_90d` surfaced separately; `insider_net_bias` (+1 buying .. −1 selling).
- `days_since_last_insider_buy` / `last_insider_type` — recency of insider activity.
- `days_since_last_8k` / `last_8k_items` (8-K item codes) and `activist_stake_flag` (13D/13G in window).
- plus `symbol`, `asof`, `lookback`, `n_form4`, `source`.

## Point-in-time, fail-safe & caching
- **Gates strictly on `filingDate <= as_of`** (the public disclosure date), never the transaction date.
- Only open-market `P` (purchase) and `S` (sale) count; grants / option exercises / tax-withholding /
  gifts are ignored (they are not informative).
- Unknown symbol, missing/odd filing, or any network/parse error → a **neutral dict** with a `note`,
  never an exception.
- The submissions JSON is memoized per process; each Form 4's parsed trades are cached per accession
  under `<DATA_DIR>/edgar/form4/`.

## How to use / extend
- `from ophir.agent.edgar import insider_signal; insider_signal("AAPL")`.
- Feeds `ResearchBrief.insider` via `gather_insider` ([agent/research.py](src/ophir/agent/research.py));
  surfaced (low weight) in the debate/manager prompts.
- **Extend:** parse 8-K item *types* (4.02 restatement, 5.02 exec exit) for a richer event signal, or
  add 13F institutional holdings (free EDGAR, but a 45-day lag + reverse-index-by-issuer cost).
