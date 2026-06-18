---
name: gather-congress
description: >-
  Use when you need a FREE US House congressional-trade disclosure signal for a ticker (e.g. NVDA) —
  cluster-buy count, signed net dollar flow (with the sell leg surfaced), or a fresh-disclosure event
  flag — under the STOCK Act. Reuse congress_signal in src/ophir/agent/congress.py, which reads the
  US House Clerk's free, no-auth yearly PTR ZIP + per-DocID PDFs and gates strictly on the DISCLOSURE
  date (never the transaction date). House-only; a slow, contested smart-money prior, not fast alpha.
  Reach for it to fetch / refresh / debug / extend the congress dimension.
---

# Gather congressional trades (US House Clerk PTRs)

Lawmakers file Periodic Transaction Reports (PTRs) under the STOCK Act. The US House Clerk publishes
them free and no-auth as one yearly ZIP (an XML filing index) plus per-DocID PDFs (the trade detail).
Predictive value is weak/contested in aggregate — the edge is concentrated in cluster buys and the
asymmetric *sell* signal — so this is a **low-weight advisory** dimension, House-only.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** US House Clerk, free + no auth — yearly
  `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/<YEAR>FD.zip` (contains `<YEAR>FD.xml`,
  an index of `{First, Last, StateDst, FilingType, FilingDate, DocID}`); the trade rows live in
  per-filing PDFs at `/public_disc/ptr-pdfs/<YEAR>/<DocID>.pdf`, parsed with `pypdf`. **Send a browser
  `User-Agent`** (some clients get a `403`, not a paywall).
- Reuse (do not re-scrape by hand):
  - [agent/congress.py:244](src/ophir/agent/congress.py:244) — `congress_signal(symbol, *, as_of=None, lookback_days=90)`.
  - [agent/congress.py](src/ophir/agent/congress.py) — `_window_index` (builds `{ticker: [tx]}` once per session),
    `_index_filings`, `_parse_pdf_transactions`, `_bucket_dollars`.
  - Reuses [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) `last_closed_session`.

## Fields
- `cluster_buy_members` / `cluster_buy_flag` — distinct House members disclosing a BUY in the same
  ticker within the trailing 30 days (the best-supported construct; flag set at ≥2).
- `net_dollar_flow_60d` — sum(buy bucket-midpoints) − sum(sell midpoints) over 60d; `sell_dollar_60d`
  surfaced separately because disclosed *sales* are the informative leg.
- `congress_net_bias` — +1 net buying .. −1 net selling.
- `days_since_last_disclosure` / `last_disclosure_type` / `new_disclosure_flag` — the recency event.
- plus `symbol`, `asof`, `lookback`, `n_disclosures`, `source`.

## Point-in-time, fail-safe & caching
- **Gates strictly on the FilingDate (public disclosure date) ≤ as_of, NEVER the transaction date** —
  the 30–45 day STOCK Act lag means using the transaction date would inject catastrophic look-ahead.
- A missing ZIP/PDF, a scanned/ticker-less filing, or an unknown symbol → a **neutral dict** with a
  `note`, never an exception.
- The yearly ZIP and each per-DocID parsed transaction list are cached under `<DATA_DIR>/congress/`.
  The first session for a window is slow (many PDF fetches); subsequent sessions hit the cache.

## Legal
- `5 U.S.C. app. 105(c)` bars using these disclosures for any *commercial* purpose (news-media excepted),
  civil penalty up to $10,000. ophir is paper-only / non-commercial — keep it that way; do not redistribute.

## How to use / extend
- `from ophir.agent.congress import congress_signal; congress_signal("NVDA")`.
- It feeds `ResearchBrief.congress` via `gather_congress` and is surfaced (low weight) in the
  debate/manager prompts.
- **Extend:** add the Senate eFD (no bulk file; Akamai + CSRF "prohibition agreement" + OCR — much
  harder, best-effort, disabled by default), or weight leadership/committee members more heavily.
