---
name: gather-attention
description: >-
  Use when you need a FREE public-attention signal for a ticker (e.g. AAPL) — Wikipedia daily
  pageviews + a trailing z-score (an attention-spike flag), plus ApeWisdom Reddit mention counts.
  Reuse attention_signal in src/ophir/agent/attention.py, which reads the no-auth Wikimedia pageviews
  REST API (point-in-time, archived since 2015) and the no-auth ApeWisdom API. Reach for it to fetch /
  refresh / debug / extend the attention dimension. Attention predicts volatility/risk more than direction.
---

# Gather public attention (Wikipedia pageviews + Reddit)

A retail/public-attention dimension ophir lacks. The primary source is **Wikimedia pageviews** — actual
unrevised daily counts, uniquely **point-in-time / backtestable** among attention proxies. A secondary,
snapshot-only Reddit read comes from ApeWisdom.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider (primary):** Wikimedia, free + no auth — pageviews REST
  `https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}`.
  Ticker→article title is resolved via Wikipedia opensearch (using the company `shortName` from
  fundamentals) and cached under `<DATA_DIR>/wiki/titles.json`.
- **Provider (secondary):** ApeWisdom, free + no auth — `https://apewisdom.io/api/v1.0/filter/all-stocks/page/1`
  (whole list, memoized per session). Snapshot-only.
- Reuse (do not re-fetch by hand):
  - [agent/attention.py:148](src/ophir/agent/attention.py:148) — `attention_signal(symbol, *, company_name=None, as_of=None)`.
  - [agent/attention.py](src/ophir/agent/attention.py) — `_resolve_title`, `_pageviews`, `_apewisdom`.

## Fields
- `wiki_pageviews` — latest daily pageviews; `wiki_pageviews_zscore` — latest vs the trailing window
  (an attention-spike flag); `wiki_title` — the resolved article.
- `reddit_mentions` / `reddit_rank` / `reddit_mentions_24h_ago` — ApeWisdom crowding read.
- plus `symbol`, `asof`, `source`.

## Point-in-time, fail-safe & caching
- Wikipedia pageviews are **point-in-time** (only days `<= as_of` are used) and genuinely backtestable.
  The ApeWisdom read is a current **snapshot** (not historical) — self-archive it for history.
- The two components are independent: each degrades to `None` on error; an unresolved ticker with no
  data yields a neutral dict with a `note`, never an exception.

## How to use / extend
- `from ophir.agent.attention import attention_signal; attention_signal("AAPL", company_name="Apple Inc")`.
- Feeds `ResearchBrief.attention` via `gather_attention` (the company name comes from the fundamentals
  `shortName`); surfaced in the debate/manager prompts.
- **Extend:** add Google Trends (free path is 429-fragile and relative-normalized — handle carefully),
  or StockTwits message volume / bull-bear (per-symbol, access under review).
