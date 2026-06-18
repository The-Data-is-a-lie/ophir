---
name: gather-sentiment
description: >-
  Use when you need a FinBERT-scored news-sentiment signal for a ticker (e.g. AAPL) — a calibrated
  net sentiment + pos/neg/neutral counts over the headlines ophir ALREADY fetches (zero new network
  fetch). Reuse news_sentiment_signal in src/ophir/agent/sentiment.py, which scores yfinance headlines
  locally with ProsusAI/finbert (transformers, on the CUDA GPU). Reach for it to fetch / refresh /
  debug / extend the scored-sentiment dimension. News sentiment is weak and short-horizon — low weight.
---

# Gather FinBERT news sentiment

ophir fetches raw Yahoo Finance headlines but never *scores* them. This turns that existing text into
a calibrated numeric sentiment layer with **zero new network fetch**: the headlines already gathered
for the brief are scored locally with FinBERT on the already-required GPU.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** `ProsusAI/finbert` via `transformers` (no auth; ~440 MB model downloaded once from
  Hugging Face, then cached). `torch`/`transformers` are already project deps. No per-ticker fetch —
  it scores the brief's own headlines.
- Reuse (do not re-implement):
  - [agent/sentiment.py:74](src/ophir/agent/sentiment.py:74) — `news_sentiment_signal(symbol, *, news=None, as_of=None)`.
  - [agent/sentiment.py](src/ophir/agent/sentiment.py) — `_classifier` (process-cached pipeline),
    `_score_headlines`. Pass the already-gathered `news` to avoid a second fetch.

## Fields
- `net_sentiment` — mean signed sentiment in [-1, 1] (positive=+1, neutral=0, negative=-1).
- `pos_count` / `neg_count` / `neutral_count`, plus `symbol`, `asof`, `n_headlines`, `source`.

## Fail-safe
- If `transformers`/the model is unavailable (no GPU, no network for the one-time download) or there
  are no headlines → a **neutral dict** with a `note`, never an exception. The model is loaded once per
  process and the failure is sticky (degrades to neutral for the rest of the run).
- The sentiment inherits yfinance news' shallow, snapshot history — a weak, short-horizon signal, wired
  as a **low-weight** advisory dimension.

## How to use / extend
- `from ophir.agent.sentiment import news_sentiment_signal; news_sentiment_signal("AAPL", news=brief.news)`.
- Feeds `ResearchBrief.sentiment` via `gather_sentiment`; surfaced in the debate/manager prompts.
- **Extend:** add a news-VOLUME / novelty spike read, or score earnings-call transcripts (needs a free
  transcript source) for management-tone sentiment.
