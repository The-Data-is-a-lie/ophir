---
name: gather-macro
description: >-
  Use when you need a FREE market-wide macro / regime read (NOT per-ticker) — the VIX term-structure
  slope (VIX / VIX3M), its z-score, and a risk_on / neutral / risk_off regime label — to give the
  LLM debate/manager and the risk gate shared cycle context. Reuse macro_signal in
  src/ophir/agent/macro.py, which pulls ^VIX and ^VIX3M from Yahoo Finance (free, no auth) and memoizes
  per session so a whole watchlist shares one download. Reach for it to fetch / refresh / debug /
  extend the macro/regime dimension.
---

# Gather macro / regime (VIX term structure)

Unlike the per-ticker signals, this is one **market-wide** read shared by every name on a given
`as_of`. The slope of the volatility term structure (VIX / VIX3M) is a forward-looking risk-on/off
regime read the spot VIX level alone misses: a ratio below ~1 (contango) is calm/risk-on, a ratio at
or above 1 (backwardation) flags acute stress.

See [docs/data-inputs.md](docs/data-inputs.md) for the full data-source catalog.

## Source
- **Provider:** Yahoo Finance, free + no auth — `^VIX` (1-month) and `^VIX3M` (3-month) via
  `yfinance.download`. Single market-wide series; no key.
- Reuse (do not re-fetch by hand):
  - [agent/macro.py:78](src/ophir/agent/macro.py:78) — `macro_signal(*, as_of=None)` (note: **no `symbol`** — market-wide).
  - [agent/macro.py](src/ophir/agent/macro.py) — `_term_structure(cutoff)` (the only network call; monkeypatch in tests)
    and `_regime(ratio)`.
  - Reuses [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) `last_closed_session`.

Secondary FRED series (free **API key**, stored via `ophir register fred-key`) and a deterministic
options-expiry gate are folded into the same once-per-session dict:
- `hy_credit_spread` (FRED `BAMLH0A0HYM2`), `nfci` (Chicago Fed financial conditions, `NFCI`),
  `yield_curve_2s10s` (`T10Y2Y`) — orthogonal to VIX; `None` until a FRED key is registered.
- `days_to_opex` — calendar days to the next monthly equity options expiry (3rd Friday); deterministic,
  needs no key or network — an event-risk gate the manager/risk layer can act on.

## Fields
- `vix`, `vix3m` — the two index levels at the cutoff session.
- `vix_ratio` — VIX / VIX3M (the term-structure slope).
- `vix_ratio_z` — the ratio's latest value vs its trailing-year mean (an anomaly read).
- `regime` — `risk_on` (ratio ≤ 0.90) / `neutral` / `risk_off` (ratio ≥ 1.00), or `unknown` on failure.
- `hy_credit_spread`, `nfci`, `yield_curve_2s10s` (FRED), `days_to_opex` (deterministic).
- plus `asof`, `n_days`, `source`.

## Fail-safe, point-in-time & caching
- Conditioned on the last completed session ≤ `as_of` (no look-ahead). FRED series publish ~T-1 and are
  revised — for look-ahead-free backtests use ALFRED vintages (not implemented here).
- Any network/shape error degrades each piece independently to `None`/`"unknown"`; `days_to_opex` is
  always present (pure date math), so the macro dimension never collapses entirely. Never raises.
- Memoized in-process by session date (`_CACHE`), so one fetch serves the whole watchlist in a run.

## How to use / extend
- `from ophir.agent.macro import macro_signal; macro_signal()` (broadcast the one dict to every ticker).
- Register the FRED key once: `ophir register fred-key <KEY>` (free, no card, from fredaccount.stlouisfed.org).
- It feeds `ResearchBrief.macro` via `gather_macro` and is surfaced in the debate/manager prompts; it is
  also a natural input to the deterministic risk gate (down-weight gross exposure in `risk_off`).
- **Extend:** add a programmatic FOMC/CPI/NFP release calendar (FRED releases/dates API) and a composite
  risk-on/off score over the VIX + credit + curve components.
