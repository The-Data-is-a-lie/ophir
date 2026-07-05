# Plan: Trading-agent foundations — data ingestion, best-practices skill, and roadmap

## Context

You're building an autonomous stock-trading agent on top of the existing **ophir**
forecasting model. Your roadmap: ingest a ticker's history → run it through ophir for a
90-day forecast → a decision layer (quant + Ollama) → LangChain research sub-agents →
bull/bear debate → a manager agent that makes the final picks.

This plan produces **three planning/research artifacts** (not the full agent code yet) so
later implementation is grounded and follows industry best practices:

1. A **detailed data-ingestion plan** (your roadmap step 1).
2. A **`trading-best-practices` skill** so future sessions automatically follow good
   trading-system hygiene.
3. A **rough overall roadmap** reconciling your 6-step vision with what already exists.

**Decisions you made (drive this plan):**
- **LLM makes the final picks** (experimental). Honored — *but* I bake in non-negotiable
  guardrails: strictly **paper / `dry_run`**, a **deterministic risk gate that can still
  veto** the LLM's picks before any (paper) order, and a full **audit log**. The skill
  documents both your approach and why the safer "LLM advisory only" pattern exists.
- **Clean-room rebuild** — build a fresh `trading_agent` package rather than extending the
  `automated-paper-trading` branch. We still **reuse ophir's model + feature contract**
  (that's the model's required input math, not trading logic — re-deriving it would be a
  correctness risk).
- **Model-ready, configurable ingestion** — default ~2 years history, `--days` flag,
  written so it feeds the model directly.

**What already exists (discovered during planning):**
- `src/ophir/ticker.py` already has the full feature pipeline: `extract_features` (the 13
  features), `StockStreamer`, `extract_model_data`, `get_splits`, `StockSplit.apply_splits`.
  `yfinance` is already a dependency.
- `ophir-bot/trading-agent-blueprint.md` is a 725-line best-practices design doc. Fresh web
  research (below) confirms its core claims. The skill distills it.
- An `automated-paper-trading` branch already wires Alpaca paper trading + schedulers — set
  aside per your clean-room choice, but a reference.

**Important model fact:** ophir consumes **365-day** windows with a **90-day** horizon
(`ui.py`: `elements_per_sample=365, response_size=90`). Your roadmap's "180 days" is *not*
enough for one inference window — hence model-ready ingestion defaults to ~2 years.

---

## Where each artifact lives

| Artifact | Path | Why here |
| --- | --- | --- |
| Best-practices skill | `C:\Users\Daniel\AppData\Local\FoundryVTT\Data\ophir\.claude\skills\trading-best-practices\SKILL.md` | **Project-level** (ophir repo), beside the `commit` skill — version-controlled. |
| Data-ingestion plan | `C:\Users\Daniel\ophir-bot\data-ingestion-plan.md` | Co-located with the existing blueprint. |
| Overall roadmap | `C:\Users\Daniel\ophir-bot\roadmap.md` | Co-located with the existing blueprint. |

The actual `trading_agent` code (the ingestion module, prediction, agents) is built in
**later steps**, following these docs. This plan's deliverables are the three documents.

---

## Deliverable A — Data-ingestion plan (content of `data-ingestion-plan.md`)

**Goal:** `ingest <TICKER> [--days N]` pulls daily OHLC from Yahoo Finance and writes it in a
format the ophir model consumes unchanged.

### Reuse map (do NOT rewrite these)
| Need | Reuse | Path |
| --- | --- | --- |
| 13-feature math | `extract_features(df)` | `src/ophir/ticker.py:259` |
| Window → model tensors | `extract_model_data(df, response_size)` | `src/ophir/ticker.py:644` |
| Slice into windows | `StockStreamer` | `src/ophir/ticker.py:340` |
| Read back / filter | `StockHanlder` | `src/ophir/ticker.py:471` |
| Data dir resolution | `get_default_data_days_dir()` | `src/ophir/register.py:233` |
| (optional) split adjust | `get_splits` / `StockSplit.apply_splits` | `src/ophir/ticker.py:139,223` |

### Pipeline (fresh orchestration in `src/agent/data/ingest.py`, reusing the above)
1. **Resolve range.** `end = today`, `start = today − days`. Default `days = 730` (~2 yr) to
   guarantee a clean 365-day window **plus** the 60-day rolling-feature warmup
   (`extract_features` uses 10/20/60-day windows). Warn if `days < 425` (window won't fill).
2. **Fetch.** `yfinance.Ticker(sym).history(start, end, interval="1d", auto_adjust=True)`.
   `auto_adjust=True` returns split/dividend-adjusted OHLC, so we **skip** ophir's
   `apply_splits` on this path (avoids double-adjustment). *(Alternative: raw +
   `apply_splits` for training parity — flagged as a verification checkpoint.)*
   Raise a clear error if the frame is empty (invalid/delisted symbol).
3. **Normalize to the `extract_features` contract.** Lowercase columns → keep
   `high`/`low`/`close`/`volume` (`open` optional, unused by features); ensure a
   **tz-naive `DatetimeIndex`** (`.tz_localize(None)`, matching ophir's convention in
   `get_splits`); sort ascending; drop NaN rows; dedupe dates.
4. **Quality / best-practice checks** (from blueprint §6.6, §4): last-bar **staleness**
   (within ~5 calendar days of today, else warn), monotonic index, no duplicate dates, and
   enough rows for a window. Log a one-line summary `(symbol, rows, date range, last close)`.
5. **Persist — the "model-ready" trick.** Write to the existing Hive layout
   `<DATA_DIR>/days/stocks/symbol=<SYMBOL>/data.parquet` with a `utc_time` column (index at a
   fixed market-close time) + `high`/`low`/`close`/`volume`. Because `stock_df` groups by
   `utc_time.dt.normalize()` and there's one row per day, the groupby is an **identity** —
   so `StockHanlder`/`StockStreamer` read it with **zero new feature code**.
6. **Bridge helper** `latest_window_tensors(symbol, seq_len=365, response_size=90)`: load
   frame → `extract_features` → last `seq_len` rows → `extract_model_data` → ready for
   `OHLCMulitClassPredictorInput`. This is the seam into the Prediction phase.
7. **Batch** `ingest_many(tickers, days)` for the ~5-stock universe.

### Output contract
- On disk: `symbol=<SYMBOL>/data.parquet` (Hive layout, model-readable).
- In memory (bridge): `feature_input (S,13)`, `targets (S,3)`, `trade_occured (S,)`,
  `response_size` — exactly what the model expects.

---

## Deliverable B — `trading-best-practices` skill (full `SKILL.md` draft)

This is the exact file I'll write on approval. It distills `trading-agent-blueprint.md` +
fresh web research into rules a future session must follow whenever it touches trading code.

```markdown
---
name: trading-best-practices
description: >-
  Industry best practices for building/modifying the ophir stock-trading agent.
  Use whenever writing or changing code that ingests market data, generates
  trading signals, sizes positions, decides buy/sell/hold, executes orders,
  backtests a strategy, or puts an LLM in the trading loop. Enforces paper-first
  / dry-run defaults, deterministic auditable execution, a risk gate + drawdown
  kill-switch, look-ahead/survivorship-bias-free backtests, and LLM-safety rails.
---

# Trading best practices

Apply these whenever you touch the trading agent. They exist to *not lose money and
not fool yourself*. Full rationale: `ophir-bot/trading-agent-blueprint.md`.

## Non-negotiable invariants
- **Paper-first.** `mode="paper"`, `dry_run=True`, `allow_live=False` are the defaults.
  Going live needs an explicit, loud opt-in. Refuse to start against a live endpoint
  unless `allow_live` is set on purpose.
- **Fail safe, not open.** On any ambiguity (stale data, disconnect, breached limit) the
  default action is *do nothing*, never *trade anyway*.
- **Broker is the source of truth.** Never trust local memory for positions/cash;
  reconcile against the broker every cycle.
- **Deterministic, idempotent execution path.** Same inputs → same orders. Deterministic
  `client_order_id` (e.g. `f"{date}:{symbol}:{side}"`) so a retry/double-run can't
  double-trade.
- **Observability.** Every decision + order goes to an append-only audit trail; you must
  be able to answer "why did it buy X on day Y?" months later.

## Data & signal
- **No look-ahead.** The signal for day T may only use data through T-1's close. Lag every
  feature; never `.shift()` the wrong way.
- **Staleness & quality checks before inference.** Verify last bar is recent, no gaps/dupes,
  tz-consistent (ophir is tz-naive). Stale feed → skip the cycle.
- **Split/dividend adjustment must be consistent** across a window (don't mix adjusted and
  raw). Reuse ophir's `extract_features` contract — don't re-derive the 13 features.
- Use **all three** ophir targets (`r_close`, `upside`, `downside`) for a risk-aware score,
  not just `r_close`.

## Position sizing & risk
- **Volatility targeting is the default** sizing method (scale each name inversely to its
  vol; scale the book to a target annual vol, e.g. 15%). Equal-weight is only a baseline.
- **Fractional Kelly only** (¼–½). Full Kelly is too aggressive and very sensitive to
  estimation error — overestimating edge → risk of ruin.
- **Constraints:** per-name cap (≤5%), sector caps, gross/net exposure bounds, liquidity
  cap (% of ADV), turnover budget + no-trade band to cut churn.
- **Drawdown kill-switch is the single most important control** — halt new risk on a
  peak-to-trough breach (e.g. 20%) and alert a human. Add a daily-loss limit.
- **Risk Gate (pre-trade):** a hard checkpoint *after* portfolio construction and *before*
  the OMS that can **veto / scale / halt**. Property to hold: no gate output can push the
  book past any configured limit.

## Execution
- Use **`alpaca-py`**, never the deprecated `alpaca-trade-api`.
- **`Decimal`, never `float`**, for share quantities and prices (float rounding → rejected
  orders).
- **Retries only on idempotent ops** (`tenacity`, backoff + jitter). A submit that times
  out is resolved by *querying order status*, not blind resubmit.
- **Reconcile vs broker truth** each cycle; handle partial fills (re-target residual) and
  rejects (log + alert).
- **Delta-reconcile, never liquidate-and-rebuy.** Sells first (free buying power), then buys.
- Prefer **marketable-limit or MOC/LOC** over naked market orders for a daily strategy.
- **Gate on the market calendar** (`pandas_market_calendars`); respect PDT (<$25k) and T+1
  settlement (size against `buying_power`, not raw cash). Single-run lock prevents
  scheduler double-fire.

## Backtesting & validation
- **Two engines:** `vectorbt` for fast signal/parameter sweeps; a small **event-driven loop
  that reuses your signal/portfolio/risk modules** for execution realism (backtest == live
  by construction).
- **The three biases that destroy backtests:** look-ahead, **survivorship** (don't backtest
  today's S&P 500 over history — use point-in-time membership), and **overfitting /
  data-snooping** (out-of-sample holdout, walk-forward, deflated Sharpe).
- **Model real costs:** commission/fees, half-spread, slippage (`base + k·size/ADV`), market
  impact. A pre-cost edge that dies after costs is the norm.
- **Validation:** walk-forward; **purged & embargoed CV** when labels overlap in time (your
  multi-day horizon leaks under naive CV); touch the out-of-sample lockbox once.
- **Benchmark vs SPY**, report Sharpe/Sortino/MaxDD/Calmar/turnover. Pin a **golden-file
  backtest** as a regression test.

## LLM in the trading loop (critical — this build lets the LLM pick)
LLMs **hallucinate**, are **non-deterministic**, and have a **knowledge cutoff**. Research
shows their trading decisions are unstable and over-sensitive to input noise. Therefore:
- **Ground every claim in tool-fetched real data**, never model memory. Require citations
  for any number or fact. No fabricated prices/figures.
- **The LLM's picks MUST pass the deterministic Risk Gate** and stay **paper / `dry_run`**.
  An LLM never sizes or places an order that bypasses the gate.
- **Log the full rationale** (inputs, research, debate, final picks) to the audit trail.
- **Reduce instability:** sample/seed deliberately, prefer selective consensus across
  multiple runs, keep prompts structured. Treat LLM output as *advisory even when it
  "decides."*
- The **safest** pattern is LLM-advisory + human-gated. If you move toward production,
  migrate decision authority back to the deterministic model + risk rules and keep the LLM
  for research and reporting.

## Anti-patterns (reject these)
Look-ahead/survivorship/overfitting; ignoring costs; trusting local state over the broker;
fire-and-forget orders; no kill-switch; secrets in git or shared paper/live keys; skipping
paper burn-in; float share quantities; naive timezones; scheduler double-fire; trading on
stale data; **letting an LLM place/size orders without a deterministic gate.**

## Recommended libraries (2026)
`alpaca-py` (broker) · `vectorbt` + `backtesting.py`/`nautilus_trader` (backtest) ·
`quantstats`/`pyfolio-reloaded` (analytics) · `cvxpy`/`riskfolio-lib` (sizing) ·
`pydantic-settings` (config) · `structlog` (logging) · `tenacity` (retries) ·
`pandas_market_calendars` (calendar) · `hypothesis` (property tests) ·
`sqlalchemy`+sqlite/postgres (state).

## References
- `ophir-bot/trading-agent-blueprint.md` (primary).
- López de Prado, *Advances in Financial Machine Learning*; Chan, *Quantitative Trading*;
  Clenow, *Trading Evolved*.
- Backtest bias & LLM-trading-risk web sources (2026), recorded in the roadmap doc.
```

---

## Deliverable C — Overall rough roadmap (content of `roadmap.md`)

Honors your vision (LLM decides) with guardrails; clean-room; paper-only. Each phase is a
clear module so it's easy to build and learn from.

- **Phase 0 — Scaffold & guardrails.** New `trading_agent` package in `ophir-bot/`
  (pyproject, depends on `ophir`). `pydantic-settings` config with `mode=paper` /
  `dry_run=True` / `allow_live=False` + the live-guard validator. `structlog` + append-only
  **audit/decision log** (matters more because an LLM decides). `Broker` Protocol +
  paper-only `alpaca-py` adapter and an in-process `PaperBroker`.
- **Phase 1 — Data ingestion** (Deliverable A). `ingest <TICKER> [--days]` → model-ready
  parquet; batch for the ~5-stock universe; quality/staleness checks.
- **Phase 2 — Prediction.** Wrap `load_base_model_ckpt` + run the latest 365-day window per
  ticker → 90-day forecast (`r_close`/`upside`/`downside`). Emit a structured `Forecast`
  (cumulative predicted return, path, upside/downside bands). *(Needs CUDA + a checkpoint.)*
- **Phase 3 — Decision layer (two tracks, for comparison).** (a) **Quant rules**: forecast →
  buy/sell/hold via thresholds + the risk gate (baseline). (b) **Ollama model**: feed the
  forecast into local `gpt-oss:20b` (already used in `ui.py`) for buy/sell/hold. Compare a↔b.
- **Phase 4 — Research sub-agents (LangChain).** A manager with a template prompt spawns
  per-stock research agents (fundamentals, news/sentiment, technicals) returning structured
  briefs. Best-practice rails: tool-grounded data + citations, no hallucinated numbers.
- **Phase 5 — Debate.** From the briefs, spawn **bull** and **bear** sub-agents per top
  candidate; structured theses.
- **Phase 6 — Final decision (LLM manager picks).** Manager ingests forecast + both decision
  tracks + briefs + debate → **final ranked picks + rationale + target weights**.
  **Guardrail:** picks pass the **deterministic risk gate** (per-name/sector caps, exposure,
  drawdown kill-switch, sanity: symbol exists, not stale, weights bounded) before any paper
  order; everything `dry_run`/paper and logged.
- **Phase 7 — Paper execution + reporting.** Final picks → target weights → delta-reconcile
  vs paper account → idempotent paper orders (dry-run default). LLM writes the daily report
  (reporting use of an LLM is always safe).
- **Phase 8 — Backtest & validation (cross-cutting).** Validate the **quant signal** with a
  walk-forward / purged-CV / realistic-cost backtest vs SPY. The **LLM layer** is evaluated
  by paper track record + decision-log review (LLM decisions are non-deterministic and
  knowledge-cutoff-bound, so they don't backtest cleanly — paper burn-in is the real test).

**Reconciliation note in the doc:** This differs from the blueprint's "LLM off the decision
path." We follow your "LLM decides" choice but keep it paper-only behind a deterministic risk
gate, so a hallucinated/unstable pick can never reach even the paper broker unchecked.

---

## Verification

Most verification applies when the ingestion *code* is built (later); for this turn the
deliverables are documents, so verification is primarily review + a sanity test of the
ingestion design.

- **Data ingestion (once implemented):**
  - `ingest AAPL --days 730` → confirm `…/.ophir/data/days/stocks/symbol=AAPL/data.parquet`
    exists with `utc_time`/`high`/`low`/`close`/`volume`.
  - Load it via `StockHanlder` and confirm `extract_features` yields the 13 feature columns +
    `trade_occured`; take a 365-row window → `extract_model_data` → assert tensor shapes
    `(365,13)` / `(365,3)` / `(365,)`.
  - Sanity-check a feature (e.g. `r_close` magnitude) against an existing training ticker.
  - Verify the `auto_adjust=True` (skip `apply_splits`) decision: spot-check a split name
    (e.g. NVDA/AAPL) shows no price discontinuity at the split date.
  - Ingestion needs **no GPU**; only Phase 2 (model forward) needs CUDA + a checkpoint.
- **Skill:** confirm `~/.claude/skills/trading-best-practices/SKILL.md` is picked up (appears
  in available skills) and triggers when a session edits trading code.
- **Docs:** review `roadmap.md` and `data-ingestion-plan.md` for alignment with intent.

## Out of scope (this turn)
Building the `trading_agent` package code, training/running the model, and wiring the
LangChain agents — those follow these docs in later steps.
