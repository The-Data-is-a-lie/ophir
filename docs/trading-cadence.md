# Trading cadence: once-a-day vs. intraday

**Decision (2026-06-18): ophir stays a once-a-day model.** Trading multiple times
in the same day is *informationally null* with the current architecture — not
merely suboptimal — until a major intraday rework happens.

The decisive fact: re-running the model intraday produces the **identical
forecast**, so a second same-day trade would act on zero new information.

## Why once-a-day is the only sensible cadence

Three independent parts of the system all assume (and require) a daily cadence.

### 1. The model is daily by construction

It is trained on **daily** OHLC bars and predicts three **daily** targets:

- `r_close = log(close / prev_close)`
- `upside  = log(high / close)`  — a *post-close* daily aggregate
- `downside = log(close / low)`  — a *post-close* daily aggregate

(see `ticker.py:295,322-323`). `upside`/`downside` are unknowable until the day
ends, so they can't be acted on as live intraday levels.

Inference conditions on the **last closed** session (T-1) and explicitly drops
today's still-forming bar (`feed.py:112`, `predict.py:78-145`). A 9:35am run and
a 2pm run therefore feed the model an *identical* 365-day input window and return
*identical* output. The 365-bar context window, the ALiBi/positional encoding,
and every rolling-window feature (10/20/60-day returns, volume, volatility) all
assume daily bars.

### 2. The data inputs don't refresh intraday

Of the 16 data inputs (see [data-inputs.md](data-inputs.md)), ~12 are
daily-or-slower: OHLC, technicals, dark-pool (daily aggregate), short interest
(biweekly), congressional trades (disclosure-gated), SEC insider/8-K
(filing-gated), and fundamentals. Only **options IV** (CBOE delayed quotes) and
**account state** are genuinely intraday — and options IV is reactive to price,
not a fresh directional signal the daily model consumes. A noon re-decision would
see essentially the same research context as the open.

### 3. The loop and execution already assume once-a-day

- The trade cycle is gated on `is_trading_day()` (`cli.py:399`) and runs once.
- Orders are plain `TimeInForce.DAY` market orders with no brackets
  (`execute.py:160-187`).
- The live `OphirRebalance` task fires once per weekday at 10:30am PT.

Multiple same-day trades would churn spread/slippage for no signal. On the
eventual **live** (non-paper) account, more than 3 day-trades in 5 business days
under $25k equity also trips the **pattern-day-trader (PDT)** rule.

## Alternatives considered (deferred, not chosen)

- **Bracket exits with no retraining** — the only no-rework way to "use intraday."
  The model already predicts next-day range (`upside`/`downside`), currently
  advisory-only (`decide.py:190`). Those could set a take-profit (upside) and stop
  (downside) on the daily entry: one decision per day with intraday *exits*, not
  intraday re-decisions. Caveat: this can't be properly backtested without
  intraday bars (the walk-forward harness rebalances ~monthly and models no
  intraday fills), so validating it is itself a data problem.
- **A true intraday model** — the "major rework." Requires an intraday data feed
  (Yahoo's intraday history is thin, so likely a paid source), reformulated
  sub-daily targets, a full retrain (positional encoding and all rolling-window
  features change), and an intraday loop/scheduler.

## When to revisit

Only when an intraday data feed **and** a retrain are actually on the table. Until
then, once-a-day is the correct and only sensible cadence.
