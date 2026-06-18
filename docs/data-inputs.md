# Ophir data inputs

This catalogs every external and derived data input the ophir trading agent consumes on a daily run, with the exact function to reuse, the fields it returns, how it is persisted/cached, and how it fails safe. Each input names the `gather-*` skill that documents it in depth (or `(derived/none)` when there is no fetch step). Use this page as the index; jump to a skill for the call recipe and extension notes.

## Price & market data

- **OHLC daily bars (yfinance)**
  - Provider: Yahoo Finance via `yfinance` — `yf.Ticker(symbol).history(start, end, interval="1d", auto_adjust=True)`.
  - Fetch: [agent/ingest.py:85](src/ophir/agent/ingest.py:85) (`ingest`), low-level fetch [agent/ingest.py:30](src/ophir/agent/ingest.py:30) (`_fetch_yahoo`). Read back with [agent/feed.py:49](src/ophir/agent/feed.py:49) (`load_daily_ohlcv`) and the session-trimmed [agent/feed.py:77](src/ophir/agent/feed.py:77) (`load_history`).
  - Fields: `high`, `low`, `close`, `volume`, indexed by `utc_time` (tz-naive, deduped, sorted ascending). `auto_adjust=True` so bars are already split/dividend-adjusted.
  - Persistence: written to the Hive layout `<DATA_DIR>/days/stocks/symbol=<SYMBOL>/data.parquet` (path via [agent/feed.py:44](src/ophir/agent/feed.py:44), `parquet_path`); re-ingest overwrites.
  - Fail-safe: empty/delisted symbol raises `ValueError`; staleness (last bar > 5 days old) and sparsity emit non-fatal warnings ([agent/ingest.py:61](src/ophir/agent/ingest.py:61)). `load_history` refuses a stale feed (`ValueError`) so callers skip the ticker.
  - Skill: gather-ohlc

- **SPY benchmark**
  - Provider: same yfinance OHLC mechanism — SPY is just another ticker ingested and read through `ingest` / `load_daily_ohlcv`.
  - Fetch: [agent/ingest.py:85](src/ophir/agent/ingest.py:85), read via [agent/feed.py:49](src/ophir/agent/feed.py:49).
  - Fields: identical to the OHLC bars above (`high`/`low`/`close`/`volume`).
  - Persistence: `symbol=SPY/data.parquet` under the same stocks root.
  - Fail-safe: same as OHLC bars.
  - Skill: gather-ohlc

- **Stock-split history (yfinance `.splits`)**
  - Provider: Yahoo Finance — `yf.Ticker(ticker).splits`.
  - Fetch: [ticker.py:139](src/ophir/ticker.py:139) (`get_splits`); apply via [ticker.py:223](src/ophir/ticker.py:223) (`StockSplit.apply_splits`).
  - Fields: per ticker a `StockSplit` of `id`, `dates` (effective split dates), `ratios`; tickers with no splits map to a `None` sentinel.
  - Persistence: pickle cache at `<DATA_DIR>/yf_splits_cache.pkl`; only missing tickers hit the network, sentinels prevent re-querying.
  - Fail-safe: a failed ticker is printed and skipped. NOTE: largely dormant in the live path — OHLC is ingested with `auto_adjust=True`, so splits are only needed for raw/unadjusted price adjustment.
  - Skill: gather-splits

- **Dark-pool / off-exchange activity (FINRA)**
  - Provider: FINRA daily short-sale-volume file (free, **no auth**) — `https://cdn.finra.org/equity/regsho/daily/CNMSshvol{YYYYMMDD}.txt`; `TotalVolume` is the off-exchange (TRF/ADF/ORF-reported) volume — a free proxy for Unusual Whales' dark-pool feed.
  - Fetch: [agent/darkpool.py:128](src/ophir/agent/darkpool.py:128) (`dark_pool_signal`); per-day download+parse [agent/darkpool.py:86](src/ophir/agent/darkpool.py:86) (`_fetch_daily`).
  - Fields: `off_exchange_volume`, `off_exchange_pct` (vs consolidated yfinance volume), `off_exchange_short_ratio`, `off_exchange_short_exempt_ratio`, `off_exchange_short_ratio_zscore`, `off_exchange_vol_zscore` (anomaly z-score), plus `asof`/`n_days`.
  - Persistence: each day's file cached under `<DATA_DIR>/finra/regsho/`; one download per session covers every ticker.
  - Fail-safe: missing/late file, `403`, or unknown symbol → neutral dict, never raises. Daily **aggregate** off-exchange volume — not individual real-time prints (no free source has those).
  - Brief dimension: `ResearchBrief.flow` via [agent/research.py](src/ophir/agent/research.py:0) (`gather_flow`).
  - Skill: gather-dark-pool

- **Consolidated short interest (FINRA, biweekly)**
  - Provider: FINRA consolidated short-interest dataset (free, **no auth**) — `POST https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest`, filtered per symbol + settlement-date range. The standing short *position* (settled twice a month) — additive to, not a duplicate of, the dark-pool intraday short *volume*.
  - Fetch: [agent/short_interest.py:?](src/ophir/agent/short_interest.py:0) (`short_interest_signal`); query+cache [agent/short_interest.py:0](src/ophir/agent/short_interest.py:0) (`_query`).
  - Fields: `short_interest` (settled short shares), `prev_short_interest`, `days_to_cover`, `si_change_pct`, `avg_daily_volume`, `settlement_date`, plus `asof`/`n_settlements`/`source`.
  - Persistence: per-symbol JSON cached under `<DATA_DIR>/finra/shortinterest/`.
  - Point-in-time: gates on the most recent `settlementDate <= as_of` (no look-ahead).
  - Fail-safe: HTTP error / unknown symbol → neutral dict, never raises.
  - Brief dimension: `ResearchBrief.short_interest` via `gather_short_interest`.
  - Skill: gather-short-interest

- **Options / IV surface (CBOE, per-name)**
  - Provider: CBOE delayed-quotes JSON (free, **no auth**, browser UA) — `https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json` (full chain with `iv` / `open_interest` / `volume` / greeks + a ready `iv30`). ophir's first **per-name** volatility input (vs the market-wide VIX).
  - Fetch: [agent/options_flow.py:133](src/ophir/agent/options_flow.py:133) (`options_flow_signal`).
  - Fields: `put_call_ratio_volume` / `put_call_ratio_oi`, `iv_skew_25d` (front-expiry 25-delta put−call IV; positive = downside fear), `iv30`, `spot`, `volume_oi_ratio`, plus `asof`/`n_contracts`/`source`.
  - Persistence: each session's snapshot cached under `<DATA_DIR>/cboe/<date>/`.
  - Point-in-time: **no free option history exists** → a LIVE-loop signal, **not backtestable**; a past `as_of` with no cached snapshot returns neutral (never live-fetches dated to the past).
  - Fail-safe: missing snapshot / unknown symbol / error → neutral dict, never raises.
  - Brief dimension: `ResearchBrief.options` via `gather_options`.
  - Skill: gather-options

## Research data

- **Fundamentals (yfinance `.info`)**
  - Provider: Yahoo Finance — `yf.Ticker(symbol).info`.
  - Fetch: [agent/research.py:107](src/ophir/agent/research.py:107) (`gather_fundamentals`).
  - Fields: curated `_FUNDAMENTAL_KEYS` subset — `sector`, `industry`, `marketCap`, `trailingPE`, `forwardPE`, `profitMargins`, `beta`, `dividendYield`, `fiftyTwoWeekHigh`, `fiftyTwoWeekLow`, `currentPrice`.
  - Persistence: none of its own; lands in `ResearchBrief.fundamentals` (audit-logged via the research event).
  - Fail-safe: any network/parse error returns `{"error": "fundamentals unavailable (...)"}` (grounded-but-empty), never raising.
  - Skill: gather-fundamentals

- **News (yfinance `.news`)**
  - Provider: Yahoo Finance — `yf.Ticker(symbol).news`.
  - Fetch: [agent/research.py:144](src/ophir/agent/research.py:144) (`gather_news`), normalized per-item by [agent/research.py:118](src/ophir/agent/research.py:118) (`_normalize_news_item`).
  - Fields: per headline `title`, `publisher`, `link`, `published` (normalized across yfinance's old flat and new nested payload shapes); limited to `research_news_limit` (default 8).
  - Persistence: none of its own; lands in `ResearchBrief.news`, with `link`s recorded in `sources`.
  - Fail-safe: any error returns `[]` (no news); the brief still synthesizes.
  - Skill: gather-news

- **Technicals (derived from OHLC + forecast)**
  - Provider: derived — no external fetch; computed from the ingested OHLC parquet and the model `Forecast`.
  - Fetch: [agent/research.py:155](src/ophir/agent/research.py:155) (`gather_technicals`), reusing [ticker.py:259](src/ophir/ticker.py:259) (`extract_features`).
  - Fields: `last_close`, `recent_return_1d`, `vol_20d`, `vol_60d`, `pct_of_52w_high`, `pct_above_52w_low`, `mom_12_1` (12-1 momentum, skip-month), `atr_pct_14` (ATR % of close), `rs_vs_spy_3m` + `rs_slope` (relative strength vs the already-ingested SPY), `amihud_illiq_60d` + `dollar_volume_20d` (liquidity / price-impact, zero new fetch); when a `Forecast` is supplied also `forecast_cum_return`, `forecast_mean_upside`, `forecast_mean_downside`.
  - Persistence: none of its own; lands in `ResearchBrief.technicals`.
  - Fail-safe: requires ingested OHLC — `load_history` raises on a missing/stale feed, which `research_many` logs and skips. The relative-strength fields fall back to `None` if SPY is missing.
  - Skill: gather-technicals

- **Congressional trades (US House Clerk PTRs)**
  - Provider: US House Clerk financial disclosures (free, **no auth**, House-only) — yearly `https://disclosures-clerk.house.gov/public_disc/financial-pdfs/<YEAR>FD.zip` (an XML filing index) plus per-`DocID` PTR PDFs at `/public_disc/ptr-pdfs/<YEAR>/<DocID>.pdf` (parsed with `pypdf`). **Send a browser `User-Agent`** (some clients get a `403`).
  - Fetch: [agent/congress.py:0](src/ophir/agent/congress.py:0) (`congress_signal`); window index [agent/congress.py:0](src/ophir/agent/congress.py:0) (`_window_index`, built once per session and cached).
  - Fields: `cluster_buy_members` / `cluster_buy_flag` (distinct members buying in 30d), `net_dollar_flow_60d` + `sell_dollar_60d` (signed bucket-midpoint flow, sell leg surfaced), `congress_net_bias`, `days_since_last_disclosure` / `last_disclosure_type` / `new_disclosure_flag`, plus `asof`/`lookback`/`n_disclosures`/`source`.
  - Persistence: yearly ZIP + per-`DocID` parsed transactions cached under `<DATA_DIR>/congress/` (first session is slow — many PDF fetches — then cached).
  - Point-in-time: **gates strictly on the filing/disclosure date, never the transaction date** (the 30–45 day STOCK Act lag would otherwise inject look-ahead). A slow, contested "smart-money" prior, wired as a **low-weight advisory** dimension.
  - Legal: `5 U.S.C. app. 105(c)` bars *commercial* use of these disclosures — ophir is paper-only / non-commercial; do not redistribute.
  - Fail-safe: missing ZIP/PDF, scanned/ticker-less filing, or unknown symbol → neutral dict, never raises.
  - Brief dimension: `ResearchBrief.congress` via `gather_congress`.
  - Skill: gather-congress

- **Earnings calendar event gate (yfinance)**
  - Provider: Yahoo Finance — `yf.Ticker(symbol).calendar`.
  - Fetch: [agent/events.py:0](src/ophir/agent/events.py:0) (`earnings_signal`).
  - Fields: `next_earnings_date` / `days_to_next_earnings` (the event-risk gate), `last_earnings_date` / `days_since_last_earnings` (the PEAD window), plus `asof`/`source`.
  - Persistence: none of its own; lands in `ResearchBrief.events`.
  - Point-in-time: the earnings *schedule* is forward-looking and revised — a **live gate**, not a look-ahead-free backtest feature.
  - Fail-safe: any network/parse error → neutral dict, never raises.
  - Brief dimension: `ResearchBrief.events` via `gather_events`.
  - Skill: gather-events

- **Corporate insider + events (SEC EDGAR)**
  - Provider: SEC EDGAR (free, **no auth**; UA-with-email, ≤10 req/s) — ticker→CIK `company_tickers.json`, per-CIK `data.sec.gov/submissions/...json` (Form 4 + 8-K + 13D/G in one fetch), raw Form 4 `<ownershipDocument>` XML. The legal-insider analog to congress, stronger evidence.
  - Fetch: [agent/edgar.py:196](src/ophir/agent/edgar.py:196) (`insider_signal`); `_cik_map` / `_submissions` / `_parse_form4`.
  - Fields: `insider_cluster_buyers` / `insider_cluster_flag` (open-market `P` buyers, 30d), `insider_net_dollar_flow_90d` + `insider_sell_dollar_90d` (actual shares×price), `insider_net_bias`, `days_since_last_insider_buy` / `last_insider_type`, `days_since_last_8k` / `last_8k_items`, `activist_stake_flag`, plus `asof`/`lookback`/`n_form4`/`source`.
  - Persistence: submissions JSON memoized per process; parsed Form 4 trades cached per accession under `<DATA_DIR>/edgar/form4/`.
  - Point-in-time: **gates on `filingDate <= as_of`** (disclosure date, never the transaction date); only open-market `P`/`S` count.
  - Fail-safe: unknown symbol / odd filing / error → neutral dict, never raises.
  - Brief dimension: `ResearchBrief.insider` via `gather_insider`.
  - Skill: gather-insider

- **News sentiment (FinBERT, derived)**
  - Provider: derived — FinBERT (`ProsusAI/finbert` via `transformers`, local GPU) scores the headlines already in the brief; **zero new fetch** (model downloaded once, then cached).
  - Fetch: [agent/sentiment.py:74](src/ophir/agent/sentiment.py:74) (`news_sentiment_signal`).
  - Fields: `net_sentiment` (mean signed, [-1,1]), `pos_count` / `neg_count` / `neutral_count`, plus `asof`/`n_headlines`/`source`.
  - Persistence: none; the FinBERT pipeline is process-cached.
  - Fail-safe: missing model / no headlines → neutral dict, never raises. Weak, short-horizon — low weight.
  - Brief dimension: `ResearchBrief.sentiment` via `gather_sentiment`.
  - Skill: gather-sentiment

- **Public attention (Wikipedia pageviews + Reddit)**
  - Provider: Wikimedia pageviews REST (free, **no auth**, archived since 2015, point-in-time) + ApeWisdom (free, no auth, snapshot). Title resolved via Wikipedia opensearch from the company `shortName`.
  - Fetch: [agent/attention.py:148](src/ophir/agent/attention.py:148) (`attention_signal`).
  - Fields: `wiki_pageviews` + `wiki_pageviews_zscore` (attention-spike flag), `wiki_title`, `reddit_mentions` / `reddit_rank` / `reddit_mentions_24h_ago`, plus `asof`/`source`.
  - Persistence: resolved titles cached under `<DATA_DIR>/wiki/titles.json`; ApeWisdom list memoized per session.
  - Point-in-time: Wikipedia pageviews gated to days `<= as_of` (backtestable); the Reddit read is a current snapshot.
  - Fail-safe: components independent; unresolved/no-data → neutral dict, never raises.
  - Brief dimension: `ResearchBrief.attention` via `gather_attention`.
  - Skill: gather-attention

## Macro & regime

- **VIX term-structure regime (yfinance)**
  - Provider: Yahoo Finance — `^VIX` / `^VIX3M` (free, **no auth**). Market-wide: one read shared by every ticker on a given `as_of` (the slope of the volatility term structure is a forward-looking risk-on/off regime read the spot level misses).
  - Fetch: [agent/macro.py:0](src/ophir/agent/macro.py:0) (`macro_signal`); download split into `_term_structure` (memoized per `as_of` so a whole watchlist shares one download).
  - Fields: `vix`, `vix3m`, `vix_ratio` (VIX / VIX3M), `vix_ratio_z` (vs the trailing year), `regime` (`risk_on` / `neutral` / `risk_off`); plus FRED `hy_credit_spread` (`BAMLH0A0HYM2`), `nfci`, `yield_curve_2s10s` (`T10Y2Y`) — `None` until `ophir register fred-key` — and the deterministic `days_to_opex` (3rd-Friday options-expiry gate); with `asof`/`n_days`/`source`.
  - Persistence: process-level memo by session date (no disk cache). FRED key stored at `<OPHIR_DIR>/.fred_key` (mirrors the MASSIVE key).
  - Point-in-time: VIX/OPEX clean; FRED series publish ~T-1 and are revised — use ALFRED vintages for backtests (not implemented).
  - Fail-safe: each component degrades independently to `None`/`"unknown"`; `days_to_opex` is always present, so the macro dimension never collapses entirely. Never raises.
  - Brief dimension: `ResearchBrief.macro` via `gather_macro`.
  - Skill: gather-macro

## Universe & calendar

- **S&P 500 constituents (Wikipedia)**
  - Provider: Wikipedia — `pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")`.
  - Fetch: [ticker.py:108](src/ophir/ticker.py:108) (`get_sp_500_symbols`).
  - Fields: a `list[str]` of ticker symbols from the constituents table's `Symbol` column.
  - Persistence: none — live HTTP request each call (invoked at import time by `ophir.ui`).
  - Fail-safe: weak — a fetch error is printed and `dfs` is left unbound, so the call then raises; treat the universe as point-in-time (survivorship caveat for backtests).
  - Skill: gather-sp500-universe

- **Curated watchlist (`ophir-bot/watchlist.txt`)**
  - Provider: operational file in the `ophir-bot` repo — one ticker per line, `#` comments and blanks ignored (~28 names today).
  - Fetch: read by `ophir-bot/rebalance.ps1`, not by `src/ophir` code; the daily run scores every name here, then the manager + risk gate pick `top_k`.
  - Fields: bare ticker symbols (e.g. `AAPL`, `NVDA`).
  - Persistence: the file itself; hand-edited.
  - Fail-safe: operational concern of the runner script, not the library.
  - Skill: (operational; no skill)

- **NYSE trading calendar (pandas-market-calendars)**
  - Provider: `pandas_market_calendars` — `mcal.get_calendar("NYSE")`.
  - Fetch: [agent/market_calendar.py:48](src/ophir/agent/market_calendar.py:48) (`last_closed_session`) and [agent/market_calendar.py:80](src/ophir/agent/market_calendar.py:80) (`is_trading_day`).
  - Fields: `last_closed_session` returns a tz-naive midnight `Timestamp` of the most recently closed session; `is_trading_day` returns a `bool`.
  - Persistence: the calendar object is `lru_cache`d per process ([agent/market_calendar.py:29](src/ophir/agent/market_calendar.py:29)); anchored on UTC/US-Eastern, never the host clock.
  - Fail-safe: no closed session in the 12-day lookback raises `ValueError`; the daily run uses `is_trading_day` to skip weekends/holidays rather than forecast on stale data.
  - Skill: gather-market-calendar

## Account & execution

- **Alpaca paper account + positions (alpaca-py)**
  - Provider: Alpaca **paper** trading via `alpaca-py` — `alpaca.trading.client.TradingClient(..., paper=True)`.
  - Fetch: [agent/execute.py:119](src/ophir/agent/execute.py:119) (`AlpacaPaperBroker`); `.get_account()` [agent/execute.py:140](src/ophir/agent/execute.py:140), `.get_positions()` [agent/execute.py:152](src/ophir/agent/execute.py:152). `PaperBroker` ([agent/execute.py:77](src/ophir/agent/execute.py:77)) is the in-memory simulated alternative for dry runs/tests.
  - Fields: `Account(equity, cash, drawdown, daily_loss)` (all `Decimal`/float); positions as `{symbol: market_value}` (`Decimal`).
  - Persistence: none locally — the broker is the source of truth, reconciled every cycle ([agent/execute.py:191](src/ophir/agent/execute.py:191), `reconcile`).
  - Fail-safe: missing `AGENT_ALPACA_KEY_ID` / `AGENT_ALPACA_SECRET_KEY` raises at construction; a single rejected order is logged and skipped, never aborting the rebalance; `dry_run=True` is the default (plans only).
  - Skill: gather-account

## Model artifacts (derived / loaded; no gather-skill)

- **13 engineered features**
  - Derived from the ingested OHLC frame — no fetch.
  - Compute: [ticker.py:259](src/ophir/ticker.py:259) (`extract_features`).
  - Fields: `time_delta`, `r_close`, the 10/20/60-day `*_norm_returns` / `*_norm_volume` / `*_volatility`, `upside`, `downside`, plus the `trade_occured` calendar-padding flag.
  - Persistence: none — recomputed per window; padded days zero-filled.
  - Fail-safe: empty input returns an empty frame.
  - Skill: (derived/none)

- **Per-ticker forecast (`Forecast`)**
  - Derived by running the trained model over the latest 365-day window — no fetch.
  - Compute: [agent/predict.py:78](src/ophir/agent/predict.py:78) (`predict_ticker` → `Forecast`), windowed by [agent/feed.py:163](src/ophir/agent/feed.py:163) (`forecast_window_tensors`).
  - Fields: `symbol`, `asof`, `horizon`, per-day `r_close`/`upside`/`downside` lists, `cum_return`, `score`.
  - Persistence: none of its own; logged to the audit trail as a `forecast` event.
  - Fail-safe: a stale/missing feed raises and `predict_many` logs + skips the ticker; requires a CUDA GPU.
  - Skill: (derived/none)

- **Trained checkpoint**
  - Loaded from disk — no fetch.
  - Load: [agent/predict.py:64](src/ophir/agent/predict.py:64) (`load_predictor`, process-cached) → [register.py:330](src/ophir/register.py:330) (`load_base_model_ckpt`, `time_version=False` for the validation-best forecast checkpoint); selection via the `basebest-*.ckpt` resolver [register.py:189](src/ophir/register.py:189) (`_latest_ckpt`).
  - Fields: a `LightningOHLCPredictor` on the GPU in eval mode.
  - Persistence: `.ckpt` files under `<MODEL_DIR>` (`src/ophir/.ophir/model/`).
  - Fail-safe: no matching checkpoint raises `FileNotFoundError`.
  - Skill: (derived/none)

## LLM

- **Local Ollama `gpt-oss:20b`**
  - Provider: a local Ollama server via `langchain_ollama.ChatOllama`; a *service dependency*, not fetched data.
  - Use: decide / research / debate / manage / report stages — e.g. the research synthesis in [agent/research.py:300](src/ophir/agent/research.py:300) and the end-of-day report in [agent/execute.py:317](src/ophir/agent/execute.py:317).
  - Config: `ollama_model` (`gpt-oss:20b`), `ollama_base_url`, `ollama_num_ctx` (16384) in [agent/config.py:36](src/ophir/agent/config.py:36).
  - Persistence: none — stateless calls; `temperature=0`, JSON-constrained where a structured reply is required.
  - Fail-safe: every LLM stage degrades to a neutral/templated result on error (e.g. research falls back to a neutral analysis, the report to a plain templated summary) while keeping the grounded data intact.
  - Skill: (service dependency; no gather-skill)

## Dormant / dead code

- **MASSIVE REST client**
  - Provider: `massive.RESTClient`, authenticated from `src/ophir/.ophir/.massive_key` (written by `ophir register massive-key`).
  - Defined at [register.py:167](src/ophir/register.py:167) (`get_massive_client`).
  - Status: **defined but never called** anywhere in `src/` — a grep for `get_massive_client(` finds only the definition (callers appear solely in docs/CLAUDE.md prose). No live or training path consumes MASSIVE data; OHLC comes entirely from yfinance. Safe to remove (function, the `massive-key` CLI command, and the doc references) when cleaning up.
  - Skill: (dead code; no skill)

## Skills

- [gather-ohlc](../.claude/skills/gather-ohlc/SKILL.md) — daily OHLC bars (and the SPY benchmark) from yfinance.
- [gather-splits](../.claude/skills/gather-splits/SKILL.md) — stock-split history from yfinance `.splits`.
- [gather-dark-pool](../.claude/skills/gather-dark-pool/SKILL.md) — free dark-pool (off-exchange) activity from FINRA daily files.
- [gather-short-interest](../.claude/skills/gather-short-interest/SKILL.md) — free FINRA biweekly consolidated short interest.
- [gather-congress](../.claude/skills/gather-congress/SKILL.md) — free US House congressional-trade disclosures (PTRs).
- [gather-macro](../.claude/skills/gather-macro/SKILL.md) — free market-wide VIX regime + FRED credit/conditions/curve + OPEX gate.
- [gather-events](../.claude/skills/gather-events/SKILL.md) — free Yahoo Finance earnings-calendar event gate.
- [gather-insider](../.claude/skills/gather-insider/SKILL.md) — free SEC EDGAR Form 4 insider + 8-K / 13D-G events.
- [gather-options](../.claude/skills/gather-options/SKILL.md) — free CBOE per-name options / IV-surface signal.
- [gather-sentiment](../.claude/skills/gather-sentiment/SKILL.md) — FinBERT-scored sentiment over existing headlines.
- [gather-attention](../.claude/skills/gather-attention/SKILL.md) — free Wikipedia pageviews + Reddit attention.
- [gather-fundamentals](../.claude/skills/gather-fundamentals/SKILL.md) — company fundamentals from yfinance `.info`.
- [gather-news](../.claude/skills/gather-news/SKILL.md) — recent headlines from yfinance `.news`.
- [gather-technicals](../.claude/skills/gather-technicals/SKILL.md) — derived indicators from OHLC + forecast.
- [gather-sp500-universe](../.claude/skills/gather-sp500-universe/SKILL.md) — S&P 500 constituents from Wikipedia.
- [gather-market-calendar](../.claude/skills/gather-market-calendar/SKILL.md) — NYSE sessions from pandas-market-calendars.
- [gather-account](../.claude/skills/gather-account/SKILL.md) — Alpaca paper account + positions via alpaca-py.
- [data-inputs](../.claude/skills/data-inputs/SKILL.md) — the main index skill mapping every input above to its gather skill.
