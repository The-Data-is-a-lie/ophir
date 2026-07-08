"""Learning metrics for the daily trading agent, computed from the audit trail.

Reads ``agent-audit.jsonl`` + ingested daily prices to answer the questions that
actually teach you where edge is (or isn't):

* **IC** — does the forecast *rank* names correctly vs. their realized forward
  returns? (cross-sectional Spearman rank correlation, per day, averaged.)
* **Decile spread** — do top-ranked names beat bottom-ranked?
* **Directional hit rate** — does a positive signal go up?
* **Decision breakdown** — BUY/SELL/HOLD by source, and (if the LLM and quant are
  both logged) which source's picks realize better returns.

These are *selection-quality* metrics — computable offline from the log + prices,
and they accrue signal in days (≈180 names/day), not months. Execution/P&L metrics
(implementation shortfall, alpha vs. SPY) need broker fill data and are a follow-up.
Forward returns are measured from each signal's ``asof`` close over N trading days
(standard IC convention: it measures signal quality, not tradeable P&L).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ophir.agent import feed

HORIZONS = (1, 3, 5, 10)
_MIN_NAMES_PER_DAY = 8  # need a cross-section to rank


def audit_path(override: str | None = None) -> Path:
    if override is not None:
        return Path(override)
    from ophir.agent.audit import _default_audit_path

    return _default_audit_path()


def load_events(path: str | None = None) -> pd.DataFrame:
    """Parse the JSONL audit trail into a DataFrame (one row per event)."""
    rows: list[dict[str, Any]] = []
    for line in audit_path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Realized forward returns (from ingested daily closes)
# ---------------------------------------------------------------------------
def _forward_returns(
    signal: pd.DataFrame, horizons: tuple[int, ...], stocks_dir: str | None
) -> pd.DataFrame:
    """Add ``fwd_{h}`` columns: h-trading-day forward return from each row's asof close."""
    cache: dict[str, pd.Series | None] = {}

    def closes(sym: str) -> pd.Series | None:
        if sym not in cache:
            try:
                cache[sym] = feed.load_daily_ohlcv(sym, stocks_dir=stocks_dir)["close"]
            except (FileNotFoundError, KeyError, OSError):
                cache[sym] = None
        return cache[sym]

    cols: dict[int, list[float]] = {h: [] for h in horizons}
    for row in signal.itertuples(index=False):
        s = closes(row.symbol)
        for h in horizons:
            cols[h].append(_fwd(s, row.asof, h))
    for h in horizons:
        signal[f"fwd_{h}"] = cols[h]
    return signal


def _fwd(close: pd.Series | None, asof: pd.Timestamp, h: int) -> float:
    if close is None or len(close) == 0:
        return float("nan")
    pos = close.index.searchsorted(pd.Timestamp(asof), side="right") - 1
    if pos < 0 or pos + h >= len(close):
        return float("nan")
    c0 = float(close.iloc[pos])
    c1 = float(close.iloc[pos + h])
    return c1 / c0 - 1.0 if c0 > 0 else float("nan")


# ---------------------------------------------------------------------------
# Cross-sectional metrics per horizon
# ---------------------------------------------------------------------------
def _daily_ic(df: pd.DataFrame, sig: str, fwd: str) -> pd.Series:
    def ic(g: pd.DataFrame) -> float:
        gg = g[[sig, fwd]].dropna()
        if len(gg) < _MIN_NAMES_PER_DAY or gg[sig].nunique() < 2 or gg[fwd].nunique() < 2:
            return float("nan")
        return float(gg[sig].corr(gg[fwd], method="spearman"))

    return df.groupby("asof").apply(ic, include_groups=False)


def _daily_decile_spread(df: pd.DataFrame, sig: str, fwd: str) -> pd.Series:
    def spread(g: pd.DataFrame) -> float:
        gg = g[[sig, fwd]].dropna()
        if len(gg) < 10:
            return float("nan")
        hi = gg[gg[sig] >= gg[sig].quantile(0.8)][fwd].mean()
        lo = gg[gg[sig] <= gg[sig].quantile(0.2)][fwd].mean()
        return float(hi - lo)

    return df.groupby("asof").apply(spread, include_groups=False)


def signal_quality(signal: pd.DataFrame, sig: str, horizons: tuple[int, ...]) -> pd.DataFrame:
    """Per-horizon: mean daily rank-IC, decile spread, directional hit rate, coverage."""
    rows = []
    for h in horizons:
        fwd = f"fwd_{h}"
        sub = signal.dropna(subset=[sig, fwd])
        ic = _daily_ic(signal, sig, fwd)
        dec = _daily_decile_spread(signal, sig, fwd)
        hit = float((np.sign(sub[sig]) == np.sign(sub[fwd])).mean()) if len(sub) else float("nan")
        rows.append(
            {
                "horizon": f"{h}d",
                "n_obs": len(sub),
                "n_days": int(ic.notna().sum()),
                "mean_IC": float(ic.mean()),
                "decile_spread_%": float(dec.mean() * 100),
                "hit_rate": hit,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def run_metrics_report(
    audit_file: str | None = None,
    *,
    stocks_dir: str | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    snapshot: bool = False,
) -> None:
    """Compute and print the daily model's learning metrics from the audit trail."""
    events = load_events(audit_file)
    if events.empty or "event" not in events.columns:
        print("[metrics] no audit events found.")
        return

    ts = pd.to_datetime(events.get("timestamp"), errors="coerce", utc=True)
    print("========================= DAILY MODEL — LEARNING METRICS =========================")
    print(f" audit: {audit_path(audit_file)}")
    print(f" events: {len(events):,}   window: {ts.min()} .. {ts.max()}")

    fc = events[events["event"] == "forecast"].copy()
    dec = events[events["event"] == "decision"].copy()
    n_days = fc["asof"].nunique() if "asof" in fc.columns else 0
    print(
        f" forecasts: {len(fc):,} over {n_days} asof-days "
        f"(~{len(fc) // max(n_days, 1)}/day, {fc['symbol'].nunique() if len(fc) else 0} symbols)"
        f"   decisions: {len(dec):,}"
    )

    # --- Forecast signal (raw model cum_return) ---
    if len(fc):
        fc["asof"] = pd.to_datetime(fc["asof"], errors="coerce")
        fc = _forward_returns(fc, horizons, stocks_dir)
        print("\n--- FORECAST SIGNAL QUALITY  (forecast.cum_return vs realized forward return) ---")
        print(signal_quality(fc, "cum_return", horizons).to_string(index=False))

    # --- Traded signal (decision.cum_return, dated by the decision timestamp) ---
    if len(dec) and "cum_return" in dec.columns:
        dec["asof"] = (
            pd.to_datetime(dec["timestamp"], errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        dec = _forward_returns(dec, horizons, stocks_dir)
        print("\n--- TRADED SIGNAL QUALITY  (decision.cum_return vs realized forward return) ---")
        print(signal_quality(dec, "cum_return", horizons).to_string(index=False))

        # --- Decision breakdown + source comparison ---
        print("\n--- DECISION BREAKDOWN ---")
        by_src = dec.groupby(["source", "action"]).size().unstack(fill_value=0)
        print(by_src.to_string())
        h = 5 if 5 in horizons else horizons[-1]
        fwd = f"fwd_{h}"
        print(f"\n  realized {h}d forward return by action (BUY should beat SELL):")
        print(dec.groupby("action")[fwd].mean().mul(100).round(3).to_string())
        sources = list(dec["source"].dropna().unique())
        if len(sources) > 1:
            print(f"\n  {h}d forward return of BUY calls, by source (who picks better?):")
            buys = dec[dec["action"] == "BUY"]
            print(
                buys.groupby("source")[fwd]
                .agg(["count", "mean"])
                .assign(mean=lambda d: (d["mean"] * 100).round(3))
                .to_string()
            )
        else:
            print(
                f"\n  (only one decision source logged: {sources or ['none']} — "
                "no LLM-vs-quant comparison available yet)"
            )

    print("\n--- READING THIS ---")
    print(" mean_IC > 0  => the model ranks names correctly. Cross-sectionally, a sustained")
    print("   rank-IC of ~0.02-0.05 is a genuine, tradeable edge; ~0 means no selection skill;")
    print("   < 0 means the signal is inverted. decile_spread_% > 0 and hit_rate > 0.5 agree.")
    print(" Only ~3 weeks of data: short-horizon (1-3d) numbers are the most trustworthy now;")
    print("   Sharpe/drawdown/alpha-vs-SPY need months + broker fills (a v2).")
    print("==================================================================================")

    if snapshot:
        append_snapshot(audit_file, stocks_dir=stocks_dir, horizons=horizons)


# ---------------------------------------------------------------------------
# Snapshot history — one row per run, so the trend firms up over time
# ---------------------------------------------------------------------------
def _history_path(override: str | None = None) -> Path:
    if override is not None:
        return Path(override)
    from ophir.register import DATA_DIR

    return Path(DATA_DIR) / "metrics-history.jsonl"


def append_snapshot(
    audit_file: str | None = None,
    *,
    stocks_dir: str | None = None,
    horizons: tuple[int, ...] = HORIZONS,
    history_file: str | None = None,
) -> Path | None:
    """Append one headline-metrics row (dated by the latest event) to the history file."""
    events = load_events(audit_file)
    if events.empty or "event" not in events.columns:
        return None
    h = 5 if 5 in horizons else horizons[-1]
    ts = pd.to_datetime(events.get("timestamp"), errors="coerce", utc=True)
    row: dict[str, object] = {"date": str(ts.max().date())}

    fc = events[events["event"] == "forecast"].copy()
    dec = events[events["event"] == "decision"].copy()
    row["n_forecasts"] = len(fc)
    row["n_decisions"] = len(dec)

    if len(fc):
        fc["asof"] = pd.to_datetime(fc["asof"], errors="coerce")
        fc = _forward_returns(fc, (h,), stocks_dir)
        row[f"fc_ic_{h}d"] = round(
            float(signal_quality(fc, "cum_return", (h,))["mean_IC"].iloc[0]), 4
        )

    if len(dec) and "cum_return" in dec.columns:
        dec["asof"] = (
            pd.to_datetime(dec["timestamp"], errors="coerce", utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        dec = _forward_returns(dec, (h,), stocks_dir)
        fwd = f"fwd_{h}"
        row[f"dec_ic_{h}d"] = round(
            float(signal_quality(dec, "cum_return", (h,))["mean_IC"].iloc[0]), 4
        )
        buy = dec.loc[dec["action"] == "BUY", fwd].mean()
        sell = dec.loc[dec["action"] == "SELL", fwd].mean()
        row[f"buy_minus_sell_{h}d_%"] = round(float((buy - sell) * 100), 3)
        bs = dec[dec["action"] == "BUY"].groupby("source")[fwd].mean()
        if "ollama" in bs.index and "quant" in bs.index:
            row[f"llm_minus_quant_buy_{h}d_%"] = round(float((bs["ollama"] - bs["quant"]) * 100), 3)

    try:  # counterfactual equity gap (best-effort; snapshot must never fail the daily run)
        books = counterfactual_books(events, stocks_dir=stocks_dir)
        if "ollama" in books and "quant" in books:
            row["llm_minus_quant_equity"] = round(
                float(
                    books["ollama"]["equity_curve"][-1][1] - books["quant"]["equity_curve"][-1][1]
                ),
                4,
            )
    except Exception:
        pass

    path = _history_path(history_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    print(f"[metrics] snapshot appended -> {path}")
    return path


def run_history_report(history_file: str | None = None) -> None:
    """Print the metrics-history trend (one row per run)."""
    path = _history_path(history_file)
    if not path.exists():
        print(f"[metrics] no history yet at {path}. Run `ophir report metrics --snapshot`.")
        return
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    print("=========================== METRICS HISTORY (per run) ===========================")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n Watch dec_ic_5d and llm_minus_quant_buy_5d_%: if they hold positive over weeks the")
    print(" selection edge / LLM value is real; if they drift toward 0 it was small-sample noise.")
    print("=================================================================================")


# ---------------------------------------------------------------------------
# Counterfactual attribution — quant-only vs LLM-only vs SPY equity curves
# ---------------------------------------------------------------------------
def counterfactual_books(
    events: pd.DataFrame,
    *,
    stocks_dir: str | None = None,
    cost_bps: float = 3.5,
    benchmark: str = "SPY",
) -> dict[str, dict[str, Any]]:
    """Per-source equal-weight BUY-book equity curves (+ SPY) from the decision log.

    For each decision source, at each run date form an equal-weighted book of that
    date's BUYs, hold to the next run date, realize the close-to-close return minus
    ``cost_bps * turnover``, and compound. Returns ``{source: {equity_curve, returns,
    n_buys, metrics}}`` plus the (cost-free) benchmark. Measures each source's
    **selection** skill, not exact traded P&L.
    """
    from ophir.agent.backtest import compute_metrics

    if "event" not in events.columns:
        return {}
    dec = events[events["event"] == "decision"].copy()
    if dec.empty:
        return {}
    dec["date"] = (
        pd.to_datetime(dec["timestamp"], errors="coerce", utc=True)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    dec = dec.dropna(subset=["date"])
    dates = sorted(dec["date"].unique())
    if len(dates) < 2:
        return {}

    cache: dict[str, pd.Series | None] = {}

    def closes(sym: str) -> pd.Series | None:
        key = sym.upper()
        if key not in cache:
            try:
                cache[key] = feed.load_daily_ohlcv(key, stocks_dir=stocks_dir)["close"]
            except (FileNotFoundError, KeyError, OSError):
                cache[key] = None
        return cache[key]

    def ret_between(sym: str, d0: object, d1: object) -> float:
        s = closes(sym)
        if s is None or len(s) == 0:
            return float("nan")
        i0 = s.index.searchsorted(pd.Timestamp(d0), side="right") - 1
        i1 = s.index.searchsorted(pd.Timestamp(d1), side="right") - 1
        if i0 < 0 or i1 < 0:
            return float("nan")
        c0, c1 = float(s.iloc[i0]), float(s.iloc[i1])
        return c1 / c0 - 1.0 if c0 > 0 else float("nan")

    cost = cost_bps / 1e4
    years = max((pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days / 365.25, 1e-9)
    ppy = max(len(dates) - 1, 1) / years

    def build_source(sub: pd.DataFrame) -> dict[str, Any]:
        eq, rets, n_buys = 1.0, [], 0
        w_old: dict[str, float] = {}
        curve = [(str(pd.Timestamp(dates[0]).date()), 1.0)]
        for i in range(len(dates) - 1):
            d0, d1 = dates[i], dates[i + 1]
            book = [
                str(s).upper()
                for s in sub.loc[(sub["date"] == d0) & (sub["action"] == "BUY"), "symbol"]
            ]
            n_buys += len(book)
            w_new = {s: 1.0 / len(book) for s in book} if book else {}
            vals = [ret_between(s, d0, d1) for s in book]
            vals = [v for v in vals if not (v is None or np.isnan(v))]
            port = float(np.mean(vals)) if vals else 0.0
            turnover = sum(
                abs(w_new.get(s, 0.0) - w_old.get(s, 0.0)) for s in set(w_new) | set(w_old)
            )
            eq *= 1.0 + (port - cost * turnover)
            rets.append(port - cost * turnover)
            curve.append((str(pd.Timestamp(d1).date()), eq))
            w_old = w_new
        return {
            "equity_curve": curve,
            "returns": rets,
            "n_buys": n_buys,
            "metrics": compute_metrics(rets, periods_per_year=ppy),
        }

    results: dict[str, dict[str, Any]] = {}
    for src in sorted(dec["source"].dropna().unique()):
        results[str(src)] = build_source(dec[dec["source"] == src])

    # Benchmark: buy-and-hold, no turnover cost.
    eq, brets = 1.0, []
    bcurve = [(str(pd.Timestamp(dates[0]).date()), 1.0)]
    for i in range(len(dates) - 1):
        r = ret_between(benchmark, dates[i], dates[i + 1])
        r = 0.0 if np.isnan(r) else r
        eq *= 1.0 + r
        brets.append(r)
        bcurve.append((str(pd.Timestamp(dates[i + 1]).date()), eq))
    results[benchmark] = {
        "equity_curve": bcurve,
        "returns": brets,
        "n_buys": None,
        "metrics": compute_metrics(brets, periods_per_year=ppy),
    }
    return results


def run_attribution_report(
    audit_file: str | None = None, *, stocks_dir: str | None = None, cost_bps: float = 3.5
) -> None:
    """Print the quant-only vs LLM-only vs SPY counterfactual equity comparison."""
    events = load_events(audit_file)
    if events.empty or "event" not in events.columns:
        print("[attribution] no audit events found.")
        return
    try:  # keep the SPY benchmark current (one-ticker fetch; best-effort, non-fatal)
        from ophir.agent.ingest import ingest

        ingest("SPY", stocks_dir=stocks_dir)
    except Exception:
        pass
    books = counterfactual_books(events, stocks_dir=stocks_dir, cost_bps=cost_bps)
    if not books:
        print("[attribution] not enough decisions across >=2 run dates yet.")
        return

    print("========== LLM-vs-QUANT COUNTERFACTUAL ATTRIBUTION (equal-weight BUY books) ==========")
    print(f" cost: {cost_bps:.1f} bps/turnover   books: {', '.join(books)}")
    summary = pd.DataFrame(
        [
            {
                "book": src,
                "n_buys": r["n_buys"],
                "final_equity": round(r["equity_curve"][-1][1], 4),
                "total_return_%": round(r["metrics"]["total_return"] * 100, 2),
                "sharpe": round(r["metrics"]["sharpe"], 2),
                "max_dd_%": round(r["metrics"]["max_drawdown"] * 100, 2),
                "hit_rate": round(r["metrics"]["hit_rate"], 2),
            }
            for src, r in books.items()
        ]
    )
    print(summary.to_string(index=False))

    eq_df = pd.DataFrame({src: dict(r["equity_curve"]) for src, r in books.items()}).sort_index()
    print("\n cumulative equity (start = 1.0):")
    print(eq_df.round(4).to_string())

    if "ollama" in books and "quant" in books:
        gap = books["ollama"]["equity_curve"][-1][1] - books["quant"]["equity_curve"][-1][1]
        print(
            f"\n LLM - quant final-equity gap: {gap:+.4f}  "
            f"({'LLM ahead' if gap > 0 else 'quant ahead'})"
        )
    print("\n Small sample (weeks): a firming-up view of SELECTION skill, not final traded P&L.")
    print("=====================================================================================")
