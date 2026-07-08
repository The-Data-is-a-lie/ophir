"""P2 baseline + the make-or-break cost gate (G2).

Assembles features+labels across the test universe, trains a regularized LightGBM
classifier to predict which longs hit the profit barrier first, selects a trade
threshold on a validation slice, and scores an untouched, purged, embargoed
**lockbox** (the most recent slice) net of a realistic round-trip cost. Prints a
pre-registered PASS/FAIL verdict.

Honesty guards: the lockbox is scored once; training samples whose label span
overlaps the validation/lockbox periods are purged, with an embargo gap; and the
report separates model *skill* (top- vs bottom-ranked spread) from mere market
drift (the always-long base rate), because a long-only strategy in an up market
earns beta that is not intraday alpha.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ophir.agent.intraday.dataset import build_training_frame, feature_columns

DEFAULT_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "JNJ",
    "WMT",
    "MA",
    "PG",
    "HD",
    "COST",
    "KO",
    "BAC",
]


def _fit_lightgbm(x: pd.DataFrame, y: np.ndarray[Any, Any], w: np.ndarray[Any, Any]) -> Any:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.03,
        min_child_samples=300,  # heavy regularization: intraday SNR is low
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=0,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x, y, sample_weight=w)
    return model


def _annualized_sharpe(net: np.ndarray[Any, Any], span_days: float) -> float:
    if len(net) < 2 or net.std(ddof=1) == 0.0 or span_days <= 0:
        return 0.0
    trades_per_year = len(net) / (span_days / 365.25)
    return float(net.mean() / net.std(ddof=1) * np.sqrt(trades_per_year))


def run_gate(
    symbols: list[str] | None = None,
    *,
    stocks_dir: str | None = None,
    cost_bps: float = 5.0,
    cost_by_symbol: dict[str, float] | None = None,
    pt_mult: float = 1.5,
    sl_mult: float = 1.5,
    max_hold: int = 26,
    vol_span: int = 26,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    embargo: str = "2D",
    pass_sharpe: float = 1.0,
) -> dict[str, Any]:
    """Run the baseline + cost gate; print the report and return the metrics."""
    symbols = symbols or DEFAULT_UNIVERSE
    print(f"[gate] assembling features+labels for {len(symbols)} symbols ...")
    df = build_training_frame(
        symbols,
        stocks_dir=stocks_dir,
        pt_mult=pt_mult,
        sl_mult=sl_mult,
        max_hold=max_hold,
        vol_span=vol_span,
    ).sort_index()
    if df.empty:
        raise RuntimeError("No training data assembled -- backfill the universe first.")

    fcols = feature_columns(df)
    end = df.index
    t1 = pd.to_datetime(df["t1"], utc=True)
    y = (df["bin_tb"] == 1.0).astype(int).to_numpy()
    w = df["weight"].to_numpy(dtype="float64")
    ret = df["ret"].to_numpy(dtype="float64")
    x = df[fcols]
    if cost_by_symbol:
        cost_row = (
            df["symbol"].map(lambda s: cost_by_symbol.get(s, cost_bps)).to_numpy(dtype="float64")
            / 1e4
        )
    else:
        cost_row = np.full(len(df), cost_bps / 1e4)

    n = len(df)
    val_start = end[int(n * (1 - val_frac - test_frac))]
    test_start = end[int(n * (1 - test_frac))]
    emb = pd.Timedelta(embargo)

    m_test = np.asarray(end >= test_start)
    m_val = np.asarray((end >= val_start) & (end < test_start)) & np.asarray(t1 < test_start)
    m_train = np.asarray((end < val_start) & (end < val_start - emb)) & np.asarray(t1 < val_start)

    print(
        f"[gate] rows={n:,}  train={m_train.sum():,}  val={m_val.sum():,}  "
        f"lockbox={m_test.sum():,}\n"
        f"[gate] lockbox window: {end[m_test].min()} .. {end[m_test].max()}"
    )

    model = _fit_lightgbm(x[m_train], y[m_train], w[m_train])
    p = model.predict_proba(x)[:, 1]

    # Pick the trade threshold on the validation slice (maximize net expectancy).
    pv, rv, cv = p[m_val], ret[m_val], cost_row[m_val]
    best_thr, best_net = float(np.quantile(pv, 0.8)), -np.inf
    for q in np.linspace(0.50, 0.95, 19):
        thr = float(np.quantile(pv, q))
        sel = pv >= thr
        if sel.sum() < 200:
            continue
        net_mean = float((rv[sel] - cv[sel]).mean())
        if net_mean > best_net:
            best_net, best_thr = net_mean, thr
    thr = best_thr

    # Score the lockbox once (per-trade cost applied by symbol).
    pt_, rt_, ct = p[m_test], ret[m_test], cost_row[m_test]
    sel = pt_ >= thr
    net = rt_[sel] - ct[sel]
    span_days = float((end[m_test].max() - end[m_test].min()).days)
    mean_cost_bps = float(ct.mean() * 1e4)

    base_gross = float(rt_.mean())
    q80, q20 = float(np.quantile(pt_, 0.8)), float(np.quantile(pt_, 0.2))
    top_net = float((rt_[pt_ >= q80] - ct[pt_ >= q80]).mean())
    bot_net = float((rt_[pt_ <= q20] - ct[pt_ <= q20]).mean())
    sel_net_mean = float(net.mean()) if len(net) else 0.0
    sharpe = _annualized_sharpe(net, span_days)
    hit = float((rt_[sel] > ct[sel]).mean()) if sel.sum() else 0.0

    passed = bool(sel_net_mean > 0.0 and sharpe >= pass_sharpe and (top_net - bot_net) > 0.0)

    bps = 1e4
    cost_label = (
        f"measured avg {mean_cost_bps:.2f} bps"
        if cost_by_symbol
        else f"{cost_bps:.1f} bps (assumed)"
    )
    print("\n================= G2 COST GATE (lockbox, scored once) =================")
    print(f" universe:            {len(symbols)} names   round-trip cost: {cost_label}")
    print(f" trade threshold:     p >= {thr:.3f}  (chosen on validation)")
    print(
        f" lockbox trades:      {int(sel.sum()):,} of {int(m_test.sum()):,} events "
        f"over ~{span_days / 365.25:.2f}y"
    )
    print(
        f" base rate (always-long, gross):   {base_gross * bps:+.2f} bps/trade  "
        f"(net {base_gross * bps - mean_cost_bps:+.2f})   <- market drift, not alpha"
    )
    print(f" model top-quintile (net):         {top_net * bps:+.2f} bps/trade")
    print(f" model bottom-quintile (net):      {bot_net * bps:+.2f} bps/trade")
    print(
        f" skill spread (top - bottom):      {(top_net - bot_net) * bps:+.2f} bps  "
        f"(>0 = ranking has signal)"
    )
    print(f" SELECTED trades (net):            {sel_net_mean * bps:+.2f} bps/trade")
    print(f" annualized net Sharpe:            {sharpe:.2f}   (hit rate {hit:.1%})")
    print(" cost sensitivity (selected net bps/trade):")
    for cb in (0.0, 2.0, 5.0, 10.0):
        c = cb / 1e4
        s2 = rt_[sel] - c
        print(
            f"     @ {cb:4.1f} bps: {float(s2.mean()) * bps:+.2f} bps/trade   Sharpe "
            f"{_annualized_sharpe(s2, span_days):.2f}"
        )
    verdict = (
        "PASS -> proceed to P3 (deep model)" if passed else "FAIL -> stop; no net-of-cost edge"
    )
    print(f"\n VERDICT: {verdict}")
    print("=======================================================================\n")

    return {
        "passed": passed,
        "threshold": thr,
        "n_trades": int(sel.sum()),
        "selected_net_bps": sel_net_mean * bps,
        "sharpe": sharpe,
        "skill_spread_bps": (top_net - bot_net) * bps,
        "base_gross_bps": base_gross * bps,
        "cost_bps": cost_bps,
    }
