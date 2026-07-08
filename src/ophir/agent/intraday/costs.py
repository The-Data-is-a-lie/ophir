"""Measure realistic round-trip trading cost from historical NBBO quotes (free SIP).

Samples a handful of short RTH quote windows spread across each symbol's history,
computes the median relative bid-ask spread (bps), and reports a round-trip cost
= spread + a small slippage buffer. Replaces the flat cost assumption in the G2
gate with a measured, per-name number. Read-only; free-tier SIP quotes are
available for any window ending >15 min ago.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ophir.agent.intraday import data as _data

_ET = "America/New_York"
_SNAP_TIMES_ET = ("10:00", "13:00", "15:00")  # morning / midday / afternoon


def measure_spread_bps(
    symbol: str,
    client: object,
    start_date: dt.date,
    end_date: dt.date,
    *,
    n_days: int = 10,
    quotes_per_snap: int = 300,
) -> float:
    """Median relative bid-ask spread (bps) over sampled RTH quote windows."""
    from alpaca.data.enums import DataFeed
    from alpaca.data.requests import StockQuotesRequest

    step = max((end_date - start_date).days // max(n_days, 1), 1)
    spreads: list[float] = []
    used, day = 0, start_date
    while day <= end_date and used < n_days:
        if day.weekday() < 5:  # weekday
            got = False
            for t in _SNAP_TIMES_ET:
                start = pd.Timestamp(f"{day} {t}", tz=_ET).tz_convert("UTC")
                end = start + pd.Timedelta(minutes=2)
                try:
                    q = client.get_stock_quotes(  # type: ignore[attr-defined]
                        StockQuotesRequest(
                            symbol_or_symbols=[symbol],
                            start=start.to_pydatetime(),
                            end=end.to_pydatetime(),
                            feed=DataFeed.SIP,
                            limit=quotes_per_snap,
                        )
                    ).df
                except Exception:
                    continue
                if q is None or q.empty or "bid_price" not in q.columns:
                    continue
                bid = q["bid_price"].to_numpy(dtype="float64")
                ask = q["ask_price"].to_numpy(dtype="float64")
                mid = (bid + ask) / 2.0
                ok = (bid > 0) & (ask > bid) & (mid > 0)
                if ok.any():
                    spreads.extend(((ask[ok] - bid[ok]) / mid[ok] * 1e4).tolist())
                    got = True
            if got:
                used += 1
        day += dt.timedelta(days=step)
    return float(np.median(spreads)) if spreads else float("nan")


def measure_round_trip_costs(
    symbols: list[str],
    *,
    buffer_bps: float = 0.5,
    n_days: int = 10,
    fallback_bps: float = 6.0,
) -> dict[str, float]:
    """Per-name round-trip cost (bps) = median spread + slippage buffer."""
    client = _data._client()
    costs: dict[str, float] = {}
    for sym in symbols:
        try:
            bars = _data.load_dollar_bars(sym)
        except FileNotFoundError:
            print(f"[cost] {sym}: no bars -> fallback {fallback_bps:.1f} bps")
            costs[sym] = fallback_bps
            continue
        et = pd.to_datetime(bars["end"], utc=True).dt.tz_convert(_ET)
        spread = measure_spread_bps(sym, client, et.min().date(), et.max().date(), n_days=n_days)
        rt = spread + buffer_bps if np.isfinite(spread) else fallback_bps
        costs[sym] = rt
        print(f"[cost] {sym}: median spread {spread:.2f} bps -> round-trip {rt:.2f} bps")
    return costs
