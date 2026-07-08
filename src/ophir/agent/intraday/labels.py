"""Triple-barrier labeling, meta-labels, and average-uniqueness sample weights.

Implements the Lopez de Prado triple-barrier method on intraday dollar bars: for
each event bar a volatility-scaled profit-take and stop barrier, plus a vertical
(time / flat-by-EOD) barrier; the label is which barrier the forward path hits
first. Sample weights down-weight events whose holding spans overlap, so the
heavily-overlapping intraday labels are not treated as IID. See blueprint P1/§6.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

_ET = "America/New_York"
_LABEL_COLS = ["t1", "ret", "bin_tb", "meta", "weight"]


def bar_volatility(close: pd.Series, span: int = 26) -> pd.Series:
    """EWMA volatility of per-bar log returns — the barrier-width scale."""
    r = np.log(close / close.shift(1))
    return r.ewm(span=span, min_periods=max(2, span // 2)).std()


def _day_last_index(end_utc: pd.Series) -> np.ndarray[Any, Any]:
    """For each bar, the positional index of the last bar sharing its ET date."""
    dates = pd.to_datetime(end_utc, utc=True).dt.tz_convert(_ET).dt.date.to_numpy()
    pos = pd.Series(np.arange(len(dates)))
    return cast("np.ndarray[Any, Any]", pos.groupby(dates).transform("max").to_numpy())


def _average_uniqueness(t1_pos: np.ndarray[Any, Any], n: int) -> np.ndarray[Any, Any]:
    """AFML average uniqueness: down-weight events with overlapping [entry, t1] spans."""
    concurrency = np.zeros(n, dtype="float64")
    for i in range(n):
        if t1_pos[i] >= 0:
            concurrency[i : t1_pos[i] + 1] += 1.0
    uniq = np.full(n, np.nan)
    for i in range(n):
        if t1_pos[i] < 0:
            continue
        c = concurrency[i : t1_pos[i] + 1]
        c = c[c > 0]
        if len(c):
            uniq[i] = float(np.mean(1.0 / c))
    mean_u = float(np.nanmean(uniq)) if np.isfinite(uniq).any() else 1.0
    return uniq / mean_u if mean_u > 0 else uniq


def triple_barrier_labels(
    bars: pd.DataFrame,
    *,
    pt_mult: float = 1.5,
    sl_mult: float = 1.5,
    max_hold: int = 26,
    vol_span: int = 26,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """Label each dollar bar by the first barrier its forward path touches.

    For a long (``side`` defaults to +1): upper = ``close*exp(pt_mult*sigma)``,
    lower = ``close*exp(-sl_mult*sigma)``; vertical barrier = ``min(i+max_hold,
    last bar of the ET day)`` (flat-by-EOD). Returns a frame indexed by bar_end:
    ``t1`` (ISO touch time), ``ret`` (side-adjusted log return to t1), ``bin_tb``
    (+1 profit / -1 stop / 0 timeout), ``meta`` (1 if ``ret>0`` else 0),
    ``weight`` (mean-normalized average-uniqueness sample weight). Events with no
    forward bar in their day (last bar of a session) are left unlabeled (NaN).
    """
    n = len(bars)
    if n == 0:
        return pd.DataFrame(columns=_LABEL_COLS)

    end = pd.to_datetime(bars["end"], utc=True)
    close = bars["close"].to_numpy(dtype="float64")
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    sigma = bar_volatility(bars["close"], vol_span).to_numpy(dtype="float64")
    side_a = np.ones(n, dtype="float64") if side is None else side.to_numpy(dtype="float64")
    day_last = _day_last_index(bars["end"])

    bin_tb = np.full(n, np.nan)
    ret = np.full(n, np.nan)
    t1_pos = np.full(n, -1, dtype="int64")

    for i in range(n):
        s = sigma[i]
        if not math.isfinite(s) or s <= 0.0:
            continue
        vbar = min(i + max_hold, int(day_last[i]))
        if vbar <= i:  # last bar of the session -> no forward path, leave unlabeled
            continue
        sd = side_a[i]
        entry = close[i]
        up = entry * math.exp(pt_mult * s)
        dn = entry * math.exp(-sl_mult * s)
        outcome = 0.0
        tj = vbar
        for j in range(i + 1, vbar + 1):
            hit_up = high[j] >= up
            hit_dn = low[j] <= dn
            if sd > 0:
                if hit_up:
                    outcome, tj = 1.0, j
                    break
                if hit_dn:
                    outcome, tj = -1.0, j
                    break
            elif hit_dn:
                outcome, tj = 1.0, j
                break
            elif hit_up:
                outcome, tj = -1.0, j
                break
        bin_tb[i] = outcome
        t1_pos[i] = tj
        ret[i] = sd * math.log(close[tj] / entry)

    weight = _average_uniqueness(t1_pos, n)
    meta = np.where(np.isfinite(ret), (ret > 0.0).astype("float64"), np.nan)
    t1 = np.array([end.iloc[p].isoformat() if p >= 0 else None for p in t1_pos], dtype=object)

    out = pd.DataFrame({"t1": t1, "ret": ret, "bin_tb": bin_tb, "meta": meta, "weight": weight})
    out.index = pd.DatetimeIndex(end, name="bar_end")
    return out
