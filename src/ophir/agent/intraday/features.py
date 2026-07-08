"""Intraday feature extraction on dollar bars (look-ahead-safe).

Adapts the philosophy of :func:`ophir.ticker.extract_features` (log returns,
multi-window normalized returns & volatility, upside/downside excursions) to
intraday dollar bars, and adds microstructure-lite and session-seasonality
features. Every feature at bar ``t`` uses only information available at bar
``t``'s close, so the matrix is causal (verified in tests/test_intraday_features.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

_ET = "America/New_York"
_RTH_MINUTES = 390.0
_OPEN_MINUTE = 9 * 60 + 30  # 09:30 ET, minutes since midnight

# ~1/3 day, ~1 day, ~3 days at ~26 dollar bars per RTH day.
DEFAULT_WINDOWS: tuple[int, ...] = (8, 26, 78)


def extract_intraday_features(
    bars: pd.DataFrame,
    *,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    eps: float = 1e-3,
) -> pd.DataFrame:
    """Return a causal feature matrix indexed by each dollar bar's close time.

    ``bars`` is the output of :func:`ophir.agent.intraday.data.build_dollar_bars`
    (columns ``open, high, low, close, volume, trade_count, dollar_volume, vwap,
    start, end, n_minutes``). ``eps`` matches the intraday per-bar return scale
    (~1e-3, ~10-30x smaller than the daily model's).
    """
    if bars.empty:
        return pd.DataFrame()

    df = bars.reset_index(drop=True)
    end = pd.to_datetime(df["end"], utc=True)
    close = df["close"].astype("float64")
    high = df["high"].astype("float64")
    low = df["low"].astype("float64")
    r = np.log(close / close.shift(1))

    feat: dict[str, pd.Series] = {"r_close": r}
    for w in windows:
        vol = r.rolling(w, min_periods=max(2, w // 2)).std()
        feat[f"norm_ret_{w}"] = r / (vol + eps)
        feat[f"vol_{w}"] = vol

    # Per-bar excursion envelope (mirrors ophir's upside/downside targets as features).
    feat["upside"] = np.log(high / close)
    feat["downside"] = np.log(close / low)
    feat["range"] = np.log(high / low)

    vwap = df["vwap"].astype("float64")
    vwap = vwap.where(vwap > 0, close)
    feat["vwap_dist"] = np.log(close / vwap)

    # Microstructure-lite (activity / participation).
    n_minutes = df["n_minutes"].astype("float64")
    feat["trade_intensity"] = df["trade_count"].astype("float64") / n_minutes.where(
        n_minutes > 0, 1.0
    )
    feat["bar_minutes"] = n_minutes

    med = windows[len(windows) // 2]
    volume = df["volume"].astype("float64")
    feat[f"rel_vol_{med}"] = volume / volume.rolling(med, min_periods=1).mean()
    dvol = df["dollar_volume"].astype("float64")
    feat[f"rel_dvol_{med}"] = dvol / dvol.rolling(med, min_periods=1).mean()

    # Session seasonality (from the bar's close time in ET).
    end_et = end.dt.tz_convert(_ET)
    minutes_since_open = end_et.dt.hour * 60 + end_et.dt.minute - _OPEN_MINUTE
    feat["tod"] = (minutes_since_open / _RTH_MINUTES).clip(0.0, 1.0)
    feat["dow"] = end_et.dt.dayofweek.astype("float64")

    out = pd.DataFrame(feat)
    out.index = pd.DatetimeIndex(end, name="bar_end")
    return out
