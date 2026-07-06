"""Offline tests for triple-barrier labeling + sample weights."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ophir.agent.intraday.labels import triple_barrier_labels


def _bars(closes: np.ndarray, day: str = "2026-06-08") -> pd.DataFrame:
    n = len(closes)
    idx = pd.date_range(f"{day} 13:30", periods=n, freq="5min", tz="UTC")
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": close * 1.0002,
            "low": close * 0.9998,
            "close": close,
            "volume": np.full(n, 1e6),
            "trade_count": np.full(n, 1e4),
            "dollar_volume": close * 1e6,
            "vwap": close,
            "start": idx,
            "end": idx,
            "n_minutes": np.full(n, 5.0),
        }
    )


def test_last_bar_of_day_is_unlabeled() -> None:
    rng = np.random.default_rng(0)
    c1 = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.001, 30)))
    c2 = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.001, 30)))
    bars = pd.concat([_bars(c1, "2026-06-08"), _bars(c2, "2026-06-09")], ignore_index=True)
    lab = triple_barrier_labels(bars, max_hold=26, vol_span=8)
    b = lab["bin_tb"].to_numpy()
    assert np.isnan(b[29]) and np.isnan(b[-1])  # last bar of each session -> no forward path


def test_labels_domain_and_weights() -> None:
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.001, 200)))
    lab = triple_barrier_labels(_bars(c), max_hold=26, vol_span=8)
    m = lab.dropna(subset=["bin_tb"])
    assert set(np.unique(m["bin_tb"])).issubset({-1.0, 0.0, 1.0})
    assert (m["weight"] > 0).all()
    assert abs(float(m["weight"].mean()) - 1.0) < 0.1  # mean-normalized uniqueness
    assert m["ret"].notna().all()


def test_strong_uptrend_favors_profit() -> None:
    rng = np.random.default_rng(2)
    c = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.0005, 200)))  # strong up drift
    lab = triple_barrier_labels(_bars(c), pt_mult=1.5, sl_mult=1.5, max_hold=26, vol_span=8)
    assert int((lab["bin_tb"] == 1.0).sum()) > int((lab["bin_tb"] == -1.0).sum())


def test_strong_downtrend_favors_stop() -> None:
    rng = np.random.default_rng(3)
    c = 100 * np.exp(np.cumsum(rng.normal(-0.002, 0.0005, 200)))  # strong down drift
    lab = triple_barrier_labels(_bars(c), pt_mult=1.5, sl_mult=1.5, max_hold=26, vol_span=8)
    assert int((lab["bin_tb"] == -1.0).sum()) > int((lab["bin_tb"] == 1.0).sum())
