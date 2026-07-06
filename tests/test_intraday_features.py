"""Offline tests for intraday feature extraction (causality is the key property)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ophir.agent.intraday.features import extract_intraday_features


def _synth_bars(n: int = 120) -> pd.DataFrame:
    """Deterministic, OHLC-coherent synthetic dollar bars."""
    idx = pd.date_range("2026-06-08 13:30", periods=n, freq="5min", tz="UTC")
    close = 100.0 + np.cumsum(np.sin(np.arange(n) / 5.0)) * 0.1
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(openp, close) + 0.05
    low = np.minimum(openp, close) - 0.05
    vol = 1000.0 + (np.arange(n) % 7) * 100.0
    return pd.DataFrame(
        {
            "open": openp, "high": high, "low": low, "close": close,
            "volume": vol, "trade_count": vol / 10.0, "dollar_volume": close * vol,
            "vwap": close, "start": idx, "end": idx, "n_minutes": np.full(n, 5.0),
        }
    )


def test_features_are_causal() -> None:
    # Feature row k must be identical whether computed on the full series or a prefix.
    bars = _synth_bars(120)
    full = extract_intraday_features(bars)
    k = 90
    prefix = extract_intraday_features(bars.iloc[:k])
    pd.testing.assert_series_equal(prefix.iloc[-1], full.iloc[k - 1], check_names=False)


def test_feature_shape_and_warmup() -> None:
    bars = _synth_bars(120)
    feat = extract_intraday_features(bars)
    assert len(feat) == 120
    assert feat.shape[1] >= 15
    assert not feat.iloc[80:].isna().any().any()  # no NaN past the longest warmup window


def test_empty_bars_return_empty() -> None:
    assert extract_intraday_features(pd.DataFrame()).empty
