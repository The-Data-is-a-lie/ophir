"""Offline unit tests for the intraday data layer (no network, deterministic)."""

from __future__ import annotations

import pandas as pd
import pytest

from ophir.agent.intraday import data


def _minutes(
    day: str, times: list[str], *, price: float = 100.0, vol: float = 100.0
) -> pd.DataFrame:
    """Build a UTC-indexed minute frame from ET wall-clock ``times`` (HH:MM)."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f"{day} {t}", tz="America/New_York").tz_convert("UTC") for t in times],
        name="utc_time",
    )
    n = len(times)
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price + 1.0] * n,
            "low": [price - 1.0] * n,
            "close": [price] * n,
            "volume": [vol] * n,
            "trade_count": [10.0] * n,
            "vwap": [price] * n,
        },
        index=idx,
    )


def test_filter_rth_keeps_only_regular_hours() -> None:
    df = _minutes("2026-06-08", ["08:00", "09:29", "09:30", "12:00", "15:59", "16:00", "18:00"])
    out = data._filter_rth(df)
    et = [t.tz_convert("America/New_York").strftime("%H:%M") for t in out.index]
    assert et == ["09:30", "12:00", "15:59"]  # pre/post-market + the 16:00 close excluded


def test_build_dollar_bars_threshold_and_counts() -> None:
    # dv/min = vwap(100)*vol(100) = 10_000 -> cum 10k,20k,30k,40k -> bar=cum//15k = 0,1,2,2
    df = _minutes("2026-06-08", ["09:30", "09:31", "09:32", "09:33"])
    bars = data.build_dollar_bars(df, dollar_threshold=15_000.0)
    assert len(bars) == 3
    assert list(bars["n_minutes"]) == [1, 1, 2]
    assert list(bars["volume"]) == [100.0, 100.0, 200.0]


def test_build_dollar_bars_resets_each_day() -> None:
    df = pd.concat(
        [_minutes("2026-06-08", ["09:30", "09:31"]), _minutes("2026-06-09", ["09:30", "09:31"])]
    ).sort_index()
    bars = data.build_dollar_bars(df, dollar_threshold=15_000.0)
    assert len(bars) == 4  # cum resets across the overnight gap -> 2 bars each day


def test_build_dollar_bars_ohlc_coherent() -> None:
    df = _minutes("2026-06-08", ["09:30", "09:31", "09:32"])
    bars = data.build_dollar_bars(df, dollar_threshold=5_000.0)
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["high"] >= bars["close"]).all()
    assert (bars["low"] <= bars["close"]).all()


def test_empty_inputs_are_safe() -> None:
    assert data.build_dollar_bars(data._empty_minute_frame(), 1_000.0).empty
    assert data.auto_dollar_threshold(data._empty_minute_frame()) == 0.0


def test_auto_threshold_scales_with_target() -> None:
    df = _minutes("2026-06-08", ["09:30", "09:31", "09:32", "09:33"])  # total dv = 40_000
    assert data.auto_dollar_threshold(df, target_bars_per_day=2) == pytest.approx(20_000.0)
