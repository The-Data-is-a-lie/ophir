"""Tests for the FINRA dark-pool signal (offline; network + calendar monkeypatched)."""

import datetime as dt

import pandas as pd

from ophir.agent import darkpool, market_calendar


def _ts(s):
    return pd.Timestamp(s)


def test_parse_handles_pipe_header_and_fractions():
    text = (
        "Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        "20260616|AAPL|5841836.036110|120098|12926062.774208|B,Q,N\n"
        "20260616|MSFT|1000000|50|2000000|B,Q,N\n"
        "garbage line without pipes\n"
    )
    rows = darkpool._parse(text)
    assert set(rows) == {"AAPL", "MSFT"}
    assert rows["AAPL"].total_volume == 12926062.774208
    assert rows["AAPL"].short_volume == 5841836.03611
    assert rows["MSFT"].total_volume == 2000000.0


def test_signal_computes_metrics(monkeypatch):
    sessions = [
        _ts("2026-06-10"),
        _ts("2026-06-11"),
        _ts("2026-06-12"),
        _ts("2026-06-15"),
        _ts("2026-06-16"),
    ]
    monkeypatch.setattr(market_calendar, "recent_sessions", lambda end=None, count=20: sessions)
    data = {
        dt.date(2026, 6, 10): {"AAPL": darkpool.DayRow(3.6e6, 1e5, 9.0e6)},
        dt.date(2026, 6, 11): {"AAPL": darkpool.DayRow(4.0e6, 1e5, 1.0e7)},
        dt.date(2026, 6, 12): {"AAPL": darkpool.DayRow(4.4e6, 1e5, 1.1e7)},
        dt.date(2026, 6, 15): {"AAPL": darkpool.DayRow(4.0e6, 1e5, 1.0e7)},
        dt.date(2026, 6, 16): {"AAPL": darkpool.DayRow(6.0e6, 1.2e5, 1.5e7)},  # off-exchange spike
    }
    monkeypatch.setattr(darkpool, "_fetch_daily", lambda d: data.get(d, {}))
    monkeypatch.setattr(darkpool, "_consolidated_volume", lambda sym, sess: 3.0e7)

    sig = darkpool.dark_pool_signal("aapl", lookback=5)
    assert sig["symbol"] == "AAPL"
    assert sig["asof"] == "2026-06-16"
    assert sig["n_days"] == 5
    assert sig["off_exchange_volume"] == 15000000
    assert sig["off_exchange_short_ratio"] == 0.4  # 6.0e6 / 1.5e7
    assert sig["off_exchange_short_exempt_ratio"] == 0.008  # 1.2e5 / 1.5e7 (free-lunch field)
    assert sig["off_exchange_pct"] == 0.5  # 1.5e7 / 3.0e7
    assert sig["off_exchange_vol_zscore"] > 3  # the latest day is a clear anomaly


def test_signal_pct_none_without_ohlc(monkeypatch):
    monkeypatch.setattr(
        market_calendar, "recent_sessions", lambda end=None, count=20: [_ts("2026-06-16")]
    )
    monkeypatch.setattr(
        darkpool, "_fetch_daily", lambda d: {"AAPL": darkpool.DayRow(4e6, 1e5, 1e7)}
    )
    monkeypatch.setattr(darkpool, "_consolidated_volume", lambda s, sess: None)
    sig = darkpool.dark_pool_signal("AAPL", lookback=1)
    assert sig["off_exchange_volume"] == 10000000
    assert sig["off_exchange_pct"] is None
    assert sig["off_exchange_vol_zscore"] is None  # < 3 days of data


def test_signal_failsafe_no_data(monkeypatch):
    monkeypatch.setattr(
        market_calendar, "recent_sessions", lambda end=None, count=20: [_ts("2026-06-16")]
    )
    monkeypatch.setattr(darkpool, "_fetch_daily", lambda d: {})  # file unavailable for every day
    sig = darkpool.dark_pool_signal("ZZZZ", lookback=5)
    assert sig["n_days"] == 0
    assert sig["off_exchange_volume"] is None
    assert "note" in sig
