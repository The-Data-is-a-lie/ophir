"""Tests for ``ophir.agent.data`` ingestion (Yahoo Finance is mocked).

Network is patched out via ``yfinance.Ticker``; the parquet root is redirected
to ``tmp_path`` so nothing touches the package's ``.ophir/`` layout.
"""

import pandas as pd
import pytest

from ophir.agent.feed import latest_window_tensors, load_daily_ohlcv
from ophir.agent.ingest import ingest, ingest_many


def _yahoo_frame(make_ohlcv, n_days=500):
    """A yfinance-like frame: Title-case OHLCV, tz-aware ``Date`` index."""
    base = make_ohlcv(n_days=n_days)
    df = base.rename(columns={c: c.capitalize() for c in base.columns})
    df.insert(0, "Open", df["Close"].shift(1).fillna(df["Close"]))
    df.index = df.index.tz_localize("UTC")
    df.index.name = "Date"
    return df


def _yahoo_download_frame(make_ohlcv, symbols, n_days=120):
    """A ``yf.download(group_by="ticker")``-style MultiIndex (ticker, field) frame."""
    per = {sym: _yahoo_frame(make_ohlcv, n_days=n_days) for sym in symbols}
    return pd.concat(per, axis=1)


@pytest.fixture
def fake_yahoo(monkeypatch, make_ohlcv):
    """Patch ``yfinance.Ticker`` to return a deterministic 500-day frame."""
    frame = _yahoo_frame(make_ohlcv, n_days=500)

    class _FakeTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return frame.copy()

    monkeypatch.setattr("yfinance.Ticker", _FakeTicker)
    return frame


def test_ingest_writes_model_ready_parquet(fake_yahoo, tmp_path):
    dest = ingest("test", days=730, stocks_dir=str(tmp_path))

    assert dest == tmp_path / "symbol=TEST" / "data.parquet"
    assert dest.exists()
    out = pd.read_parquet(dest)
    assert list(out.columns) == ["utc_time", "high", "low", "close", "volume"]
    assert len(out) == 500
    assert out["utc_time"].dt.tz is None  # tz-naive, per ophir convention


def test_load_daily_ohlcv_roundtrip(fake_yahoo, tmp_path):
    ingest("TEST", days=730, stocks_dir=str(tmp_path))

    df = load_daily_ohlcv("TEST", stocks_dir=str(tmp_path))
    assert list(df.columns) == ["high", "low", "close", "volume"]
    assert df.index.name == "utc_time"
    assert df.index.is_monotonic_increasing


def test_latest_window_tensors_shapes(fake_yahoo, tmp_path):
    ingest("TEST", days=730, stocks_dir=str(tmp_path))

    md = latest_window_tensors("TEST", seq_len=30, response_size=10, stocks_dir=str(tmp_path))
    assert tuple(md["feature_input"].shape) == (30, 13)
    assert tuple(md["targets"].shape) == (30, 3)
    assert tuple(md["trade_occured"].shape) == (30,)


def test_ingest_many_skips_failures(monkeypatch, make_ohlcv, tmp_path):
    # A delisted/invalid ticker is simply absent from the batch download -> skipped.
    def _fake_download(tickers, **kwargs):
        present = [t for t in tickers if t != "BAD"]
        return _yahoo_download_frame(make_ohlcv, present, n_days=120)

    monkeypatch.setattr("yfinance.download", _fake_download)

    paths = ingest_many(["good", "BAD"], days=200, stocks_dir=str(tmp_path))
    assert set(paths) == {"GOOD"}
    assert paths["GOOD"].exists()


def test_ingest_many_batches_and_dedupes(monkeypatch, make_ohlcv, tmp_path):
    chunks = []

    def _fake_download(tickers, **kwargs):
        chunks.append(list(tickers))
        return _yahoo_download_frame(make_ohlcv, list(tickers), n_days=120)

    monkeypatch.setattr("yfinance.download", _fake_download)

    paths = ingest_many(
        ["AAA", "BBB", "CCC", "aaa"], days=200, stocks_dir=str(tmp_path), chunk_size=2
    )
    assert set(paths) == {"AAA", "BBB", "CCC"}  # duplicate "aaa" collapsed
    assert chunks == [["AAA", "BBB"], ["CCC"]]  # deduped, then chunked by 2


def test_ingest_many_normalizes_dotted_symbols(monkeypatch, make_ohlcv, tmp_path):
    seen = {}

    def _fake_download(tickers, **kwargs):
        seen["tickers"] = list(tickers)
        return _yahoo_download_frame(make_ohlcv, list(tickers), n_days=120)

    monkeypatch.setattr("yfinance.download", _fake_download)

    paths = ingest_many(["brk.b"], days=200, stocks_dir=str(tmp_path))
    assert seen["tickers"] == ["BRK-B"]  # dot -> dash, upper-cased before fetch
    assert set(paths) == {"BRK-B"}
    assert paths["BRK-B"] == tmp_path / "symbol=BRK-B" / "data.parquet"


def test_ingest_many_retries_then_succeeds(monkeypatch, make_ohlcv, tmp_path):
    calls = {"n": 0}

    def _flaky_download(tickers, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limited")
        return _yahoo_download_frame(make_ohlcv, list(tickers), n_days=120)

    monkeypatch.setattr("yfinance.download", _flaky_download)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    paths = ingest_many(["AAA"], days=200, stocks_dir=str(tmp_path))
    assert set(paths) == {"AAA"}
    assert calls["n"] == 2  # failed once, backed off, retried, succeeded


def test_ingest_many_all_fail_is_safe(monkeypatch, tmp_path):
    def _always_fail(tickers, **kwargs):
        raise RuntimeError("yahoo down")

    monkeypatch.setattr("yfinance.download", _always_fail)
    monkeypatch.setattr("time.sleep", lambda *_: None)

    paths = ingest_many(["AAA", "BBB"], days=200, stocks_dir=str(tmp_path), max_retries=2)
    assert paths == {}  # never raises; just an empty result


def test_ingest_unknown_symbol_raises(monkeypatch, tmp_path):
    class _EmptyTicker:
        def __init__(self, symbol):
            self.symbol = symbol

        def history(self, **kwargs):
            return pd.DataFrame()

    monkeypatch.setattr("yfinance.Ticker", _EmptyTicker)

    with pytest.raises(ValueError, match="No Yahoo Finance data"):
        ingest("ZZZZ", days=730, stocks_dir=str(tmp_path))
