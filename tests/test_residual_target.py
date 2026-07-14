"""Tests for the beta-residualized r_close target (beta -> alpha lever).

Offline + CPU-only: they exercise the residual math, the target-swap in
``extract_model_data``, the streamer's warm-up handling, and the benchmark
loader / exclusion in ``build_split_handlers`` — no checkpoint, CUDA, or network.
"""

import numpy as np
import pandas as pd

from ophir.ticker.inputs import extract_model_data
from ophir.ticker.residual import residualize_r_close, trailing_beta
from ophir.ticker.streamer import StockStreamer
from ophir.train import _load_residual_benchmark, build_split_handlers

_SPLIT_KW = {
    "seq_len": 30,
    "offset": 5,
    "min_volume": 0.0,
    "train_min_year": 2019,
    "train_max_year": 2020,
    "val_min_year": 2021,
    "val_max_year": 2022,
    "use_sp500": False,
}


def test_trailing_beta_is_lookahead_safe_and_recovers_slope():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    bench = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300), index=idx)
    stock = 2.0 * bench  # exact beta of 2
    beta = trailing_beta(stock, bench, window=60)
    assert beta.iloc[:60].isna().all()  # warm-up + one-day shift => NaN
    assert np.allclose(beta.iloc[120:].to_numpy(), 2.0, atol=1e-6)


def test_residualize_removes_benchmark_component():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    bench = pd.Series(np.random.default_rng(1).normal(0, 0.01, 300), index=idx)
    resid = residualize_r_close(2.0 * bench, bench, window=60)
    assert np.allclose(resid.iloc[120:].to_numpy(), 0.0, atol=1e-6)


def _mini_window(*, with_resid: bool) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    data = {
        "r_close": np.arange(5, dtype=float),
        "upside": np.ones(5),
        "downside": np.ones(5) * 0.5,
        "trade_occured": np.ones(5, dtype=bool),
        "feature_valid": np.ones(5, dtype=bool),
    }
    if with_resid:
        data["r_close_resid"] = np.arange(5, dtype=float) - 10.0
    return pd.DataFrame(data, index=idx)


def test_extract_model_data_uses_raw_target_without_resid():
    md = extract_model_data(_mini_window(with_resid=False), response_size=2)
    assert md["feature_input"].shape[1] == 3  # r_close, upside, downside
    assert np.allclose(md["targets"][:, 0].numpy(), np.arange(5))


def test_extract_model_data_swaps_residual_into_target_only():
    md = extract_model_data(_mini_window(with_resid=True), response_size=2)
    # r_close_resid is excluded from the feature sweep (still 3 features)...
    assert md["feature_input"].shape[1] == 3
    # ...but replaces the r_close TARGET channel; upside is untouched.
    assert np.allclose(md["targets"][:, 0].numpy(), np.arange(5) - 10.0)
    assert np.allclose(md["targets"][:, 1].numpy(), np.ones(5))


def test_streamer_residual_adds_column_and_extends_warmup(make_ohlcv):
    ohlc = make_ohlcv(n_days=400, seed=7)
    bench_ret = np.log(make_ohlcv(n_days=400, seed=8)["close"]).diff()
    raw = StockStreamer(ohlc_df=ohlc, seq_len=90, offset=30)
    resid = StockStreamer(
        ohlc_df=ohlc, seq_len=90, offset=30, benchmark_returns=bench_ret, beta_window=120
    )
    assert "r_close_resid" not in raw.preprocessed_ohlc_df.columns
    assert "r_close_resid" in resid.preprocessed_ohlc_df.columns
    assert resid.size < raw.size  # the extended beta warm-up drops the earliest windows


def test_load_residual_benchmark_etf_and_universe(parquet_dir):
    base_path, _paths = parquet_dir
    etf = _load_residual_benchmark("AAA", base_path=base_path, universe=None)
    assert etf.notna().any()
    uni = _load_residual_benchmark("@universe", base_path=base_path, universe=["AAA"])
    # the equal-weight mean over a single name is that name's return series
    pd.testing.assert_series_equal(uni, etf, check_names=False)


def test_build_split_handlers_excludes_residual_benchmark(parquet_dir):
    base_path, _paths = parquet_dir
    _train, val = build_split_handlers(
        base_path=base_path,
        symbols=["AAA", "BBB", "CCC"],
        residual_benchmark="CCC",
        beta_window=5,
        **_SPLIT_KW,
    )
    assert "CCC" not in val.stocks
    assert set(val.stocks) <= {"AAA", "BBB"}
