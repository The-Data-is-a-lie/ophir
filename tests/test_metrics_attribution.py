"""Offline tests for the counterfactual LLM-vs-quant attribution (no network/GPU)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ophir.agent.metrics import counterfactual_books

_DATES = ["2026-06-01", "2026-06-08", "2026-06-15"]


def _write_daily(stocks_dir: Path, symbol: str, closes: pd.Series) -> None:
    """Write a synthetic daily OHLCV parquet in the layout load_daily_ohlcv expects."""
    df = pd.DataFrame(
        {
            "utc_time": closes.index,
            "high": closes.to_numpy() * 1.001,
            "low": closes.to_numpy() * 0.999,
            "close": closes.to_numpy(),
            "volume": np.full(len(closes), 1e6),
        }
    )
    path = stocks_dir / f"symbol={symbol}" / "data.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _decision(symbol: str, source: str, action: str, date: str) -> dict:
    return {
        "event": "decision",
        "symbol": symbol,
        "source": source,
        "action": action,
        "confidence": 1.0,
        "cum_return": 0.05,
        "timestamp": f"{date}T12:00:00Z",
    }


def _setup_prices(tmp_path: Path) -> None:
    days = pd.bdate_range("2026-05-25", "2026-06-20")
    _write_daily(tmp_path, "RISE", pd.Series(np.linspace(100.0, 130.0, len(days)), index=days))
    _write_daily(tmp_path, "FALL", pd.Series(np.linspace(100.0, 80.0, len(days)), index=days))
    _write_daily(tmp_path, "SPY", pd.Series(np.linspace(100.0, 105.0, len(days)), index=days))


def test_quant_beats_llm_when_it_picks_the_winner(tmp_path: Path) -> None:
    _setup_prices(tmp_path)
    events = pd.DataFrame(
        [_decision("RISE", "quant", "BUY", d) for d in _DATES]
        + [_decision("FALL", "ollama", "BUY", d) for d in _DATES]
    )
    books = counterfactual_books(events, stocks_dir=str(tmp_path), cost_bps=0.0)
    assert books["quant"]["equity_curve"][-1][1] > 1.0  # bought the riser -> up
    assert books["ollama"]["equity_curve"][-1][1] < 1.0  # bought the faller -> down
    assert books["quant"]["equity_curve"][-1][1] > books["ollama"]["equity_curve"][-1][1]
    assert "SPY" in books  # benchmark present


def test_cost_reduces_return(tmp_path: Path) -> None:
    _setup_prices(tmp_path)
    events = pd.DataFrame([_decision("RISE", "quant", "BUY", d) for d in _DATES])
    free = counterfactual_books(events, stocks_dir=str(tmp_path), cost_bps=0.0)
    costed = counterfactual_books(events, stocks_dir=str(tmp_path), cost_bps=50.0)
    assert costed["quant"]["equity_curve"][-1][1] < free["quant"]["equity_curve"][-1][1]


def test_empty_and_single_date_are_safe() -> None:
    assert counterfactual_books(pd.DataFrame()) == {}
    one = pd.DataFrame([_decision("RISE", "quant", "BUY", _DATES[0])])
    assert counterfactual_books(one) == {}  # need >=2 run dates
