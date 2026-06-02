"""Model-driven trading strategy for Ophir.

Turns the trained base checkpoint's per-ticker return predictions into a
top-K ranking and a concrete set of buy/sell decisions, plus a MASSIVE-backed
data-refresh step so inference runs on current prices.

The heavy pieces (data refresh, batch inference) need a live MASSIVE key, a
CUDA GPU, and a trained checkpoint, so they are exercised only through
``ophir trade rebalance``. The pure decision helpers
(:func:`rank_top_k`, :func:`compute_rebalance`) are unit-tested.

Notes
-----
``build_forecast_input`` constructs a genuinely forward-looking window: the
trailing :data:`RESPONSE_SIZE` rows are zeroed future placeholders, so the
model's predictions for that region cannot leak from their own features (the
``r_close`` feature is also a target). Confirm this is sound for the loaded
checkpoint before trusting signals -- see the validation gate in the plan.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import pandas as pd  # type: ignore[import-untyped]
import torch

from ophir import register
from ophir.ticker import (
    StockHanlder,
    StockStreamer,
    extract_model_data,
    get_splits,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from ophir.training_models import LightningOHLCPredictor

SEQ_LEN = 365
RESPONSE_SIZE = 90
HISTORY_SIZE = SEQ_LEN - RESPONSE_SIZE
_FEATURE_PAD = 0.0


def _stocks_base_path() -> str:
    """Return the per-symbol parquet partition root used by the pipeline."""
    return os.path.join(register.DATA_DIR, "days", "stocks")


def refresh_universe(symbols: Iterable[str], lookback_days: int = 800) -> list[str]:
    """Download recent daily bars from MASSIVE and write the parquet layout.

    For each symbol, fetches ``lookback_days`` of daily aggregates and writes
    ``<DATA_DIR>/days/stocks/symbol=<SYMBOL>/data.parquet`` with the
    ``utc_time/high/low/close/volume`` columns the reader expects.

    Parameters
    ----------
    symbols : Iterable[str]
        Ticker symbols to refresh.
    lookback_days : int, optional
        Calendar days of history to request. Defaults to ``800`` (comfortably
        more than the model's ``SEQ_LEN`` calendar window plus warmup).

    Returns
    -------
    list[str]
        Symbols whose data was successfully written.
    """
    client = register.get_massive_client()
    base_path = _stocks_base_path()
    to_date = date.today()
    from_date = to_date - timedelta(days=lookback_days)

    refreshed: list[str] = []
    for symbol in symbols:
        try:
            aggs = client.get_aggs(symbol, 1, "day", from_date, to_date, adjusted=True)
            rows = [
                {
                    "utc_time": pd.to_datetime(agg.timestamp, unit="ms"),
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume,
                }
                for agg in aggs
            ]
            if not rows:
                continue
            df = pd.DataFrame(rows)
            partition = os.path.join(base_path, f"symbol={symbol}")
            os.makedirs(partition, exist_ok=True)
            df.to_parquet(os.path.join(partition, "data.parquet"))
            refreshed.append(symbol)
        except Exception as exc:  # one bad symbol must not abort the whole run
            print(f"refresh failed for {symbol}: {exc}")
    return refreshed


def build_handler(symbols: Iterable[str]) -> StockHanlder:
    """Build a :class:`~ophir.ticker.StockHanlder` over the given universe.

    Mirrors the dashboard's handler configuration and restricts it to
    ``symbols`` that actually have parquet data.

    Parameters
    ----------
    symbols : Iterable[str]
        Universe to keep (e.g. the S&P 500 constituents).

    Returns
    -------
    StockHanlder
        A streamer-producing handler filtered to the available universe.
    """
    symbols = list(symbols)
    splits = get_splits(symbols)
    handler = StockHanlder(
        seq_len=SEQ_LEN,
        base_path=_stocks_base_path(),
        return_stock_id=False,
        return_streamer=True,
        stock_splits=splits,
        offset=RESPONSE_SIZE,
        min_year=2024,
        min_volume=1000,
        winsorize_returns=True,
    )
    handler.keep_stocks(symbols)
    return handler


def build_forecast_input(streamer: StockStreamer) -> dict[str, Any] | None:
    """Build a forward-looking model input from a streamer's history.

    Takes the trailing :data:`HISTORY_SIZE` real feature rows and appends
    :data:`RESPONSE_SIZE` zeroed future placeholder rows (``trade_occured`` is
    ``False``), so the model's predictions for the response region are a genuine
    forecast rather than a reconstruction of known days.

    Parameters
    ----------
    streamer : StockStreamer
        A streamer whose ``preprocessed_ohlc_df`` holds the 13 features plus
        the ``trade_occured`` mask.

    Returns
    -------
    dict or None
        The kwargs for :class:`~ophir.model_data.OHLCMulitClassPredictorInput`,
        or ``None`` if the stock lacks enough history.
    """
    if not hasattr(streamer, "preprocessed_ohlc_df"):
        return None
    history = streamer.preprocessed_ohlc_df
    if len(history) < HISTORY_SIZE:
        return None

    history = history.iloc[-HISTORY_SIZE:].reset_index(drop=True)
    future = pd.DataFrame(
        _FEATURE_PAD,
        index=range(RESPONSE_SIZE),
        columns=history.columns,
    )
    future["trade_occured"] = False
    future = future.astype(history.dtypes)
    window = pd.concat([history, future], ignore_index=True)
    return extract_model_data(window, response_size=RESPONSE_SIZE)


def score_universe(
    model: LightningOHLCPredictor,
    handler: StockHanlder,
    horizon: int = 5,
) -> dict[str, float]:
    """Score every available symbol by predicted near-term return.

    Runs the model on each symbol's forward window and scores it by the
    cumulative predicted ``r_close`` (log return) over the first ``horizon``
    forecast days. Symbols without enough history or that error out are skipped.

    Parameters
    ----------
    model : LightningOHLCPredictor
        A loaded, CUDA-resident model in eval mode.
    handler : StockHanlder
        A streamer-producing handler filtered to the universe.
    horizon : int, optional
        Number of forecast days to accumulate into the score. Defaults to ``5``.

    Returns
    -------
    dict[str, float]
        Mapping of symbol to score; higher is more bullish.
    """
    scores: dict[str, float] = {}
    skipped = 0
    for symbol in handler.stocks:
        try:
            streamer = handler[symbol]
            assert isinstance(streamer, StockStreamer)
            model_input = build_forecast_input(streamer)
            if model_input is None:
                skipped += 1
                continue
            with torch.no_grad():
                output = model(model_input)
            scores[symbol] = float(output.predicted_r_close[0, :horizon].sum().item())
        except Exception as exc:  # skip bad symbols, keep scoring the rest
            print(f"scoring failed for {symbol}: {exc}")
            skipped += 1
    print(f"scored {len(scores)} symbols, skipped {skipped}")
    return scores


def rank_top_k(scores: Mapping[str, float], k: int, min_score: float = 0.0) -> list[str]:
    """Return the top ``k`` symbols by score above ``min_score``.

    Parameters
    ----------
    scores : Mapping[str, float]
        Symbol-to-score mapping (e.g. from :func:`score_universe`).
    k : int
        Maximum number of symbols to return.
    min_score : float, optional
        Exclude symbols at or below this score. Defaults to ``0.0`` (only
        positive-return predictions qualify).

    Returns
    -------
    list[str]
        Up to ``k`` symbols, highest score first.
    """
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [symbol for symbol, score in ranked if score > min_score][:k]


def compute_rebalance(
    current: Sequence[str],
    targets: Sequence[str],
    buying_power: float,
    budget_frac: float = 0.95,
) -> tuple[list[str], list[tuple[str, float]]]:
    """Reconcile held positions against target symbols into orders.

    Positions not in ``targets`` are sold in full; target symbols not already
    held are bought, splitting ``buying_power * budget_frac`` equally by
    notional across the new names.

    Parameters
    ----------
    current : Sequence[str]
        Symbols currently held.
    targets : Sequence[str]
        Desired symbols to hold (e.g. from :func:`rank_top_k`).
    buying_power : float
        Available buying power, in dollars.
    budget_frac : float, optional
        Fraction of buying power to deploy across new buys. Defaults to
        ``0.95``.

    Returns
    -------
    tuple[list[str], list[tuple[str, float]]]
        ``(sells, buys)`` where ``buys`` pairs each symbol with a notional
        dollar amount.
    """
    target_set = set(targets)
    held_set = set(current)
    sells = [symbol for symbol in current if symbol not in target_set]
    new = [symbol for symbol in targets if symbol not in held_set]
    if not new:
        return sells, []
    notional = buying_power * budget_frac / len(new)
    buys = [(symbol, notional) for symbol in new]
    return sells, buys
