"""Beta-residualized ``r_close`` target (market-neutral / alpha target).

Turns the model's absolute ``r_close`` target into a residual return by removing
each name's benchmark-explained component, ``r_close - beta * benchmark``, with a
trailing (look-ahead-safe) beta. Ranking the residual target isolates
cross-sectional *selection* skill from market/sector *beta* — see the
market-alpha plan for the rationale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


def trailing_beta(stock_ret: pd.Series, bench_ret: pd.Series, window: int) -> pd.Series:
    """Trailing OLS beta of ``stock_ret`` on ``bench_ret`` (look-ahead-safe).

    ``beta(t)`` is estimated from the ``window``-observation block ENDING at
    ``t-1`` (the rolling estimate is shifted one day), so it uses only
    information available before ``t`` and never peeks at the return it will help
    residualize.

    Parameters
    ----------
    stock_ret, bench_ret : pandas.Series
        Aligned daily return series (same index).
    window : int
        Trailing window length in observations.

    Returns
    -------
    pandas.Series
        Beta aligned to the input index; ``NaN`` over the warm-up (first
        ``window`` rows) and where the benchmark variance is zero.
    """
    cov = stock_ret.rolling(window).cov(bench_ret)
    var = bench_ret.rolling(window).var()
    beta = (cov / var).replace([np.inf, -np.inf], np.nan)
    return beta.shift(1)


def residualize_r_close(r_close: pd.Series, bench_ret: pd.Series, window: int) -> pd.Series:
    """Residualize ``r_close`` against a benchmark: ``r_close - beta * bench``.

    Parameters
    ----------
    r_close : pandas.Series
        The stock's daily log returns (date-indexed, trading days only).
    bench_ret : pandas.Series
        Daily benchmark log returns; reindexed onto ``r_close``'s index.
    window : int
        Trailing beta window (see :func:`trailing_beta`).

    Returns
    -------
    pandas.Series
        The residual return aligned to ``r_close.index``; ``NaN`` where the
        trailing beta or the benchmark return is undefined.
    """
    bench = bench_ret.reindex(r_close.index)
    beta = trailing_beta(r_close, bench, window)
    return r_close - beta * bench
