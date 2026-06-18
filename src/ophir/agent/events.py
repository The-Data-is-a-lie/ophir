"""Free earnings-calendar event gate from Yahoo Finance.

The single highest-leverage event read for a daily trader: how many sessions
until the next earnings report. Holding into a binary earnings print is a
distinct risk the price/vol features do not capture, so the manager/risk gate can
use ``days_to_next_earnings`` to avoid (or down-size) a name reporting imminently,
and the post-earnings window is where the documented drift (PEAD) lives.

Source is ``yfinance`` (``yf.Ticker(sym).calendar``), free and no-auth. NOTE: the
earnings *schedule* is forward-looking and gets revised, so this is a live trading
gate, not a look-ahead-free backtest feature. Fails safe to a neutral dict.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

_SOURCE = "Yahoo Finance earnings calendar"


def _empty(symbol: str, as_of: Any, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "next_earnings_date": None,
        "days_to_next_earnings": None,
        "last_earnings_date": None,
        "days_since_last_earnings": None,
        "source": _SOURCE,
        "note": note,
    }


def _to_date(value: Any) -> dt.date | None:
    """Coerce a yfinance calendar entry into a ``date`` (or ``None``)."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def _fetch_calendar(symbol: str) -> Any:
    """Return yfinance's ``.calendar`` for ``symbol`` (split out for monkeypatching)."""
    import yfinance as yf

    return yf.Ticker(symbol).calendar


def _extract_dates(calendar: Any) -> list[dt.date]:
    """Pull earnings dates from yfinance's dict- or frame-shaped ``.calendar``."""
    raw: Any = None
    if isinstance(calendar, dict):
        raw = calendar.get("Earnings Date")
    elif hasattr(calendar, "loc"):  # legacy DataFrame shape
        try:
            raw = calendar.loc["Earnings Date"].tolist()
        except (KeyError, AttributeError):
            raw = None
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    return [d for d in (_to_date(v) for v in items) if d is not None]


def earnings_signal(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Return the next/last earnings dates and day-distances for ``symbol``.

    Conditioned on the last completed session on or before ``as_of`` (default the
    last closed NYSE session). ``days_to_next_earnings`` is the event-risk gate.

    Returns
    -------
    dict
        ``next_earnings_date`` / ``days_to_next_earnings`` (the upcoming print),
        ``last_earnings_date`` / ``days_since_last_earnings`` (the PEAD window),
        plus ``symbol`` / ``asof`` / ``source``. Missing data yields a neutral
        dict, never an exception.
    """
    from ophir.agent import market_calendar as cal

    symbol = symbol.upper().strip()
    try:
        asof_date = cal.last_closed_session(as_of).date()
    except ValueError:
        return _empty(symbol, as_of, "no NYSE session resolved")

    try:
        dates = _extract_dates(_fetch_calendar(symbol))
    except Exception as exc:  # any network / shape failure -> neutral
        return _empty(symbol, asof_date, f"earnings unavailable ({type(exc).__name__})")

    if not dates:
        return _empty(symbol, asof_date, "no earnings dates for symbol")

    upcoming = sorted(d for d in dates if d >= asof_date)
    past = sorted(d for d in dates if d < asof_date)
    next_date = upcoming[0] if upcoming else None
    last_date = past[-1] if past else None

    return {
        "symbol": symbol,
        "asof": str(asof_date),
        "next_earnings_date": str(next_date) if next_date else None,
        "days_to_next_earnings": (next_date - asof_date).days if next_date else None,
        "last_earnings_date": str(last_date) if last_date else None,
        "days_since_last_earnings": (asof_date - last_date).days if last_date else None,
        "source": _SOURCE,
    }
