"""Free FINRA biweekly consolidated short-interest signal.

The standing short *position* (settled twice a month) complements the intraday
off-exchange short *volume* the :mod:`ophir.agent.darkpool` signal already reads --
they measure different things, so this is additive, not duplicative. Source is
FINRA's free, no-auth consolidated short-interest dataset
(``https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest``),
queried per symbol over a trailing settlement window and gated to the most recent
settlement on or before ``as_of`` (point-in-time, no look-ahead).

One small JSON query per ticker, cached on disk under
``<DATA_DIR>/finra/shortinterest/``. Everything fails safe: a missing/late file,
an HTTP error, or an unknown symbol yields a neutral dict, never an exception.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
_TIMEOUT = 20.0
_USER_AGENT = "Mozilla/5.0 (compatible; ophir-shortinterest/1.0)"
_SOURCE = "FINRA consolidated short interest (biweekly)"
# Trailing days to pull so at least a couple of bi-monthly settlements land in range.
_WINDOW_DAYS = 150


def _cache_dir() -> Path:
    """Return (creating) the on-disk cache dir for short-interest query results."""
    from ophir.register import DATA_DIR

    path = Path(DATA_DIR) / "finra" / "shortinterest"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _query(symbol: str, start: str, end: str) -> list[dict[str, Any]]:
    """Query FINRA consolidated short interest for ``symbol`` in ``[start, end]``.

    Returns the parsed rows (cached per symbol+end date); ``[]`` if unavailable.
    """
    cache = _cache_dir() / f"{symbol}_{end.replace('-', '')}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (ValueError, OSError):
            return []
    body = json.dumps(
        {
            "limit": 20,
            "compareFilters": [
                {"fieldName": "symbolCode", "fieldValue": symbol, "compareType": "equal"}
            ],
            "dateRangeFilters": [
                {"fieldName": "settlementDate", "startDate": start, "endDate": end}
            ],
        }
    ).encode("utf-8")
    request = Request(
        _URL,
        data=body,
        headers={
            "User-Agent": _USER_AGENT,
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=_TIMEOUT) as resp:
            rows = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (URLError, OSError, ValueError):
        return []
    if not isinstance(rows, list):
        return []
    cache.write_text(json.dumps(rows), encoding="utf-8")
    return rows


def _num(value: Any) -> float | None:
    """Coerce to a finite float, or ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _empty(symbol: str, as_of: Any, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "settlement_date": None,
        "n_settlements": 0,
        "short_interest": None,
        "prev_short_interest": None,
        "days_to_cover": None,
        "si_change_pct": None,
        "avg_daily_volume": None,
        "source": _SOURCE,
        "note": note,
    }


def short_interest_signal(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Return the latest FINRA consolidated short-interest signal for ``symbol``.

    Picks the most recent settlement on or before ``as_of`` (default: the last
    closed NYSE session -- no look-ahead) within a trailing window.

    Returns
    -------
    dict
        ``short_interest`` (settled short shares), ``days_to_cover`` (short / avg
        daily volume), ``si_change_pct`` (vs the prior settlement), the
        ``settlement_date`` it is conditioned on, plus ``symbol`` / ``asof`` /
        ``n_settlements`` / ``source``. Missing data yields a neutral dict, never
        an exception.
    """
    from ophir.agent import market_calendar as cal

    symbol = symbol.upper().strip()
    try:
        end_ts = cal.last_closed_session(as_of)
    except ValueError:
        return _empty(symbol, as_of, "no NYSE session resolved")
    end = str(end_ts.date())
    start = str(end_ts.date() - dt.timedelta(days=_WINDOW_DAYS))

    rows = _query(symbol, start, end)
    # Keep only settlements at or before the as_of cutoff (point-in-time).
    settled = [r for r in rows if isinstance(r, dict) and str(r.get("settlementDate", "")) <= end]
    if not settled:
        return _empty(symbol, as_of, "no FINRA short interest for symbol")

    latest = max(settled, key=lambda r: str(r.get("settlementDate")))
    short = _num(latest.get("currentShortPositionQuantity"))
    prev = _num(latest.get("previousShortPositionQuantity"))
    dtc = _num(latest.get("daysToCoverQuantity"))
    change_pct = _num(latest.get("changePercent"))
    adv = _num(latest.get("averageDailyVolumeQuantity"))

    return {
        "symbol": symbol,
        "asof": end,
        "settlement_date": str(latest.get("settlementDate")),
        "n_settlements": len(settled),
        "short_interest": round(short) if short is not None else None,
        "prev_short_interest": round(prev) if prev is not None else None,
        "days_to_cover": round(dtc, 2) if dtc is not None else None,
        "si_change_pct": round(change_pct, 2) if change_pct is not None else None,
        "avg_daily_volume": round(adv) if adv is not None else None,
        "source": _SOURCE,
    }
