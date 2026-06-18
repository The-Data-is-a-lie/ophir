"""Free per-name options / implied-volatility-surface signal from CBOE.

ophir's only volatility input is the market-wide VIX term structure; this adds a
**per-name** options read from CBOE's free, no-auth delayed-quotes JSON
(``https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json``) -- one
file per ticker with the spot, a ready 30-day IV (``iv30``), and every contract's
``iv`` / ``open_interest`` / ``volume`` / greeks. From it we derive the
**put/call ratio** (positioning), the **25-delta IV skew** (downside-fear pricing;
the steepest-skew names underperform -- Xing-Zhang-Zhao), and the **IV level**.

**Point-in-time catch (important):** there is **no free historical option chain**,
so this is a LIVE-loop signal, not backtestable. We only ever use a snapshot we
actually collected: the file is cached per session under ``<DATA_DIR>/cboe/<date>/``
and a *past* ``as_of`` with no cached snapshot yields a neutral dict (never a live
fetch dated to the past). ``iv_rank``/``iv_percentile`` would need self-collected
history and are deliberately omitted until enough days accrue. Fails safe always.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol}.json"
_TIMEOUT = 30.0
_USER_AGENT = "Mozilla/5.0 (compatible; ophir-options/1.0)"
_SOURCE = "CBOE delayed option quotes (per-name IV surface)"
# Contract symbol tail: 6-digit expiry (YYMMDD), C/P, 8-digit strike (x1000).
_OPT_RE = re.compile(r"(\d{6})([CP])(\d{8})$")
# Front-ish expiries for the skew read (skip 0DTE noise and far LEAPS).
_SKEW_MIN_DAYS = 7
_SKEW_MAX_DAYS = 60


def _cache_dir(as_of_date: str) -> Path:
    """Return (creating) the per-date on-disk cache dir for CBOE snapshots."""
    from ophir.register import DATA_DIR

    path = Path(DATA_DIR) / "cboe" / as_of_date
    path.mkdir(parents=True, exist_ok=True)
    return path


def _load_chain(symbol: str, as_of_date: str, *, live_ok: bool) -> dict[str, Any] | None:
    """Return the CBOE chain dict for ``symbol`` on ``as_of_date``.

    Uses the cached snapshot if present; otherwise fetches live ONLY when
    ``live_ok`` (the as_of is the current session). ``None`` if unavailable.
    """
    cache = _cache_dir(as_of_date) / f"{symbol}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
        except (ValueError, OSError):
            return None
    if not live_ok:
        return None
    request = Request(_URL.format(symbol=symbol), headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (URLError, OSError, ValueError):
        return None
    if not isinstance(payload, dict) or "data" not in payload:
        return None
    with contextlib.suppress(OSError):
        cache.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _num(value: Any) -> float | None:
    """Coerce to a finite float, or ``None``."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _skew_25d(contracts: list[dict[str, Any]], ref: dt.date) -> float | None:
    """25-delta IV skew: front-expiry put IV at delta -0.25 minus call IV at +0.25."""
    best_put: tuple[float, float] | None = None  # (|delta+0.25| distance, iv)
    best_call: tuple[float, float] | None = None
    for c in contracts:
        match = _OPT_RE.search(str(c.get("option", "")))
        iv = _num(c.get("iv"))
        delta = _num(c.get("delta"))
        if match is None or not iv or delta is None:
            continue
        try:
            expiry = dt.datetime.strptime(match.group(1), "%y%m%d").date()
        except ValueError:
            continue
        days = (expiry - ref).days
        if days < _SKEW_MIN_DAYS or days > _SKEW_MAX_DAYS:
            continue
        if match.group(2) == "P":
            dist = abs(delta - (-0.25))
            if best_put is None or dist < best_put[0]:
                best_put = (dist, iv)
        else:
            dist = abs(delta - 0.25)
            if best_call is None or dist < best_call[0]:
                best_call = (dist, iv)
    if best_put is None or best_call is None:
        return None
    return best_put[1] - best_call[1]


def _empty(symbol: str, as_of: Any, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "spot": None,
        "iv30": None,
        "put_call_ratio_volume": None,
        "put_call_ratio_oi": None,
        "iv_skew_25d": None,
        "volume_oi_ratio": None,
        "n_contracts": 0,
        "source": _SOURCE,
        "note": note,
    }


def options_flow_signal(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Return a compact per-name options / IV-surface signal for ``symbol``.

    LIVE-loop signal (no free option history): uses a cached CBOE snapshot for the
    ``as_of`` session, or fetches live only when ``as_of`` is the current session.

    Returns
    -------
    dict
        ``put_call_ratio_volume`` / ``put_call_ratio_oi`` (positioning),
        ``iv_skew_25d`` (front-expiry 25-delta put-minus-call IV; positive = downside
        fear), ``iv30`` (CBOE 30-day IV), ``spot``, ``volume_oi_ratio`` (an activity
        read), plus ``symbol`` / ``asof`` / ``n_contracts`` / ``source``. Missing data
        yields a neutral dict, never an exception.
    """
    from ophir.agent import market_calendar as cal

    symbol = symbol.upper().strip()
    try:
        as_of_date = cal.last_closed_session(as_of).date()
        live_ok = as_of is None or as_of_date == cal.last_closed_session(None).date()
    except ValueError:
        return _empty(symbol, as_of, "no NYSE session resolved")

    key = str(as_of_date)
    payload = _load_chain(symbol, key, live_ok=live_ok)
    if payload is None:
        return _empty(symbol, key, "no CBOE option snapshot for session")
    data = payload.get("data", {})
    contracts = data.get("options", []) if isinstance(data, dict) else []
    if not contracts:
        return _empty(symbol, key, "no option contracts for symbol")

    call_vol = put_vol = call_oi = put_oi = 0.0
    total_vol = total_oi = 0.0
    for c in contracts:
        match = _OPT_RE.search(str(c.get("option", "")))
        if match is None:
            continue
        vol = _num(c.get("volume")) or 0.0
        oi = _num(c.get("open_interest")) or 0.0
        total_vol += vol
        total_oi += oi
        if match.group(2) == "P":
            put_vol += vol
            put_oi += oi
        else:
            call_vol += vol
            call_oi += oi

    return {
        "symbol": symbol,
        "asof": key,
        "spot": round(_num(data.get("current_price")) or 0.0, 2) or None,
        "iv30": round(_num(data.get("iv30")) or 0.0, 2) or None,
        "put_call_ratio_volume": round(put_vol / call_vol, 3) if call_vol > 0 else None,
        "put_call_ratio_oi": round(put_oi / call_oi, 3) if call_oi > 0 else None,
        "iv_skew_25d": round(s, 4) if (s := _skew_25d(contracts, as_of_date)) is not None else None,
        "volume_oi_ratio": round(total_vol / total_oi, 4) if total_oi > 0 else None,
        "n_contracts": len(contracts),
        "source": _SOURCE,
    }
