"""Free market-wide macro / regime signal: VIX term structure + FRED + OPEX gate.

Unlike the per-ticker signals, this is one **market-wide** read shared by every
name on a given ``as_of``. It combines:

* the slope of the volatility term structure (``^VIX`` / ``^VIX3M`` from Yahoo,
  free/no-auth) -- contango = risk-on, backwardation = acute stress;
* three orthogonal FRED series (free API key): the HY credit spread, the Chicago
  Fed financial-conditions index (NFCI), and the 2s10s yield-curve slope; and
* a deterministic ``days_to_opex`` event-risk gate (the monthly 3rd-Friday equity
  options expiry), needing no key or network.

The result is memoized per ``as_of`` for the process so a whole watchlist shares a
single fetch. The FRED fields are ``None`` until ``ophir register fred-key`` is run;
everything fails safe to neutral fields, never an exception. (For look-ahead-free
backtests the FRED series should use ALFRED vintages -- not implemented here.)
"""

from __future__ import annotations

import datetime as dt
import json
import statistics
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import pandas as pd  # type: ignore[import-untyped]

_SOURCE = "Yahoo VIX term structure + FRED credit / conditions / curve"
_LOOKBACK_DAYS = 420  # ~252 trading sessions of baseline for the z-score
_RISK_ON_MAX = 0.90  # ratio at/below this = deep contango = risk-on
_RISK_OFF_MIN = 1.00  # ratio at/above this = backwardation = risk-off

# FRED series (free key): HY credit spread, financial-conditions index, 2s10s curve slope.
_FRED_URL = (
    "https://api.stlouisfed.org/fred/series/observations?series_id={series}"
    "&api_key={key}&file_type=json&observation_end={end}&sort_order=desc&limit=8"
)
_FRED_SERIES = {
    "hy_credit_spread": "BAMLH0A0HYM2",
    "nfci": "NFCI",
    "yield_curve_2s10s": "T10Y2Y",
}
_FRED_TIMEOUT = 15.0

# Process-level memo so one download serves the whole watchlist in a session.
_CACHE: dict[str, dict[str, Any]] = {}


def _empty(as_of: Any, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "asof": str(as_of) if as_of is not None else None,
        "n_days": 0,
        "vix": None,
        "vix3m": None,
        "vix_ratio": None,
        "vix_ratio_z": None,
        "regime": "unknown",
        "hy_credit_spread": None,
        "nfci": None,
        "yield_curve_2s10s": None,
        "days_to_opex": None,
        "source": _SOURCE,
        "note": note,
    }


def _regime(ratio: float) -> str:
    """Bucket the term-structure ratio into a risk-on / neutral / risk-off label."""
    if ratio >= _RISK_OFF_MIN:
        return "risk_off"
    if ratio <= _RISK_ON_MAX:
        return "risk_on"
    return "neutral"


def _fred_latest(series: str, end: str, key: str) -> float | None:
    """Latest FRED observation for ``series`` on/before ``end``; ``None`` on error."""
    url = _FRED_URL.format(series=series, key=key, end=end)
    request = Request(url, headers={"User-Agent": "ophir-macro/1.0"})
    try:
        with urlopen(request, timeout=_FRED_TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
    except (URLError, OSError, ValueError):
        return None
    for obs in data.get("observations", []):  # already sorted newest-first, gated to <= end
        try:
            return float(obs["value"])
        except (KeyError, TypeError, ValueError):
            continue  # FRED uses "." for missing days -> skip to the prior observation
    return None


def _days_to_opex(as_of: dt.date) -> int:
    """Calendar days from ``as_of`` to the next monthly equity options expiry (3rd Friday)."""

    def third_friday(year: int, month: int) -> dt.date:
        first = dt.date(year, month, 1)
        # 0=Mon..4=Fri; first Friday then +14 days = third Friday.
        first_friday = first + dt.timedelta(days=(4 - first.weekday()) % 7)
        return first_friday + dt.timedelta(days=14)

    opex = third_friday(as_of.year, as_of.month)
    if opex < as_of:
        nxt = as_of.replace(day=1) + dt.timedelta(days=32)
        opex = third_friday(nxt.year, nxt.month)
    return (opex - as_of).days


def _macro_extras(cutoff: Any) -> dict[str, Any]:
    """FRED credit/conditions/curve levels (if a key is registered) + the OPEX gate."""
    from ophir.agent.fred import get_fred_key

    end = str(cutoff.date())
    out: dict[str, Any] = dict.fromkeys(_FRED_SERIES)
    key = get_fred_key()
    if key:
        for name, series in _FRED_SERIES.items():
            value = _fred_latest(series, end, key)
            out[name] = round(value, 4) if value is not None else None
    out["days_to_opex"] = _days_to_opex(cutoff.date())
    return out


def _term_structure(cutoff: Any) -> tuple[list[float], float, float] | None:
    """Download VIX/VIX3M closes through ``cutoff``; ``(ratios, vix, vix3m)`` or ``None``.

    Split out so tests can monkeypatch the only network call.
    """
    import yfinance as yf

    start = (cutoff - pd.Timedelta(days=_LOOKBACK_DAYS)).date()
    raw = yf.download(
        ["^VIX", "^VIX3M"],
        start=str(start),
        end=str((cutoff + pd.Timedelta(days=1)).date()),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    close = raw["Close"][["^VIX", "^VIX3M"]].dropna()
    close = close[close.index <= cutoff]
    if close.empty:
        return None
    ratios = (close["^VIX"] / close["^VIX3M"]).tolist()
    return ratios, float(close["^VIX"].iloc[-1]), float(close["^VIX3M"].iloc[-1])


def macro_signal(*, as_of: Any = None) -> dict[str, Any]:
    """Return the market-wide VIX term-structure regime signal.

    Conditioned on the last completed session on or before ``as_of`` (default the
    last closed NYSE session -- no look-ahead). Market-wide: the same dict applies
    to every ticker on a given date.

    Returns
    -------
    dict
        ``vix`` / ``vix3m`` levels, ``vix_ratio`` (VIX / VIX3M), ``vix_ratio_z``
        (latest vs the trailing-year mean), a ``regime`` label; plus the FRED
        ``hy_credit_spread`` / ``nfci`` / ``yield_curve_2s10s`` (``None`` until a
        free key is registered via ``ophir register fred-key``) and the
        deterministic ``days_to_opex`` event gate, with ``asof`` / ``n_days`` /
        ``source``. Missing data yields neutral fields, never raises.
    """
    from ophir.agent import market_calendar as cal

    try:
        cutoff = cal.last_closed_session(as_of)
    except ValueError:
        return _empty(as_of, "no NYSE session resolved")
    key = str(cutoff.date())
    if key in _CACHE:
        return _CACHE[key]

    result: dict[str, Any] = {"asof": key, "source": _SOURCE}
    result.update(_vix_regime(cutoff))
    result.update(_macro_extras(cutoff))
    _CACHE[key] = result
    return result


def _vix_regime(cutoff: Any) -> dict[str, Any]:
    """The VIX term-structure regime fields (``None`` / ``"unknown"`` on failure)."""
    base: dict[str, Any] = {
        "n_days": 0,
        "vix": None,
        "vix3m": None,
        "vix_ratio": None,
        "vix_ratio_z": None,
        "regime": "unknown",
    }
    try:
        term = _term_structure(cutoff)
    except Exception:  # any network / shape / parse failure -> neutral VIX part
        return base
    if not term or term[2] <= 0:
        return base
    ratios, vix_last, vix3m_last = term
    ratio = vix_last / vix3m_last
    z: float | None = None
    if len(ratios) >= 30:
        baseline = ratios[:-1]
        mean = statistics.fmean(baseline)
        std = statistics.pstdev(baseline)
        z = (ratios[-1] - mean) / std if std > 0 else 0.0
    base.update(
        {
            "n_days": len(ratios),
            "vix": round(vix_last, 2),
            "vix3m": round(vix3m_last, 2),
            "vix_ratio": round(ratio, 4),
            "vix_ratio_z": round(z, 2) if z is not None else None,
            "regime": _regime(ratio),
        }
    )
    return base
