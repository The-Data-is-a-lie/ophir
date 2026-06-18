"""Free dark-pool activity signal from FINRA daily short-sale-volume files.

Unusual Whales' per-print dark-pool feed comes from the paid consolidated SIP tape; this
approximates the actionable part of it from FINRA's free, no-auth daily file at
``https://cdn.finra.org/equity/regsho/daily/CNMSshvol<YYYYMMDD>.txt``. That file's
``TotalVolume`` column is the **off-exchange** (TRF/ADF/ORF-reported) volume per symbol --
dark pools plus internalizers -- so it yields a daily dark-pool participation and
short-pressure signal.

This is daily **aggregate** off-exchange volume, not individual prints (no free source has
the per-print firehose). One ~few-MB file covers the whole market, so it is fetched once per
session and cached under ``<DATA_DIR>/finra/regsho/``. Everything fails safe: a missing/late
file or an unknown symbol yields a neutral result, never an exception.
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    import datetime as dt

_BASE_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"
_TIMEOUT = 15.0
# FINRA's CDN returns 403 to the default Python-urllib agent; send a browser-like one.
_USER_AGENT = "Mozilla/5.0 (compatible; ophir-darkpool/1.0)"
_SOURCE = "FINRA CNMS daily short-sale volume (off-exchange)"


class DayRow(NamedTuple):
    """One symbol's off-exchange (TRF/ADF/ORF) volume for a single session."""

    short_volume: float
    short_exempt_volume: float
    total_volume: float


def _cache_dir() -> Path:
    """Return (creating) the on-disk cache dir for downloaded FINRA daily files."""
    from ophir.register import DATA_DIR

    path = Path(DATA_DIR) / "finra" / "regsho"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _download(date_str: str) -> str | None:
    """Return the raw daily-file text for ``YYYYMMDD`` (cached); ``None`` if unavailable."""
    cache = _cache_dir() / f"CNMSshvol{date_str}.txt"
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    request = Request(_BASE_URL.format(date=date_str), headers={"User-Agent": _USER_AGENT})
    try:
        with urlopen(request, timeout=_TIMEOUT) as resp:
            text: str = resp.read().decode("utf-8", errors="replace")
    except (URLError, OSError, ValueError):
        return None
    lines = text.splitlines()
    if not lines or not lines[0].lower().startswith("date|symbol"):
        return None  # not a real data file (404 page, empty, etc.) -- do not cache
    cache.write_text(text, encoding="utf-8")
    return text


def _parse(text: str) -> dict[str, DayRow]:
    """Parse the pipe-delimited daily file into ``{SYMBOL: DayRow}`` (off-exchange volume)."""
    rows: dict[str, DayRow] = {}
    for line in text.splitlines()[1:]:  # skip the header
        parts = line.split("|")
        if len(parts) < 5:
            continue
        symbol = parts[1].strip().upper()
        if not symbol:
            continue
        try:
            rows[symbol] = DayRow(float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
    return rows


def _fetch_daily(date: dt.date) -> dict[str, DayRow]:
    """Off-exchange volume for every symbol on ``date``; ``{}`` if the file is unavailable."""
    text = _download(date.strftime("%Y%m%d"))
    return _parse(text) if text else {}


def _consolidated_volume(symbol: str, session: Any) -> float | None:
    """Consolidated total volume for ``symbol`` on the ``session`` date (ingested OHLC).

    Looks up the exact session bar in the raw parquet (no staleness guard); returns
    ``None`` if the ticker is not ingested or that date's bar is absent -- so the
    participation ratio is simply omitted rather than raising.
    """
    from ophir.agent.feed import load_daily_ohlcv

    try:
        df = load_daily_ohlcv(symbol)
    except (FileNotFoundError, OSError, ValueError):
        return None
    key = session.normalize()
    if key not in df.index:
        return None
    vol = float(df.loc[key, "volume"])
    return vol if vol > 0 else None


def _empty(symbol: str, as_of: Any, lookback: int, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "lookback": lookback,
        "n_days": 0,
        "off_exchange_volume": None,
        "off_exchange_pct": None,
        "off_exchange_short_ratio": None,
        "off_exchange_short_exempt_ratio": None,
        "off_exchange_short_ratio_zscore": None,
        "off_exchange_vol_zscore": None,
        "source": _SOURCE,
        "note": note,
    }


def _trailing_zscore(values: list[float]) -> float | None:
    """Z-score of the latest value vs the trailing baseline (>=3 points), else ``None``."""
    if len(values) < 3:
        return None
    baseline = values[:-1]
    mean = statistics.fmean(baseline)
    std = statistics.pstdev(baseline)
    return (values[-1] - mean) / std if std > 0 else 0.0


def dark_pool_signal(symbol: str, *, as_of: Any = None, lookback: int = 20) -> dict[str, Any]:
    """Return a compact dark-pool (off-exchange) activity signal for ``symbol``.

    Over the last ``lookback`` NYSE sessions ending at ``as_of`` (default the last closed
    session -- no look-ahead), reads each day's FINRA off-exchange volume and summarizes the
    most recent session against the trailing baseline.

    Returns
    -------
    dict
        ``off_exchange_volume`` (latest off-exchange shares), ``off_exchange_pct`` (that over
        the consolidated volume from the ingested OHLC, or ``None``), ``off_exchange_short_ratio``
        (off-exchange short / total), ``off_exchange_short_exempt_ratio`` (short-exempt / total),
        ``off_exchange_short_ratio_zscore`` (the short ratio's latest-vs-baseline anomaly),
        ``off_exchange_vol_zscore`` (latest vs the trailing mean, an anomaly flag), plus
        ``symbol`` / ``asof`` / ``lookback`` / ``n_days`` / ``source``. Missing data yields a
        neutral dict, never an exception.
    """
    from ophir.agent import market_calendar as cal

    symbol = symbol.upper().strip()
    sessions = cal.recent_sessions(end=as_of, count=lookback)
    if not sessions:
        return _empty(symbol, as_of, lookback, "no NYSE sessions resolved")

    entries: list[tuple[Any, DayRow]] = []
    for session in sessions:
        row = _fetch_daily(session.date()).get(symbol)
        if row is not None and row.total_volume > 0:
            entries.append((session, row))
    if not entries:
        return _empty(symbol, as_of, lookback, "no FINRA off-exchange data for symbol")

    latest_session, latest = entries[-1]
    totals = [row.total_volume for _, row in entries]

    short_ratio = latest.short_volume / latest.total_volume if latest.total_volume > 0 else None
    short_exempt_ratio = (
        latest.short_exempt_volume / latest.total_volume if latest.total_volume > 0 else None
    )

    consolidated = _consolidated_volume(symbol, latest_session)
    off_pct = latest.total_volume / consolidated if consolidated else None

    zscore = _trailing_zscore(totals)
    # Per-day off-exchange short ratio, then the latest vs its trailing baseline -- a
    # short-pressure anomaly flag riding on the same already-parsed file (no extra I/O).
    short_ratios = [
        row.short_volume / row.total_volume for _, row in entries if row.total_volume > 0
    ]
    short_ratio_z = _trailing_zscore(short_ratios) if len(short_ratios) == len(entries) else None

    return {
        "symbol": symbol,
        "asof": str(latest_session.date()),
        "lookback": lookback,
        "n_days": len(entries),
        "off_exchange_volume": round(latest.total_volume),
        "off_exchange_pct": round(off_pct, 4) if off_pct is not None else None,
        "off_exchange_short_ratio": round(short_ratio, 4) if short_ratio is not None else None,
        "off_exchange_short_exempt_ratio": (
            round(short_exempt_ratio, 4) if short_exempt_ratio is not None else None
        ),
        "off_exchange_short_ratio_zscore": (
            round(short_ratio_z, 2) if short_ratio_z is not None else None
        ),
        "off_exchange_vol_zscore": round(zscore, 2) if zscore is not None else None,
        "source": _SOURCE,
    }
