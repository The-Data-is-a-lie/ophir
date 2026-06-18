"""Free US House congressional-trade disclosure signal (STOCK Act PTRs).

Lawmakers must file Periodic Transaction Reports (PTRs) under the STOCK Act. The
US House Clerk publishes them free and **no-auth** as one yearly ZIP
(``.../financial-pdfs/<YEAR>FD.zip``) whose ``<YEAR>FD.xml`` indexes every filing
(member, FilingType, **FilingDate**, DocID); the trade-level detail (ticker,
buy/sell, amount bucket) lives in per-DocID PDFs (``.../ptr-pdfs/<YEAR>/<DocID>.pdf``)
that we parse with ``pypdf``.

**Point-in-time rule (load-bearing):** every feature is gated on the FilingDate
(public *disclosure* date), NEVER the transaction date -- the transaction precedes
disclosure by 30-45 days, so using it would inject look-ahead. The window index is
built once per session and cached on disk; each PDF is parsed once per DocID.

Predictive value is weak/contested in aggregate (the edge is concentrated in
cluster buys and the asymmetric *sell* signal), and the 30-45 day disclosure lag
makes this a slow "smart-money" prior, not fast alpha -- so it is a low-weight
advisory dimension. House-only. Everything fails safe to a neutral dict.

Legal note: 5 U.S.C. app. 105(c) bars *commercial* use of these disclosures;
ophir is paper-only / non-commercial -- keep it that way and do not redistribute.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

_ZIP_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{docid}.pdf"
_TIMEOUT = 30.0
_USER_AGENT = "Mozilla/5.0 (compatible; ophir-congress/1.0)"
_SOURCE = "US House Clerk financial disclosure (PTR)"

_CLUSTER_WINDOW = 30  # days for the cluster-buy detector
_FLOW_WINDOW = 60  # days for the signed net-dollar flow
_FRESH_WINDOW = 5  # days for the "new disclosure" event flag

# One transaction row in the flattened PTR text: ticker, asset-type code, type, dates, amount.
_TX_RX = re.compile(
    r"\((?P<ticker>[A-Z][A-Z.]{0,5})\)\s*"
    r"\[(?P<code>[A-Z]{1,3})\]\s*"
    r"(?P<type>[PSE])\s*"
    r"\d{2}/\d{2}/\d{4}\d{2}/\d{2}/\d{4}\s*"
    r"\$(?P<amt>[\d,]+)"
)
# STOCK Act amount buckets, keyed by their (unique) lower bound -> midpoint dollars.
_BUCKET_MID: dict[int, float] = {
    1001: 8000.0,
    15001: 32500.0,
    50001: 75000.0,
    100001: 175000.0,
    250001: 375000.0,
    500001: 750000.0,
    1000001: 3000000.0,
    5000001: 15000000.0,
    25000001: 37500000.0,
    50000001: 75000000.0,
}

# Process-level memo: (as_of_iso, lookback) -> {ticker: [transaction, ...]}.
_WINDOW_CACHE: dict[tuple[str, int], dict[str, list[dict[str, Any]]]] = {}


def _cache_dir() -> Path:
    """Return (creating) the on-disk cache dir for House disclosure files."""
    from ophir.register import DATA_DIR

    path = Path(DATA_DIR) / "congress"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(url: str) -> bytes | None:
    """GET ``url`` with a browser UA; ``None`` on any network error."""
    try:
        with urlopen(Request(url, headers={"User-Agent": _USER_AGENT}), timeout=_TIMEOUT) as resp:
            data: bytes = resp.read()
            return data
    except (URLError, OSError, ValueError):
        return None


def _index_filings(year: int) -> list[dict[str, Any]]:
    """Return the year's PTR filings from the cached ``<YEAR>FD.zip`` index.

    Each entry: ``docid`` / ``year`` / ``filing_date`` (date) / ``member`` (a
    distinctness key). Non-PTR filings and unparseable rows are skipped; any
    failure yields ``[]``.
    """
    cache = _cache_dir() / f"{year}FD.zip"
    raw = cache.read_bytes() if cache.exists() else _fetch(_ZIP_URL.format(year=year))
    if not raw:
        return []
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
        xml = archive.read(f"{year}FD.xml")
    except (zipfile.BadZipFile, KeyError, OSError):
        return []
    if not cache.exists():
        with contextlib.suppress(OSError):
            cache.write_bytes(raw)
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return []

    filings: list[dict[str, Any]] = []
    for member in root:
        if (member.findtext("FilingType") or "") != "P":
            continue
        docid = member.findtext("DocID")
        filed = _parse_date(member.findtext("FilingDate"))
        if not docid or filed is None:
            continue
        parts = (member.findtext("First"), member.findtext("Last"), member.findtext("StateDst"))
        name = " ".join(part for part in parts if part)
        filings.append({"docid": docid, "year": year, "filing_date": filed, "member": name})
    return filings


def _parse_date(value: str | None) -> dt.date | None:
    """Parse a ``MM/DD/YYYY`` House FilingDate into a ``date`` (or ``None``)."""
    if not value:
        return None
    try:
        return dt.datetime.strptime(value.strip(), "%m/%d/%Y").date()
    except ValueError:
        return None


def _parse_pdf_transactions(year: int, docid: str) -> list[dict[str, Any]]:
    """Return ``[{ticker, action, dollars}, ...]`` for one PTR (cached per DocID).

    Reads the per-DocID PDF, extracts text with ``pypdf``, and matches stock rows.
    A scanned/handwritten or ticker-less PTR simply yields ``[]``; never raises.
    """
    cache = _cache_dir() / "ptr" / f"{docid}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (ValueError, OSError):
            return []

    transactions: list[dict[str, Any]] = []
    raw = _fetch(_PDF_URL.format(year=year, docid=docid))
    if raw:
        try:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:  # pypdf raises a variety of errors on odd PDFs
            text = ""
        for match in _TX_RX.finditer(text):
            action = {"P": "purchase", "S": "sale", "E": "exchange"}.get(match.group("type"))
            if action is None:
                continue
            transactions.append(
                {
                    "ticker": match.group("ticker"),
                    "action": action,
                    "dollars": _bucket_dollars(match.group("amt")),
                }
            )

    cache.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        cache.write_text(json.dumps(transactions), encoding="utf-8")
    return transactions


def _bucket_dollars(amount: str) -> float | None:
    """Map a STOCK Act amount-range lower bound to its bucket midpoint dollars."""
    try:
        low = int(amount.replace(",", ""))
    except ValueError:
        return None
    return _BUCKET_MID.get(low, float(low))


def _window_index(as_of: dt.date, lookback: int) -> dict[str, list[dict[str, Any]]]:
    """Build (memoized) ``{ticker: [transactions]}`` disclosed in the trailing window.

    Every transaction carries its filing's disclosure date and member, gated to
    ``filing_date <= as_of`` and within ``lookback`` days -- strictly point-in-time.
    """
    key = (as_of.isoformat(), lookback)
    if key in _WINDOW_CACHE:
        return _WINDOW_CACHE[key]

    earliest = as_of - dt.timedelta(days=lookback)
    years = sorted({earliest.year, as_of.year})
    by_ticker: dict[str, list[dict[str, Any]]] = {}
    for year in years:
        for filing in _index_filings(year):
            filed: dt.date = filing["filing_date"]
            if filed > as_of or filed < earliest:
                continue
            for tx in _parse_pdf_transactions(filing["year"], filing["docid"]):
                ticker = tx["ticker"]
                by_ticker.setdefault(ticker, []).append(
                    {
                        "action": tx["action"],
                        "dollars": tx["dollars"],
                        "disclosed": filed,
                        "member": filing["member"],
                    }
                )
    _WINDOW_CACHE[key] = by_ticker
    return by_ticker


def _empty(symbol: str, as_of: Any, lookback: int, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "lookback": lookback,
        "n_disclosures": 0,
        "cluster_buy_members": 0,
        "cluster_buy_flag": False,
        "net_dollar_flow_60d": None,
        "sell_dollar_60d": None,
        "congress_net_bias": None,
        "days_since_last_disclosure": None,
        "last_disclosure_type": None,
        "new_disclosure_flag": False,
        "source": _SOURCE,
        "note": note,
    }


def congress_signal(symbol: str, *, as_of: Any = None, lookback_days: int = 90) -> dict[str, Any]:
    """Return a compact congressional-trade disclosure signal for ``symbol``.

    Over House PTRs whose **disclosure date** is on or before ``as_of`` (default
    the last closed NYSE session -- no look-ahead) and within ``lookback_days``,
    summarizes the per-ticker disclosed activity.

    Returns
    -------
    dict
        ``cluster_buy_members`` / ``cluster_buy_flag`` (distinct members buying in
        the last 30d), ``net_dollar_flow_60d`` and ``sell_dollar_60d`` (signed
        bucket-midpoint flow, sells surfaced separately as the informative leg),
        ``congress_net_bias`` (+1 net buying .. -1 net selling), and the
        ``days_since_last_disclosure`` / ``last_disclosure_type`` /
        ``new_disclosure_flag`` recency event, plus ``symbol`` / ``asof`` /
        ``lookback`` / ``n_disclosures`` / ``source``. Missing data yields a
        neutral dict, never an exception.
    """
    from ophir.agent import market_calendar as cal

    symbol = symbol.upper().strip()
    try:
        as_of_date = cal.last_closed_session(as_of).date()
    except ValueError:
        return _empty(symbol, as_of, lookback_days, "no NYSE session resolved")

    try:
        index = _window_index(as_of_date, lookback_days)
    except Exception as exc:  # belt-and-braces: the build is already fail-safe
        note = f"congress unavailable ({type(exc).__name__})"
        return _empty(symbol, as_of_date, lookback_days, note)

    rows = index.get(symbol, [])
    if not rows:
        return _empty(symbol, as_of_date, lookback_days, "no disclosed House trades for symbol")

    cluster_cut = as_of_date - dt.timedelta(days=_CLUSTER_WINDOW)
    flow_cut = as_of_date - dt.timedelta(days=_FLOW_WINDOW)
    fresh_cut = as_of_date - dt.timedelta(days=_FRESH_WINDOW)

    cluster_members = {
        r["member"] for r in rows if r["action"] == "purchase" and r["disclosed"] >= cluster_cut
    }
    buy_dollars = sum(
        r["dollars"]
        for r in rows
        if r["action"] == "purchase" and r["disclosed"] >= flow_cut and r["dollars"]
    )
    sell_dollars = sum(
        r["dollars"]
        for r in rows
        if r["action"] == "sale" and r["disclosed"] >= flow_cut and r["dollars"]
    )
    n_buys = sum(1 for r in rows if r["action"] == "purchase")
    n_sells = sum(1 for r in rows if r["action"] == "sale")
    net_bias = (n_buys - n_sells) / (n_buys + n_sells) if (n_buys + n_sells) else None

    latest = max(rows, key=lambda r: r["disclosed"])

    return {
        "symbol": symbol,
        "asof": str(as_of_date),
        "lookback": lookback_days,
        "n_disclosures": len(rows),
        "cluster_buy_members": len(cluster_members),
        "cluster_buy_flag": len(cluster_members) >= 2,
        "net_dollar_flow_60d": round(buy_dollars - sell_dollars),
        "sell_dollar_60d": round(sell_dollars),
        "congress_net_bias": round(net_bias, 3) if net_bias is not None else None,
        "days_since_last_disclosure": (as_of_date - latest["disclosed"]).days,
        "last_disclosure_type": latest["action"],
        "new_disclosure_flag": latest["disclosed"] >= fresh_cut,
        "source": _SOURCE,
    }
