"""Free SEC EDGAR signal: corporate-insider (Form 4) + material-event flags.

The legal-insider analog to :mod:`ophir.agent.congress`, but with stronger evidence:
real issuer insiders (CEO/CFO/directors) trading their own stock at disclosed prices
with a ~2-day filing deadline. SEC EDGAR is free and **no-auth** (a User-Agent with a
contact email is required). One per-CIK submissions JSON
(``https://data.sec.gov/submissions/CIK##########.json``) indexes every recent filing,
so Form 4 (insider trades), 8-K (material events), and 13D/13G (activist stakes) all
come from a SINGLE fetch; the Form 4 trade detail is in clean per-filing XML.

**Point-in-time rule:** every field gates on the **filingDate** (public disclosure
date) ``<= as_of`` -- never the transaction date -- so there is no look-ahead. Only
open-market purchases (code ``P``) and sales (code ``S``) count as signal; grants,
option exercises, tax-withholding, and gifts are ignored. Everything fails safe to a
neutral dict; the submissions JSON is memoized per process and Form 4 trade detail is
cached per accession on disk.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
_TIMEOUT = 25.0
# SEC requires a descriptive User-Agent with a contact email; it 403s a bare urllib agent.
_USER_AGENT = "ophir-edgar/1.0 (danielgrkinich@gmail.com)"
_SOURCE = "SEC EDGAR (Form 4 insider + 8-K / 13D-G events)"

_CLUSTER_WINDOW = 30  # days for the insider cluster-buy detector
_FLOW_WINDOW = 90  # days for the signed net-dollar flow

# Process-level memos so one fetch serves the whole watchlist in a session.
_CIK_MAP: dict[str, str] = {}
_SUBMISSIONS: dict[str, dict[str, Any]] = {}


def _cache_dir() -> Path:
    """Return (creating) the on-disk cache dir for parsed Form 4 trade detail."""
    from ophir.register import DATA_DIR

    path = Path(DATA_DIR) / "edgar" / "form4"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch(url: str) -> bytes | None:
    """GET ``url`` with the SEC-required UA; ``None`` on any network error."""
    try:
        with urlopen(Request(url, headers={"User-Agent": _USER_AGENT}), timeout=_TIMEOUT) as resp:
            data: bytes = resp.read()
            return data
    except (URLError, OSError, ValueError):
        return None


def _cik_map() -> dict[str, str]:
    """Return (memoized) ``{TICKER: zero-padded-10-digit CIK}`` from SEC's ticker file."""
    if _CIK_MAP:
        return _CIK_MAP
    raw = _fetch(_TICKERS_URL)
    if not raw:
        return {}
    try:
        table = json.loads(raw)
    except ValueError:
        return {}
    for row in table.values():
        ticker = str(row.get("ticker", "")).upper().strip()
        cik = row.get("cik_str")
        if ticker and cik is not None:
            _CIK_MAP[ticker] = str(cik).zfill(10)
    return _CIK_MAP


def _submissions(cik: str) -> dict[str, Any]:
    """Return (memoized) the SEC submissions JSON for a zero-padded ``cik``; ``{}`` on failure."""
    if cik in _SUBMISSIONS:
        return _SUBMISSIONS[cik]
    raw = _fetch(_SUBMISSIONS_URL.format(cik=cik))
    data: dict[str, Any] = {}
    if raw:
        with contextlib.suppress(ValueError):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                data = parsed
    _SUBMISSIONS[cik] = data
    return data


def _parse_date(value: str | None) -> dt.date | None:
    """Parse an ISO ``YYYY-MM-DD`` EDGAR date into a ``date`` (or ``None``)."""
    if not value:
        return None
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError:
        return None


def _text(node: Any, path: str) -> str | None:
    """Return the stripped text at ``path`` under ``node``, or ``None``."""
    found = node.find(path)
    return found.text.strip() if found is not None and found.text else None


def _num(value: str | None) -> float | None:
    """Coerce to a finite float, or ``None``."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _parse_form4(cik: str, accession: str, primary_doc: str) -> list[dict[str, Any]]:
    """Return ``[{owner, action, dollars}, ...]`` of open-market trades in one Form 4.

    Cached per accession. Reads the raw ``<ownershipDocument>`` XML (the submissions
    ``primaryDocument`` is the XSL-rendered HTML; the raw XML is the same dir with the
    XSL prefix stripped). Only open-market purchases (``P``) and sales (``S``) count;
    a scanned/odd/empty filing yields ``[]``; never raises.
    """
    acc_nodash = accession.replace("-", "")
    cache = _cache_dir() / f"{acc_nodash}.json"
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (ValueError, OSError):
            return []

    trades: list[dict[str, Any]] = []
    doc = primary_doc.split("/")[-1]  # strip the xslF345X__/ render prefix -> raw XML
    raw = _fetch(_ARCHIVE_URL.format(cik=int(cik), acc=acc_nodash, doc=doc))
    if raw:
        try:
            root = ET.fromstring(raw)
        except ET.ParseError:
            root = None
        if root is not None and root.tag == "ownershipDocument":
            owner = _text(root, "reportingOwner/reportingOwnerId/rptOwnerName") or "?"
            for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
                code = _text(txn, "transactionCoding/transactionCode")
                action = {"P": "purchase", "S": "sale"}.get(code or "")
                if action is None:
                    continue
                shares = _num(_text(txn, "transactionAmounts/transactionShares/value"))
                price = _num(_text(txn, "transactionAmounts/transactionPricePerShare/value"))
                dollars = shares * price if shares is not None and price is not None else None
                trades.append({"owner": owner, "action": action, "dollars": dollars})

    with contextlib.suppress(OSError):
        cache.write_text(json.dumps(trades), encoding="utf-8")
    return trades


def _empty(symbol: str, as_of: Any, lookback: int, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "lookback": lookback,
        "n_form4": 0,
        "insider_cluster_buyers": 0,
        "insider_cluster_flag": False,
        "insider_net_dollar_flow_90d": None,
        "insider_sell_dollar_90d": None,
        "insider_net_bias": None,
        "days_since_last_insider_buy": None,
        "last_insider_type": None,
        "days_since_last_8k": None,
        "last_8k_items": None,
        "activist_stake_flag": False,
        "source": _SOURCE,
        "note": note,
    }


def _recent_filings(sub: dict[str, Any]) -> dict[str, list[Any]]:
    """Return the ``filings.recent`` columns dict (or empty lists) from a submissions JSON."""
    recent = sub.get("filings", {}).get("recent", {})
    return recent if isinstance(recent, dict) else {}


def insider_signal(symbol: str, *, as_of: Any = None, lookback_days: int = 90) -> dict[str, Any]:
    """Return a compact SEC-EDGAR insider + corporate-event signal for ``symbol``.

    Over the issuer's filings whose **disclosure date** is on or before ``as_of``
    (default the last closed NYSE session -- no look-ahead) and within ``lookback_days``,
    summarizes open-market insider trades (Form 4) plus 8-K and 13D/13G event recency.

    Returns
    -------
    dict
        ``insider_cluster_buyers`` / ``insider_cluster_flag`` (distinct insiders buying
        in 30d), ``insider_net_dollar_flow_90d`` and ``insider_sell_dollar_90d`` (signed
        actual-dollar flow, sells surfaced separately), ``insider_net_bias``,
        ``days_since_last_insider_buy`` / ``last_insider_type``, plus the
        ``days_since_last_8k`` / ``last_8k_items`` and ``activist_stake_flag`` event
        recency, and ``symbol`` / ``asof`` / ``lookback`` / ``n_form4`` / ``source``.
        Missing data yields a neutral dict, never an exception.
    """
    from ophir.agent import market_calendar as cal

    symbol = symbol.upper().strip()
    try:
        as_of_date = cal.last_closed_session(as_of).date()
    except ValueError:
        return _empty(symbol, as_of, lookback_days, "no NYSE session resolved")

    cik = _cik_map().get(symbol)
    if not cik:
        return _empty(symbol, as_of_date, lookback_days, "no SEC CIK for symbol")

    recent = _recent_filings(_submissions(cik))
    forms = recent.get("form") or []
    if not forms:
        return _empty(symbol, as_of_date, lookback_days, "no EDGAR filings for symbol")

    dates = recent.get("filingDate") or []
    accessions = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []
    items = recent.get("items") or []
    earliest = as_of_date - dt.timedelta(days=lookback_days)

    trades: list[dict[str, Any]] = []
    last_8k: dt.date | None = None
    last_8k_items: str | None = None
    activist = False
    for i, form in enumerate(forms):
        filed = _parse_date(dates[i] if i < len(dates) else None)
        if filed is None or filed > as_of_date:
            continue  # point-in-time: only filings disclosed on/before as_of
        if form == "8-K" and (last_8k is None or filed > last_8k):
            last_8k = filed
            last_8k_items = (items[i] if i < len(items) else "") or None
        if form.startswith(("SC 13D", "SC 13G")) and filed >= earliest:
            activist = True
        if form == "4" and filed >= earliest and i < len(accessions) and i < len(docs):
            for txn in _parse_form4(cik, accessions[i], docs[i]):
                trades.append({**txn, "disclosed": filed})

    cluster_cut = as_of_date - dt.timedelta(days=_CLUSTER_WINDOW)
    flow_cut = as_of_date - dt.timedelta(days=_FLOW_WINDOW)

    def _flow(action: str) -> float:
        return float(
            sum(
                t["dollars"]
                for t in trades
                if t["action"] == action and t["disclosed"] >= flow_cut and t["dollars"]
            )
        )

    cluster = {
        t["owner"] for t in trades if t["action"] == "purchase" and t["disclosed"] >= cluster_cut
    }
    buy_dollars, sell_dollars = _flow("purchase"), _flow("sale")
    buys = [t for t in trades if t["action"] == "purchase"]
    n_buys, n_sells = len(buys), sum(1 for t in trades if t["action"] == "sale")
    net_bias = (n_buys - n_sells) / (n_buys + n_sells) if (n_buys + n_sells) else None
    last_buy = max((t["disclosed"] for t in buys), default=None)
    last_trade = max((t["disclosed"] for t in trades), default=None)

    return {
        "symbol": symbol,
        "asof": str(as_of_date),
        "lookback": lookback_days,
        "n_form4": len(trades),
        "insider_cluster_buyers": len(cluster),
        "insider_cluster_flag": len(cluster) >= 2,
        "insider_net_dollar_flow_90d": round(buy_dollars - sell_dollars) if trades else None,
        "insider_sell_dollar_90d": round(sell_dollars) if trades else None,
        "insider_net_bias": round(net_bias, 3) if net_bias is not None else None,
        "days_since_last_insider_buy": (as_of_date - last_buy).days if last_buy else None,
        "last_insider_type": (
            max(trades, key=lambda t: t["disclosed"])["action"] if last_trade else None
        ),
        "days_since_last_8k": (as_of_date - last_8k).days if last_8k else None,
        "last_8k_items": last_8k_items,
        "activist_stake_flag": activist,
        "source": _SOURCE,
    }
