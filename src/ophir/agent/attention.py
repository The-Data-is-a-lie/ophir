"""Free public-attention signal: Wikipedia pageviews (+ Reddit mentions).

A retail/public-attention dimension ophir lacks entirely. The primary source is the
**Wikimedia pageviews** REST API (free, no-auth) -- actual unrevised daily counts
archived since 2015, so it is uniquely **point-in-time / backtestable** among
attention proxies. A spike in a company's Wikipedia traffic precedes volatility and
retail flow. A secondary, snapshot-only Reddit read comes from ApeWisdom (free,
no-auth). Attention predicts *volatility/risk* more than direction, so this is a
low-weight advisory dimension.

Everything fails safe: the two components are independent and each degrades to
``None`` on error; an unresolved ticker or missing data yields a neutral dict.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_OPENSEARCH = "https://en.wikipedia.org/w/api.php?action=opensearch&search={q}&limit=1&namespace=0&format=json"
_PAGEVIEWS = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/all-agents/{title}/daily/{start}/{end}"
)
_APEWISDOM = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
_TIMEOUT = 20.0
_USER_AGENT = "ophir-attention/1.0 (danielgrkinich@gmail.com)"
_SOURCE = "Wikipedia pageviews + ApeWisdom Reddit mentions"
_WINDOW_DAYS = 35

_TITLE_CACHE: dict[str, str | None] = {}
_APEWISDOM_CACHE: dict[str, dict[str, Any]] = {}
_APEWISDOM_DONE = False


def _cache_dir() -> Path:
    """Return (creating) the on-disk cache dir for resolved Wikipedia titles."""
    from ophir.register import DATA_DIR

    path = Path(DATA_DIR) / "wiki"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_json(url: str) -> Any:
    """GET and parse JSON with the courtesy UA; ``None`` on any error."""
    try:
        with urlopen(Request(url, headers={"User-Agent": _USER_AGENT}), timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (URLError, OSError, ValueError):
        return None


def _resolve_title(symbol: str, company_name: str | None) -> str | None:
    """Resolve a ticker (+ optional company name) to a Wikipedia article key (cached)."""
    if symbol in _TITLE_CACHE:
        return _TITLE_CACHE[symbol]
    disk = _cache_dir() / "titles.json"
    titles: dict[str, str] = {}
    if disk.exists():
        with contextlib.suppress(ValueError, OSError):
            loaded = json.loads(disk.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                titles = loaded
    if symbol in titles:
        _TITLE_CACHE[symbol] = titles[symbol]
        return titles[symbol]

    query = company_name or symbol
    result = _get_json(_OPENSEARCH.format(q=quote(query)))
    title: str | None = None
    if isinstance(result, list) and len(result) >= 4 and result[3]:
        # the canonical article key is the URL path segment after /wiki/
        title = str(result[3][0]).rsplit("/wiki/", 1)[-1] or None
    _TITLE_CACHE[symbol] = title
    if title:
        titles[symbol] = title
        with contextlib.suppress(OSError):
            disk.write_text(json.dumps(titles), encoding="utf-8")
    return title


def _pageviews(title: str, as_of: dt.date) -> tuple[int | None, float | None]:
    """Return ``(latest_views, zscore)`` over the trailing window ending ``as_of``."""
    start = (as_of - dt.timedelta(days=_WINDOW_DAYS)).strftime("%Y%m%d")
    end = as_of.strftime("%Y%m%d")
    data = _get_json(_PAGEVIEWS.format(title=title, start=start, end=end))
    items = data.get("items", []) if isinstance(data, dict) else []
    series: list[float] = []
    for it in items:
        stamp = str(it.get("timestamp", ""))[:8]
        try:
            day = dt.datetime.strptime(stamp, "%Y%m%d").date()
        except ValueError:
            continue
        if day <= as_of:  # point-in-time
            views = it.get("views")
            if isinstance(views, (int, float)):
                series.append(float(views))
    if not series:
        return None, None
    latest = series[-1]
    zscore: float | None = None
    if len(series) >= 7:
        baseline = series[:-1]
        mean = statistics.fmean(baseline)
        std = statistics.pstdev(baseline)
        zscore = (latest - mean) / std if std > 0 else 0.0
    return round(latest), round(zscore, 2) if zscore is not None else None


def _apewisdom(symbol: str) -> dict[str, Any]:
    """Return ApeWisdom Reddit mentions for ``symbol`` (whole list memoized per session)."""
    global _APEWISDOM_DONE
    if not _APEWISDOM_DONE:
        _APEWISDOM_DONE = True
        data = _get_json(_APEWISDOM)
        for row in (data or {}).get("results", []):
            ticker = str(row.get("ticker", "")).upper()
            if ticker:
                _APEWISDOM_CACHE[ticker] = row
    return _APEWISDOM_CACHE.get(symbol, {})


def _empty(symbol: str, as_of: Any, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "wiki_title": None,
        "wiki_pageviews": None,
        "wiki_pageviews_zscore": None,
        "reddit_mentions": None,
        "reddit_rank": None,
        "reddit_mentions_24h_ago": None,
        "source": _SOURCE,
        "note": note,
    }


def attention_signal(
    symbol: str, *, company_name: str | None = None, as_of: Any = None
) -> dict[str, Any]:
    """Return a public-attention signal (Wikipedia pageviews + Reddit mentions) for ``symbol``.

    Wikipedia pageviews are point-in-time (gated to days ``<= as_of``); the ApeWisdom
    Reddit read is a current snapshot (not historical).

    Returns
    -------
    dict
        ``wiki_pageviews`` (latest daily) + ``wiki_pageviews_zscore`` (vs the trailing
        window -- an attention spike flag), ``reddit_mentions`` / ``reddit_rank`` /
        ``reddit_mentions_24h_ago``, plus ``symbol`` / ``asof`` / ``wiki_title`` /
        ``source``. Missing data yields neutral fields, never an exception.
    """
    from ophir.agent import market_calendar as cal

    symbol = symbol.upper().strip()
    try:
        as_of_date = cal.last_closed_session(as_of).date()
    except ValueError:
        return _empty(symbol, as_of, "no NYSE session resolved")

    title = _resolve_title(symbol, company_name)
    views, views_z = _pageviews(title, as_of_date) if title else (None, None)
    reddit = _apewisdom(symbol)

    result = _empty(symbol, as_of_date, "")
    result.pop("note", None)
    result.update(
        {
            "wiki_title": title,
            "wiki_pageviews": views,
            "wiki_pageviews_zscore": views_z,
            "reddit_mentions": reddit.get("mentions"),
            "reddit_rank": reddit.get("rank"),
            "reddit_mentions_24h_ago": reddit.get("mentions_24h_ago"),
        }
    )
    if views is None and not reddit:
        result["note"] = "no attention data for symbol"
    return result
