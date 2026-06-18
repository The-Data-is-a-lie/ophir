"""Trade-tracker report: every executed trade + date, and windowed account P&L.

Pulls the **Alpaca paper account** (the source of truth -- there is no local trade
ledger) for the list of filled trades and the daily equity curve, and writes a
regenerable ``trade-tracker/`` folder under the reports dir:

* ``README.md`` -- account snapshot, a **gains/losses** table over day / week /
  month / YTD / 1-year windows (portfolio-level: realized + unrealized), and a
  table of every trade with its date.
* ``trades.csv`` -- the raw filled-trade rows for spreadsheets.

P&L is the account *equity* change over each window (from portfolio history), so it
reflects the true paper-account return. Everything fails safe: a missing broker
method, an API hiccup, or **no fills yet** (the live run may be gated) still writes a
tracker that says so, never raising.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ophir.agent import audit

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings

_SOURCE = "Alpaca paper account (orders + portfolio history)"
_WINDOWS = ("Day", "Week", "Month", "YTD", "1 Year")


def _base_report_dir(settings: AgentSettings) -> Path:
    """Resolve the base reports directory (``report_dir`` or ``<DATA_DIR>/reports``)."""
    if settings.report_dir:
        return Path(settings.report_dir)
    from ophir.register import DATA_DIR

    return Path(DATA_DIR) / "reports"


def _repo_dir() -> Path | None:
    """The project root (parent of ``src/``) when running from a source checkout, else ``None``."""
    try:
        root = Path(__file__).resolve().parents[3]
    except IndexError:
        return None
    return root if (root / "pyproject.toml").exists() else None


def _base_at(series: list[tuple[dt.date, float]], boundary: dt.date) -> tuple[dt.date, float]:
    """The last ``(date, equity)`` on or before ``boundary``; the first point (inception)
    when ``boundary`` predates the whole series."""
    chosen = series[0]
    for point in series:
        if point[0] <= boundary:
            chosen = point
        else:
            break
    return chosen


def _pnl(base: tuple[dt.date, float], end_equity: float) -> dict[str, Any]:
    """A window P&L record from a base point to the latest equity."""
    base_date, base_equity = base
    pnl = end_equity - base_equity
    return {
        "start_date": base_date.isoformat(),
        "start_equity": base_equity,
        "end_equity": end_equity,
        "pnl": pnl,
        "pnl_pct": (pnl / base_equity if base_equity > 0 else None),
    }


def window_pnl(series: list[tuple[dt.date, float]]) -> dict[str, dict[str, Any]]:
    """Account P&L over Day / Week / Month / YTD / 1-Year from a daily equity series.

    ``series`` is ascending ``(date, equity)``. Day is measured against the prior
    session; Week/Month/1-Year are trailing 7/30/365 calendar days; YTD is measured
    from the last close of the prior year. Each base falls back to inception (the
    first point) when the window predates the account. Non-positive points (a broker's
    pre-funding zero padding) are ignored. ``{}`` for an empty series.
    """
    series = [point for point in series if point[1] > 0]
    if not series:
        return {}
    end_date, end_equity = series[-1]
    prior_session = series[-2] if len(series) >= 2 else series[0]
    return {
        "Day": _pnl(prior_session, end_equity),
        "Week": _pnl(_base_at(series, end_date - dt.timedelta(days=7)), end_equity),
        "Month": _pnl(_base_at(series, end_date - dt.timedelta(days=30)), end_equity),
        "YTD": _pnl(
            _base_at(series, dt.date(end_date.year, 1, 1) - dt.timedelta(days=1)), end_equity
        ),
        "1 Year": _pnl(_base_at(series, end_date - dt.timedelta(days=365)), end_equity),
    }


def _money(value: Any) -> str:
    """Format a dollar amount, or ``n/a``."""
    return f"${float(value):,.2f}" if value is not None else "n/a"


def _signed(value: Any) -> str:
    """Format a signed dollar P&L, or ``n/a``."""
    return f"{float(value):+,.2f}" if value is not None else "n/a"


def _pct(value: Any) -> str:
    """Format a signed percentage, or ``n/a``."""
    return f"{float(value):+.2%}" if value is not None else "n/a"


def _render_markdown(
    asof: str,
    account: Any,
    windows: dict[str, dict[str, Any]],
    trades: list[dict[str, Any]],
) -> str:
    """Render the trade-tracker markdown (header + P&L table + trades table)."""
    equity = _money(getattr(account, "equity", None)) if account is not None else "n/a"
    cash = _money(getattr(account, "cash", None)) if account is not None else "n/a"
    lines = [
        "# Ophir paper-trading tracker",
        "",
        f"- **As of:** {asof}",
        f"- **Account equity:** {equity}  |  **Cash:** {cash}",
        f"- **Source:** {_SOURCE}",
        "- Paper trading only. P&L is account equity change over each window "
        "(realized + unrealized). This populates as the agent executes fills.",
        "",
        "## Gains / losses",
        "",
        "| Window | Start date | Start equity | End equity | P&L ($) | Return |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label in _WINDOWS:
        w = windows.get(label)
        if not w:
            lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {w['start_date']} | {_money(w['start_equity'])} | "
            f"{_money(w['end_equity'])} | {_signed(w['pnl'])} | {_pct(w['pnl_pct'])} |"
        )

    lines += ["", f"## Trades ({len(trades)})", ""]
    if not trades:
        lines.append("_No fills yet._")
    else:
        lines += [
            "| Date | Symbol | Side | Qty | Fill price | Notional | Status |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
        for t in sorted(trades, key=lambda r: str(r.get("filled_at")), reverse=True):
            qty = f"{float(t['qty']):g}" if t.get("qty") is not None else "n/a"
            lines.append(
                f"| {t.get('filled_at', 'n/a')} | {t.get('symbol', '')} | {t.get('side', '')} | "
                f"{qty} | {_money(t.get('fill_price'))} | {_money(t.get('notional'))} | "
                f"{t.get('status', '')} |"
            )
    return "\n".join(lines) + "\n"


def _render_csv(trades: list[dict[str, Any]]) -> str:
    """Render the filled trades as CSV (header always present)."""
    header = "filled_at,symbol,side,qty,fill_price,notional,status"
    rows = [header]
    for t in sorted(trades, key=lambda r: str(r.get("filled_at"))):
        rows.append(
            ",".join(
                str(t.get(key, ""))
                for key in (
                    "filled_at",
                    "symbol",
                    "side",
                    "qty",
                    "fill_price",
                    "notional",
                    "status",
                )
            )
        )
    return "\n".join(rows) + "\n"


def write_trade_tracker(
    broker: Any,
    *,
    settings: AgentSettings | None = None,
    out_dir: Path | None = None,
    lookback_days: int = 370,
) -> Path:
    """Write the ``trade-tracker/`` folder (README.md + trades.csv); return its path.

    ``broker`` must expose ``filled_orders(lookback_days=...)`` and
    ``equity_series(lookback_days=...)`` (the :class:`~ophir.agent.execute.AlpacaPaperBroker`
    does); ``get_account`` is used for the header if present. Never raises -- on any
    failure it writes a tracker noting the empty state.

    Writes to ``<out_dir or report_dir>/trade-tracker``. When ``out_dir`` is not given
    (the default / live runs) it also keeps a copy at ``<project root>/trade-tracker``
    so the tracker is always visible inside the repo, not only under the reports dir.
    The Alpaca account is fetched once and rendered into every destination.
    """
    if settings is None:
        from ophir.agent.config import get_settings

        settings = get_settings()

    trades = _safe_call(lambda: broker.filled_orders(lookback_days=lookback_days)) or []
    series = _safe_call(lambda: broker.equity_series(lookback_days=lookback_days)) or []
    account = _safe_call(broker.get_account) if hasattr(broker, "get_account") else None

    windows = window_pnl(series)
    asof = series[-1][0].isoformat() if series else _today()
    readme = _render_markdown(asof, account, windows, trades)
    csv = _render_csv(trades)

    bases = [out_dir if out_dir is not None else _base_report_dir(settings)]
    if out_dir is None:  # also mirror into the project so it lives "within this area"
        repo = _repo_dir()
        if repo is not None and repo not in bases:
            bases.append(repo)

    primary = bases[0] / "trade-tracker"
    for base in bases:
        target = base / "trade-tracker"
        target.mkdir(parents=True, exist_ok=True)
        (target / "README.md").write_text(readme, encoding="utf-8", newline="\n")
        (target / "trades.csv").write_text(csv, encoding="utf-8", newline="\n")
    audit.log_event(
        "trade_tracker_written",
        dir=str(primary),
        dirs=[str(b / "trade-tracker") for b in bases],
        n_trades=len(trades),
        asof=asof,
    )
    return primary


def _safe_call(fn: Any) -> Any:
    """Call ``fn`` returning its value, or ``None`` on any error."""
    try:
        return fn()
    except Exception:
        return None


def _today() -> str:
    """Last closed NYSE session as an ISO date (fallback: today)."""
    from ophir.agent import market_calendar as cal

    try:
        return str(cal.last_closed_session().date().isoformat())
    except Exception:
        return dt.date.today().isoformat()
