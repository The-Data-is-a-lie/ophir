"""Offline tests for the trade-tracker (stub broker; no network)."""

import datetime as dt
from types import SimpleNamespace

from ophir.agent import trades
from ophir.agent.execute import _enum_str, _opt_float


def _linear_series(start: dt.date, n: int, step: float = 50.0, base: float = 100_000.0):
    """An ascending ``(date, equity)`` series, one point per calendar day."""
    return [(start + dt.timedelta(days=i), base + i * step) for i in range(n)]


def test_window_pnl_boundaries():
    # 400 daily points ending 2026-06-17 at equity 119,950 (+50/day).
    series = _linear_series(dt.date(2025, 5, 14), 400)
    w = trades.window_pnl(series)
    assert w["Day"]["start_date"] == "2026-06-16" and w["Day"]["pnl"] == 50.0
    assert w["Week"]["start_date"] == "2026-06-10" and w["Week"]["pnl"] == 350.0  # 7d
    assert w["Month"]["start_date"] == "2026-05-18" and w["Month"]["pnl"] == 1500.0  # 30d
    assert w["YTD"]["start_date"] == "2025-12-31" and w["YTD"]["pnl"] == 8400.0  # prior year-end
    assert w["1 Year"]["start_date"] == "2025-06-17" and w["1 Year"]["pnl"] == 18250.0  # 365d
    assert w["Day"]["pnl_pct"] == 50.0 / w["Day"]["start_equity"]


def test_window_pnl_inception_clamp():
    # Only 5 days of history -> every window but Day clamps to inception (first point).
    series = _linear_series(dt.date(2026, 6, 13), 5)
    w = trades.window_pnl(series)
    assert w["1 Year"]["start_date"] == "2026-06-13"  # inception, not 365d ago
    assert w["Month"]["start_date"] == "2026-06-13"
    assert w["Day"]["start_date"] == "2026-06-16"  # prior session still works


def test_window_pnl_empty():
    assert trades.window_pnl([]) == {}


def test_window_pnl_ignores_prefunding_zero_padding():
    # Alpaca portfolio history pads pre-funding days with equity 0; those must not
    # become the window base (which would report a bogus +full-equity gain).
    padding = [(dt.date(2025, 6, 1) + dt.timedelta(days=i), 0.0) for i in range(20)]
    funded = _linear_series(dt.date(2026, 6, 13), 5)  # account starts 2026-06-13
    w = trades.window_pnl(padding + funded)
    assert w["1 Year"]["start_equity"] > 0  # inception is the first FUNDED point
    assert w["1 Year"]["start_date"] == "2026-06-13"
    assert w["Month"]["start_equity"] == 100_000.0


class _StubBroker:
    def __init__(self, trades_, series, *, equity=119_950.0, cash=5_000.0, has_account=True):
        self._trades = trades_
        self._series = series
        self._equity = equity
        self._cash = cash
        self._has_account = has_account

    def filled_orders(self, *, lookback_days=370):
        return self._trades

    def equity_series(self, *, lookback_days=370):
        return self._series

    def get_account(self):
        if not self._has_account:
            return None
        return SimpleNamespace(equity=self._equity, cash=self._cash)


def test_write_trade_tracker_writes_folder(tmp_path):
    fills = [
        {
            "symbol": "AAPL",
            "side": "buy",
            "qty": 10,
            "fill_price": 200.0,
            "notional": 2000.0,
            "filled_at": "2026-06-15",
            "status": "filled",
        },
        {
            "symbol": "NVDA",
            "side": "sell",
            "qty": 5,
            "fill_price": 210.0,
            "notional": 1050.0,
            "filled_at": "2026-06-17",
            "status": "filled",
        },
    ]
    series = _linear_series(dt.date(2025, 5, 14), 400)
    broker = _StubBroker(fills, series)

    path = trades.write_trade_tracker(broker, out_dir=tmp_path)
    assert path == tmp_path / "trade-tracker"
    readme = (path / "README.md").read_text(encoding="utf-8")
    csv = (path / "trades.csv").read_text(encoding="utf-8")
    # P&L table + both trades present; newest-first in the markdown
    assert "## Gains / losses" in readme and "| Day |" in readme and "| 1 Year |" in readme
    assert "AAPL" in readme and "NVDA" in readme
    trades_md = readme.split("## Trades")[1]  # scope to the trades table (gains table reuses dates)
    assert trades_md.index("2026-06-17") < trades_md.index("2026-06-15")  # newest first
    assert "Trades (2)" in readme
    assert csv.splitlines()[0] == "filled_at,symbol,side,qty,fill_price,notional,status"
    assert len(csv.strip().splitlines()) == 3  # header + 2 fills


def test_write_trade_tracker_empty_is_failsafe(tmp_path):
    broker = _StubBroker([], [])
    path = trades.write_trade_tracker(broker, out_dir=tmp_path)
    readme = (path / "README.md").read_text(encoding="utf-8")
    assert "_No fills yet._" in readme
    assert "| Day | n/a |" in readme  # empty windows render n/a, never raise
    assert (path / "trades.csv").read_text(encoding="utf-8").strip() == (
        "filled_at,symbol,side,qty,fill_price,notional,status"
    )


def test_write_trade_tracker_mirrors_into_repo(monkeypatch, tmp_path):
    # Default runs (out_dir=None) write to the reports base AND a project-local copy.
    primary_base = tmp_path / "reports"
    repo_base = tmp_path / "repo"
    monkeypatch.setattr(trades, "_base_report_dir", lambda settings: primary_base)
    monkeypatch.setattr(trades, "_repo_dir", lambda: repo_base)
    broker = _StubBroker([], _linear_series(dt.date(2026, 6, 13), 5))

    primary = trades.write_trade_tracker(broker)  # no out_dir -> dual write
    assert primary == primary_base / "trade-tracker"
    assert (primary_base / "trade-tracker" / "README.md").exists()
    assert (repo_base / "trade-tracker" / "README.md").exists()  # in-repo copy too


def test_write_trade_tracker_splices_live_equity_midsession(tmp_path):
    # Mid-session: Alpaca's daily series lags (tail dated before today) but the live
    # account equity is current -> splice it so the As-of date, every window's End
    # equity, and the header all reflect the live mark, flagged as intraday.
    today = dt.date.today()
    series = [(today - dt.timedelta(days=4), 99_680.82)]
    broker = _StubBroker([], series, equity=99_806.03, cash=84_757.58)

    path = trades.write_trade_tracker(broker, out_dir=tmp_path)
    readme = (path / "README.md").read_text(encoding="utf-8")

    assert f"- **As of:** {today.isoformat()}" in readme
    assert "(live intraday mark" in readme
    assert "not a settled end-of-day snapshot" in readme
    assert "$99,806.03" in readme  # live header equity
    for label in ("Day", "Week", "Month", "YTD", "1 Year"):
        row = next(ln for ln in readme.splitlines() if ln.startswith(f"| {label} |"))
        assert "$99,806.03" in row  # End equity == live mark
        assert "$99,680.82" in row  # Start equity == settled base


def test_write_trade_tracker_no_live_equity_keeps_settled(tmp_path):
    # No live equity (broker account unavailable): no splice, no live label, and the
    # As-of stays the settled series tail -- identical to the pre-splice behavior.
    today = dt.date.today()
    tail = today - dt.timedelta(days=3)
    series = [(today - dt.timedelta(days=10), 100_000.0), (tail, 99_900.0)]
    broker = _StubBroker([], series, has_account=False)

    path = trades.write_trade_tracker(broker, out_dir=tmp_path)
    readme = (path / "README.md").read_text(encoding="utf-8")

    assert f"- **As of:** {tail.isoformat()}" in readme
    assert "live intraday mark" not in readme
    assert "not a settled end-of-day snapshot" not in readme


def test_splice_live_equity_replaces_same_day_tail():
    # Alpaca already posted today's point: the live mark replaces it in place
    # (no duplicate row, series length unchanged), keeping the tail's date.
    today = dt.date.today()
    series = [(today - dt.timedelta(days=1), 99_500.0), (today, 99_700.0)]
    spliced, live = trades._splice_live_equity(series, SimpleNamespace(equity=99_806.03))
    assert live is True
    assert len(spliced) == len(series)
    assert spliced[-1] == (today, 99_806.03)
    assert spliced[-2] == series[-2]


def test_splice_live_equity_noops_without_live_equity():
    today = dt.date.today()
    series = [(today - dt.timedelta(days=2), 100_000.0)]
    splice = trades._splice_live_equity
    assert splice(series, None) == (series, False)  # no account
    assert splice([], SimpleNamespace(equity=120_000.0)) == ([], False)  # empty series
    assert splice(series, SimpleNamespace(equity=0.0))[1] is False  # non-positive
    assert splice(series, SimpleNamespace(equity=float("nan")))[1] is False  # NaN


def test_broker_enum_and_float_helpers():
    assert _enum_str("OrderSide.BUY") == "buy"
    assert _opt_float("12.5") == 12.5
    assert _opt_float(None) is None
    assert _opt_float("nope") is None
