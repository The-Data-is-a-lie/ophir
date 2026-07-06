"""Intraday market-data layer: Alpaca SIP historical minute bars + dollar bars.

Fetches full-SIP historical minute bars for the intraday research pipeline,
filters to regular trading hours (RTH), and persists them to a separate Hive
parquet root that mirrors the daily ``agent.ingest`` layout
(``<DATA_DIR>/intraday/stocks/symbol=<SYM>/minute.parquet``). **Dollar bars**
(activity-based sampling; blueprint §4.2) are derived from the minute bars.

Free-tier note: the Basic plan serves full **SIP** history as long as the query
``end`` is at least ~15 minutes old, so backfill defaults to ``feed="sip"`` and
clamps ``end`` accordingly. This module is read-only against the market and
writes only under the intraday root; it never touches the daily data tree.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

from ophir.agent.config import get_settings

if TYPE_CHECKING:
    from alpaca.data.historical import StockHistoricalDataClient

# Alpaca minute-bar schema (confirmed live): OHLCV + trade count + VWAP.
_BAR_COLS = ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
_ET = "America/New_York"
_RTH_OPEN = "09:30"
_RTH_CLOSE = "16:00"
# Free-tier SIP cannot see the most recent 15 min; use 16 for margin.
_FREE_SIP_DELAY = dt.timedelta(minutes=16)
# Chunk long backfills so each request stays bounded (alpaca-py paginates within).
_CHUNK_DAYS = 90


# ---------------------------------------------------------------------------
# Paths (mirror agent.feed.resolve_stocks_root, under an `intraday/` sibling)
# ---------------------------------------------------------------------------
def intraday_root(override: str | None = None) -> Path:
    """Return the ``.../intraday/stocks`` root holding ``symbol=<SYM>/`` partitions."""
    if override is not None:
        return Path(override)
    from ophir.register import get_default_data_days_dir

    # get_default_data_days_dir() -> <DATA_DIR>/days ; put intraday beside it.
    return Path(get_default_data_days_dir()).parent / "intraday" / "stocks"


def minute_parquet_path(symbol: str, *, override: str | None = None) -> Path:
    """Path to a symbol's raw minute-bar parquet."""
    return intraday_root(override) / f"symbol={symbol.upper()}" / "minute.parquet"


def dollar_parquet_path(symbol: str, *, override: str | None = None) -> Path:
    """Path to a symbol's derived dollar-bar parquet."""
    return intraday_root(override) / f"symbol={symbol.upper()}" / "dollar.parquet"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------
def _as_utc(ts: object) -> dt.datetime:
    """Coerce a timestamp-like to a tz-aware UTC ``datetime``."""
    out = pd.Timestamp(ts)
    out = out.tz_localize("UTC") if out.tzinfo is None else out.tz_convert("UTC")
    return out.to_pydatetime()


def _client() -> StockHistoricalDataClient:
    """Build an Alpaca historical-data client from the agent settings."""
    from alpaca.data.historical import StockHistoricalDataClient

    s = get_settings()
    if s.alpaca_key_id is None or s.alpaca_secret_key is None:
        raise RuntimeError(
            "Alpaca credentials missing: set AGENT_ALPACA_KEY_ID / AGENT_ALPACA_SECRET_KEY "
            "(the ophir-bot/.env values) before backfilling intraday data."
        )
    return StockHistoricalDataClient(
        s.alpaca_key_id.get_secret_value(), s.alpaca_secret_key.get_secret_value()
    )


def _filter_rth(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only regular-hours bars (09:30–16:00 ET, close-exclusive); index stays UTC."""
    et = df.tz_convert(_ET)
    et = et.between_time(_RTH_OPEN, _RTH_CLOSE, inclusive="left")
    return et.tz_convert("UTC")


def fetch_minute_bars(
    symbol: str,
    start: object,
    end: object,
    *,
    feed: str | None = None,
    rth_only: bool | None = None,
    client: StockHistoricalDataClient | None = None,
) -> pd.DataFrame:
    """Fetch minute bars for ``symbol`` in ``[start, end]``, RTH-filtered.

    Returns a UTC-indexed frame with columns ``open, high, low, close, volume,
    trade_count, vwap`` (empty frame if the window has no data). ``feed`` defaults
    to the configured ``data_feed_backfill`` ("sip"); ``end`` is clamped to respect
    the free-tier 15-minute SIP delay.
    """
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    s = get_settings()
    feed = feed or s.data_feed_backfill
    rth_only = s.rth_only if rth_only is None else rth_only
    client = client or _client()

    start_dt = _as_utc(start)
    end_dt = min(_as_utc(end), dt.datetime.now(dt.timezone.utc) - _FREE_SIP_DELAY)
    if end_dt <= start_dt:
        return _empty_minute_frame()

    req = StockBarsRequest(
        symbol_or_symbols=[symbol.upper()],
        timeframe=TimeFrame.Minute,
        start=start_dt,
        end=end_dt,
        feed=DataFeed(feed),
        adjustment=Adjustment.ALL,
    )
    raw = client.get_stock_bars(req).df
    if raw is None or raw.empty:
        return _empty_minute_frame()

    df = raw.reset_index()
    if "symbol" in df.columns:
        df = df[df["symbol"] == symbol.upper()]
    df = df.set_index("timestamp").sort_index()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[[c for c in _BAR_COLS if c in df.columns]].astype("float64")
    df = df[~df.index.duplicated(keep="last")]
    return _filter_rth(df) if rth_only else df


def _empty_minute_frame() -> pd.DataFrame:
    idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in _BAR_COLS}, index=idx)


# ---------------------------------------------------------------------------
# Backfill (chunked) + persistence
# ---------------------------------------------------------------------------
def backfill_symbol(
    symbol: str,
    *,
    days: int | None = None,
    feed: str | None = None,
    stocks_dir: str | None = None,
    client: StockHistoricalDataClient | None = None,
) -> tuple[Path, int]:
    """Backfill and persist a symbol's RTH minute bars; returns (parquet_path, rows).

    Fetches ``days`` (default ``intraday_history_days``) of history in ``_CHUNK_DAYS``
    windows to keep each request bounded, then writes one ``minute.parquet``.
    """
    symbol = symbol.upper().strip()
    s = get_settings()
    days = days or s.intraday_history_days
    client = client or _client()

    end = dt.datetime.now(dt.timezone.utc) - _FREE_SIP_DELAY
    start = end - dt.timedelta(days=days)

    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + dt.timedelta(days=_CHUNK_DAYS), end)
        chunk = fetch_minute_bars(symbol, cursor, chunk_end, feed=feed, client=client)
        if not chunk.empty:
            frames.append(chunk)
        cursor = chunk_end

    df = pd.concat(frames).sort_index() if frames else _empty_minute_frame()
    df = df[~df.index.duplicated(keep="last")]

    path = minute_parquet_path(symbol, override=stocks_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.reset_index().rename(columns={"timestamp": "utc_time", "index": "utc_time"})
    out.to_parquet(path, index=False)
    if len(df):
        print(
            f"[intraday] {symbol}: {len(df)} RTH minute bars "
            f"{df.index.min()}..{df.index.max()} -> {path}"
        )
    else:
        print(f"[intraday] {symbol}: no bars returned -> {path} (empty)")
    return path, len(df)


def load_minute_bars(symbol: str, *, stocks_dir: str | None = None) -> pd.DataFrame:
    """Read back a symbol's persisted minute bars as a UTC-indexed frame."""
    path = minute_parquet_path(symbol, override=stocks_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No intraday minute data for {symbol!r} at {path}. "
            f"Run `ophir intraday backfill {symbol}` first."
        )
    df = pd.read_parquet(path)
    df["utc_time"] = pd.to_datetime(df["utc_time"], utc=True)
    return df.set_index("utc_time").sort_index()[_BAR_COLS]


def load_dollar_bars(symbol: str, *, stocks_dir: str | None = None) -> pd.DataFrame:
    """Read back a symbol's persisted dollar bars (start/end coerced to UTC)."""
    path = dollar_parquet_path(symbol, override=stocks_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"No intraday dollar bars for {symbol!r} at {path}. "
            f"Run `ophir intraday backfill {symbol}` first."
        )
    df = pd.read_parquet(path)
    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def backfill_many(
    symbols: list[str],
    *,
    days: int | None = None,
    feed: str | None = None,
    stocks_dir: str | None = None,
) -> dict[str, Path]:
    """Backfill several symbols (one shared client); failures are logged and skipped."""
    client = _client()
    out: dict[str, Path] = {}
    for symbol in symbols:
        try:
            path, rows = backfill_symbol(
                symbol, days=days, feed=feed, stocks_dir=stocks_dir, client=client
            )
            if rows:
                out[symbol] = path
        except Exception as exc:  # noqa: BLE001 -- one bad symbol must not abort the batch
            print(f"[intraday] {symbol}: FAILED -- {type(exc).__name__}: {exc}")
    return out


# ---------------------------------------------------------------------------
# Dollar bars (activity-based sampling, computed within each RTH day)
# ---------------------------------------------------------------------------
def _dollar_volume(df: pd.DataFrame) -> pd.Series:
    """Per-bar dollar volume, using VWAP when present, else close."""
    price = df["vwap"].where(df["vwap"] > 0, df["close"])
    return price * df["volume"]


def auto_dollar_threshold(minute_df: pd.DataFrame, target_bars_per_day: int = 26) -> float:
    """Pick a dollar threshold giving ~``target_bars_per_day`` bars (median RTH day)."""
    if minute_df.empty:
        return 0.0
    dv = _dollar_volume(minute_df)
    et_date = dv.index.tz_convert(_ET).date
    daily = dv.groupby(et_date).sum()
    median_daily = float(daily.median())
    return median_daily / max(target_bars_per_day, 1)


def build_dollar_bars(minute_df: pd.DataFrame, dollar_threshold: float) -> pd.DataFrame:
    """Aggregate RTH minute bars into within-day dollar bars (vectorized).

    Each dollar bar closes once cumulative dollar volume (reset every trading day)
    crosses ``dollar_threshold``. Columns: ``open, high, low, close, volume,
    trade_count, dollar_volume, vwap, start, end, n_minutes``.
    """
    if minute_df.empty or dollar_threshold <= 0:
        return _empty_dollar_frame()

    df = minute_df.reset_index().rename(columns={"index": "utc_time", "timestamp": "utc_time"})
    df["dv"] = _dollar_volume(minute_df).to_numpy()
    df["date"] = df["utc_time"].dt.tz_convert(_ET).dt.date
    df["cum"] = df.groupby("date")["dv"].cumsum()
    df["bar"] = (df["cum"] // dollar_threshold).astype("int64")

    agg = df.groupby(["date", "bar"]).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
        dollar_volume=("dv", "sum"),
        start=("utc_time", "first"),
        end=("utc_time", "last"),
        n_minutes=("utc_time", "size"),
    )
    agg["vwap"] = agg["dollar_volume"] / agg["volume"].where(agg["volume"] > 0, 1.0)
    return agg.reset_index(drop=True)


def _empty_dollar_frame() -> pd.DataFrame:
    cols = [
        "open", "high", "low", "close", "volume", "trade_count",
        "dollar_volume", "start", "end", "n_minutes", "vwap",
    ]
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in cols})


def build_and_persist_dollar_bars(
    symbol: str,
    *,
    target_bars_per_day: int = 26,
    stocks_dir: str | None = None,
) -> tuple[Path, int]:
    """Load a symbol's minute bars, derive dollar bars, and persist them."""
    minute_df = load_minute_bars(symbol, stocks_dir=stocks_dir)
    threshold = auto_dollar_threshold(minute_df, target_bars_per_day=target_bars_per_day)
    bars = build_dollar_bars(minute_df, threshold)
    path = dollar_parquet_path(symbol, override=stocks_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(path, index=False)
    print(
        f"[intraday] {symbol}: {len(bars)} dollar bars "
        f"(threshold ${threshold:,.0f}, ~{target_bars_per_day}/day) -> {path}"
    )
    return path, len(bars)
