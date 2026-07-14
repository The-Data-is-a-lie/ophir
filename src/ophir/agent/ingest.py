"""Fetch daily OHLC from Yahoo Finance and persist it model-ready.

Pulls history for a ticker, normalizes it to the schema
:func:`ophir.ticker.extract_features` consumes, runs non-fatal quality checks,
and writes it to the Hive layout ``symbol=<SYMBOL>/data.parquet`` so the
existing ``StockHandler`` / ``StockStreamer`` pipeline reads it unchanged. A
``=`` in a Yahoo symbol (continuous futures like ``CL=F``) is mapped to ``_``
for the on-disk partition (``symbol=CL_F``) so the ``split("=")`` partition
parser stays unambiguous; the Yahoo *fetch* keeps the ``=``.

Yahoo's ``auto_adjust=True`` output is already split/dividend-adjusted, so the
separate split back-adjustment in :mod:`ophir.ticker` is intentionally skipped
on this path (applying it again would double-adjust).
"""

from __future__ import annotations

import datetime as dt
from typing import TYPE_CHECKING

import pandas as pd  # type: ignore[import-untyped]

from ophir.agent.feed import parquet_path

if TYPE_CHECKING:
    from pathlib import Path

_OHLCV_COLS = ["high", "low", "close", "volume"]
_MIN_MODEL_DAYS = 455  # 365-day window + ~90 calendar days of rolling-feature warmup
_STALE_DAYS = 5

#: Default fetch depth (~26 years). Deep by default because ``auto_adjust``
#: rescales the WHOLE history on every dividend/split: a full re-fetch keeps
#: the stored frame internally consistent, and it is the same single Yahoo
#: request as a shallow one. Keep in sync with the ``ophir ingest`` CLI default.
DEEP_DAYS = 9500


def _norm_symbol(symbol: str) -> str:
    """Normalize a ticker to Yahoo form: upper/stripped, '.' -> '-' (BRK.B -> BRK-B)."""
    return symbol.strip().upper().replace(".", "-")


def _store_symbol(symbol: str) -> str:
    """Partition-safe store symbol: sanitize the Hive-key separator ``=``.

    The parquet store is a Hive layout (``symbol=<X>``) parsed by
    ``path.split("=")[-1]``, so a Yahoo symbol containing ``=`` (the
    continuous-futures ``CL=F``) would be mangled to ``F``. Map ``=`` -> ``_``
    for the on-disk symbol (``CL=F`` -> ``CL_F``); the Yahoo *fetch* symbol keeps
    its ``=``. Symbols without ``=`` are returned unchanged.
    """
    return symbol.replace("=", "_")


def _fetch_yahoo(symbol: str, days: int) -> pd.DataFrame:
    """Fetch ``days`` of split/dividend-adjusted daily bars from Yahoo Finance."""
    import yfinance as yf

    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    raw = yf.Ticker(symbol).history(
        start=start.isoformat(),
        end=end.isoformat(),
        interval="1d",
        auto_adjust=True,
    )
    if raw.empty:
        raise ValueError(
            f"No Yahoo Finance data for {symbol!r} between {start} and {end} "
            "(invalid or delisted symbol?)."
        )
    return raw


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Coerce a yfinance frame to a tz-naive, deduplicated daily OHLCV frame."""
    df = raw.rename(columns=str.lower)[_OHLCV_COLS].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "utc_time"
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["high", "low", "close"])


def _quality_warnings(df: pd.DataFrame, days: int) -> list[str]:
    """Return human-readable staleness / sparsity warnings (non-fatal)."""
    warnings: list[str] = []
    last = df.index.max()
    age = (pd.Timestamp.today().normalize() - last.normalize()).days
    if age > _STALE_DAYS:
        warnings.append(f"last bar {last.date()} is {age} days old (stale feed?)")
    if days < _MIN_MODEL_DAYS or len(df) < 365:
        warnings.append(
            f"only {len(df)} rows for {days} requested days -- the model's "
            f"365-day window may not fully populate (want ~{_MIN_MODEL_DAYS} days)"
        )
    return warnings


def _persist(df: pd.DataFrame, symbol: str, override: str | None) -> Path:
    """Write ``df`` to ``symbol=<SYMBOL>/data.parquet``, never shrinking history.

    Invariant: a persist may not lose stored rows. A fetch reaching at least as
    far back as the stored frame replaces it wholesale (an ``auto_adjust``
    re-fetch is internally consistent). A shallower fetch is spliced onto the
    stored rows that precede it — new rows win on overlapping dates — so a
    short explicit refresh can never truncate deep history. The splice can
    leave an adjustment seam if a split/dividend occurred since the stored
    rows were fetched; a default (deep) ingest heals it.
    """
    dest = parquet_path(symbol, override=override)
    if dest.exists():
        try:
            existing = pd.read_parquet(dest).set_index("utc_time").sort_index()
        except Exception as exc:  # unreadable store: replace it, but say so
            print(f"[ingest] {symbol}: WARNING existing parquet unreadable ({exc}); overwriting")
            existing = pd.DataFrame()
        if len(existing) and existing.index.min() < df.index.min():
            head = existing.loc[existing.index < df.index.min(), _OHLCV_COLS]
            print(
                f"[ingest] {symbol}: WARNING shallow refresh (from {df.index.min().date()}) "
                f"spliced onto stored history (from {existing.index.min().date()}); rows "
                "before the splice keep their old adjustment basis -- a default deep "
                "ingest heals any seam"
            )
            df = pd.concat([head, df[_OHLCV_COLS]]).sort_index()
    out = df.reset_index()[["utc_time", *_OHLCV_COLS]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(dest, index=False)
    return dest


def ingest(symbol: str, days: int = DEEP_DAYS, *, stocks_dir: str | None = None) -> Path:
    """Ingest one ticker's daily OHLC into a model-ready parquet.

    Parameters
    ----------
    symbol : str
        Ticker symbol (case-insensitive).
    days : int, optional
        Calendar days of history to fetch. Defaults to :data:`DEEP_DAYS`
        (~26 years) so every refresh keeps the full training history
        self-consistent; pass a smaller value only for a quick spot fetch
        (``_persist`` splices it onto deeper stored history, never truncating).
    stocks_dir : str, optional
        Override for the parquet root. Defaults to ophir's
        ``<DATA_DIR>/days/stocks``.

    Returns
    -------
    pathlib.Path
        The written parquet path.
    """
    symbol = _norm_symbol(symbol)
    df = _normalize(_fetch_yahoo(symbol, days))
    if df.empty:
        raise ValueError(f"No usable rows for {symbol!r} after normalization.")
    for warning in _quality_warnings(df, days):
        print(f"[ingest] {symbol}: WARNING {warning}")
    dest = _persist(df, _store_symbol(symbol), stocks_dir)
    print(
        f"[ingest] {symbol}: {len(df)} rows "
        f"{df.index.min().date()}..{df.index.max().date()} "
        f"(last close {df['close'].iloc[-1]:.2f}) -> {dest}"
    )
    return dest


def _split_chunk(raw: pd.DataFrame, symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Split a multi-ticker ``yf.download(group_by="ticker")`` frame into per-symbol frames.

    Tickers absent from the download (delisted/invalid) are omitted. A single-ticker
    download returns a flat (non-MultiIndex) frame, handled as the lone symbol.
    """
    out: dict[str, pd.DataFrame] = {}
    columns = raw.columns
    if getattr(columns, "nlevels", 1) > 1:  # MultiIndex (ticker, field)
        present = set(columns.get_level_values(0))
        for symbol in symbols:
            if symbol in present:
                out[symbol] = raw[symbol]
    elif len(symbols) == 1:
        out[symbols[0]] = raw
    return out


def _download_chunk(symbols: list[str], days: int, max_retries: int) -> dict[str, pd.DataFrame]:
    """Batch-download a chunk of tickers from Yahoo with retry/backoff; ``{}`` if all fail."""
    import time

    import yfinance as yf

    end = dt.date.today()
    start = end - dt.timedelta(days=days)
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                symbols,
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:  # network / rate-limit / parse error -- retry
            last_error, raw = exc, None
        if raw is not None and not raw.empty:
            return _split_chunk(raw, symbols)
        if attempt < max_retries - 1:
            time.sleep(2.0**attempt)
    print(
        f"[ingest] chunk ({len(symbols)} symbols) FAILED after {max_retries} tries -- {last_error}"
    )
    return {}


def ingest_many(
    symbols: list[str],
    days: int = DEEP_DAYS,
    *,
    stocks_dir: str | None = None,
    chunk_size: int = 50,
    max_retries: int = 3,
) -> dict[str, Path]:
    """Ingest several tickers via chunked batch downloads; returns ``{symbol: parquet_path}``.

    Symbols are normalized to Yahoo form (``BRK.B`` -> ``BRK-B``) and de-duplicated,
    then fetched in batches of ``chunk_size`` (one ``yf.download`` per batch, with
    retry/backoff) -- far fewer Yahoo requests than one call per ticker, which matters
    at S&P-500 scale. A symbol that fails to download or normalize is reported and
    skipped so one bad ticker (or a throttled chunk) never aborts the batch.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for raw_symbol in symbols:
        symbol = _norm_symbol(raw_symbol)
        if symbol and symbol not in seen:
            seen.add(symbol)
            unique.append(symbol)

    paths: dict[str, Path] = {}
    for offset in range(0, len(unique), chunk_size):
        chunk = unique[offset : offset + chunk_size]
        frames = _download_chunk(chunk, days, max_retries)
        for symbol in chunk:
            raw = frames.get(symbol)
            if raw is None:
                print(f"[ingest] {symbol}: FAILED -- no data returned")
                continue
            try:
                df = _normalize(raw)
                if df.empty:
                    print(f"[ingest] {symbol}: FAILED -- no usable rows after normalization")
                    continue
                for warning in _quality_warnings(df, days):
                    print(f"[ingest] {symbol}: WARNING {warning}")
                store = _store_symbol(symbol)
                dest = _persist(df, store, stocks_dir)
                paths[store] = dest
                print(
                    f"[ingest] {symbol}: {len(df)} rows "
                    f"{df.index.min().date()}..{df.index.max().date()} "
                    f"(last close {df['close'].iloc[-1]:.2f}) -> {dest}"
                )
            except (ValueError, OSError) as exc:
                print(f"[ingest] {symbol}: FAILED -- {exc}")
    return paths
