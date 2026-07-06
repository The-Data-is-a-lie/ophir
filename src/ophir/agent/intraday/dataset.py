"""Assemble the intraday training frame: features + triple-barrier labels per symbol.

Joins :func:`ophir.agent.intraday.features.extract_intraday_features` with
:func:`ophir.agent.intraday.labels.triple_barrier_labels` on each symbol's dollar
bars, concatenates across the universe, and drops warmup / unlabeled rows. The
result is a single frame indexed by ``bar_end`` (times collide across symbols --
that is expected; the ``symbol`` column disambiguates) ready for the P2 baseline.
"""

from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]

from ophir.agent.intraday import data
from ophir.agent.intraday.features import DEFAULT_WINDOWS, extract_intraday_features
from ophir.agent.intraday.labels import triple_barrier_labels

LABEL_COLS = ["t1", "ret", "bin_tb", "meta", "weight"]
_NON_FEATURE = {*LABEL_COLS, "symbol"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Feature column names (everything that is not a label/meta column)."""
    return [c for c in df.columns if c not in _NON_FEATURE]


def build_symbol_frame(
    symbol: str,
    *,
    stocks_dir: str | None = None,
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    pt_mult: float = 1.5,
    sl_mult: float = 1.5,
    max_hold: int = 26,
    vol_span: int = 26,
) -> pd.DataFrame:
    """Features joined to triple-barrier labels for one symbol, indexed by bar_end."""
    bars = data.load_dollar_bars(symbol, stocks_dir=stocks_dir)
    feats = extract_intraday_features(bars, windows=windows)
    labs = triple_barrier_labels(
        bars, pt_mult=pt_mult, sl_mult=sl_mult, max_hold=max_hold, vol_span=vol_span
    )
    out = feats.join(labs[LABEL_COLS])
    out["symbol"] = symbol
    return out


def build_training_frame(
    symbols: list[str],
    *,
    stocks_dir: str | None = None,
    dropna: bool = True,
    **kwargs: object,
) -> pd.DataFrame:
    """Concatenate per-symbol frames (sorted by bar_end); drop warmup + unlabeled rows."""
    frames: list[pd.DataFrame] = []
    for sym in symbols:
        try:
            frames.append(build_symbol_frame(sym, stocks_dir=stocks_dir, **kwargs))  # type: ignore[arg-type]
        except FileNotFoundError:
            print(f"[dataset] skip {sym}: no dollar bars")
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames).sort_index()
    if dropna:
        cols = feature_columns(df)
        df = df.dropna(subset=[*cols, "bin_tb"])
    return df
