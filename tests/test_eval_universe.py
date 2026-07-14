"""Tests for the eval-universe watchlist filter (commodity baseline).

Offline + CPU-only: they exercise the ``symbols`` restriction on
:func:`ophir.train.build_split_handlers` and the :func:`ophir.evaluate._read_watchlist`
parser without loading a checkpoint, touching CUDA, or hitting the network.
"""

from ophir.evaluate import _read_watchlist
from ophir.train import build_split_handlers

# A tiny, embargo-valid split config shared by the handler tests. The fixture
# data is 2020-based, so the exact years only need to satisfy the one-window
# embargo gap (``ceil(seq_len / 365) == 1``); the tests assert the symbol
# filter, not the streamed contents.
_SPLIT_KW = {
    "seq_len": 30,
    "offset": 5,
    "min_volume": 0.0,
    "train_min_year": 2019,
    "train_max_year": 2020,
    "val_min_year": 2021,
    "val_max_year": 2022,
    "use_sp500": False,
}


def test_read_watchlist_parses_normalizes_and_dedupes(tmp_path):
    path = tmp_path / "commodities.txt"
    path.write_text("uso\n# a comment\n\n  gld \nUSO\nslv\n", encoding="utf-8")
    assert _read_watchlist(str(path)) == ["USO", "GLD", "SLV"]


def test_build_split_handlers_symbols_restricts_to_named_universe(parquet_dir):
    base_path, _paths = parquet_dir
    _train, val = build_split_handlers(base_path=base_path, symbols=["AAA"], **_SPLIT_KW)
    assert val.stocks == ["AAA"]


def test_build_split_handlers_symbols_unknown_yields_empty(parquet_dir):
    base_path, _paths = parquet_dir
    _train, val = build_split_handlers(base_path=base_path, symbols=["ZZZ"], **_SPLIT_KW)
    assert val.stocks == []
