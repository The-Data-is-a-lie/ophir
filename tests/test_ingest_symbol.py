"""Tests for ingest symbol handling (partition-safe continuous-futures symbols).

Offline + pure: no network. The Yahoo *fetch* symbol must keep ``=`` (``CL=F``)
while the on-disk *store* symbol drops it (``CL_F``) so the Hive
``split("=")`` partition parser stays unambiguous.
"""

from ophir.agent.ingest import _norm_symbol, _store_symbol


def test_store_symbol_sanitizes_equals_for_partition():
    assert _store_symbol("CL=F") == "CL_F"
    assert _store_symbol("GC=F") == "GC_F"
    assert _store_symbol("AAPL") == "AAPL"  # no '=', unchanged


def test_norm_symbol_keeps_equals_for_yahoo_fetch():
    assert _norm_symbol("cl=f") == "CL=F"  # fetch form retains '='
    assert _norm_symbol("BRK.B") == "BRK-B"  # '.' -> '-' still applies
