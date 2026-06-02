"""Tests for the pure decision helpers in :mod:`ophir.strategy`."""

from ophir.strategy import compute_rebalance, rank_top_k

# --------------------------------------------------------------------------- #
# rank_top_k
# --------------------------------------------------------------------------- #


def test_rank_top_k_orders_by_score_descending():
    scores = {"AAA": 0.01, "BBB": 0.05, "CCC": 0.03}
    assert rank_top_k(scores, k=3) == ["BBB", "CCC", "AAA"]


def test_rank_top_k_respects_k():
    scores = {"AAA": 0.01, "BBB": 0.05, "CCC": 0.03}
    assert rank_top_k(scores, k=2) == ["BBB", "CCC"]


def test_rank_top_k_filters_min_score():
    scores = {"AAA": -0.01, "BBB": 0.05, "CCC": 0.0}
    # default min_score=0.0 keeps only strictly positive scores
    assert rank_top_k(scores, k=10) == ["BBB"]


def test_rank_top_k_custom_min_score():
    scores = {"AAA": 0.02, "BBB": 0.05, "CCC": 0.03}
    assert rank_top_k(scores, k=10, min_score=0.025) == ["BBB", "CCC"]


def test_rank_top_k_empty():
    assert rank_top_k({}, k=5) == []


# --------------------------------------------------------------------------- #
# compute_rebalance
# --------------------------------------------------------------------------- #


def test_compute_rebalance_sells_unwanted_and_buys_new():
    sells, buys = compute_rebalance(
        current=["AAA", "BBB"],
        targets=["BBB", "CCC", "DDD"],
        buying_power=1000.0,
        budget_frac=1.0,
    )
    assert sells == ["AAA"]
    assert buys == [("CCC", 500.0), ("DDD", 500.0)]


def test_compute_rebalance_budget_fraction_applied():
    _, buys = compute_rebalance(
        current=[],
        targets=["AAA", "BBB"],
        buying_power=1000.0,
        budget_frac=0.5,
    )
    assert buys == [("AAA", 250.0), ("BBB", 250.0)]


def test_compute_rebalance_no_new_buys_when_already_held():
    sells, buys = compute_rebalance(
        current=["AAA", "BBB"],
        targets=["AAA", "BBB"],
        buying_power=1000.0,
    )
    assert sells == []
    assert buys == []


def test_compute_rebalance_sells_everything_when_no_targets():
    sells, buys = compute_rebalance(
        current=["AAA", "BBB"],
        targets=[],
        buying_power=1000.0,
    )
    assert sells == ["AAA", "BBB"]
    assert buys == []


def test_compute_rebalance_preserves_target_order_in_buys():
    _, buys = compute_rebalance(
        current=[],
        targets=["CCC", "AAA", "BBB"],
        buying_power=300.0,
        budget_frac=1.0,
    )
    assert [symbol for symbol, _ in buys] == ["CCC", "AAA", "BBB"]
