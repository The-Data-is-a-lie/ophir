"""Offline tests for the agent input signals (network + calendar monkeypatched).

Covers fail-safe behaviour and -- most importantly -- the point-in-time /
no-look-ahead gating that keeps these signals safe to condition a forecast on.
"""

import datetime as dt

import numpy as np
import pandas as pd

from ophir.agent import (
    attention,
    congress,
    edgar,
    events,
    macro,
    market_calendar,
    research,
    sentiment,
)
from ophir.agent import options_flow as opt
from ophir.agent import short_interest as si


def _pin_session(monkeypatch, date_str):
    monkeypatch.setattr(
        market_calendar, "last_closed_session", lambda now=None: pd.Timestamp(date_str)
    )


# --------------------------------------------------------------------------- short interest
def test_short_interest_picks_latest_settlement_point_in_time(monkeypatch):
    _pin_session(monkeypatch, "2026-06-10")
    rows = [
        {
            "settlementDate": "2026-05-15",
            "currentShortPositionQuantity": 100,
            "previousShortPositionQuantity": 90,
            "daysToCoverQuantity": 2.0,
            "changePercent": 11.1,
            "averageDailyVolumeQuantity": 50,
        },
        {
            "settlementDate": "2026-05-29",
            "currentShortPositionQuantity": 120,
            "previousShortPositionQuantity": 100,
            "daysToCoverQuantity": 2.4,
            "changePercent": 20.0,
            "averageDailyVolumeQuantity": 50,
        },
        {
            "settlementDate": "2026-06-15",
            "currentShortPositionQuantity": 999,  # AFTER as_of
            "previousShortPositionQuantity": 120,
            "daysToCoverQuantity": 9.9,
            "changePercent": 700.0,
            "averageDailyVolumeQuantity": 50,
        },
    ]
    monkeypatch.setattr(si, "_query", lambda symbol, start, end: rows)
    sig = si.short_interest_signal("aapl")
    assert sig["settlement_date"] == "2026-05-29"  # latest <= as_of; 06-15 excluded (look-ahead)
    assert sig["short_interest"] == 120
    assert sig["days_to_cover"] == 2.4
    assert sig["si_change_pct"] == 20.0


def test_short_interest_failsafe(monkeypatch):
    _pin_session(monkeypatch, "2026-06-10")
    monkeypatch.setattr(si, "_query", lambda symbol, start, end: [])
    sig = si.short_interest_signal("ZZZZ")
    assert sig["n_settlements"] == 0
    assert sig["short_interest"] is None
    assert "note" in sig


# --------------------------------------------------------------------------- macro
def test_macro_regime_buckets():
    assert macro._regime(1.05) == "risk_off"
    assert macro._regime(0.80) == "risk_on"
    assert macro._regime(0.95) == "neutral"


def test_macro_signal_from_term_structure(monkeypatch):
    macro._CACHE.clear()
    _pin_session(monkeypatch, "2026-06-17")
    # a varied baseline (std > 0) so the z-score is well-defined, then a low latest ratio
    ratios = [0.9, 0.95] * 20 + [0.7]
    monkeypatch.setattr(macro, "_term_structure", lambda cutoff: (ratios, 14.0, 20.0))
    monkeypatch.setattr(macro, "_macro_extras", lambda cutoff: {"days_to_opex": 5})
    sig = macro.macro_signal()
    assert sig["vix"] == 14.0
    assert sig["vix3m"] == 20.0
    assert sig["vix_ratio"] == 0.7
    assert sig["regime"] == "risk_on"
    assert sig["vix_ratio_z"] is not None and sig["vix_ratio_z"] < 0  # latest below baseline
    macro._CACHE.clear()


def test_macro_signal_failsafe(monkeypatch):
    macro._CACHE.clear()
    _pin_session(monkeypatch, "2026-06-17")
    monkeypatch.setattr(macro, "_term_structure", lambda cutoff: None)
    monkeypatch.setattr(macro, "_macro_extras", lambda cutoff: {"days_to_opex": 2})
    sig = macro.macro_signal()
    assert sig["regime"] == "unknown"  # VIX unavailable
    assert sig["vix_ratio"] is None
    assert sig["days_to_opex"] == 2  # deterministic gate still present
    macro._CACHE.clear()


def test_days_to_opex_is_third_friday():
    # 2026-06-17 (Wed) -> 3rd Friday of June 2026 is 2026-06-19 -> 2 days out
    assert macro._days_to_opex(dt.date(2026, 6, 17)) == 2
    # on/after this month's OPEX -> rolls to next month's 3rd Friday (2026-07-17)
    assert macro._days_to_opex(dt.date(2026, 6, 20)) == 27


# --------------------------------------------------------------------------- events
def test_events_extract_dates_dict_and_frame():
    assert events._extract_dates({"Earnings Date": [dt.date(2026, 7, 30)]}) == [
        dt.date(2026, 7, 30)
    ]
    assert events._extract_dates(None) == []
    assert events._extract_dates({}) == []


def test_earnings_signal(monkeypatch):
    _pin_session(monkeypatch, "2026-06-17")
    monkeypatch.setattr(
        events,
        "_fetch_calendar",
        lambda symbol: {"Earnings Date": [dt.date(2026, 7, 30), dt.date(2026, 4, 30)]},
    )
    sig = events.earnings_signal("AAPL")
    assert sig["next_earnings_date"] == "2026-07-30"
    assert sig["days_to_next_earnings"] == 43
    assert sig["last_earnings_date"] == "2026-04-30"
    assert sig["days_since_last_earnings"] == 48


def test_earnings_signal_failsafe(monkeypatch):
    _pin_session(monkeypatch, "2026-06-17")
    monkeypatch.setattr(events, "_fetch_calendar", lambda symbol: {"Earnings Date": []})
    sig = events.earnings_signal("ZZZZ")
    assert sig["days_to_next_earnings"] is None
    assert "note" in sig


# --------------------------------------------------------------------------- congress
def test_bucket_dollars_midpoints():
    assert congress._bucket_dollars("15,001") == 32500.0
    assert congress._bucket_dollars("1,000,001") == 3000000.0
    assert congress._bucket_dollars("bogus") is None


def test_tx_regex_parses_stock_row():
    text = "Apple Inc (AAPL) [ST] P 12/23/202512/23/2025$15,001 - $50,000"
    matches = list(congress._TX_RX.finditer(text))
    assert matches and matches[0].group("ticker") == "AAPL"
    assert matches[0].group("type") == "P"
    assert matches[0].group("amt") == "15,001"


def test_congress_gates_on_disclosure_date(monkeypatch):
    congress._WINDOW_CACHE.clear()
    _pin_session(monkeypatch, "2026-03-31")
    filings = [
        {"docid": "A", "year": 2026, "filing_date": dt.date(2026, 3, 20), "member": "Alice TX01"},
        {"docid": "B", "year": 2026, "filing_date": dt.date(2026, 3, 25), "member": "Bob NY02"},
        {
            "docid": "FUTURE",
            "year": 2026,
            "filing_date": dt.date(2026, 4, 10),
            "member": "Carol CA03",
        },
        {"docid": "OLD", "year": 2026, "filing_date": dt.date(2025, 12, 1), "member": "Dave FL04"},
    ]
    txs = {
        "A": [{"ticker": "NVDA", "action": "purchase", "dollars": 32500.0}],
        "B": [{"ticker": "NVDA", "action": "purchase", "dollars": 175000.0}],
        "FUTURE": [{"ticker": "NVDA", "action": "purchase", "dollars": 3000000.0}],  # look-ahead
        "OLD": [{"ticker": "NVDA", "action": "sale", "dollars": 8000.0}],  # before window
    }
    monkeypatch.setattr(
        congress, "_index_filings", lambda year: [f for f in filings if f["year"] == year]
    )
    monkeypatch.setattr(congress, "_parse_pdf_transactions", lambda year, docid: txs.get(docid, []))

    sig = congress.congress_signal("nvda", as_of="2026-03-31", lookback_days=30)
    assert sig["n_disclosures"] == 2  # A + B; FUTURE (after as_of) and OLD (before window) excluded
    assert sig["cluster_buy_members"] == 2
    assert sig["cluster_buy_flag"] is True
    assert sig["net_dollar_flow_60d"] == 207500  # 32500 + 175000, no sells in window
    assert sig["sell_dollar_60d"] == 0
    assert sig["last_disclosure_type"] == "purchase"
    congress._WINDOW_CACHE.clear()


def test_congress_failsafe_unknown_symbol(monkeypatch):
    congress._WINDOW_CACHE.clear()
    _pin_session(monkeypatch, "2026-03-31")
    monkeypatch.setattr(congress, "_index_filings", lambda year: [])
    monkeypatch.setattr(congress, "_parse_pdf_transactions", lambda year, docid: [])
    sig = congress.congress_signal("NVDA", lookback_days=30)
    assert sig["n_disclosures"] == 0
    assert sig["cluster_buy_flag"] is False
    assert "note" in sig
    congress._WINDOW_CACHE.clear()


# --------------------------------------------------------------------------- research wiring
def test_advisory_signals_filters_empty_and_errored():
    brief = research.ResearchBrief(
        "AAPL",
        "2026-06-17",
        {},
        [],
        {},
        research.ResearchAnalysis(),
        flow={"off_exchange_pct": 0.5, "source": "x"},
        short_interest={"note": "no data"},  # excluded (fail-safe note)
        macro={"error": "boom"},  # excluded (error)
        events={},  # excluded (empty)
        congress={"cluster_buy_flag": True, "source": "y"},
    )
    sig = brief.advisory_signals()
    assert set(sig) == {"flow", "congress"}


def test_price_factors_derived_no_data():
    close = pd.Series(np.linspace(100.0, 160.0, 300))
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close})
    out = research._price_factors(df, float(close.iloc[-1]))
    assert out["mom_12_1"] is not None
    assert out["atr_pct_14"] is not None and out["atr_pct_14"] > 0


def test_liquidity_amihud_and_dollar_volume():
    n = 80
    close = pd.Series(np.linspace(100.0, 120.0, n))
    vol = pd.Series(np.full(n, 1e6))
    df = pd.DataFrame({"high": close + 1, "low": close - 1, "close": close, "volume": vol})
    out = research._liquidity(df)
    assert out["dollar_volume_20d"] is not None and out["dollar_volume_20d"] > 0
    assert out["amihud_illiq_60d"] is not None
    # too-short frame -> neutral
    assert research._liquidity(df.iloc[:10]) == {
        "amihud_illiq_60d": None,
        "dollar_volume_20d": None,
    }


def test_relative_strength_vs_spy(monkeypatch):
    idx = pd.date_range("2025-01-01", periods=120, freq="D")
    stock = pd.Series(np.linspace(100.0, 130.0, 120), index=idx)
    spy = pd.DataFrame({"close": pd.Series(np.linspace(400.0, 440.0, 120), index=idx)})
    monkeypatch.setattr(research, "load_history", lambda symbol, stocks_dir=None: spy)
    out = research._relative_strength(stock, stocks_dir=None)
    assert out["rs_vs_spy_3m"] is not None  # stock outran SPY -> positive excess
    assert out["rs_vs_spy_3m"] > 0
    assert out["rs_slope"] is not None


def test_relative_strength_failsafe_no_spy(monkeypatch):
    def _raise(symbol, stocks_dir=None):
        raise FileNotFoundError("no SPY")

    monkeypatch.setattr(research, "load_history", _raise)
    out = research._relative_strength(pd.Series([1.0, 2.0, 3.0]), stocks_dir=None)
    assert out == {"rs_vs_spy_3m": None, "rs_slope": None}


# --------------------------------------------------------------------------- SEC EDGAR insider
def test_insider_gates_on_filing_date_and_clusters(monkeypatch):
    edgar._SUBMISSIONS.clear()
    edgar._CIK_MAP.clear()
    _pin_session(monkeypatch, "2026-03-31")
    monkeypatch.setattr(edgar, "_cik_map", lambda: {"NVDA": "0001045810"})
    submissions = {
        "filings": {
            "recent": {
                "form": ["4", "4", "4", "8-K", "SC 13D"],
                "filingDate": [
                    "2026-03-20",
                    "2026-03-25",
                    "2026-04-10",
                    "2026-03-15",
                    "2026-03-10",
                ],
                "accessionNumber": ["A", "B", "FUTURE", "K", "D"],
                "primaryDocument": ["a.xml", "b.xml", "f.xml", "k.htm", "d.htm"],
                "items": ["", "", "", "2.02,9.01", ""],
            }
        }
    }
    monkeypatch.setattr(edgar, "_submissions", lambda cik: submissions)
    txns = {
        "A": [{"owner": "Alice", "action": "purchase", "dollars": 100000.0}],
        "B": [{"owner": "Bob", "action": "purchase", "dollars": 250000.0}],
        "FUTURE": [{"owner": "Carol", "action": "purchase", "dollars": 9000000.0}],  # look-ahead
    }
    monkeypatch.setattr(edgar, "_parse_form4", lambda cik, acc, doc: txns.get(acc, []))

    sig = edgar.insider_signal("nvda", as_of="2026-03-31", lookback_days=30)
    assert sig["n_form4"] == 2  # A + B; FUTURE (filingDate after as_of) excluded
    assert sig["insider_cluster_buyers"] == 2
    assert sig["insider_cluster_flag"] is True
    assert sig["insider_net_dollar_flow_90d"] == 350000  # 100k + 250k, no sells
    assert sig["insider_sell_dollar_90d"] == 0
    assert sig["days_since_last_8k"] == 16  # 2026-03-31 - 2026-03-15
    assert sig["last_8k_items"] == "2.02,9.01"
    assert sig["activist_stake_flag"] is True  # SC 13D filed 2026-03-10, within window
    edgar._SUBMISSIONS.clear()


def test_insider_failsafe_unknown_symbol(monkeypatch):
    edgar._CIK_MAP.clear()
    _pin_session(monkeypatch, "2026-03-31")
    monkeypatch.setattr(edgar, "_cik_map", lambda: {})
    sig = edgar.insider_signal("ZZZZ")
    assert sig["n_form4"] == 0
    assert sig["insider_cluster_flag"] is False
    assert "note" in sig


# --------------------------------------------------------------------------- options / IV surface
def test_options_flow_computes_ratios_and_skew(monkeypatch):
    _pin_session(monkeypatch, "2026-06-17")
    # one near-25-delta call (iv 0.30) and put (iv 0.36) expiring ~4 weeks out
    payload = {
        "data": {
            "current_price": 140.0,
            "iv30": 33.0,
            "options": [
                {
                    "option": "AAA260715C00150000",
                    "delta": 0.25,
                    "iv": 0.30,
                    "volume": 100,
                    "open_interest": 200,
                },
                {
                    "option": "AAA260715P00120000",
                    "delta": -0.25,
                    "iv": 0.36,
                    "volume": 80,
                    "open_interest": 150,
                },
            ],
        }
    }
    monkeypatch.setattr(opt, "_load_chain", lambda symbol, key, live_ok: payload)
    sig = opt.options_flow_signal("AAA")
    assert sig["put_call_ratio_volume"] == 0.8  # 80 / 100
    assert sig["put_call_ratio_oi"] == 0.75  # 150 / 200
    assert sig["iv_skew_25d"] == 0.06  # 0.36 put IV - 0.30 call IV (downside fear)
    assert sig["iv30"] == 33.0
    assert sig["n_contracts"] == 2


def test_options_flow_failsafe_no_snapshot(monkeypatch):
    _pin_session(monkeypatch, "2026-06-17")
    monkeypatch.setattr(opt, "_load_chain", lambda symbol, key, live_ok: None)
    sig = opt.options_flow_signal("AAPL")
    assert sig["n_contracts"] == 0
    assert sig["put_call_ratio_oi"] is None
    assert "note" in sig


# --------------------------------------------------------------------------- sentiment (FinBERT)
def test_news_sentiment_aggregates_scores(monkeypatch):
    monkeypatch.setattr(sentiment, "_score_headlines", lambda titles: [1.0, -1.0, 0.0])
    news = [{"title": "beats"}, {"title": "lawsuit"}, {"title": "neutral note"}]
    sig = sentiment.news_sentiment_signal("AAPL", news=news)
    assert sig["n_headlines"] == 3
    assert sig["net_sentiment"] == 0.0  # (1 - 1 + 0) / 3
    assert sig["pos_count"] == 1 and sig["neg_count"] == 1 and sig["neutral_count"] == 1


def test_news_sentiment_failsafe_no_model(monkeypatch):
    monkeypatch.setattr(sentiment, "_score_headlines", lambda titles: [])  # model unavailable
    sig = sentiment.news_sentiment_signal("AAPL", news=[{"title": "x"}])
    assert sig["n_headlines"] == 0
    assert sig["net_sentiment"] is None
    assert "note" in sig


def test_news_sentiment_no_headlines():
    sig = sentiment.news_sentiment_signal("AAPL", news=[])
    assert sig["n_headlines"] == 0
    assert "note" in sig


# --------------------------------------------------------------------------- attention
def test_attention_combines_wiki_and_reddit(monkeypatch):
    _pin_session(monkeypatch, "2026-06-17")
    monkeypatch.setattr(attention, "_resolve_title", lambda symbol, company_name: "Apple_Inc.")
    monkeypatch.setattr(attention, "_pageviews", lambda title, as_of_date: (9400, -0.59))
    monkeypatch.setattr(attention, "_apewisdom", lambda symbol: {"mentions": 42, "rank": 28})
    sig = attention.attention_signal("AAPL", company_name="Apple Inc")
    assert sig["wiki_title"] == "Apple_Inc."
    assert sig["wiki_pageviews"] == 9400
    assert sig["wiki_pageviews_zscore"] == -0.59
    assert sig["reddit_mentions"] == 42
    assert "note" not in sig


def test_attention_failsafe_unresolved(monkeypatch):
    _pin_session(monkeypatch, "2026-06-17")
    monkeypatch.setattr(attention, "_resolve_title", lambda symbol, company_name: None)
    monkeypatch.setattr(attention, "_apewisdom", lambda symbol: {})
    sig = attention.attention_signal("ZZZZ")
    assert sig["wiki_pageviews"] is None
    assert "note" in sig
