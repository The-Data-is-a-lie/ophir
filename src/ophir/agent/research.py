"""Per-ticker research briefs: grounded data + a cited LLM synthesis.

For each ticker we gather three dimensions of *real* data deterministically --
**fundamentals** (Yahoo Finance ``.info``), **news** (Yahoo Finance ``.news``),
and **technicals** (ophir's computed features plus the model
:class:`~ophir.agent.predict.Forecast`) -- and then ask the local ``gpt-oss:20b``
model to summarize *only that data* into a structured brief. The model never
fetches or invents numbers; if it is unreachable or replies badly the brief
falls back to a neutral analysis while keeping the grounded data intact.

Per the ``trading-best-practices`` skill: ground every claim, no fabricated
figures, fail safe, and audit every brief.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel

from ophir.agent import audit
from ophir.agent.config import get_settings
from ophir.agent.decide import _extract_json_object
from ophir.agent.feed import load_history

if TYPE_CHECKING:
    from ophir.agent.config import AgentSettings
    from ophir.agent.predict import Forecast

Stance = Literal["bullish", "neutral", "bearish"]

_FUNDAMENTAL_KEYS = [
    "shortName",
    "sector",
    "industry",
    "marketCap",
    "trailingPE",
    "forwardPE",
    "profitMargins",
    "beta",
    "dividendYield",
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "currentPrice",
]


class ResearchAnalysis(BaseModel):
    """The LLM's validated synthesis of the grounded research data."""

    fundamentals_summary: str = ""
    news_summary: str = ""
    technicals_summary: str = ""
    overall_stance: Stance = "neutral"
    overall_summary: str = ""


@dataclass(frozen=True, slots=True)
class ResearchBrief:
    """Grounded research data for one ticker plus the LLM analysis.

    All advisory signal dimensions (``flow`` / ``short_interest`` / ``macro`` /
    ``events`` / ``congress`` / ``insider`` / ``options`` / ``sentiment`` /
    ``attention``) default to an empty dict and never block the brief; collect the
    populated ones with :meth:`advisory_signals`.

    Attributes
    ----------
    symbol : str
        Ticker symbol.
    asof : str
        ISO date the data is conditioned on.
    fundamentals : dict
        Deterministically fetched fundamentals (Yahoo Finance ``.info`` subset).
    news : list[dict]
        Recent headlines (``title`` / ``publisher`` / ``link`` / ``published``).
    technicals : dict
        ophir-computed indicators plus the model forecast (when supplied).
    flow : dict
        Free FINRA off-exchange (dark-pool) activity signal for the ticker.
    short_interest : dict
        Free FINRA biweekly consolidated short-interest signal.
    macro : dict
        Free market-wide VIX term-structure regime signal (shared across tickers).
    events : dict
        Free Yahoo Finance earnings-calendar event gate (days to next earnings).
    congress : dict
        Free US House congressional-trade disclosure signal (advisory, low weight).
    insider : dict
        Free SEC EDGAR corporate-insider (Form 4) + 8-K / 13D-G event signal.
    options : dict
        Free CBOE per-name options / IV-surface signal (put/call, 25-delta skew, iv30).
    sentiment : dict
        FinBERT-scored sentiment over the brief's own headlines (zero new fetch).
    attention : dict
        Free Wikipedia-pageviews (+ Reddit) public-attention signal.
    analysis : ResearchAnalysis
        The LLM's summary; a neutral default when ``llm_ok`` is ``False``.
    sources : list[str]
        Provenance of every datum in the brief.
    llm_ok : bool
        ``True`` when the LLM synthesis succeeded.
    """

    symbol: str
    asof: str
    fundamentals: dict[str, Any]
    news: list[dict[str, Any]]
    technicals: dict[str, Any]
    analysis: ResearchAnalysis
    flow: dict[str, Any] = field(default_factory=dict)
    short_interest: dict[str, Any] = field(default_factory=dict)
    macro: dict[str, Any] = field(default_factory=dict)
    events: dict[str, Any] = field(default_factory=dict)
    congress: dict[str, Any] = field(default_factory=dict)
    insider: dict[str, Any] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    sentiment: dict[str, Any] = field(default_factory=dict)
    attention: dict[str, Any] = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    llm_ok: bool = True

    def advisory_signals(self) -> dict[str, dict[str, Any]]:
        """Return the populated advisory signal dimensions (skipping empty/errored).

        Used by the debate and manager prompts to surface flow / short interest /
        macro / events / congress / insider as low-weight advisory color.
        """
        out: dict[str, dict[str, Any]] = {}
        names = (
            "flow",
            "short_interest",
            "macro",
            "events",
            "congress",
            "insider",
            "options",
            "sentiment",
            "attention",
        )
        for name in names:
            signal = getattr(self, name)
            if signal and not signal.get("error") and not signal.get("note"):
                out[name] = signal
        return out


def _num(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` if missing / non-finite."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def gather_fundamentals(symbol: str) -> dict[str, Any]:
    """Fetch a curated subset of Yahoo Finance fundamentals (best-effort)."""
    import yfinance as yf

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:  # network / parse failure -> grounded-but-empty
        return {"error": f"fundamentals unavailable ({type(exc).__name__})"}
    return {key: info.get(key) for key in _FUNDAMENTAL_KEYS}


def _normalize_news_item(item: dict[str, Any]) -> dict[str, Any]:
    """Normalize a yfinance news item across its old (flat) and new (nested) shapes."""
    content = item.get("content")
    if isinstance(content, dict):  # newer nested shape
        provider = content.get("provider")
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else ""
        url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        link = url_obj.get("url", "") if isinstance(url_obj, dict) else ""
        return {
            "title": content.get("title", ""),
            "publisher": publisher,
            "link": link,
            "published": str(content.get("pubDate", "")),
        }
    ts = item.get("providerPublishTime")  # older flat shape
    published = ""
    if isinstance(ts, (int, float)):
        published = time.strftime("%Y-%m-%d", time.gmtime(ts))
    return {
        "title": item.get("title", ""),
        "publisher": item.get("publisher", ""),
        "link": item.get("link", ""),
        "published": published,
    }


def gather_news(symbol: str, limit: int = 8) -> list[dict[str, Any]]:
    """Fetch recent Yahoo Finance headlines (best-effort, normalized)."""
    import yfinance as yf

    try:
        raw = yf.Ticker(symbol).news or []
    except Exception:  # network / parse failure -> no news
        return []
    return [_normalize_news_item(item) for item in raw[:limit] if isinstance(item, dict)]


def gather_flow(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Fetch the free FINRA off-exchange (dark-pool) activity signal (best-effort).

    Thin wrapper over :func:`ophir.agent.darkpool.dark_pool_signal`, which already
    fails safe to a neutral dict; this only guards the import so a missing module
    can never break a brief.
    """
    try:
        from ophir.agent.darkpool import dark_pool_signal

        return dark_pool_signal(symbol, as_of=as_of)
    except Exception as exc:  # never let the flow dimension break the brief
        return {"error": f"flow unavailable ({type(exc).__name__})"}


def gather_short_interest(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Fetch the free FINRA biweekly consolidated short-interest signal (best-effort)."""
    try:
        from ophir.agent.short_interest import short_interest_signal

        return short_interest_signal(symbol, as_of=as_of)
    except Exception as exc:
        return {"error": f"short interest unavailable ({type(exc).__name__})"}


def gather_macro(*, as_of: Any = None) -> dict[str, Any]:
    """Fetch the free market-wide VIX term-structure regime signal (best-effort)."""
    try:
        from ophir.agent.macro import macro_signal

        return macro_signal(as_of=as_of)
    except Exception as exc:
        return {"error": f"macro unavailable ({type(exc).__name__})"}


def gather_events(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Fetch the free Yahoo Finance earnings-calendar event gate (best-effort)."""
    try:
        from ophir.agent.events import earnings_signal

        return earnings_signal(symbol, as_of=as_of)
    except Exception as exc:
        return {"error": f"events unavailable ({type(exc).__name__})"}


def gather_congress(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Fetch the free US House congressional-trade disclosure signal (best-effort)."""
    try:
        from ophir.agent.congress import congress_signal

        return congress_signal(symbol, as_of=as_of)
    except Exception as exc:
        return {"error": f"congress unavailable ({type(exc).__name__})"}


def gather_insider(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Fetch the free SEC EDGAR insider (Form 4) + corporate-event signal (best-effort)."""
    try:
        from ophir.agent.edgar import insider_signal

        return insider_signal(symbol, as_of=as_of)
    except Exception as exc:
        return {"error": f"insider unavailable ({type(exc).__name__})"}


def gather_options(symbol: str, *, as_of: Any = None) -> dict[str, Any]:
    """Fetch the free CBOE per-name options / IV-surface signal (best-effort)."""
    try:
        from ophir.agent.options_flow import options_flow_signal

        return options_flow_signal(symbol, as_of=as_of)
    except Exception as exc:
        return {"error": f"options unavailable ({type(exc).__name__})"}


def gather_sentiment(
    symbol: str, *, news: list[dict[str, Any]] | None = None, as_of: Any = None
) -> dict[str, Any]:
    """Score the brief's headlines with FinBERT (best-effort; zero new fetch)."""
    try:
        from ophir.agent.sentiment import news_sentiment_signal

        return news_sentiment_signal(symbol, news=news, as_of=as_of)
    except Exception as exc:
        return {"error": f"sentiment unavailable ({type(exc).__name__})"}


def gather_attention(
    symbol: str, *, company_name: str | None = None, as_of: Any = None
) -> dict[str, Any]:
    """Fetch the free Wikipedia-pageviews + Reddit attention signal (best-effort)."""
    try:
        from ophir.agent.attention import attention_signal

        return attention_signal(symbol, company_name=company_name, as_of=as_of)
    except Exception as exc:
        return {"error": f"attention unavailable ({type(exc).__name__})"}


def _signal_source(signal: dict[str, Any]) -> str | None:
    """The provenance string for a signal dict, or ``None`` on an error/empty result."""
    if signal.get("error") or signal.get("note"):
        return None
    source = signal.get("source")
    return source if isinstance(source, str) else None


def _price_factors(df: pd.DataFrame, last_close: float) -> dict[str, Any]:
    """12-1 momentum and a 14-day ATR%-of-close from the OHLC frame (None if short)."""
    out: dict[str, Any] = {"mom_12_1": None, "atr_pct_14": None}
    close = df["close"]
    if len(close) >= 252:
        p_recent, p_year = float(close.iloc[-21]), float(close.iloc[-252])
        if p_recent > 0 and p_year > 0:
            out["mom_12_1"] = round(math.log(p_recent / p_year), 5)
    if len(df) >= 15 and last_close > 0:
        prev_close = close.shift(1)
        high, low = df["high"], df["low"]
        true_range = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        out["atr_pct_14"] = round(float(true_range.iloc[-14:].mean()) / last_close, 5)
    return out


def _relative_strength(close: pd.Series, *, stocks_dir: str | None) -> dict[str, Any]:
    """Relative strength vs SPY: 3-month excess return and the price/SPY ratio slope.

    SPY is already ingested; fails safe to ``None`` fields when it is missing/stale.
    """
    out: dict[str, Any] = {"rs_vs_spy_3m": None, "rs_slope": None}
    try:
        spy = load_history("SPY", stocks_dir=stocks_dir)["close"]
    except (ValueError, FileNotFoundError, OSError):
        return out
    joined = pd.concat([close.rename("s"), spy.rename("b")], axis=1, join="inner").dropna()
    if len(joined) >= 63:
        s, b = joined["s"], joined["b"]
        if float(s.iloc[-63]) > 0 and float(b.iloc[-63]) > 0:
            stock_ret = float(s.iloc[-1]) / float(s.iloc[-63]) - 1.0
            bench_ret = float(b.iloc[-1]) / float(b.iloc[-63]) - 1.0
            out["rs_vs_spy_3m"] = round(stock_ret - bench_ret, 5)
        if len(joined) >= 21:
            ratio_now = float(s.iloc[-1]) / float(b.iloc[-1]) if float(b.iloc[-1]) > 0 else 0.0
            ratio_then = float(s.iloc[-21]) / float(b.iloc[-21]) if float(b.iloc[-21]) > 0 else 0.0
            if ratio_now > 0 and ratio_then > 0:
                out["rs_slope"] = round(math.log(ratio_now / ratio_then), 5)
    return out


def _liquidity(df: pd.DataFrame) -> dict[str, Any]:
    """Amihud illiquidity and 20-day average dollar volume from the OHLCV frame."""
    out: dict[str, Any] = {"amihud_illiq_60d": None, "dollar_volume_20d": None}
    if len(df) < 21:
        return out
    dollar_vol = df["close"] * df["volume"]
    out["dollar_volume_20d"] = round(float(dollar_vol.iloc[-20:].mean()))
    ret = df["close"].pct_change().abs()
    illiq = (ret / dollar_vol.where(dollar_vol > 0)).iloc[-60:].dropna()
    if len(illiq):
        # Amihud (2002): |return| per dollar traded, scaled to a readable magnitude.
        out["amihud_illiq_60d"] = round(float(illiq.mean()) * 1e9, 4)
    return out


def gather_technicals(
    symbol: str, forecast: Forecast | None = None, *, stocks_dir: str | None = None
) -> dict[str, Any]:
    """Compute grounded technical indicators from ingested OHLC (+ the forecast)."""
    from ophir.ticker import extract_features

    df = load_history(symbol, stocks_dir=stocks_dir)
    feats = extract_features(df)
    real = feats[feats["trade_occured"]]
    last = real.iloc[-1]
    close = df["close"]
    recent = close.iloc[-252:]
    last_close = float(close.iloc[-1])
    tech: dict[str, Any] = {
        "last_close": round(last_close, 4),
        "recent_return_1d": _num(last["r_close"]),
        "vol_20d": _num(last["20_volatility"]),
        "vol_60d": _num(last["60_volatility"]),
        "pct_of_52w_high": round(last_close / float(recent.max()), 4),
        "pct_above_52w_low": round(last_close / float(recent.min()), 4),
    }
    tech.update(_price_factors(df, last_close))
    tech.update(_relative_strength(close, stocks_dir=stocks_dir))
    tech.update(_liquidity(df))
    if forecast is not None:
        tech["forecast_cum_return"] = round(forecast.cum_return, 5)
        tech["forecast_mean_upside"] = round(_mean(forecast.upside), 5)
        tech["forecast_mean_downside"] = round(_mean(forecast.downside), 5)
    return tech


def _build_messages(
    symbol: str,
    asof: str,
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]],
    technicals: dict[str, Any],
) -> tuple[Any, Any]:
    """Build the grounded system + human messages for the synthesis call."""
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(
        content=(
            "You are an equity research assistant. Summarize ONLY the data provided "
            "below -- never invent prices, ratios, headlines, or figures, and do not "
            "use outside knowledge. Base your stance solely on the provided "
            "fundamentals, news headlines, and technicals. Reply with ONLY a JSON "
            'object of the form {"fundamentals_summary": "...", "news_summary": "...", '
            '"technicals_summary": "...", "overall_stance": "bullish|neutral|bearish", '
            '"overall_summary": "..."} and nothing else.'
        )
    )
    news_lines = (
        "\n".join(
            f"  - {n['title']} ({n['publisher']}, {n['published']})" for n in news if n["title"]
        )
        or "  (no recent headlines)"
    )
    human = HumanMessage(
        content=(
            f"Ticker: {symbol}\nAs-of date: {asof}\n\n"
            f"FUNDAMENTALS:\n{json.dumps(fundamentals, indent=2, default=str)}\n\n"
            f"NEWS HEADLINES:\n{news_lines}\n\n"
            f"TECHNICALS:\n{json.dumps(technicals, indent=2, default=str)}"
        )
    )
    return system, human


def _synthesize(
    symbol: str,
    asof: str,
    fundamentals: dict[str, Any],
    news: list[dict[str, Any]],
    technicals: dict[str, Any],
    llm: Any,
) -> tuple[ResearchAnalysis, bool]:
    """Call the LLM to summarize the grounded data; fail safe to neutral."""
    system, human = _build_messages(symbol, asof, fundamentals, news, technicals)
    try:
        content = llm.invoke([system, human]).content
        text = content if isinstance(content, str) else str(content)
        data = json.loads(_extract_json_object(text))
        if not isinstance(data, dict):
            raise ValueError("LLM did not return a JSON object")
        stance = data.get("overall_stance")
        if isinstance(stance, str):
            data["overall_stance"] = stance.strip().lower()
        return ResearchAnalysis(**data), True
    except Exception as exc:  # fail safe: keep the grounded data, neutral analysis
        note = f"synthesis unavailable ({type(exc).__name__})"
        return (
            ResearchAnalysis(
                fundamentals_summary=note,
                news_summary=note,
                technicals_summary=note,
                overall_stance="neutral",
                overall_summary=note,
            ),
            False,
        )


def research_ticker(
    symbol: str,
    *,
    forecast: Forecast | None = None,
    llm: Any = None,
    settings: AgentSettings | None = None,
    stocks_dir: str | None = None,
) -> ResearchBrief:
    """Build a grounded research brief for one ticker.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    forecast : Forecast, optional
        The model forecast; adds forecast fields to the technicals dimension.
    llm : object, optional
        A chat model exposing ``.invoke(messages) -> obj.content``; defaults to a
        ``ChatOllama(model=settings.ollama_model, temperature=0)``. Inject a fake
        in tests.
    settings : AgentSettings, optional
        Defaults to the cached :func:`~ophir.agent.config.get_settings`.
    stocks_dir : str, optional
        Override for the parquet root.

    Returns
    -------
    ResearchBrief
        Grounded data plus the (possibly fail-safe) LLM analysis.
    """
    settings = settings or get_settings()
    symbol = symbol.upper().strip()

    fundamentals = gather_fundamentals(symbol)
    news = gather_news(symbol, limit=settings.research_news_limit)
    technicals = gather_technicals(symbol, forecast, stocks_dir=stocks_dir)
    asof = (
        forecast.asof
        if forecast is not None
        else str(load_history(symbol, stocks_dir=stocks_dir).index.max().date())
    )
    flow = gather_flow(symbol, as_of=asof)
    short_interest = gather_short_interest(symbol, as_of=asof)
    macro = gather_macro(as_of=asof)
    events = gather_events(symbol, as_of=asof)
    congress = gather_congress(symbol, as_of=asof)
    insider = gather_insider(symbol, as_of=asof)
    options = gather_options(symbol, as_of=asof)
    sentiment = gather_sentiment(symbol, news=news, as_of=asof)
    attention = gather_attention(symbol, company_name=fundamentals.get("shortName"), as_of=asof)

    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            format="json",
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )

    analysis, llm_ok = _synthesize(symbol, asof, fundamentals, news, technicals, llm)

    sources = ["yfinance:info"]
    sources += [n["link"] for n in news if n["link"]]
    sources.append("ophir:features")
    if forecast is not None:
        sources.append("ophir:forecast")
    advisory = (
        flow,
        short_interest,
        macro,
        events,
        congress,
        insider,
        options,
        sentiment,
        attention,
    )
    for signal in advisory:
        src = _signal_source(signal)
        if src and src not in sources:
            sources.append(src)

    brief = ResearchBrief(
        symbol=symbol,
        asof=asof,
        fundamentals=fundamentals,
        news=news,
        technicals=technicals,
        analysis=analysis,
        flow=flow,
        short_interest=short_interest,
        macro=macro,
        events=events,
        congress=congress,
        insider=insider,
        options=options,
        sentiment=sentiment,
        attention=attention,
        sources=sources,
        llm_ok=llm_ok,
    )
    audit.log_event(
        "research", symbol=symbol, stance=analysis.overall_stance, llm_ok=llm_ok, n_news=len(news)
    )
    return brief


def research_many(
    symbols: list[str],
    *,
    forecasts: list[Forecast] | None = None,
    llm: Any = None,
    settings: AgentSettings | None = None,
    stocks_dir: str | None = None,
) -> list[ResearchBrief]:
    """Build briefs for several tickers; failures are logged and skipped."""
    settings = settings or get_settings()
    by_symbol = {f.symbol.upper(): f for f in forecasts} if forecasts else {}

    if llm is None:
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=settings.ollama_model,
            temperature=0,
            format="json",
            base_url=settings.ollama_base_url,
            num_ctx=settings.ollama_num_ctx,
        )

    briefs: list[ResearchBrief] = []
    for symbol in symbols:
        try:
            briefs.append(
                research_ticker(
                    symbol,
                    forecast=by_symbol.get(symbol.upper().strip()),
                    llm=llm,
                    settings=settings,
                    stocks_dir=stocks_dir,
                )
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            audit.log_event("research_failed", symbol=symbol, error=str(exc))
            print(f"[research] {symbol}: FAILED -- {exc}")
    return briefs
