"""Free FinBERT-scored news-sentiment signal over headlines ophir already has.

ophir fetches raw Yahoo Finance headlines but never *scores* them. This turns that
existing text into a calibrated numeric sentiment layer with **zero new network
fetch**: the headlines already gathered for the research brief are scored locally
with FinBERT (``ProsusAI/finbert`` via ``transformers``) on the already-required GPU.

The model is loaded once per process (lazy). Everything fails safe -- if
``transformers``/the model is unavailable or there are no headlines, a neutral dict
is returned, never an exception. News sentiment is a weak, short-horizon signal, so
it is wired as a low-weight advisory dimension.
"""

from __future__ import annotations

from typing import Any

_SOURCE = "FinBERT sentiment over Yahoo Finance headlines"
_MODEL = "ProsusAI/finbert"
_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# Process-level singleton for the (heavy) FinBERT pipeline.
_PIPELINE: Any = None
_PIPELINE_FAILED = False


def _classifier() -> Any:
    """Return the cached FinBERT text-classification pipeline, or ``None`` if unavailable."""
    global _PIPELINE, _PIPELINE_FAILED
    if _PIPELINE is not None or _PIPELINE_FAILED:
        return _PIPELINE
    try:
        from transformers import pipeline

        _PIPELINE = pipeline("text-classification", model=_MODEL, top_k=None)
    except Exception:  # missing dep, no model, no network -> degrade to neutral forever
        _PIPELINE_FAILED = True
    return _PIPELINE


def _score_headlines(titles: list[str]) -> list[float]:
    """Score each headline to a signed sentiment in [-1, 1]; ``[]`` if scoring is unavailable."""
    clf = _classifier()
    if clf is None or not titles:
        return []
    try:
        outputs = clf(titles)
    except Exception:
        return []
    scores: list[float] = []
    for out in outputs:
        # top_k=None yields a list of {label, score} per input; take the argmax label.
        if isinstance(out, list) and out:
            best = max(out, key=lambda d: d.get("score", 0.0))
            scores.append(_SCORE.get(str(best.get("label", "")).lower(), 0.0))
    return scores


def _empty(symbol: str, as_of: Any, note: str) -> dict[str, Any]:
    """A neutral signal dict for the fail-safe path."""
    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "n_headlines": 0,
        "net_sentiment": None,
        "pos_count": 0,
        "neg_count": 0,
        "neutral_count": 0,
        "source": _SOURCE,
        "note": note,
    }


def news_sentiment_signal(
    symbol: str, *, news: list[dict[str, Any]] | None = None, as_of: Any = None
) -> dict[str, Any]:
    """Return a FinBERT sentiment summary over ``symbol``'s recent headlines.

    Pass the already-gathered ``news`` (list of ``{title, ...}``) to avoid a second
    fetch; if omitted, the headlines are fetched fresh.

    Returns
    -------
    dict
        ``net_sentiment`` (mean signed sentiment in [-1, 1]), ``pos_count`` /
        ``neg_count`` / ``neutral_count``, plus ``symbol`` / ``asof`` /
        ``n_headlines`` / ``source``. No headlines or no model yields a neutral dict,
        never an exception.
    """
    symbol = symbol.upper().strip()
    if news is None:
        try:
            from ophir.agent.research import gather_news

            news = gather_news(symbol)
        except Exception as exc:
            return _empty(symbol, as_of, f"news unavailable ({type(exc).__name__})")

    titles = [str(item.get("title", "")).strip() for item in news if item.get("title")]
    if not titles:
        return _empty(symbol, as_of, "no headlines for symbol")

    scores = _score_headlines(titles)
    if not scores:
        return _empty(symbol, as_of, "sentiment scoring unavailable")

    return {
        "symbol": symbol,
        "asof": str(as_of) if as_of is not None else None,
        "n_headlines": len(scores),
        "net_sentiment": round(sum(scores) / len(scores), 4),
        "pos_count": sum(1 for s in scores if s > 0),
        "neg_count": sum(1 for s in scores if s < 0),
        "neutral_count": sum(1 for s in scores if s == 0),
        "source": _SOURCE,
    }
