"""``ophir intraday`` — backfill SIP minute bars and derive dollar bars (Option B, P0).

Only ``typer`` is imported at module load; the data layer (pandas + alpaca) is
imported lazily inside each command so mounting this sub-app in ``ophir.cli`` does
not slow or affect the daily ``ophir trade`` path.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(help="Intraday research pipeline: SIP minute bars + dollar bars.")


def _resolve_symbols(symbols: list[str] | None, watchlist: str | None) -> list[str]:
    """Upper-cased, de-duplicated symbols from CLI args and/or a watchlist file."""
    collected: list[str] = list(symbols or [])
    if watchlist:
        for line in Path(watchlist).read_text().splitlines():
            token = line.strip()
            if token and not token.startswith("#"):
                collected.append(token)
    seen: set[str] = set()
    unique: list[str] = []
    for sym in collected:
        norm = sym.upper().strip()
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


@app.command()
def backfill(
    symbols: list[str] | None = typer.Argument(None, help="Tickers; omit to use --watchlist"),
    watchlist: str | None = typer.Option(None, "--watchlist", "-w", help="Watchlist file path"),
    days: int | None = typer.Option(None, help="Days of history (default: config)"),
    dollar_bars: bool = typer.Option(True, help="Also derive dollar bars after backfill"),
    target_bars_per_day: int = typer.Option(26, help="Dollar-bar cadence target"),
    stocks_dir: str | None = typer.Option(None, help="Override the intraday parquet root"),
) -> None:
    """Backfill RTH minute bars (SIP) and optionally derive dollar bars."""
    from ophir.agent.intraday import data

    syms = _resolve_symbols(symbols, watchlist)
    if not syms:
        raise typer.BadParameter("No symbols: pass tickers or --watchlist.")
    typer.echo(f"[intraday] backfilling {len(syms)} symbol(s): {', '.join(syms)}")
    paths = data.backfill_many(syms, days=days, stocks_dir=stocks_dir)
    if dollar_bars:
        for sym in paths:
            data.build_and_persist_dollar_bars(
                sym, target_bars_per_day=target_bars_per_day, stocks_dir=stocks_dir
            )
    typer.echo(f"[intraday] done: {len(paths)}/{len(syms)} symbol(s) have data.")


@app.command(name="dollar-bars")
def dollar_bars_cmd(
    symbols: list[str] = typer.Argument(..., help="Tickers with existing minute data"),
    target_bars_per_day: int = typer.Option(26, help="Dollar-bar cadence target"),
    stocks_dir: str | None = typer.Option(None, help="Override the intraday parquet root"),
) -> None:
    """(Re)derive dollar bars from already-backfilled minute bars."""
    from ophir.agent.intraday import data

    for sym in symbols:
        data.build_and_persist_dollar_bars(
            sym, target_bars_per_day=target_bars_per_day, stocks_dir=stocks_dir
        )


@app.command()
def gate(
    symbols: list[str] | None = typer.Argument(None, help="Universe; omit for the default 20"),
    cost_bps: float = typer.Option(5.0, help="Assumed round-trip cost in basis points"),
    stocks_dir: str | None = typer.Option(None, help="Override the intraday parquet root"),
) -> None:
    """Run the P2 baseline + G2 cost gate and print the PASS/FAIL verdict."""
    from ophir.agent.intraday.gate import run_gate

    run_gate(symbols=symbols or None, cost_bps=cost_bps, stocks_dir=stocks_dir)
