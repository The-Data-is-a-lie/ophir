"""Alpaca paper-trading integration for Ophir.

Stores API credentials under ``.ophir/alpaca_paper.json`` (key + secret only,
never logged) and exposes a Typer sub-application (:data:`app`) with commands
for configuring credentials, checking account status, and placing orders.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Annotated, Any, cast

if TYPE_CHECKING:
    import requests

import typer

from ophir.register import OPHIR_DIR

app = typer.Typer(help="Alpaca paper-trading commands")

_CREDS_FILE = os.path.join(OPHIR_DIR, "alpaca_paper.json")
_BASE_URL = "https://paper-api.alpaca.markets/v2"


# ---------------------------------------------------------------------------
# Credential helpers
# ---------------------------------------------------------------------------


def _save_creds(key: str, secret: str) -> None:
    with open(_CREDS_FILE, "w") as f:
        json.dump({"key": key, "secret": secret}, f)


def _load_creds() -> tuple[str, str]:
    if not os.path.exists(_CREDS_FILE):
        typer.echo(
            "No Alpaca credentials found. Run `ophir trade configure` first.",
            err=True,
        )
        raise typer.Exit(1)
    with open(_CREDS_FILE) as f:
        data = json.load(f)
    return data["key"], data["secret"]


def _headers(key: str, secret: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
        "Content-Type": "application/json",
    }


def _raise(resp: requests.Response) -> None:
    if not resp.ok:
        try:
            msg = resp.json().get("message", resp.text)
        except Exception:
            msg = resp.text
        typer.echo(f"Error {resp.status_code}: {msg}", err=True)
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Alpaca REST helpers
# ---------------------------------------------------------------------------


def _get(path: str) -> Any:
    import requests

    key, secret = _load_creds()
    resp = requests.get(f"{_BASE_URL}{path}", headers=_headers(key, secret))
    _raise(resp)
    return resp.json()


def _post(path: str, payload: dict[str, object]) -> Any:
    import requests

    key, secret = _load_creds()
    resp = requests.post(f"{_BASE_URL}{path}", headers=_headers(key, secret), json=payload)
    _raise(resp)
    return resp.json()


def get_account() -> dict[str, Any]:
    """Return the Alpaca paper-trading account object."""
    return cast("dict[str, Any]", _get("/account"))


def list_positions() -> list[dict[str, Any]]:
    """Return all current Alpaca paper-trading positions."""
    return cast("list[dict[str, Any]]", _get("/positions"))


def place_market_order(
    symbol: str,
    side: str,
    *,
    qty: float | None = None,
    notional: float | None = None,
) -> dict[str, Any]:
    """Place a market order (``qty`` shares or ``notional`` dollars)."""
    payload: dict[str, object] = {
        "symbol": symbol.upper(),
        "side": side,
        "type": "market",
        "time_in_force": "day",
    }
    if notional is not None:
        payload["notional"] = str(notional)
    else:
        payload["qty"] = str(qty)
    return cast("dict[str, Any]", _post("/orders", payload))


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def configure(
    key: Annotated[
        str, typer.Option(prompt=True, hide_input=False, help="Alpaca paper API key ID")
    ],
    secret: Annotated[
        str, typer.Option(prompt=True, hide_input=True, help="Alpaca paper secret key")
    ],
) -> None:
    """Save Alpaca paper-trading credentials to disk."""
    _save_creds(key, secret)
    typer.echo(f"Credentials saved to {_CREDS_FILE}")


@app.command()
def account() -> None:
    """Print a summary of your Alpaca paper-trading account."""
    data = get_account()

    typer.echo(f"Account:       {data['account_number']}")
    typer.echo(f"Status:        {data['status']}")
    typer.echo(f"Cash:          ${float(data['cash']):,.2f}")
    typer.echo(f"Portfolio:     ${float(data['portfolio_value']):,.2f}")
    typer.echo(f"Buying power:  ${float(data['buying_power']):,.2f}")


@app.command()
def buy(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol, e.g. AAPL")],
    qty: Annotated[float, typer.Option(help="Number of shares (fractional OK)")] = 1.0,
    notional: Annotated[
        float | None, typer.Option(help="Dollar amount to buy instead of qty")
    ] = None,
) -> None:
    """Place a paper market buy order."""
    order = place_market_order(symbol, "buy", qty=qty, notional=notional)
    typer.echo(f"Order ID:  {order['id']}")
    typer.echo(f"Symbol:    {order['symbol']}")
    typer.echo(f"Side:      {order['side']}")
    typer.echo(f"Qty:       {order.get('qty') or order.get('notional')}")
    typer.echo(f"Status:    {order['status']}")


@app.command()
def sell(
    symbol: Annotated[str, typer.Argument(help="Ticker symbol, e.g. AAPL")],
    qty: Annotated[float, typer.Option(help="Number of shares to sell")] = 1.0,
) -> None:
    """Place a paper market sell order."""
    order = place_market_order(symbol, "sell", qty=qty)
    typer.echo(f"Order ID:  {order['id']}")
    typer.echo(f"Symbol:    {order['symbol']}")
    typer.echo(f"Side:      {order['side']}")
    typer.echo(f"Qty:       {order['qty']}")
    typer.echo(f"Status:    {order['status']}")


@app.command()
def orders() -> None:
    """List all open paper-trading orders."""
    order_list = _get("/orders")

    if not order_list:
        typer.echo("No open orders.")
        return

    for o in order_list:
        qty_or_notional = o.get("qty") or f"${o.get('notional')}"
        typer.echo(
            f"{o['id']}  {o['symbol']:6s}  {o['side']:4s}  {qty_or_notional:>10}  {o['status']}"
        )


@app.command()
def positions() -> None:
    """List all current paper-trading positions."""
    pos_list = list_positions()

    if not pos_list:
        typer.echo("No open positions.")
        return

    typer.echo(f"{'Symbol':<8} {'Qty':>8} {'Avg Entry':>12} {'Market Val':>12} {'P&L':>10}")
    typer.echo("-" * 54)
    for p in pos_list:
        typer.echo(
            f"{p['symbol']:<8}"
            f" {float(p['qty']):>8.4f}"
            f" {float(p['avg_entry_price']):>12.2f}"
            f" {float(p['market_value']):>12.2f}"
            f" {float(p['unrealized_pl']):>10.2f}"
        )


@app.command()
def rebalance(
    top_k: Annotated[int, typer.Option(help="Number of symbols to hold")] = 10,
    horizon: Annotated[int, typer.Option(help="Forecast days to score over")] = 5,
    budget_frac: Annotated[float, typer.Option(help="Fraction of buying power to deploy")] = 0.95,
    min_score: Annotated[float, typer.Option(help="Minimum predicted return to buy")] = 0.0,
    refresh: Annotated[bool, typer.Option(help="Refresh OHLC data from MASSIVE first")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run/--execute", help="Print the plan without placing orders")
    ] = True,
) -> None:
    """Score the S&P 500 with the model and rebalance into the top-K names.

    Requires a CUDA GPU and a trained base checkpoint. Defaults to a dry run;
    pass ``--execute`` to actually place orders on the paper account.
    """
    from ophir import strategy
    from ophir.register import load_base_model_ckpt
    from ophir.ticker import get_sp_500_symbols

    symbols = get_sp_500_symbols()

    if refresh:
        typer.echo(f"Refreshing OHLC data for {len(symbols)} symbols from MASSIVE...")
        refreshed = strategy.refresh_universe(symbols)
        typer.echo(f"Refreshed {len(refreshed)}/{len(symbols)} symbols.")

    typer.echo("Loading model...")
    model = load_base_model_ckpt().cuda().eval()
    handler = strategy.build_handler(symbols)

    typer.echo(f"Scoring {len(handler.stocks)} symbols...")
    scores = strategy.score_universe(model, handler, horizon=horizon)
    targets = strategy.rank_top_k(scores, top_k, min_score=min_score)

    pos_list = list_positions()
    held = [p["symbol"] for p in pos_list]
    buying_power = float(get_account()["buying_power"])
    sells, buys = strategy.compute_rebalance(held, targets, buying_power, budget_frac)

    typer.echo("\nTop-K targets:")
    for symbol in targets:
        typer.echo(f"  {symbol:<8} score={scores[symbol]:+.5f}")
    typer.echo("\nSells:")
    for symbol in sells:
        typer.echo(f"  SELL {symbol}")
    typer.echo("\nBuys:")
    for symbol, amount in buys:
        typer.echo(f"  BUY  {symbol:<8} ${amount:,.2f}")

    if dry_run:
        typer.echo("\nDry run -- no orders placed. Re-run with --execute to trade.")
        return

    for symbol in sells:
        pos = next(p for p in pos_list if p["symbol"] == symbol)
        order = place_market_order(symbol, "sell", qty=float(pos["qty"]))
        typer.echo(f"SELL {symbol}: {order['status']}")
    for symbol, amount in buys:
        order = place_market_order(symbol, "buy", notional=amount)
        typer.echo(f"BUY  {symbol}: {order['status']}")
