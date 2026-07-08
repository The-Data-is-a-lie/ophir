"""``ophir report`` — offline analytics over the trading-agent audit trail.

Only ``typer`` is imported at module load; the metrics engine (pandas + price
data) is imported lazily inside each command, so mounting this sub-app in
``ophir.cli`` does not affect the daily trading path.
"""

from __future__ import annotations

import typer

app = typer.Typer(help="Offline analytics over the trading-agent audit trail.")


@app.command()
def metrics(
    audit_file: str | None = typer.Option(
        None, help="Path to agent-audit.jsonl (default: <DATA_DIR>)"
    ),
    stocks_dir: str | None = typer.Option(None, help="Override the daily stocks parquet root"),
    snapshot: bool = typer.Option(
        False, "--snapshot", help="Append a headline row to the metrics history for trend-tracking"
    ),
) -> None:
    """Learning metrics (IC, decile spread, hit rate, decision breakdown) from the audit trail."""
    from ophir.agent.metrics import run_metrics_report

    run_metrics_report(audit_file, stocks_dir=stocks_dir, snapshot=snapshot)


@app.command()
def history() -> None:
    """Show the metrics-history trend (how IC / LLM-vs-quant firm up over runs)."""
    from ophir.agent.metrics import run_history_report

    run_history_report()


@app.command()
def attribution(
    cost_bps: float = typer.Option(3.5, help="Cost per unit turnover, in basis points"),
    audit_file: str | None = typer.Option(
        None, help="Path to agent-audit.jsonl (default: <DATA_DIR>)"
    ),
    stocks_dir: str | None = typer.Option(None, help="Override the daily stocks parquet root"),
) -> None:
    """Counterfactual equity: quant-only vs LLM-only vs SPY, from the decision log."""
    from ophir.agent.metrics import run_attribution_report

    run_attribution_report(audit_file, stocks_dir=stocks_dir, cost_bps=cost_bps)
