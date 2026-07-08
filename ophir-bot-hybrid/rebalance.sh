#!/usr/bin/env bash
# Daily hybrid-model paper-trade rebalance for the ISOLATED fresh Alpaca account.
#
# Run from the separate ~/ophir-hybrid checkout (its own .ophir data/model/audit
# and reports), so nothing here can touch the primary burn-in's numbers. Linux
# mirror of ../ophir-bot/rebalance.ps1 (the primary, Windows-scheduled account).
#
# MONEY SAFETY: --broker alpaca always hits Alpaca's PAPER endpoint (paper=True is
# hardcoded in AlpacaPaperBroker). MODE decides whether orders are actually placed:
#   MODE=--dry-run  (default) prints the plan only; nothing is filled.
#   MODE=--execute  submits paper orders so a fresh equity curve accrues.
# Flip to --execute only after a dry-run you have reviewed.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$HERE/.." && pwd)"          # the ~/ophir-hybrid checkout root
ENV_FILE="$HERE/.env"

# ---- knobs (env-overridable) ------------------------------------------------
WATCHLIST="${WATCHLIST:-$HERE/watchlist.txt}"
TOP_K="${TOP_K:-20}"
BROKER="${BROKER:-alpaca}"          # "alpaca" = fresh paper account; "paper" = in-proc sim
MODE="${MODE:---dry-run}"           # --dry-run plans only; --execute places paper orders
# -----------------------------------------------------------------------------

LOG_DIR="$HERE/logs"; mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/rebalance-$(date +%F).log"
log() { echo "=== $(date '+%F %T') $* ===" | tee -a "$LOG"; }

cd "$PROJECT"

# Load AGENT_* from the gitignored .env (config.py reads real env vars + a CWD .env).
if [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a
else
  log "WARNING .env not found at $ENV_FILE -- '--broker alpaca' will fail"
fi
# Isolate the per-stock dossiers + trade tracker under this bot dir.
export AGENT_REPORT_DIR="${AGENT_REPORT_DIR:-$HERE/reports}"
mkdir -p "$AGENT_REPORT_DIR"

# Read the watchlist (skip blanks + #comments, strip CR).
mapfile -t SYMBOLS < <(sed 's/\r$//' "$WATCHLIST" | grep -vE '^[[:space:]]*(#|$)' | awk '{$1=$1};1')
if [ "${#SYMBOLS[@]}" -eq 0 ]; then log "ABORT no symbols in $WATCHLIST"; exit 1; fi

# 1) Refresh today's bars for the whole watchlist in ONE process.
log "data refresh start (${#SYMBOLS[@]} symbols)"
OPHIR_SYMS="$(IFS=,; echo "${SYMBOLS[*]}")" \
  uv run python -c "import os; from ophir.agent.ingest import ingest_many; ingest_many([s for s in os.environ['OPHIR_SYMS'].split(',') if s])" >>"$LOG" 2>&1 \
  || log "data refresh FAILED -- continuing with existing data"

# 2) Score + rebalance (dry-run by default).
log "rebalance start (broker=$BROKER $MODE top_k=$TOP_K report_dir=$AGENT_REPORT_DIR)"
uv run ophir trade "${SYMBOLS[@]}" --top-k "$TOP_K" --broker "$BROKER" "$MODE" >>"$LOG" 2>&1
code=$?
log "rebalance exit $code"

# 3) Learning-metrics snapshot (non-fatal; never changes the exit code).
uv run ophir report metrics --snapshot >>"$LOG" 2>&1 || true
log "metrics snapshot done"

exit "$code"
