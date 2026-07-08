# Hybrid-model runbook — isolated retrain + fresh paper account (WSL/Linux)

Branch: `hybrid_model_trade`. Goal: train the adopted upstream model on your data
and run it on a **brand-new Alpaca paper account**, in a **separate checkout** so
its data / model / audit / reports never touch the primary burn-in's numbers.

Why a separate checkout: `OPHIR_DIR` (hence `DATA_DIR`, `MODEL_DIR`, and
`agent-audit.jsonl`) is hardcoded to `<repo>/src/ophir/.ophir` with no env
override, so a second checkout at a different path *is* the isolation boundary.

General model-training details are in `docs/retrain-runbook.md`; this file adds
the isolation + fresh-account specifics.

## 1. Separate checkout (the isolation boundary)

```bash
# CUDA-in-WSL must work first:
nvidia-smi

git clone /mnt/c/Users/Daniel/AppData/Local/FoundryVTT/Data/ophir ~/ophir-hybrid
cd ~/ophir-hybrid
git switch hybrid_model_trade          # the branch you created on Windows
uv sync --group dev
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # -> True
```

`git clone` copies only tracked files, so `~/ophir-hybrid/src/ophir/.ophir` starts
empty — a clean, isolated root.

## 2. Seed the isolated DATA_DIR with price data

```bash
# Fast path: copy the already-ingested parquet from the Windows tree.
cp -r /mnt/c/Users/Daniel/AppData/Local/FoundryVTT/Data/ophir/src/ophir/.ophir/data \
      ~/ophir-hybrid/src/ophir/.ophir/
# (Alternative: re-ingest your watchlist with ingest_many.)
```

## 3. Train the hybrid model (compiled Triton path)

```bash
uv run ophir curate --use-sp500
uv run ophir train --val-identity --use-quality-allowlist --use-sp500 --epochs 10
# --val-identity selects the checkpoint on val_rank_ic_near (the free ~2x skill win).
# best-epoch candidates -> ~/ophir-hybrid/src/ophir/.ophir/model/candidates/
```

## 4. Promote + evaluate

```bash
ls -lh src/ophir/.ophir/model/candidates/
cp src/ophir/.ophir/model/candidates/<chosen-val_rank_ic_near...>.ckpt \
   src/ophir/.ophir/model/ophir-ohlc-base-best.ckpt

uv run ophir evaluate      # paste rank_ic_near + the IC-decay table back for review
uv run ophir predict AAPL  # confirms the promoted checkpoint loads + forecasts
```

**Gate:** don't trade until `rank_ic_near` is clearly positive / above the null.

## 5. Configure the fresh paper account (keys stay git-ignored)

```bash
cp ophir-bot-hybrid/.env.example ophir-bot-hybrid/.env
# Edit ophir-bot-hybrid/.env: paste the NEW paper key/secret, keep
#   AGENT_MODE=paper  AGENT_DRY_RUN=true  AGENT_ALLOW_LIVE=false
chmod +x ophir-bot-hybrid/rebalance.sh
cp /mnt/c/Users/Daniel/ophir-bot/watchlist.txt ophir-bot-hybrid/watchlist.txt   # or curate your own
```

## 6. Dry-run, then go live (paper)

```bash
# Dry-run: prints the gated book; places NOTHING. Paste the output for review.
MODE=--dry-run bash ophir-bot-hybrid/rebalance.sh

# After the dry-run looks right, start the fresh account's equity curve:
MODE=--execute bash ophir-bot-hybrid/rebalance.sh
```

Verify isolation afterward: `~/ophir-hybrid/ophir-bot-hybrid/reports` and
`~/ophir-hybrid/src/ophir/.ophir/agent-audit.jsonl` are the only things written;
the Windows `ophir-bot/reports` and `.ophir` are untouched. `ophir report-trades`
shows a fresh, separate tracker for the new account.

## Notes

- `paper=True` is hardcoded in `AlpacaPaperBroker`; there is no real-money path.
- Rotate the paper keys in Alpaca once set up (they were shared in chat).
- To schedule the hybrid rebalance in WSL, add a cron entry calling
  `MODE=--execute bash ~/ophir-hybrid/ophir-bot-hybrid/rebalance.sh`.
