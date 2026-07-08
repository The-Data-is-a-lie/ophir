# Retrain runbook — WSL/Linux (branch `adopt-upstream-model`)

After adopting upstream's model, the old checkpoint no longer loads (`FEATURE_DIM`
13→12 + architecture change). This produces a fresh canonical base checkpoint on
a Triton-capable Linux/WSL box, where flex-attention compiles at full speed
(native Windows lacks Triton and must fall back to the slow eager path).

## 0. One-time WSL setup

```bash
# In WSL2 with the NVIDIA CUDA driver installed (RTX 4080):
nvidia-smi                      # must succeed inside WSL

# Option A (recommended): reuse the existing Windows working tree + its ingested
# data. The branch is already checked out and the .ophir/data parquet is present.
cd /mnt/c/Users/Daniel/AppData/Local/FoundryVTT/Data/ophir

# Option B: fresh WSL-native clone (faster training I/O, but re-ingest needed —
#   git clone <origin-url> ophir && cd ophir && git switch adopt-upstream-model
#   then re-ingest your universe with your usual backfill before step 2).

uv sync --group dev             # installs the Linux torch cu130 wheels + Triton
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # -> True
```

## 1. Curate the training universe

Writes the quality allowlist + stats JSON (skips thin-history / junk tickers):

```bash
uv run ophir curate --use-sp500        # drop --use-sp500 to keep every ingested name
```

## 2. Train the base model — with the free ~2× skill win

`--val-identity` turns on `val_rank_ic` + selects the best checkpoint on
`val_rank_ic_near` (near-horizon IC), instead of the `val_loss` criterion that is
anti-aligned with cross-sectional skill (~0.5× peak). This is the operating-point
fix from his forecast-ceiling investigation.

```bash
uv run ophir train --val-identity --use-quality-allowlist --use-sp500 --epochs 10
# candidates are written to:  src/ophir/.ophir/model/candidates/
```

Optional, higher-quality (slower): search hyperparameters first, then train the
winner:

```bash
uv run ophir sweep --trials 50 --confirm-top 5     # Optuna, scored on val_rank_ic
```

## 3. Promote the best candidate to canonical

There is no auto-selection — promotion is an explicit copy. **Prefer a
`val_rank_ic_near`-named candidate** over a `val_loss` one:

```bash
ls -lh src/ophir/.ophir/model/candidates/
cp src/ophir/.ophir/model/candidates/<chosen-val_rank_ic_near...>.ckpt \
   src/ophir/.ophir/model/ophir-ohlc-base-best.ckpt      # register.BASE_BEST_CKPT
```

## 4. Evaluate

```bash
uv run ophir evaluate      # reports rank_ic_near + the near-horizon IC-decay table
```

Sanity bar (from his investigation): pooled peak rank-IC ~0.027, with skill
concentrated at the near horizon (a costless 1-day reversal rule ~0.053 is the
ceiling). Confirm `rank_ic_near` is clearly positive and above the null.

## 5. Smoke-test inference

```bash
uv run ophir predict AAPL   # loads ophir-ohlc-base-best.ckpt, prints a forecast
```

## 6. Back on Windows — Phase 6 switch (do this yourself)

1. If you trained via `/mnt/c`, the canonical `.ckpt` is already in the Windows
   tree. If you trained on a WSL-native clone, copy
   `src/ophir/.ophir/model/ophir-ohlc-base-best.ckpt` back into the Windows repo's
   `.ophir/model/` dir. (`.ophir/` is git-ignored / machine-local.)
2. Point the `ophir-bot` daily runtime at the `adopt-upstream-model` branch.
3. Dry-run the daily flow and diff against the old model before going live:
   ```bash
   uv run ophir trade <symbols> --broker alpaca --dry-run
   ```
4. Only then execute. Instant rollback the whole time:
   `git switch start-to-trade_2`.

## Notes

- Your daily `ophir trade` command is unchanged. His deterministic trading core
  is available separately under `ophir trading` (propose/gate/record/…).
- Old `val_loss` candidate zoo cleanup (optional, frees disk) is in
  `docs/checkpoint-promotion.md`.
```
