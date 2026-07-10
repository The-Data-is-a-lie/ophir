# ε recalibration — 2026-07-08

Per the harness spec (ε = 2·SE from ≥3 baseline seeds before any unattended
session). First recalibration on the **full-depth store** (same-day deep
re-ingest: 507 tickers, median ~6.5k rows, ~26y; the prior store was the
trading loop's rolling 730-day window, so earlier smoke numbers are void).

## Setup

- Machine: RTX 4080 SUPER (16 GB), torch 2.10.0+cu130 + triton-windows 3.6.
- Baseline: unmodified `train_experiment.py` at `c809640`, 10k steps.
- Scorer: `eval_harness.py` acceptance split (2024–25), seeded 64/507 panel,
  n = 15,744 rows per eval.
- Train wall time: ~820–920 s per seed (drove the 600→1500 s time-box fix).

## Results (`rank_ic_near`)

| seed | rank_ic_near | best ckpt step |
| ---- | ------------ | -------------- |
| 0    | 0.14535      | 8500           |
| 1    | 0.06826      | 3500           |
| 2    | 0.07544      | 5000           |

- mean = **0.09635**, std (ddof=1) = **0.04259**
- **ε = 2·std = 0.0852** (floor 0.0069 not binding)

## Implications

- Single-seed measurement noise at 10k steps is large (seed spread 0.077);
  ε = 0.0852 is intentionally conservative — only large, real effects clear
  the acceptance bar. Session launch: `--epsilon 0.0852`.
- Follow-up worth considering if sessions keep discarding plausible wins:
  average 2–3 seeds per trial in the loop (3× GPU cost) to shrink ε.

## Amendment (2026-07-08 evening): 3-seed trials

Adopted the follow-up above — the loop now trains seeds (0, 1, 2) per trial
and decides on the **mean** `rank_ic_near` (session `s1-20260708` discarded
all four single-seed proposals; the ε=0.0852 bar only admits transformative
effects).

- SE of a 3-seed mean = 0.04259 / √3 = **0.02459** → **ε = 2·SE ≈ 0.049**.
- Trial cost: ~3 × 15 min train + 3 × 10 s eval ≈ 46 min GPU (plus proposer).
- The in-session baseline (iteration 0) now measures the same three seeds as
  this recalibration, so it must reproduce mean **0.09635** exactly
  (deterministic same-store re-run) — a drift means the store changed.
- Seed-count tradeoff on the RTX 4080 SUPER: ε ∝ 1/√n, wall-clock ∝ n —
  5 seeds would give ε ≈ 0.038 at ~77 min/trial (~6 trials/night vs ~9).
  Search stays at 3 seeds; reserve ≥5 seeds for graduating a champion.

## Amendment (2026-07-10): persistent workers + concurrent seeds → ε = 0.033

`persistent_workers=True` (commit `b545356`) changed the streaming interleave,
so the operating point was re-measured (same store as s4, seed 0/1/2, 10k
steps each):

| seed | rank_ic_near |
| ---- | ------------ |
| 0    | 0.08437      |
| 1    | 0.04411      |
| 2    | 0.09843      |

- mean = **0.07564**, std (ddof=1) = **0.02819** → SE of the 3-seed mean =
  0.01628 → **ε = 2·SE = 0.033** (one 3-seed draw, same rigor as the
  original derivation; re-derive if the operating point changes again).
- Timing: sequential 3-seed batch **898 s** (was ~2430 s — worker respawn was
  ~63% of “training” wall). Concurrent 3-way (`--concurrent-seeds 3`)
  reproduced the same seeds **bit-for-bit at 458 s** → adopted as default.
  A full trial is now ~10–13 min including the proposer (was ~45–50 min).
