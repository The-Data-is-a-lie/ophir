# program.md — human search directives for the autoresearch loop

You (the proposer) improve a BERT-style masked transformer that forecasts
three forward OHLC targets per day. **Objective: maximize `rank_ic_near`** —
pooled daily cross-sectional Spearman rank-IC of `r_close` at trading-day
offsets 1–5, on a fixed held-out split. Current baseline is iteration 0 of
`results.tsv`; the 3-seed reference on the full-depth store (2026-07-08) is
mean ≈ +0.096 with LARGE single-seed spread (0.068–0.145 across seeds), so
only changes with a real mechanism will clear ε.

## Ground rules

- One focused change per iteration. Keep `train_experiment.py` runnable and
  self-consistent. Each trial trains seeds 0/1/2 and is judged on the MEAN;
  each seed's run is killed at 25 minutes (the baseline uses ~15), so a
  change that slows training much beyond that wastes the whole trial.
- You may inline any ophir component into `train_experiment.py` (e.g. copy a
  method into `ExperimentPredictor` and modify it) — but never edit files
  under `src/ophir`.
- If you give `ExperimentPredictor` its own `__init__`, it MUST call
  `super().__init__(...)` and `self.save_hyperparameters()`, or your
  checkpoint cannot be reloaded for scoring and the trial is wasted.
- Never touch the sealed `from _sealed import ...` line; never write
  year-like literals (the split lives in the pinned `_sealed.py`).
- Never construct `StockHandler` directly; go through `build_split_handlers`
  (the loop rejects direct `StockHandler` references).
- Simplicity rule: a marginal gain does not justify added complexity. On a
  near-tie, prefer the simpler variant. Reverting a kept-but-marginal
  complexity increase is a valid proposal.
- Every feature must be knowable strictly before the prediction timestamp.
  Never introduce anything that peeks into the response block.

## Known results (do not re-litigate)

- `rezero_lr` dominates hyperparameter importance; `lr`, `loss_decay` matter.
- `rezero_init` tuning does NOT help (multi-seed confirmed). Do not tune it.
- Skill concentrates at offsets 1–5 and dies by offset ~10; the pooled
  90-day objective dilutes it. That is WHY the metric is `rank_ic_near`.
- Plain hyperparameter grid-walking is the Optuna sweep's job, not yours —
  only propose a hyperparameter change with a mechanistic rationale.
- Session 2026-07-08 (4 trials, ε=0.0852, seed-0 baseline 0.145 — a high
  draw of the seed distribution): ALL discarded.
  - Near-horizon loss concentration (exponential half-life, ~37% of loss
    mass on offsets 1–5) scored 0.084 — clearly WORSE than baseline. This
    family has now failed twice; do not re-try loss re-weighting toward the
    near band without a genuinely different mechanism.
  - Volatility-standardization family — per-name Huber knee (0.122),
    vol-normalized regression target (0.101), causal vol-normalized input
    channels (0.115) — clustered below the baseline draw. Single-seed noise
    is ±0.04, so read as "no evidence of gain", not proof of harm. The
    vol-normalized target's h1=0.264 and the input-normalization's
    h5=0.168 were the best sub-metrics seen; unconfirmed.
- Session 2026-07-09 overnight (3-seed means, ε=0.049, baseline 0.09635):
  6 trials, ALL discarded.
  - RankNet pairwise ranking loss on offsets 1–5 at weight 0.01 → 0.088.
    A light auxiliary weight adds nothing; if retrying this family, use a
    materially stronger weight or a listwise objective — not 0.01 again.
  - Shrinking the training response block 90 → 20 days → **−0.016, the
    worst result recorded.** The eval harness scores 90-day response
    windows, so training on a different response geometry is a train/eval
    mismatch, not a framing win. Do NOT change RESPONSE_SIZE.
  - EMA (Polyak, decay 0.999) of weights swapped in for validation and
    checkpointing → **0.105, the best challenger yet** (baseline 0.096;
    bar 0.145). Mechanism directly targets the 0.068–0.145 seed spread.
    Extending it (different decay, averaging window, or combined with
    another variance reducer) is the most promising known lead.
  - Lower-variance checkpoint selection (150 val batches every 1000 steps)
    → 0.092 (neutral). Streamer cache 8 → 32 for batch decorrelation
    → 0.073 (no gain).
  - Interrupted before training (queued, untested): ablate the upside/
    downside auxiliary heads (loss weights 1.0/0.5/0.5 → 1.0/0.0/0.0) so
    the full trunk serves the one channel the metric scores.

## Promising directions (highest leverage first)

1. **Variance reduction on the final iterate.** EMA-for-validation was the
   best challenger (0.105 vs baseline 0.096) — extend that mechanism
   (decay/window variants, or EMA combined with the queued aux-head
   ablation), don't re-propose it verbatim.
2. **Aux-head ablation (queued, untested).** Zero the upside/downside loss
   weights so the trunk serves only `r_close` — interrupted before
   training on 2026-07-09; still the top untested single edit.
3. **Ranking the cross-section** — only with a materially stronger weight
   or a listwise objective (0.01-weight RankNet already showed nothing).
4. **Feature-side ideas** with strict causal lagging — but note the
   vol-normalization family already showed no gain (see Known results);
   prefer a different feature mechanism.
5. Architecture changes last — the evidence says the ceiling is framing,
   not capacity. Do NOT touch `RESPONSE_SIZE` (train/eval geometry
   mismatch; see Known results).

## Measurement honesty (why some wins don't count)

- Acceptance needs mean-over-3-seeds `rank_ic_near > best + ε` (ε set by the
  runner, ≈0.049 — 2·SE of a 3-seed mean at 10k steps). Most true small
  gains will still not clear it, and that is intentional.
- A 10k-step win can be a proxy artifact; champions face more seeds and
  full-budget re-runs at graduation. Prefer changes with a mechanism, not
  a lucky number.

## When stuck

If 3+ consecutive proposals are discarded, switch families (e.g. from loss
shaping to ranking) rather than iterating on the failed idea; consider a
revert-to-simpler proposal if recent kept changes look like noise.
