# Research Plan: from Reproduction to Sparse+Sink

Working document — updated 2026-08-20. Repo: this one. Compute: SNAP/ILC
(`infolab`, 4×A100 jobs; per-user cap 10×A100 / 2×B200-unusable-multi-GPU),
HAIC as overflow.

## Where we are

- **Reproduction done** (see README Results): both seed pairs favor Full
  AttnRes (+0.0211 / +0.0045 nats, noise floor 0.0080), dynamics signatures
  reproduced. Naive Full AttnRes costs **6.6× step time** at 124M.
- **New measurement** (not in the paper): learned depth-attention decomposes
  into a **static sparse skeleton + thin near-uniform tail**. Deep consumers:
  top-8 static set holds ~80% mass, effective sources ~8-12/25; per-token
  concentration ≈ token-averaged concentration (token_top4 0.61 vs mean_top4
  0.57) and argmax rarely moves → **routing is NOT token-adaptive at this
  scale**. Consistent across both seeds.
- **Design implied by the data — "Sparse+Sink" depth attention**:
  `h_l = softmax over { sink = uniform running sum of all sources (O(1));
  k static skip sources (O(k)) }`. Static *wiring*, dynamic *weights*
  (keys stay input-dependent). Dodges both failure modes the paper tested:
  sliding-window (no long-range) and DenseFormer (static weights).
  Contains standard residual (k=0) and Full AttnRes (k=L−1) as limits;
  zero-init function preservation carries over.
- **Novelty check (2026-08-20)**: DAR (arXiv:2607.18730) is dual-stream
  reciprocal addressing — no sparsity, no wiring analysis. CLSA/IndexCache
  sparsify the *sequence* dimension. Our niche (skeleton analysis + learned
  static wiring + sink decomposition) is open. Field moves fast → arXiv early.

## Target

Primary: **ICML 2027** (deadline ~late Jan 2027). Fallback: **COLM 2027**
(~Mar 2027). Staging: an efficient-training workshop + arXiv preprint as soon
as multi-scale results exist. Evidence bar set by DAR: 0.1-1B dense + ablations
+ analysis (their 7B MoE is aspirational for us, not required if the
efficiency/analysis story is sharp).

## Phase A — free experiments (this week, no training)

All run on ILC from existing checkpoints. Tool: `analysis/wiring_stability.py`.

| # | Experiment | Command sketch | Decision rule |
|---|---|---|---|
| A1 | Cross-seed wiring agreement | `wiring_stability.py --ckpts out/attnres_124m/latest.pt out/attnres_124m_s42/latest.pt` | deep-band top-8 overlap ≥0.7 → wiring is a property of architecture+data; seed-union wiring is a principled static choice. <0.5 → wiring is run-specific → Sparse+Sink must learn wiring end-to-end |
| A2 | Wiring crystallization over training | `--ckpts ckpt_001201 ckpt_002401 ckpt_003601 latest` (attnres_124m) | overlap(t, final) ≥0.8 by step 2400 → "crystallizes early" → profile-then-prune schedule is viable + paper figure |
| A3 | Domain shift of routing | prepare code + zh val sets (`data/prepare_domain_val.py`), then `--ckpts latest --data_dirs fineweb_edu domain_code domain_zh` | small TV distance → routing is universal (strengthens static wiring); large → domain-adaptive routing is itself a finding |

## Phase B — Sparse+Sink at 124M (decision gate, ~1 day of compute)

1. Implement in `attnres/depth_attention.py` (new mode `sparse_sink`, wiring
   from Phase A1 union or learned; zero-init equivalence test extended).
2. Runs (seed 1337 first): k=4 and k=8, everything else identical to the
   existing protocol.
3. **Gate**: val loss ≤ 3.253 (within noise of Full AttnRes's 3.2453) AND
   step time ≤ 1.4× baseline. Pass → commit to Phase C. Fail → write the
   analysis + honest negative as a workshop paper; project still complete.

## Phase C — the paper's evidence (conditional on B, ~4-6 weeks queue time)

- Scales: 350M (~16h/run) and 760M (~3.2d/run) × {baseline, Full AttnRes,
  Sparse+Sink}, 2 seeds at 124M/350M, 1 seed at 760M + fitted trends.
- Baselines reviewers will demand: Block AttnRes (implement, it's simple),
  mHC (public code), DAR (cite; implement only if time allows).
- Skeleton analysis repeated at every scale (does the structure persist? does
  token-adaptivity emerge?) — this is the analysis section.
- Optional strengtheners: Muon arm, 8192-ctx arm.
- Budget: ~400-700 A100-hours total. Within ILC quota if queued steadily.

## Phase D — engineering + writing

- Fused kernel (Triton) for sparse+sink: target ≤1.2× step time; also fix the
  Full AttnRes fp32-promotion/checkpoint waste for an honest speed baseline.
- arXiv preprint the moment 350M results are in; workshop submission in
  parallel; main-conference draft by early Jan.

## Risk register

- **Scooping** (field ships monthly): mitigate with early arXiv + narrow,
  well-measured claims. Re-run the novelty search before each milestone.
- **Effect sizes near noise** (0.01-0.03 nats): every claim paired-seed,
  matched data order, noise floor reported. Never claim from n=1.
- **Adaptivity may emerge at scale**: scope claims to tested scales; measure
  per-token stats at each scale (cheap, same tooling).
- **Sparse+Sink may tie but not beat Full**: that IS the win (same quality,
  ~5× cheaper depth attention) — frame as quality-per-cost from day one.
- **ILC queue contention**: ampere fleet is busy; submit long-running arms
  early, keep HAIC as overflow, avoid blackwell1 for multi-GPU (NCCL hang).

## Immediate next actions

1. Run A1/A2/A3 on ILC (commands above; scripts in repo).
2. Report numbers back into this doc under "Phase A results".
3. Implement Sparse+Sink behind a config flag; extend equivalence tests.
4. Queue Phase B runs.

## Phase A results

(to be filled)
