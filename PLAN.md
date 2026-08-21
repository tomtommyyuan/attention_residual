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
- **Overhead-vs-scale crossover figure**: depth-attention step-time overhead
  shrinks ~linearly-in-width relative to matmuls (O(k·d) traffic vs O(d²)
  compute) while the quality gain persists per the paper's scaling law.
  Measure the ratio at 124M/350M/760M with the fused kernel and plot the
  net-win crossover. Honest framing: at 124M every depth-attention variant
  (incl. Full) is net-negative on wall-clock; the contribution is pushing the
  crossover down. The 1.4× Phase-B gate is an engineering gate
  (implementation ≈ algorithm cost), not a net-efficiency claim.
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

## Phase B: COMPLETE — both gates passed (2026-08-22)

- **Quality**: sparse_sink k=8 val 3.2485 (85% of Full's gain, gate ≤3.253).
- **Speed**: fused Triton kernel v2 (bf16 candidate stack, blocked query-grad
  reduction) → 7.0× op speedup, end-to-end **1.35×** baseline step time
  (gate ≤1.4×; was 4.29× eager). Parity: fp32 ≤2e-4 elementwise, bf16
  rel-L2 <2% (eager and kernel round in different places; kernel is the more
  precise side).
- Open: seed-42 k8 replication (queued), then Phase C.

## Phase B result: QUALITY GATE PASSED (2026-08-21)

**sparse_sink k=8: val 3.2485** — recovers **85% of Full AttnRes's gain**
(gate ≤3.253 passed with 0.0045 margin) while reading 9 candidates instead of
up to 25. Dose-response across arms confirms the mass→gain correspondence:
k4 57% mass → 56% gain; k8 ~80% mass → 85% gain; Full 100% → 100%.
Single seed so far — seed-42 replication queued (deliberately reusing the
seed-1337 wiring file: a cross-seed wiring-transfer test, predicted to work
by Phase-A1's 0.83 overlap). Speed remains the open item (eager k8 is 4.3×
baseline step time; checkpoint recompute + unfused op chain) → Triton kernel.

## Phase B interim results (2026-08-21)

- **sparse_sink k=4: val 3.2547** (baseline 3.2664, Full 3.2453; gate ≤3.253).
  Beats baseline by 0.0117 nats = recovers **55% of Full's gain** while its
  wiring captures ~57% of attention mass — an almost linear mass→gain
  correspondence. Testable prediction: k=8 (~80% mass) → val ≈ 3.250 → pass.
  k=8 run queued.
- **Speed**: torch.compile lifts baseline 411k→700k tok/s (54% MFU!) and
  sparse k4 168k→348k. Compiled-vs-compiled ratio **2.0×** (eager was 2.45×;
  gate 1.4× not yet met). Remaining cost: fp32-promoted candidate stream +
  unfused per-consumer op chains → next: bf16 candidate stream (fp32 softmax
  kept) + fused Triton kernel (Phase D pulled forward).
- Decision: science arms stay eager/uncompiled for comparability until the
  kernel lands; then re-baseline everything compiled in Phase C.

## Phase A results (2026-08-20, k=8, deep band = S≥13; raw: wiring_A*.json)

| Exp | Key numbers | Verdict vs decision rule |
|---|---|---|
| A1 cross-seed | deep overlap **0.827**, all 0.881; worst deep consumer 0.625; tv_deep 0.164 | **PASS** (≥0.7): wiring is a property of architecture+data → static wiring from a trained model's top-8 is principled |
| A2 crystallization | deep overlap vs final: step1200 **0.817** → 2400 **0.914** → 3600 **0.962** (monotone; tv shrinking 0.126→0.018) | **PASS** (≥0.8 by 2400): wiring crystallizes by ~25-50% of training → profile-then-prune viable; paper figure |
| A3 domain shift | top-8 overlap vs web: code **0.904** (tv 0.107), zh **0.856** (tv 0.170); shift grows with domain distance | **PASS**: wiring is domain-universal; weight shifts are modest and ordered (code < zh) — a secondary observation worth reporting |

Implications locked in for Phase B: static wiring sourced from seed-1337
final top-8 sets (available in results/attnres_124m/dynamics/dynamics.json);
profile-then-prune becomes a Phase-C refinement; all three findings are
paper-figure material.
