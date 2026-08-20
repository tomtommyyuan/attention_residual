"""Stability of the learned depth-attention wiring across seeds / training
time / data domains.

Computes token-averaged alpha per consumer for every (checkpoint, data_dir)
combination, then reports pairwise agreement between combinations:

  topk_overlap   |top-k(A) ∩ top-k(B)| / min(k, S) per consumer, averaged
                 (how much of the wiring skeleton is shared)
  tv_distance    0.5 * Σ|alpha_A − alpha_B| per consumer, averaged
                 (how different the full weight distributions are)

One tool covers the three Phase-A experiments (see PLAN.md):
  A1 cross-seed:      --ckpts <seed1337 ckpt> <seed42 ckpt>
  A2 crystallization: --ckpts ckpt_001201.pt ckpt_002401.pt ... latest.pt
  A3 domain shift:    --ckpts latest.pt --data_dirs fineweb_edu domain_code ...

Usage:
    python analysis/wiring_stability.py \
        --ckpts out/attnres_124m/latest.pt out/attnres_124m_s42/latest.pt
"""

import argparse
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attnres import GPT, ModelConfig  # noqa: E402
from attnres.dataloader import TokenShardReader  # noqa: E402

DEEP_MIN_SOURCES = 13  # summary band: consumers late enough for selection to matter


def mean_alphas(ckpt_path, data_dir, batches, batch_size, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mc = ModelConfig(**ckpt["config"]["model"])
    assert mc.residual_mode == "attnres", f"{ckpt_path} is not an attnres checkpoint"
    model = GPT(mc)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    reader = TokenShardReader(data_dir, "val", batch_size, mc.seq_len)
    sums = None
    with torch.no_grad():
        for b in range(batches):
            x, _ = reader.batch(b)
            _, _, stats = model(x.to(device), return_stats=True)
            if sums is None:
                sums = [a.double() for a in stats.alphas]
            else:
                sums = [s + a.double() for s, a in zip(sums, stats.alphas)]
    del model
    return [(s / batches).float() for s in sums]


def label_for(ckpt_path, data_dir, multi_data):
    run = os.path.basename(os.path.dirname(ckpt_path))
    step = os.path.splitext(os.path.basename(ckpt_path))[0]
    lab = f"{run}/{step}"
    if multi_data:
        lab += f"@{os.path.basename(os.path.normpath(data_dir))}"
    return lab


def topk_set(alpha, k):
    return set(torch.topk(alpha, min(k, alpha.numel())).indices.tolist())


def compare(alphas_a, alphas_b, k):
    rows = []
    for i, (a, b) in enumerate(zip(alphas_a, alphas_b)):
        s = a.numel()
        if s < 4 or s != b.numel():
            continue
        ov = len(topk_set(a, k) & topk_set(b, k)) / min(k, s)
        tv = 0.5 * (a - b).abs().sum().item()
        rows.append({"consumer": i, "sources": s, "topk_overlap": round(ov, 4), "tv": round(tv, 4)})
    return rows


def band_mean(rows, key, deep):
    vals = [r[key] for r in rows if not deep or r["sources"] >= DEEP_MIN_SOURCES]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--data_dirs", nargs="+", default=["data/fineweb_edu"])
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--detail", action="store_true", help="print per-consumer rows for each pair")
    parser.add_argument("--out", type=str, default="wiring_stability.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    combos = []
    multi_data = len(args.data_dirs) > 1
    for ckpt in args.ckpts:
        for dd in args.data_dirs:
            lab = label_for(ckpt, dd, multi_data)
            print(f"measuring {lab} ...", flush=True)
            combos.append((lab, mean_alphas(ckpt, dd, args.batches, args.batch_size, device)))

    results = []
    print(f"\npairwise agreement (k={args.k}; deep band = consumers with S>={DEEP_MIN_SOURCES}):")
    print(f"{'pair':60s} {'ov_all':>7} {'ov_deep':>8} {'tv_all':>7} {'tv_deep':>8}")
    for (la, aa), (lb, ab) in itertools.combinations(combos, 2):
        rows = compare(aa, ab, args.k)
        deep_rows = [r for r in rows if r["sources"] >= DEEP_MIN_SOURCES]
        summary = {
            "pair": f"{la} vs {lb}",
            "topk_overlap_all": round(band_mean(rows, "topk_overlap", deep=False), 4),
            "topk_overlap_deep": round(band_mean(rows, "topk_overlap", deep=True), 4) if deep_rows else None,
            "tv_all": round(band_mean(rows, "tv", deep=False), 4),
            "tv_deep": round(band_mean(rows, "tv", deep=True), 4) if deep_rows else None,
            "consumers": rows,
        }
        results.append(summary)
        ovd = f"{summary['topk_overlap_deep']:.3f}" if summary["topk_overlap_deep"] is not None else "  n/a"
        tvd = f"{summary['tv_deep']:.3f}" if summary["tv_deep"] is not None else "  n/a"
        print(
            f"{summary['pair']:60s} {summary['topk_overlap_all']:7.3f} {ovd:>8} "
            f"{summary['tv_all']:7.3f} {tvd:>8}"
        )
        if args.detail:
            for r in rows:
                print(f"    consumer {r['consumer']:3d} (S={r['sources']:2d})  overlap {r['topk_overlap']:.2f}  tv {r['tv']:.3f}")

    with open(args.out, "w") as f:
        json.dump({"k": args.k, "batches": args.batches, "pairs": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
