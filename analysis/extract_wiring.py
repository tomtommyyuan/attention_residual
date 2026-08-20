"""Extract static top-k wiring from a trained Full-AttnRes model's mean
depth-attention weights (as saved by plot_dynamics.py in dynamics.json).

Writes a ranked wiring file consumed by residual_mode="sparse_sink": for each
consumer, the source indices sorted by descending mean alpha, truncated to
--k_max. Configs pick a prefix via model.depth_attn_k, so one file serves
k=4 and k=8. Phase-A results justify using one seed's wiring (cross-seed
top-8 overlap 0.83; domain-universal).

Usage:
    python analysis/extract_wiring.py \
        --dynamics results/attnres_124m/dynamics/dynamics.json \
        --out configs/wiring_124m.json
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamics", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--k_max", type=int, default=8)
    args = parser.parse_args()

    with open(args.dynamics) as f:
        d = json.load(f)
    alphas = d["alphas"]
    assert alphas, "dynamics.json has no alphas (not an attnres run?)"

    wiring = []
    for a in alphas:
        ranked = sorted(range(len(a)), key=lambda i: a[i], reverse=True)
        wiring.append(ranked[: args.k_max])

    with open(args.out, "w") as f:
        json.dump(
            {
                "source_dynamics": args.dynamics,
                "source_ckpt": d.get("ckpt"),
                "k_max": args.k_max,
                "wiring": wiring,
            },
            f,
            indent=2,
        )
    kept = [len(w) for w in wiring]
    print(f"wrote {args.out}: {len(wiring)} consumers, wired sources per consumer {min(kept)}-{max(kept)}")


if __name__ == "__main__":
    main()
