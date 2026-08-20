"""Per-token depth-attention concentration analysis.

The token-AVERAGED alpha (dynamics.json) showed moderate concentration for
deep consumers (top-4 mass ~0.57, ~8-12 effective sources of 25). That mean
can hide two very different realities:

  (a) per-token alpha is equally diffuse            -> static sparse wiring
      has a limited ceiling; budgeted-k designs only
  (b) per-token alpha is sharp but peaks move across
      tokens (averaging smears them)                -> routing is genuinely
      token-adaptive; dynamic top-k / conditioned queries are the right design

This script measures which. Decisive statistics per consumer:
  token_top4      median over tokens of top-4 alpha mass within the token
                  (compare against mean_top4: a large excess signals (b))
  static8_overlap median over tokens of the mass a token places on the
                  GLOBAL top-8 source set (from mean alpha). High => a static
                  8-wide wiring captures per-token mass even if peaks move
                  within the set; low + high token_top4 => dynamic routing.
  argmax_moves    number of distinct sources that are some token's top-1
                  (>1% of tokens) -- how much the peak position wanders.

Usage:
    python analysis/per_token_alpha.py --ckpt out/attnres_124m/latest.pt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attnres import GPT, ModelConfig  # noqa: E402
from attnres.dataloader import TokenShardReader  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/fineweb_edu")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--out", type=str, default=None, help="default: <ckpt dir>/dynamics/per_token_alpha.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    mc = ModelConfig(**ckpt["config"]["model"])
    assert mc.residual_mode == "attnres", "per-token alpha only exists in attnres mode"
    model = GPT(mc)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # capture every consumer's per-token alpha by wrapping _alpha; under
    # no_grad the checkpointed path is bypassed, so each consumer fires once
    captured = {}

    def wrap(idx, orig):
        def hooked(v):
            alpha = orig(v)
            captured.setdefault(idx, []).append(alpha.detach().float().cpu())
            return alpha

        return hooked

    for idx, da in enumerate(model.depth_attns):
        da._alpha = wrap(idx, da._alpha)

    reader = TokenShardReader(args.data_dir, "val", args.batch_size, mc.seq_len)
    with torch.no_grad():
        for b in range(args.batches):
            x, _ = reader.batch(b)
            model(x.to(device))

    rows = []
    print(
        f"{'consumer':>8} {'S':>3} {'mean_top4':>9} {'tok_top1':>8} {'tok_top4':>8} "
        f"{'tok_top8':>8} {'tok_eff':>7} {'static8':>8} {'argmax_moves':>12}"
    )
    for idx in sorted(captured):
        # [S, n_tokens] with tokens pooled across batches and positions
        a = torch.cat([t.flatten(1) for t in captured[idx]], dim=1)
        S, n = a.shape
        if S < 6:
            continue
        mean_alpha = a.mean(dim=1)
        mean_top4 = mean_alpha.sort(descending=True).values[:4].sum().item()
        srt = a.sort(dim=0, descending=True).values
        tok_top1 = srt[0].median().item()
        tok_top4 = srt[:4].sum(dim=0).median().item()
        tok_top8 = srt[:8].sum(dim=0).median().item() if S >= 8 else 1.0
        eff = (a.sum(dim=0).pow(2) / a.pow(2).sum(dim=0)).median().item()
        static8 = mean_alpha.argsort(descending=True)[: min(8, S)]
        static8_overlap = a[static8].sum(dim=0).median().item()
        counts = torch.bincount(a.argmax(dim=0), minlength=S).float() / n
        argmax_moves = int((counts > 0.01).sum().item())
        rows.append(
            dict(
                consumer=idx, sources=S, mean_top4=round(mean_top4, 4),
                token_top1=round(tok_top1, 4), token_top4=round(tok_top4, 4),
                token_top8=round(tok_top8, 4), token_eff_sources=round(eff, 2),
                static8_overlap=round(static8_overlap, 4), argmax_moves=argmax_moves,
            )
        )
        print(
            f"{idx:>8} {S:>3} {mean_top4:9.3f} {tok_top1:8.3f} {tok_top4:8.3f} "
            f"{tok_top8:8.3f} {eff:7.1f} {static8_overlap:8.3f} {argmax_moves:>12}"
        )

    deep = [r for r in rows if r["sources"] >= 13]
    if deep:
        med = lambda k: sorted(r[k] for r in deep)[len(deep) // 2]  # noqa: E731
        print("\ndeep consumers (S>=13), medians:")
        print(f"  mean_top4       {med('mean_top4'):.3f}   (token-averaged concentration)")
        print(f"  token_top4      {med('token_top4'):.3f}   (per-token concentration)")
        print(f"  static8_overlap {med('static8_overlap'):.3f}   (mass captured by a static 8-source wiring)")
        print("verdict hints: token_top4 >> mean_top4 -> peaks are token-adaptive;")
        print("               static8_overlap >= ~0.85 -> static sparse wiring suffices anyway")

    out_path = args.out or os.path.join(os.path.dirname(args.ckpt), "dynamics", "per_token_alpha.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"ckpt": args.ckpt, "batches": args.batches, "rows": rows}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
