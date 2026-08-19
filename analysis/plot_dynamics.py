"""Training-dynamics diagnostics from a checkpoint (paper Figs. 5 & 8 analogs).

Produces, under <run_dir>/dynamics/:
  alpha_heatmap.png   learned depth-attention weights, consumer x source (attnres only)
  source_norms.png    RMS of each residual source (embedding + every sublayer output)
  grad_norms.png      per-block parameter gradient norm from one backward pass
  dynamics.json       raw numbers behind the plots

Usage:
    python analysis/plot_dynamics.py --ckpt out/attnres_124m/latest.pt
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attnres import GPT, ModelConfig  # noqa: E402
from attnres.dataloader import TokenShardReader  # noqa: E402

# Reference palette (light mode): validated categorical slots + chrome.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Single-hue sequential ramp (blue 100 -> 700) for the heatmap.
BLUES = LinearSegmentedColormap.from_list("blues_ref", ["#cde2fb", "#3987e5", "#0d366b"])
BLUES.set_bad(SURFACE)


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/fineweb_edu")
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_dir", type=str, default=None, help="default: <ckpt dir>/dynamics")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location="cpu")
    mc = ModelConfig(**ckpt["config"]["model"])
    model = GPT(mc)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    is_attnres = mc.residual_mode == "attnres"

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), "dynamics")
    os.makedirs(out_dir, exist_ok=True)

    reader = TokenShardReader(args.data_dir, "val", args.batch_size, mc.seq_len)

    # ---- forward stats, averaged over batches -------------------------------
    n_sources = 2 * mc.n_layer + 1
    rms_sum = np.zeros(n_sources)
    alpha_sums = [np.zeros(i + 1) for i in range(n_sources)] if is_attnres else None
    with torch.no_grad():
        for b in range(args.batches):
            x, _ = reader.batch(b)
            _, _, stats = model(x.to(device), return_stats=True)
            rms_sum += np.array(stats.source_rms)
            if is_attnres:
                for i, a in enumerate(stats.alphas):
                    alpha_sums[i] += a.numpy()
    source_rms = rms_sum / args.batches

    # ---- one backward pass for per-block grad norms -------------------------
    x, y = reader.batch(args.batches)
    _, loss = model(x.to(device), y.to(device))
    loss.backward()
    grad_norms = []
    for i in range(mc.n_layer):
        sq = sum(
            p.grad.detach().float().pow(2).sum().item()
            for name, p in model.named_parameters()
            if name.startswith(f"blocks.{i}.") and p.grad is not None
        )
        grad_norms.append(float(np.sqrt(sq)))
    model.zero_grad(set_to_none=True)

    # ---- plots --------------------------------------------------------------
    series = SERIES[1] if is_attnres else SERIES[0]
    label = "AttnRes" if is_attnres else "baseline"

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style_axes(ax)
    ax.plot(range(n_sources), source_rms, color=series, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("residual source index (0 = token embedding)", color=MUTED)
    ax.set_ylabel("output RMS", color=MUTED)
    ax.set_title(f"Residual-source magnitudes ({label})", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "source_norms.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), facecolor=SURFACE)
    style_axes(ax)
    ax.plot(range(mc.n_layer), grad_norms, color=series, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("transformer block index", color=MUTED)
    ax.set_ylabel("parameter grad norm", color=MUTED)
    ax.set_title(f"Per-block gradient norms ({label})", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "grad_norms.png"), dpi=200)
    plt.close(fig)

    alphas_json = None
    if is_attnres:
        mat = np.full((n_sources, n_sources), np.nan)
        for i, a in enumerate(alpha_sums):
            mat[i, : i + 1] = a / args.batches
        fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
        ax.set_facecolor(SURFACE)
        im = ax.imshow(mat, cmap=BLUES, aspect="auto", vmin=0.0)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.set_label("mean attention weight", color=MUTED)
        cbar.ax.tick_params(colors=MUTED, labelsize=8)
        ax.set_xlabel("source sublayer (0 = embedding)", color=MUTED)
        ax.set_ylabel("consumer sublayer (last row = final aggregation)", color=MUTED)
        ax.set_title("Learned depth-attention weights", color=INK, fontsize=11)
        ax.tick_params(colors=MUTED, labelsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "alpha_heatmap.png"), dpi=200)
        plt.close(fig)
        alphas_json = [list(a / args.batches) for a in alpha_sums]

    with open(os.path.join(out_dir, "dynamics.json"), "w") as f:
        json.dump(
            {
                "ckpt": args.ckpt,
                "residual_mode": mc.residual_mode,
                "source_rms": list(source_rms),
                "block_grad_norms": grad_norms,
                "alphas": alphas_json,
            },
            f,
            indent=2,
        )
    print(f"wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
