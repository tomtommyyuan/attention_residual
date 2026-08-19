"""Overlay validation-loss curves from several runs' log.jsonl files.

Usage:
    python analysis/compare_runs.py out/baseline_124m out/attnres_124m --out compare.png
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reference palette (light mode): fixed categorical order, assigned by
# position on the command line -- never cycled or re-ranked.
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def read_curve(run_dir: str, metric: str):
    # dedup by step (keep the last occurrence -- crash-resumes re-log steps)
    # and sort by tokens so resumed logs don't zigzag on the x-axis
    recs = {}
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        for line in f:
            rec = json.loads(line)
            if metric in rec:
                recs[rec["step"]] = (rec["tokens"], rec[metric])
    pts = sorted(recs.values())
    return [p[0] for p in pts], [p[1] for p in pts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="run directories containing log.jsonl")
    parser.add_argument("--labels", nargs="*", default=None)
    parser.add_argument("--metric", type=str, default="val_loss")
    parser.add_argument("--out", type=str, default="compare.png")
    args = parser.parse_args()

    assert len(args.run_dirs) <= len(SERIES), f"at most {len(SERIES)} runs per chart"
    labels = args.labels or [os.path.basename(os.path.normpath(d)) for d in args.run_dirs]

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.75)
    ax.set_axisbelow(True)

    finals = []
    for i, (run_dir, label) in enumerate(zip(args.run_dirs, labels)):
        xs, ys = read_curve(run_dir, args.metric)
        assert xs, f"no {args.metric!r} records in {run_dir}/log.jsonl"
        ax.plot(xs, ys, color=SERIES[i], linewidth=2, label=label)
        # selective direct label at the line end, in text ink (identity is
        # carried by the adjacent colored line, not by colored text)
        ax.annotate(
            f" {label}: {ys[-1]:.4f}",
            xy=(xs[-1], ys[-1]),
            fontsize=9,
            color=INK,
            va="center",
        )
        finals.append((label, xs[-1], ys[-1]))

    ax.set_xlabel("training tokens", color=MUTED)
    ax.set_ylabel(args.metric.replace("_", " "), color=MUTED)
    ax.set_title(f"{args.metric.replace('_', ' ')} vs tokens", color=INK, fontsize=11)
    ax.legend(frameon=False, labelcolor=INK, fontsize=9)
    ax.margins(x=0.12)  # room for the end-of-line labels
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)

    print(f"{'run':30s} {'tokens':>14s} {args.metric:>12s}")
    for label, tok, val in finals:
        print(f"{label:30s} {tok:14,d} {val:12.5f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
