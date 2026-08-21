"""Micro-benchmark: eager vs fused-Triton sparse+sink consumer op.

Times forward+backward of a single deep consumer at training shapes and
extrapolates the per-step cost across all consumers/micro-steps.

Usage (on a GPU node):
    python analysis/bench_depth_attn.py --impl eager
    python analysis/bench_depth_attn.py --impl triton
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attnres.depth_attention import SparseSinkDepthAttention  # noqa: E402


def bench(impl, B, T, dim, k, n_sources, iters, dtype):
    module = SparseSinkDepthAttention(
        dim=dim, wiring=list(range(0, 2 * k, 2))[:k], n_sources=n_sources, impl=impl
    ).cuda()
    with torch.no_grad():
        module.query.normal_(0, 0.5)
    values = [
        torch.randn(B, T, dim, device="cuda", dtype=dtype, requires_grad=True)
        for _ in range(n_sources)
    ]
    running = torch.stack(values).float().sum(dim=0)

    def step():
        out = module(values, running)
        out.backward(torch.randn_like(out))
        for v in values:
            v.grad = None

    for _ in range(5):
        step()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["eager", "triton", "both"], default="both")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--n_sources", type=int, default=25)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                        help="use fp32 on pre-Ampere GPUs (no bf16)")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    impls = ["eager", "triton"] if args.impl == "both" else [args.impl]
    results = {}
    for impl in impls:
        ms = bench(impl, args.batch, args.seq, args.dim, args.k, args.n_sources, args.iters, dtype)
        results[impl] = ms
        n_consumers, micro = 2 * 12 + 1, 4
        print(
            f"{impl:7s} fwd+bwd {ms:7.2f} ms/consumer  "
            f"-> ~{ms * n_consumers * micro / 1e3:5.2f} s/step across {n_consumers} consumers x {micro} micro"
        )
    if len(results) == 2:
        print(f"speedup: {results['eager'] / results['triton']:.1f}x")


if __name__ == "__main__":
    main()
