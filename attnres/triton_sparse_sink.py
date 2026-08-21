"""Fused Triton kernels for Sparse+Sink depth attention.

The eager path costs ~30 memory round-trips per consumer (stack, fp32
promotion, key-norm chain, two einsums, checkpoint recompute). The fused
kernel does one pass per candidate row held in registers: computes the sink
row from running_sum - sum(wired) on the fly, normalizes keys, applies the
sink-biased softmax, and writes the mixed output -- ~4 round-trips, one
launch. Backward is a mirrored kernel; only alpha ([C, N] fp32, a few MB) is
saved, so no activation checkpointing is needed.

Numerics match the eager path: all math in fp32 registers, same sink-bias
formulation, same eps placement. Parity is enforced by
tests/test_triton_parity.py (CUDA only).
"""

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # CPU-only environments (tests skip the triton path)
    HAS_TRITON = False

MAX_C = 16  # compile-time cap on candidates (sink + wired); k<=15


if HAS_TRITON:

    @triton.jit
    def _fwd_kernel(
        WIRED, RUN, QG, OUT, ALPHA,
        K, n_rest, sink_bias, eps, N, D,
        HAS_SINK: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        cidx = tl.arange(0, MAX_C)

        qg = tl.load(QG + offs, mask=mask, other=0.0).to(tl.float32)
        logits = tl.full([MAX_C], float("-inf"), tl.float32)
        sum_w = tl.zeros([BLOCK_D], tl.float32)

        # pass 1: wired logits (+ accumulate the wired sum for the sink)
        for c in range(K):
            v = tl.load(WIRED + c * N * D + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            sum_w += v
            rms = tl.sqrt(tl.sum(v * v) / D + eps)
            logit = tl.sum(qg * v) / rms
            slot = c + 1 if HAS_SINK else c
            logits = tl.where(cidx == slot, logit, logits)

        if HAS_SINK:
            run = tl.load(RUN + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            sink = (run - sum_w) / n_rest
            rms = tl.sqrt(tl.sum(sink * sink) / D + eps)
            logit = tl.sum(qg * sink) / rms + sink_bias
            logits = tl.where(cidx == 0, logit, logits)

        m = tl.max(logits)
        p = tl.exp(logits - m)
        alpha = p / tl.sum(p)

        # pass 2: weighted sum (wired rows re-read from L2)
        out = tl.zeros([BLOCK_D], tl.float32)
        if HAS_SINK:
            out += tl.sum(tl.where(cidx == 0, alpha, 0.0)) * sink
        for c in range(K):
            v = tl.load(WIRED + c * N * D + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            slot = c + 1 if HAS_SINK else c
            out += tl.sum(tl.where(cidx == slot, alpha, 0.0)) * v

        tl.store(OUT + row * D + offs, out, mask=mask)
        C = K + 1 if HAS_SINK else K
        for c in range(C):
            tl.store(ALPHA + c * N + row, tl.sum(tl.where(cidx == c, alpha, 0.0)))

    @triton.jit
    def _bwd_kernel(
        WIRED, RUN, QG, ALPHA, GOUT,
        GWIRED, GRUN, GQG_PART,
        K, n_rest, eps, N, D,
        HAS_SINK: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK_D)
        mask = offs < D
        cidx = tl.arange(0, MAX_C)

        qg = tl.load(QG + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(GOUT + row * D + offs, mask=mask, other=0.0).to(tl.float32)

        C = K + 1 if HAS_SINK else K
        alpha = tl.full([MAX_C], 0.0, tl.float32)
        for c in range(C):
            a = tl.load(ALPHA + c * N + row)
            alpha = tl.where(cidx == c, a, alpha)

        # pass 1: g_alpha_c = g . v_c  (and rebuild the sink row)
        galpha = tl.zeros([MAX_C], tl.float32)
        sum_w = tl.zeros([BLOCK_D], tl.float32)
        for c in range(K):
            v = tl.load(WIRED + c * N * D + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            sum_w += v
            slot = c + 1 if HAS_SINK else c
            galpha = tl.where(cidx == slot, tl.sum(g * v), galpha)
        sink = tl.zeros([BLOCK_D], tl.float32)
        if HAS_SINK:
            run = tl.load(RUN + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            sink = (run - sum_w) / n_rest
            galpha = tl.where(cidx == 0, tl.sum(g * sink), galpha)

        # softmax backward: g_logit = alpha * (g_alpha - sum(alpha * g_alpha))
        dot = tl.sum(alpha * galpha)
        glogit = alpha * (galpha - dot)

        # pass 2: per-candidate value + key-path grads
        gqg = tl.zeros([BLOCK_D], tl.float32)
        gsink = tl.zeros([BLOCK_D], tl.float32)
        if HAS_SINK:
            a0 = tl.sum(tl.where(cidx == 0, alpha, 0.0))
            gl0 = tl.sum(tl.where(cidx == 0, glogit, 0.0))
            rms = tl.sqrt(tl.sum(sink * sink) / D + eps)
            # logit = (qg . v) / rms ; d/dv = qg/rms - v * (qg.v) / (D * rms^3)
            qv = tl.sum(qg * sink)
            gsink = a0 * g + gl0 * (qg / rms - sink * (qv / (D * rms * rms * rms)))
            gqg += gl0 * (sink / rms)
            tl.store(GRUN + row * D + offs, gsink / n_rest, mask=mask)
        for c in range(K):
            v = tl.load(WIRED + c * N * D + row * D + offs, mask=mask, other=0.0).to(tl.float32)
            slot = c + 1 if HAS_SINK else c
            a_c = tl.sum(tl.where(cidx == slot, alpha, 0.0))
            gl_c = tl.sum(tl.where(cidx == slot, glogit, 0.0))
            rms = tl.sqrt(tl.sum(v * v) / D + eps)
            qv = tl.sum(qg * v)
            gv = a_c * g + gl_c * (qg / rms - v * (qv / (D * rms * rms * rms)))
            gqg += gl_c * (v / rms)
            if HAS_SINK:
                gv -= gsink / n_rest  # wired rows are subtracted inside the sink
            tl.store(GWIRED + c * N * D + row * D + offs, gv, mask=mask)

        tl.store(GQG_PART + row * BLOCK_D + offs, gqg, mask=mask)


class _SparseSinkFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qg, running, n_rest, sink_bias, eps, *wired):
        B, T, D = wired[0].shape
        N = B * T
        wired_stack = torch.stack([w.reshape(N, D).float() for w in wired])  # [K, N, D]
        run = running.reshape(N, D).float().contiguous()
        has_sink = n_rest > 0
        K = len(wired)
        C = K + 1 if has_sink else K
        out = torch.empty(N, D, dtype=torch.float32, device=qg.device)
        alpha = torch.empty(C, N, dtype=torch.float32, device=qg.device)
        BLOCK_D = triton.next_power_of_2(D)
        _fwd_kernel[(N,)](
            wired_stack, run, qg.float().contiguous(), out, alpha,
            K, float(max(n_rest, 1)), float(sink_bias), float(eps), N, D,
            HAS_SINK=has_sink, BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(qg, running, alpha, *wired)
        ctx.meta = (n_rest, eps, B, T, D)
        return out.view(B, T, D)

    @staticmethod
    def backward(ctx, grad_out):
        qg, running, alpha, *wired = ctx.saved_tensors
        n_rest, eps, B, T, D = ctx.meta
        N = B * T
        has_sink = n_rest > 0
        K = len(wired)
        wired_stack = torch.stack([w.reshape(N, D).float() for w in wired])
        run = running.reshape(N, D).float().contiguous()
        g = grad_out.reshape(N, D).float().contiguous()
        BLOCK_D = triton.next_power_of_2(D)
        gwired = torch.empty_like(wired_stack)
        grun = torch.zeros_like(run) if has_sink else None
        gqg_part = torch.empty(N, BLOCK_D, dtype=torch.float32, device=qg.device)
        _bwd_kernel[(N,)](
            wired_stack, run, qg.float().contiguous(), alpha, g,
            gwired, grun if has_sink else run, gqg_part,
            K, float(max(n_rest, 1)), float(eps), N, D,
            HAS_SINK=has_sink, BLOCK_D=BLOCK_D,
        )
        gqg = gqg_part[:, :D].sum(dim=0).to(qg.dtype)
        grun_out = grun.view(B, T, D).to(running.dtype) if has_sink else None
        gws = [
            gwired[i].view(B, T, D).to(wired[i].dtype) for i in range(K)
        ]
        return (gqg, grun_out, None, None, None, *gws)


def sparse_sink_attend(qg, values_wired, running_sum, n_rest, sink_bias, eps):
    """Fused sparse+sink aggregation. Returns [B, T, D] in fp32."""
    return _SparseSinkFn.apply(qg, running_sum, n_rest, sink_bias, eps, *values_wired)
