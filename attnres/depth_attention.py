"""Depth-wise attention over preceding sublayer outputs (Attention Residuals).

Standard PreNorm residuals give every sublayer the SAME uniformly-weighted sum
of the embedding and all preceding sublayer outputs -- i.e. depth-wise linear
attention with fixed unit weights. AttnRes replaces that fixed accumulation
with learned softmax attention over depth (Moonshot AI, arXiv:2603.15031):

    h_l = sum_i alpha_{i->l} * v_i,   alpha = softmax_i( q_l . RMSNorm(v_i) )

where v_0 is the token embedding, v_i are raw sublayer outputs, keys equal
values, and q_l is a single learned per-consumer d-dim pseudo-query. q_l is
input-independent by design (all consumers' weights are computable in
parallel at decode time) and zero-initialized, so training starts from a
uniform average -- which is function-preserving in a PreNorm model because
every reader of the stream is scale-invariant (RMSNorm; exact at norm eps=0).

The whole op runs with autocast DISABLED: under bf16 autocast the sources are
mixed-precision (fp32 embedding, bf16 sublayer outputs) and autocast would
silently downcast the explicitly fp32-cast einsums back to bf16. Sources are
promoted to a common dtype (fp32 under bf16 training -- matching the standard
arm, whose residual stream also accumulates in fp32), and the depth softmax
runs in at-least-fp32.
"""

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    # bf16/fp16 get an fp32 softmax; fp32/fp64 stay as-is
    # (fp64 matters for the exact-equivalence unit test).
    return torch.promote_types(dtype, torch.float32)


def _stack_sources(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Stack sources into [S, B, T, d], explicitly promoting to a common dtype
    (don't rely on torch.stack's implicit mixed-dtype behavior)."""
    dtype = values[0].dtype
    for t in values[1:]:
        dtype = torch.promote_types(dtype, t.dtype)
    return torch.stack([t.to(dtype) for t in values])


class DepthAttention(nn.Module):
    """One depth-attention aggregation: mixes all previous sublayer outputs."""

    def __init__(self, dim: int, kernel: str = "softmax", key_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        assert kernel in ("softmax", "sigmoid")
        self.kernel = kernel
        self.key_norm = key_norm
        self.eps = eps
        # Zero init => uniform attention at the start of training. The authors
        # report nonzero init causes training volatility.
        self.query = nn.Parameter(torch.zeros(dim))
        # Learnable gain of the key RMSNorm. Mathematically it folds into the
        # query (q . (g * k_hat) == (q*g) . k_hat) but is kept to match the
        # paper's parameterization; it never breaks zero-init equivalence.
        self.key_gain = nn.Parameter(torch.ones(dim)) if key_norm else None

    def _alpha(self, v: torch.Tensor) -> torch.Tensor:
        """Attention weights over depth. v: [S, B, T, d] -> alpha: [S, B, T]."""
        cd = _compute_dtype(v.dtype)
        with torch.autocast(device_type=v.device.type, enabled=False):
            k = v.to(cd)
            if self.key_norm:
                k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                q = (self.query * self.key_gain).to(cd)
            else:
                q = self.query.to(cd)
            logits = torch.einsum("d,sbtd->sbt", q, k)
            if self.kernel == "softmax":
                return torch.softmax(logits, dim=0)
            return torch.sigmoid(logits)

    def _attend(self, *values: torch.Tensor) -> torch.Tensor:
        v = _stack_sources(values)
        alpha = self._alpha(v)
        with torch.autocast(device_type=v.device.type, enabled=False):
            return torch.einsum("sbt,sbtd->btd", alpha.to(v.dtype), v)

    def forward(self, values: list[torch.Tensor]) -> torch.Tensor:
        # Checkpointed so the stacked [S, B, T, d] prefix (and its promoted
        # copies) is not retained per consumer -- Full AttnRes would otherwise
        # keep O(L^2) activation copies alive. The sources themselves are
        # already retained by autograd, so recompute here is cheap.
        if torch.is_grad_enabled() and any(v.requires_grad for v in values):
            return checkpoint(self._attend, *values, use_reentrant=False)
        return self._attend(*values)

    @torch.no_grad()
    def alpha_summary(self, values: list[torch.Tensor]) -> torch.Tensor:
        """Mean attention weight per source, averaged over batch and tokens.

        Returns a fp32 CPU tensor of shape [S]; used by analysis scripts.
        """
        alpha = self._alpha(_stack_sources(tuple(values)))
        return alpha.float().mean(dim=(1, 2)).cpu()


class SparseSinkDepthAttention(nn.Module):
    """Sparse+Sink depth attention (see PLAN.md).

    Motivated by the measured structure of Full AttnRes weights (static sparse
    skeleton + thin near-uniform tail): one softmax over K statically wired
    sources plus a "sink" candidate holding the mean of all NON-wired sources.
    The sink is maintained from a running sum, so a consumer reads O(K)
    sources instead of O(l) -- the wiring is static, but the weights stay
    input-dependent through the keys.

    The sink logit carries a fixed log(l-K) bias, so at zero init the softmax
    weights are exactly (l-K)/l on the sink and 1/l on each wired source --
    i.e. the uniform average over ALL sources. Function preservation vs
    standard residuals therefore carries over unchanged (exact at norm eps=0).
    """

    def __init__(self, dim: int, wiring: list[int], n_sources: int, eps: float = 1e-6):
        super().__init__()
        wiring = sorted(dict.fromkeys(int(i) for i in wiring))
        assert wiring and all(0 <= i < n_sources for i in wiring)
        self.wiring = wiring
        self.n_sources = n_sources
        self.n_rest = n_sources - len(wiring)  # sources represented by the sink
        self.sink_bias = math.log(self.n_rest) if self.n_rest > 0 else 0.0
        self.eps = eps
        self.query = nn.Parameter(torch.zeros(dim))  # zero init -> uniform average
        self.key_gain = nn.Parameter(torch.ones(dim))

    def _attend(self, *candidates: torch.Tensor) -> torch.Tensor:
        v = _stack_sources(candidates)  # [C, B, T, d], C = K(+1)
        cd = _compute_dtype(v.dtype)
        with torch.autocast(device_type=v.device.type, enabled=False):
            k = v.to(cd)
            k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            q = (self.query * self.key_gain).to(cd)
            logits = torch.einsum("d,sbtd->sbt", q, k)
            if self.n_rest > 0:
                bias = torch.zeros(logits.shape[0], dtype=cd, device=logits.device)
                bias[0] = self.sink_bias
                logits = logits + bias.view(-1, 1, 1)
            alpha = torch.softmax(logits, dim=0)
            return torch.einsum("sbt,sbtd->btd", alpha.to(v.dtype), v)

    def forward(self, values: list[torch.Tensor], running_sum: torch.Tensor) -> torch.Tensor:
        wired = [values[i] for i in self.wiring]
        if self.n_rest > 0:
            sink = (running_sum - torch.stack(wired).sum(dim=0)) / self.n_rest
            candidates = (sink, *wired)
        else:
            candidates = tuple(wired)
        # Checkpointed like Full AttnRes: without it every consumer retains
        # ~2.5 fp32 copies of its candidate stack for backward, which OOMs an
        # 80GB A100 at k=8 (25 consumers x 10 candidates); the recompute of
        # this small op chain is nearly free.
        if torch.is_grad_enabled() and any(c.requires_grad for c in candidates):
            return checkpoint(self._attend, *candidates, use_reentrant=False)
        return self._attend(*candidates)
