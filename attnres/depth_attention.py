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
