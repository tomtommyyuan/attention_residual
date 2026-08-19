"""Exact function-preservation test.

With every depth-attention query zero-initialized, AttnRes computes a uniform
average of the residual sources. A uniform average differs from the standard
residual sum only by a positive per-token scalar, and every reader of the
stream (attention/MLP pre-norms, the final norm) is scale-invariant RMSNorm --
so the logits of a zero-init AttnRes model must EXACTLY match a standard-
residual model with the same weights. We verify this in float64.
"""

import pytest
import torch

from attnres import GPT, ModelConfig


def small_cfg(**kwargs) -> ModelConfig:
    # norm_eps=0 because the scale-invariance argument is exact only without
    # the eps term (see ModelConfig.norm_eps); random weights make zero-norm
    # activations impossible in practice.
    base = dict(
        vocab_size=256, n_layer=4, n_head=4, dim=64, ffn_dim=128, seq_len=32, norm_eps=0.0
    )
    base.update(kwargs)
    return ModelConfig(**base)


def build_pair(**attnres_kwargs):
    torch.manual_seed(0)
    baseline = GPT(small_cfg())
    attnres = GPT(small_cfg(residual_mode="attnres", **attnres_kwargs))
    missing, unexpected = attnres.load_state_dict(baseline.state_dict(), strict=False)
    assert unexpected == []
    assert missing and all("depth_attns" in k for k in missing)
    return baseline.double(), attnres.double()


def sample_input(cfg_vocab=256, batch=2, seq=32):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, cfg_vocab, (batch, seq), generator=g)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"depth_attn_kernel": "sigmoid"},  # sigmoid(0)=0.5 -> still a uniform scale
        {"depth_attn_key_norm": False},
    ],
)
def test_zero_init_matches_standard_residuals(kwargs):
    baseline, attnres = build_pair(**kwargs)
    x = sample_input()
    with torch.no_grad():
        logits_base, _ = baseline(x)
        logits_attn, _ = attnres(x)
    assert torch.allclose(logits_base, logits_attn, atol=1e-8, rtol=1e-8)


def test_nonzero_query_changes_the_function():
    baseline, attnres = build_pair()
    with torch.no_grad():
        attnres.depth_attns[5].query.add_(
            torch.randn(64, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
        )
    x = sample_input()
    with torch.no_grad():
        logits_base, _ = baseline(x)
        logits_attn, _ = attnres(x)
    assert not torch.allclose(logits_base, logits_attn, atol=1e-6, rtol=1e-6)


def test_losses_match_too():
    baseline, attnres = build_pair()
    x = sample_input()
    y = sample_input().clone()
    with torch.no_grad():
        _, loss_base = baseline(x, y)
        _, loss_attn = attnres(x, y)
    assert torch.allclose(loss_base, loss_attn, atol=1e-8, rtol=1e-8)
