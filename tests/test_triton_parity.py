"""Numerical parity of the fused Triton sparse+sink kernel vs the eager path.

CUDA-only (skipped elsewhere). Tolerances are fp32-reduction-order level.
"""

import json

import pytest
import torch

from attnres import GPT, ModelConfig
from attnres.depth_attention import SparseSinkDepthAttention
from attnres.triton_sparse_sink import HAS_TRITON

cuda_only = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_TRITON), reason="needs CUDA + triton"
)

ATOL, RTOL = 2e-4, 2e-4


def make_inputs(n_sources, B=2, T=128, dim=768, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    values = [
        torch.randn(B, T, dim, device="cuda", generator=g, requires_grad=True)
        for _ in range(n_sources)
    ]
    running = torch.stack(values).sum(dim=0)
    return values, running


def run_module(impl, wiring, n_sources, values, running, seed=1):
    module = SparseSinkDepthAttention(
        dim=values[0].shape[-1], wiring=wiring, n_sources=n_sources, impl=impl
    ).cuda()
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        module.query.copy_(torch.randn(module.query.shape, generator=g) * 0.5)
        module.key_gain.copy_(1.0 + 0.1 * torch.randn(module.key_gain.shape, generator=g))
    out = module(values, running)
    grad_out = torch.randn(out.shape, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    grads = torch.autograd.grad(
        out, [*values, module.query, module.key_gain], grad_outputs=grad_out
    )
    return out.detach(), [gr.detach() for gr in grads]


@cuda_only
@pytest.mark.parametrize("wiring,n_sources", [([0, 3, 7, 11, 14, 17, 20, 22], 25), ([0, 2, 4], 25), (list(range(8)), 8)])
def test_kernel_matches_eager(wiring, n_sources):
    values, running = make_inputs(n_sources)
    out_e, grads_e = run_module("eager", wiring, n_sources, values, running)
    out_t, grads_t = run_module("triton", wiring, n_sources, values, running)
    assert torch.allclose(out_e, out_t, atol=ATOL, rtol=RTOL), (out_e - out_t).abs().max()
    for i, (ge, gt) in enumerate(zip(grads_e, grads_t)):
        assert torch.allclose(ge, gt, atol=ATOL, rtol=RTOL), f"grad {i}: {(ge - gt).abs().max()}"


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / (a.norm() + 1e-8)).item()


@cuda_only
def test_kernel_bf16_inputs_close_to_eager():
    # Training path: bf16 wired inputs. The two implementations round in
    # different places (eager casts alpha and the output to bf16; the kernel
    # keeps fp32 registers throughout), so elementwise comparison trips over
    # single-ULP bf16 disagreements. Relative L2 error is the right gauge:
    # "same math up to bf16 quantization noise" = well under 2%.
    n_sources, wiring = 25, [0, 3, 7, 11, 14, 17, 20, 22]
    g = torch.Generator(device="cuda").manual_seed(0)
    values = [
        torch.randn(2, 128, 768, device="cuda", generator=g, dtype=torch.float32).bfloat16().requires_grad_()
        for _ in range(n_sources)
    ]
    running = torch.stack(values).float().sum(dim=0)
    out_e, grads_e = run_module("eager", wiring, n_sources, values, running)
    out_t, grads_t = run_module("triton", wiring, n_sources, values, running)
    assert torch.isfinite(out_t).all()
    assert rel_l2(out_e, out_t) < 0.02, rel_l2(out_e, out_t)
    for i, (ge, gt) in enumerate(zip(grads_e, grads_t)):
        assert torch.isfinite(gt.float()).all(), f"grad {i}: nonfinite"
        assert rel_l2(ge, gt) < 0.02, f"grad {i}: rel_l2 {rel_l2(ge, gt):.4f}"


@cuda_only
def test_full_model_parity(tmp_path):
    wiring = [list(range(c + 1))[::-1][:8] for c in range(2 * 4 + 1)]
    wf = tmp_path / "wiring.json"
    wf.write_text(json.dumps({"wiring": wiring}))
    cfg = dict(
        vocab_size=256, n_layer=4, n_head=4, dim=64, ffn_dim=128, seq_len=32,
        residual_mode="sparse_sink", depth_attn_k=8, depth_wiring_file=str(wf),
    )
    torch.manual_seed(0)
    eager = GPT(ModelConfig(**cfg, depth_attn_impl="eager")).cuda()
    triton_m = GPT(ModelConfig(**cfg, depth_attn_impl="triton")).cuda()
    triton_m.load_state_dict(eager.state_dict())
    # nudge queries off zero so the depth attention actually routes
    with torch.no_grad():
        for m in (eager, triton_m):
            for da in m.depth_attns:
                da.query.normal_(
                    0, 0.3, generator=torch.Generator(device="cuda").manual_seed(3)
                )
    x = torch.randint(0, 256, (2, 32), device="cuda")
    le, _ = eager(x)
    lt, _ = triton_m(x)
    assert torch.allclose(le, lt, atol=5e-4, rtol=5e-4), (le - lt).abs().max()
