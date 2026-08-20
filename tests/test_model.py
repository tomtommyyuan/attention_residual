"""Sanity tests: shapes, gradient flow, alpha normalization, optimizer coverage."""

import json

import pytest
import torch

from attnres import GPT, DepthAttention, ModelConfig
from attnres.depth_attention import SparseSinkDepthAttention


def small_cfg(**kwargs) -> ModelConfig:
    base = dict(vocab_size=256, n_layer=4, n_head=4, dim=64, ffn_dim=128, seq_len=32)
    base.update(kwargs)
    return ModelConfig(**base)


@pytest.mark.parametrize("mode", ["standard", "attnres"])
def test_forward_backward(mode):
    torch.manual_seed(0)
    model = GPT(small_cfg(residual_mode=mode))
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, 256)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name} (DDP would hang on unused params)"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_softmax_alpha_sums_to_one():
    torch.manual_seed(0)
    da = DepthAttention(dim=16)
    with torch.no_grad():
        da.query.normal_()
    values = [torch.randn(2, 8, 16) for _ in range(5)]
    alpha = da._alpha(torch.stack(values))
    assert alpha.shape == (5, 2, 8)
    assert torch.allclose(alpha.sum(dim=0), torch.ones(2, 8), atol=1e-6)
    summary = da.alpha_summary(values)
    assert summary.shape == (5,)
    assert torch.allclose(summary.sum(), torch.tensor(1.0), atol=1e-5)


def test_optimizer_covers_all_params():
    model = GPT(small_cfg(residual_mode="attnres"))
    opt = model.configure_optimizer(0.1, 1e-3, (0.9, 0.95), device_type="cpu")
    in_groups = {id(p) for g in opt.param_groups for p in g["params"]}
    for name, p in model.named_parameters():
        assert id(p) in in_groups, f"{name} missing from optimizer"
    # depth-attention queries/gains are 1-D and must not be weight-decayed
    no_decay_ids = {id(p) for p in opt.param_groups[1]["params"]}
    for i, da in enumerate(model.depth_attns):
        assert id(da.query) in no_decay_ids
        assert id(da.key_gain) in no_decay_ids


def test_depth_param_overhead_is_tiny():
    base = GPT(small_cfg())
    attn = GPT(small_cfg(residual_mode="attnres"))
    n_base = base.num_params()["total"]
    n_attn = attn.num_params()["total"]
    consumers = 2 * 4 + 1  # 2 per block + final aggregation
    assert n_attn - n_base == consumers * (64 + 64)  # query + key gain each


def _toy_wiring_file(tmp_path, n_layer=4, k=8):
    wiring = [list(range(c + 1))[::-1][:k] for c in range(2 * n_layer + 1)]
    path = tmp_path / "wiring.json"
    path.write_text(json.dumps({"wiring": wiring}))
    return str(path)


def test_sparse_sink_forward_backward(tmp_path):
    torch.manual_seed(0)
    model = GPT(
        small_cfg(
            residual_mode="sparse_sink",
            depth_attn_k=2,
            depth_wiring_file=_toy_wiring_file(tmp_path, k=2),
        )
    )
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    logits, loss = model(x, y)
    assert logits.shape == (2, 32, 256)
    assert torch.isfinite(loss)
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"no grad for {name} (DDP would hang on unused params)"
        assert torch.isfinite(p.grad).all(), f"non-finite grad for {name}"


def test_sparse_sink_init_output_is_uniform_average():
    # zero-init + log(n_rest) sink bias must yield the exact mean of ALL
    # sources, wired or not -- the function-preservation cornerstone
    torch.manual_seed(0)
    module = SparseSinkDepthAttention(dim=16, wiring=[0, 3], n_sources=6, eps=0.0).double()
    values = [torch.randn(2, 8, 16, dtype=torch.float64) for _ in range(6)]
    running = torch.stack(values).sum(dim=0)
    out = module(values, running)
    expected = torch.stack(values).mean(dim=0)
    assert torch.allclose(out, expected, atol=1e-10)


def test_sparse_sink_bf16_autocast(tmp_path):
    torch.manual_seed(0)
    model = GPT(
        small_cfg(
            residual_mode="sparse_sink",
            depth_attn_k=2,
            depth_wiring_file=_toy_wiring_file(tmp_path, k=2),
        )
    )
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()


@pytest.mark.parametrize("mode", ["standard", "attnres"])
def test_bf16_autocast_forward_backward(mode):
    # mirrors the real training numerics: bf16 autocast with mixed-precision
    # residual sources (fp32 embedding, bf16 sublayer outputs)
    torch.manual_seed(0)
    model = GPT(small_cfg(residual_mode=mode))
    x = torch.randint(0, 256, (2, 32))
    y = torch.randint(0, 256, (2, 32))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        _, loss = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()


def test_depth_attention_ignores_autocast():
    # autocast would silently downcast the explicitly-fp32 einsums to bf16;
    # the op must run at stream precision regardless of ambient autocast
    da = DepthAttention(dim=16)
    values = [torch.randn(2, 8, 16) for _ in range(3)]
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = da(values)
        alpha = da._alpha(torch.stack(values))
    assert out.dtype == torch.float32
    assert alpha.dtype == torch.float32


def test_return_stats():
    torch.manual_seed(0)
    model = GPT(small_cfg(residual_mode="attnres"))
    x = torch.randint(0, 256, (2, 32))
    _, _, stats = model(x, return_stats=True)
    assert len(stats.source_rms) == 2 * 4 + 1  # embedding + 8 sublayer outputs
    assert len(stats.alphas) == 2 * 4 + 1  # 8 sublayer inputs + final aggregation
    # consumer l attends over l sources; the final one sees all 9
    assert [len(a) for a in stats.alphas] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
