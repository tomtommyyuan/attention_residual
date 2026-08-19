"""Llama-style decoder-only LM with switchable residual stream.

residual_mode="standard": the usual PreNorm residual stream.
residual_mode="attnres":  Full Attention Residuals -- every attention/MLP
sublayer (and the final output aggregation) reads a learned softmax mix of the
token embedding and all preceding sublayer outputs instead of their fixed sum.

The two modes share every other parameter, so a baseline state_dict loads into
an attnres model with strict=False, which is what the exact-equivalence unit
test exploits.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .depth_attention import DepthAttention, _compute_dtype


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cd = _compute_dtype(x.dtype)
        xf = x.to(cd)
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (out * self.weight.to(cd)).to(x.dtype)


def _rope_tables(head_dim: int, seq_len: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # [T, head_dim/2]
    return torch.cos(freqs), torch.sin(freqs)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, head_dim]; NeoX-style rotation of the two halves.
    cd = _compute_dtype(x.dtype)
    xf = x.to(cd)
    x1, x2 = xf.chunk(2, dim=-1)
    cos = cos[None, None, :, :].to(cd)
    sin = sin[None, None, :, :].to(cd)
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.to(x.dtype)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.dim // cfg.n_head
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.proj.RESIDUAL_SCALE_INIT = True

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)
        self.down.RESIDUAL_SCALE_INIT = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """One transformer block. Residual wiring lives in GPT.forward so the
    attnres mode can address each sublayer output individually."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.mlp = SwiGLU(cfg)


@dataclass
class ForwardStats:
    # RMS of each residual source (token embedding + every sublayer output).
    source_rms: list[float]
    # attnres only: per-consumer mean attention weights over sources ([S] each).
    alphas: list[torch.Tensor]


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.wte.weight

        if cfg.residual_mode == "attnres":
            # One consumer per sublayer input (2 per block) + one final
            # aggregation before the output norm / LM head.
            self.depth_attns = nn.ModuleList(
                DepthAttention(
                    cfg.dim,
                    kernel=cfg.depth_attn_kernel,
                    key_norm=cfg.depth_attn_key_norm,
                    eps=cfg.norm_eps,
                )
                for _ in range(2 * cfg.n_layer + 1)
            )

        cos, sin = _rope_tables(cfg.dim // cfg.n_head, cfg.seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        # DepthAttention params are deliberately untouched (zeros/ones set at
        # construction, and RNG-free so both residual modes consume identical
        # RNG streams -> identical shared-weight init under the same seed).
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            s = std / math.sqrt(2 * self.cfg.n_layer) if hasattr(module, "RESIDUAL_SCALE_INIT") else std
            nn.init.normal_(module.weight, mean=0.0, std=s)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_stats: bool = False,
    ):
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"sequence length {T} > configured {self.cfg.seq_len}"
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        emb = self.wte(idx)
        stats = ForwardStats(source_rms=[], alphas=[]) if return_stats else None

        def record_source(v: torch.Tensor):
            if stats is not None:
                stats.source_rms.append(v.detach().float().pow(2).mean().sqrt().item())

        record_source(emb)
        if self.cfg.residual_mode == "standard":
            h = emb
            for block in self.blocks:
                v = block.attn(block.attn_norm(h), cos, sin)
                record_source(v)
                h = h + v
                v = block.mlp(block.mlp_norm(h))
                record_source(v)
                h = h + v
            h = self.final_norm(h)
        else:
            values = [emb]

            def aggregate(consumer: DepthAttention) -> torch.Tensor:
                if stats is not None:
                    stats.alphas.append(consumer.alpha_summary(values))
                return consumer(values)

            for i, block in enumerate(self.blocks):
                h = aggregate(self.depth_attns[2 * i])
                v = block.attn(block.attn_norm(h), cos, sin)
                record_source(v)
                values.append(v)
                h = aggregate(self.depth_attns[2 * i + 1])
                v = block.mlp(block.mlp_norm(h))
                record_source(v)
                values.append(v)
            h = self.final_norm(aggregate(self.depth_attns[-1]))

        logits = self.lm_head(h)
        loss = None
        if targets is not None:
            # upcast bf16 logits for a stable CE; leave fp32/fp64 untouched
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)).to(_compute_dtype(logits.dtype)),
                targets.reshape(-1),
            )
        if return_stats:
            return logits, loss, stats
        return logits, loss

    def num_params(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        # tied lm_head shares storage with wte, so subtracting wte once
        # yields the non-embedding count either way
        return {"total": total, "non_embedding": total - self.wte.weight.numel()}

    def flops_per_token(self) -> float:
        # 6N for the matmuls (fwd+bwd) + attention-score term, per nanoGPT.
        n = self.num_params()["total"]
        return 6.0 * n + 12.0 * self.cfg.n_layer * self.cfg.dim * self.cfg.seq_len

    def configure_optimizer(
        self, weight_decay: float, lr: float, betas: tuple[float, float], device_type: str
    ) -> torch.optim.AdamW:
        params = [p for p in self.parameters() if p.requires_grad]
        # 1-D params (norm gains, depth-attention queries/gains) get no decay.
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(
            groups, lr=lr, betas=betas, eps=1e-8, fused=(device_type == "cuda")
        )
