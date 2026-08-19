"""Config dataclasses and YAML loading with dot-path CLI overrides."""

from dataclasses import dataclass, field, fields
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    dim: int = 768
    ffn_dim: int = 2048
    seq_len: int = 2048
    rope_theta: float = 10000.0
    # "standard": h = h + f(norm(h)).  "attnres": each sublayer input is a
    # learned softmax-attention mix over all preceding sublayer outputs.
    residual_mode: str = "standard"
    depth_attn_kernel: str = "softmax"  # softmax | sigmoid
    depth_attn_key_norm: bool = True
    tie_embeddings: bool = True
    init_std: float = 0.02
    # RMSNorm eps. Zero-init AttnRes is EXACTLY function-preserving only at
    # eps=0: a consumer averaging S sources shrinks the stream by S, so each
    # reader RMSNorm sees an effective eps of S^2 * eps (625x at the final
    # consumer of a 12-layer model) -- deviation is O(S^2 * eps / RMS^2), about
    # 1% on logits / ~0.003 nats of CE at init-scale activations with the
    # default 1e-6, decaying as activations grow during training. Tests use
    # 0.0 for exactness; keep this in mind for the retrofit experiment, where
    # pretrained late-layer activations may be small.
    norm_eps: float = 1e-6

    def __post_init__(self):
        assert self.residual_mode in ("standard", "attnres")
        assert self.depth_attn_kernel in ("softmax", "sigmoid")
        assert self.dim % self.n_head == 0


@dataclass
class TrainConfig:
    run_name: str = "run"
    out_dir: str = "out"
    data_dir: str = "data/fineweb_edu"
    seed: int = 1337
    micro_batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 4800
    lr: float = 6.0e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_tokens: int = 10_485_760
    checkpoint_every: int = 1200
    log_every: int = 10
    wandb: bool = False
    wandb_project: str = "attnres"
    compile: bool = False


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(
        model=ModelConfig(**raw.get("model", {})),
        train=TrainConfig(**raw.get("train", {})),
    )


def _cast(value: str, current: Any) -> Any:
    if isinstance(current, bool):
        if value.lower() in ("true", "1", "yes"):
            return True
        if value.lower() in ("false", "0", "no"):
            return False
        raise ValueError(f"cannot parse bool from {value!r}")
    if current is None:
        return value
    return type(current)(value)


def apply_overrides(cfg: Config, pairs: list[str]) -> Config:
    """Apply CLI overrides of the form 'model.n_layer=4' or 'train.lr=3e-4'."""
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"override must look like section.key=value, got {pair!r}")
        dotted, value = pair.split("=", 1)
        section_name, _, key = dotted.partition(".")
        section = getattr(cfg, section_name, None)
        if section is None or not key:
            raise ValueError(f"unknown config section in override {pair!r}")
        if key not in {f.name for f in fields(section)}:
            raise ValueError(f"unknown config key in override {pair!r}")
        setattr(section, key, _cast(value, getattr(section, key)))
    # re-run validation
    section_types = {"model": ModelConfig, "train": TrainConfig}
    for name, typ in section_types.items():
        current = getattr(cfg, name)
        post = getattr(typ, "__post_init__", None)
        if post is not None:
            post(current)
    return cfg
