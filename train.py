"""DDP pretraining loop for the AttnRes reproduction.

Usage (4 GPUs):
    torchrun --standalone --nproc_per_node=4 train.py --config configs/baseline_124m.yaml
Single GPU / CPU debug:
    python train.py --config configs/baseline_124m.yaml train.micro_batch_size=2

Extra positional args are config overrides, e.g. train.seed=42 model.n_layer=6.
Data order is a pure function of (step, rank, world size, batch shape), so two
runs with the same config and world size see identical batches.
"""

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from attnres import GPT, apply_overrides, load_config
from attnres.dataloader import TokenShardReader

# Dense BF16 peak per GPU (no sparsity), used only for MFU logging.
# Auto-detected from the device name; override with PEAK_FLOPS_PER_GPU env var.
_PEAK_FLOPS = {
    "B200": 2.2e15,
    "H100": 989e12,  # SXM; PCIe H100 is 756e12 -- use the env override there
    "A100": 312e12,
    "L40S": 181e12,
}


def peak_flops_per_gpu() -> float:
    env = os.environ.get("PEAK_FLOPS_PER_GPU")
    if env:
        return float(env)
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        for key, value in _PEAK_FLOPS.items():
            if key in name:
                return value
    return _PEAK_FLOPS["L40S"]


def get_lr(step: int, cfg) -> float:
    # warmup denominator is warmup_steps+1 so the peak is hit exactly once (at
    # step==warmup_steps, t=0 in the decay branch); decay denominator is
    # max_steps-1-warmup_steps so the last executed step reaches min_lr.
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / (cfg.warmup_steps + 1)
    min_lr = cfg.lr * cfg.min_lr_ratio
    t = (step - cfg.warmup_steps) / max(1, cfg.max_steps - 1 - cfg.warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * min(t, 1.0))) * (cfg.lr - min_lr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    parser.add_argument("overrides", nargs="*", help="config overrides like train.lr=3e-4")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.overrides)
    mc, tc = cfg.model, cfg.train

    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        rank, world, local_rank = 0, 1, 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    master = rank == 0

    # Same seed on every rank -> identical model init without a broadcast.
    torch.manual_seed(tc.seed)
    torch.set_float32_matmul_precision("high")

    run_dir = os.path.join(tc.out_dir, tc.run_name)
    if master:
        os.makedirs(run_dir, exist_ok=True)

    train_reader = TokenShardReader(tc.data_dir, "train", tc.micro_batch_size, mc.seq_len, rank, world)
    val_reader = TokenShardReader(tc.data_dir, "val", tc.micro_batch_size, mc.seq_len, rank, world)

    model = GPT(mc).to(device)
    raw_model = model  # un-compiled, un-wrapped handle for checkpoints/optimizer
    optimizer = raw_model.configure_optimizer(
        tc.weight_decay, tc.lr, (tc.beta1, tc.beta2), device_type
    )

    start_step = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        # Data order is a pure function of (step, rank, world, batch shape);
        # resuming with different values silently trains on a different token
        # stream and invalidates the controlled comparison -- hard-fail.
        saved = ckpt.get("config", {})
        live = {"model": asdict(mc), "train": asdict(tc)}
        for sec, key in [
            ("train", "micro_batch_size"),
            ("train", "seed"),
            ("train", "data_dir"),
            ("model", "seq_len"),
        ]:
            if saved.get(sec, {}).get(key) != live[sec][key]:
                raise SystemExit(
                    f"resume mismatch: {sec}.{key} checkpoint="
                    f"{saved.get(sec, {}).get(key)!r} != live={live[sec][key]!r}"
                )
        # The true data-order invariant is the PRODUCT world * grad_accum:
        # any split with the same product consumes identical per-step token
        # sets with identical mean gradients (only fp reduction order moves),
        # so a world=1 checkpoint may legally resume on 2 or 4 GPUs.
        saved_product = ckpt.get("world_size", world) * saved.get("train", {}).get(
            "grad_accum_steps", tc.grad_accum_steps
        )
        live_product = world * tc.grad_accum_steps
        if saved_product != live_product:
            raise SystemExit(
                f"resume mismatch: world*grad_accum checkpoint={saved_product} "
                f"!= live={live_product} (data order would diverge)"
            )
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        if master:
            print(f"resumed from {args.resume} at step {start_step}")

    if tc.compile:
        model = torch.compile(model)
    if ddp:
        # the only buffers are the constant rope tables, identical on every
        # rank -- skip the per-forward buffer broadcast
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False)

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    tokens_per_step = tc.micro_batch_size * mc.seq_len * world * tc.grad_accum_steps
    flops_per_token = raw_model.flops_per_token()
    peak_flops = peak_flops_per_gpu()
    if master:
        n = raw_model.num_params()
        print(f"params: {n['total']/1e6:.1f}M total, {n['non_embedding']/1e6:.1f}M non-embedding")
        print(f"tokens/step: {tokens_per_step:,} | total tokens: {tokens_per_step * tc.max_steps/1e9:.2f}B")
        needed = tokens_per_step * tc.max_steps
        if train_reader.total < needed:
            print(
                f"WARNING: train split has {train_reader.total:,} tokens but the run "
                f"consumes {needed:,} ({needed / train_reader.total:.2f} epochs via wrap-around)"
            )
        log_path = os.path.join(run_dir, "log.jsonl")
        if args.resume is not None and os.path.exists(log_path):
            # drop records at/after the resume step so re-logged steps don't
            # leave duplicate/conflicting entries
            with open(log_path) as f:
                kept = [ln for ln in f if json.loads(ln).get("step", 0) < start_step]
            with open(log_path, "w") as f:
                f.writelines(kept)
        log_file = open(log_path, "a")
        if tc.wandb:
            import wandb

            wandb.init(
                project=tc.wandb_project,
                name=tc.run_name,
                id=tc.run_name,
                resume="allow",
                config=asdict(cfg),
            )

    def log(record: dict):
        if not master:
            return
        log_file.write(json.dumps(record) + "\n")
        log_file.flush()
        if tc.wandb:
            import wandb

            wandb.log(record, step=record["step"])

    def to_device(x, y):
        if device_type == "cuda":
            return (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        return x.to(device), y.to(device)

    @torch.no_grad()
    def run_val() -> float:
        model.eval()
        iters = max(1, tc.eval_tokens // (val_reader.chunk * world))
        total = torch.zeros((), device=device)
        for i in range(iters):
            x, y = to_device(*val_reader.batch(i))
            with autocast:
                _, loss = model(x, y)
            total += loss.detach()
        total /= iters
        if ddp:
            dist.all_reduce(total, op=dist.ReduceOp.AVG)
        model.train()
        return total.item()

    def save_checkpoint(step: int):
        if not master:
            return
        ckpt = {
            "model": raw_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "world_size": world,
            "config": {"model": asdict(mc), "train": asdict(tc)},
        }
        path = os.path.join(run_dir, f"ckpt_{step:06d}.pt")
        torch.save(ckpt, path)
        tmp = os.path.join(run_dir, "latest.pt.tmp")
        torch.save(ckpt, tmp)
        os.replace(tmp, os.path.join(run_dir, "latest.pt"))  # atomic on POSIX
        print(f"saved checkpoint {path}")

    model.train()
    t_last = time.time()
    last_log_step = start_step - 1
    for step in range(start_step, tc.max_steps):
        lr = get_lr(step, tc)
        for g in optimizer.param_groups:
            g["lr"] = lr

        loss_accum = torch.zeros((), device=device)
        for micro in range(tc.grad_accum_steps):
            x, y = to_device(*train_reader.batch(step * tc.grad_accum_steps + micro))
            if ddp:
                # sync gradients only on the last micro-step
                model.require_backward_grad_sync = micro == tc.grad_accum_steps - 1
            with autocast:
                _, loss = model(x, y)
            loss = loss / tc.grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        last = step == tc.max_steps - 1
        if step % tc.log_every == 0 or last:
            if device_type == "cuda":
                torch.cuda.synchronize()
            if ddp:
                dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
            now = time.time()
            dt = (now - t_last) / max(1, step - last_log_step)
            t_last = now
            last_log_step = step
            tok_per_s = tokens_per_step / dt
            mfu = flops_per_token * tok_per_s / (world * peak_flops)
            record = {
                "step": step,
                "tokens": (step + 1) * tokens_per_step,
                "train_loss": round(loss_accum.item(), 5),
                "lr": lr,
                "grad_norm": round(float(grad_norm), 4),
                "dt_ms": round(dt * 1e3, 1),
                "tok_per_s": round(tok_per_s),
                "mfu": round(mfu, 4),
            }
            log(record)
            if master:
                print(
                    f"step {step:5d} | loss {record['train_loss']:.4f} | lr {lr:.2e} "
                    f"| {record['tok_per_s']:,} tok/s | mfu {mfu*100:.1f}%"
                )

        if (step % tc.eval_every == 0 and step > start_step) or last:
            pause = time.time()
            val_loss = run_val()
            log({"step": step, "tokens": (step + 1) * tokens_per_step, "val_loss": round(val_loss, 5)})
            if master:
                print(f"step {step:5d} | val loss {val_loss:.4f}")
            t_last += time.time() - pause  # keep val time out of the MFU window

        if (step % tc.checkpoint_every == 0 and step > start_step) or last:
            pause = time.time()
            save_checkpoint(step + 1)
            t_last += time.time() - pause

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
