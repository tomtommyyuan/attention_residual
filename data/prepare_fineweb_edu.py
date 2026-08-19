"""Download and tokenize FineWeb-Edu into flat uint16 token shards.

Streams HuggingFaceFW/fineweb-edu (sample-10BT by default), tokenizes with the
GPT-2 BPE (tiktoken), and writes headerless little-endian uint16 shards:

    data/fineweb_edu/val_000000.bin      first --val_tokens tokens
    data/fineweb_edu/train_000000.bin    subsequent shards of --shard_size tokens
    ...

Each document is encoded as [<|endoftext|>] + tokens, so the EOT id (50256)
delimits documents. Defaults produce ~2.6B train tokens (~1-2 hours, mostly
tokenization; needs ~5.3 GB of disk).

Usage:
    python data/prepare_fineweb_edu.py --out_dir data/fineweb_edu
"""

import argparse
import json
import multiprocessing as mp
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset

_enc = None


def _init_worker():
    global _enc
    _enc = tiktoken.get_encoding("gpt2")


def _tokenize(text: str) -> np.ndarray:
    tokens = [_enc.eot_token] + _enc.encode_ordinary(text)
    assert max(tokens) < 2**16, "token id exceeds uint16 range"
    return np.array(tokens, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="data/fineweb_edu")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--remote_name", type=str, default="sample-10BT")
    parser.add_argument("--shard_size", type=int, default=100_000_000, help="tokens per train shard")
    parser.add_argument("--val_tokens", type=int, default=52_428_800)
    parser.add_argument("--train_tokens", type=int, default=2_600_000_000)
    parser.add_argument("--nproc", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = load_dataset(args.dataset, name=args.remote_name, split="train", streaming=True)

    # (split name, total token budget for the split)
    plan = [("val", args.val_tokens), ("train", args.train_tokens)]
    plan_idx = 0
    split, budget = plan[plan_idx]
    written_in_split = 0
    shard_idx = 0
    shard_files = []

    buf = np.empty(args.shard_size, dtype=np.uint16)
    pos = 0

    def flush():
        nonlocal pos, shard_idx, written_in_split
        if pos == 0:
            return
        path = os.path.join(args.out_dir, f"{split}_{shard_idx:06d}.bin")
        buf[:pos].tofile(path)
        shard_files.append({"path": os.path.basename(path), "tokens": int(pos)})
        print(f"wrote {path} ({pos:,} tokens)", flush=True)
        written_in_split += pos
        shard_idx += 1
        pos = 0

    def shard_capacity() -> int:
        # never exceed the split budget, even mid-shard
        return min(args.shard_size, budget - written_in_split) - pos

    done = False
    with mp.Pool(args.nproc, initializer=_init_worker) as pool:
        for tokens in pool.imap(_tokenize, (doc["text"] for doc in ds), chunksize=32):
            offset = 0
            while offset < len(tokens):
                take = min(len(tokens) - offset, shard_capacity())
                buf[pos : pos + take] = tokens[offset : offset + take]
                pos += take
                offset += take
                if shard_capacity() == 0:
                    flush()
                    if written_in_split >= budget:
                        plan_idx += 1
                        if plan_idx >= len(plan):
                            done = True
                            break
                        split, budget = plan[plan_idx]
                        written_in_split = 0
                        shard_idx = 0
                        # drop the rest of the straddling document so splits
                        # stay document-disjoint and each split starts at an
                        # EOT boundary
                        break
            if done:
                break
    if not done:
        flush()  # dataset exhausted before budget; keep what we have
        print(
            f"WARNING: dataset exhausted at {written_in_split:,}/{budget:,} tokens "
            f"for split {split!r}; adjust --train_tokens or use a larger remote_name"
        )

    meta = {
        "tokenizer": "gpt2 (tiktoken)",
        "eot_token": 50256,
        "dataset": args.dataset,
        "remote_name": args.remote_name,
        "shards": shard_files,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("done")
    sys.stdout.flush()
    # HF streaming's background prefetch threads can abort Python during
    # interpreter finalization (PyGILState_Release) AFTER everything is
    # written, turning a successful run into a non-zero exit that kills
    # sbatch afterok dependency chains -- skip finalization entirely.
    os._exit(0)


if __name__ == "__main__":
    main()
