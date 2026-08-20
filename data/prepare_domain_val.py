"""Tokenize a small out-of-domain validation set into a val_*.bin shard.

Produces the same headerless uint16 GPT-2-BPE format as prepare_fineweb_edu.py
so TokenShardReader can point at it directly. Used by the Phase-A3 domain-
shift experiment (see PLAN.md): measure how the learned depth-attention
routing changes on code / non-English text.

Usage:
    python data/prepare_domain_val.py --dataset codeparrot/codeparrot-clean-valid \
        --text_field content --out_dir data/domain_code
    python data/prepare_domain_val.py --dataset wikimedia/wikipedia \
        --remote_name 20231101.zh --out_dir data/domain_zh
"""

import argparse
import json
import os
import sys

import numpy as np
import tiktoken
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--remote_name", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_field", type=str, default="text")
    parser.add_argument("--tokens", type=int, default=10_485_760)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset(args.dataset, name=args.remote_name, split=args.split, streaming=True)

    buf = np.empty(args.tokens, dtype=np.uint16)
    pos = 0
    for doc in ds:
        text = doc.get(args.text_field)
        if not text:
            continue
        tokens = [enc.eot_token] + enc.encode_ordinary(text)
        assert max(tokens) < 2**16, "token id exceeds uint16 range"
        take = min(len(tokens), args.tokens - pos)
        buf[pos : pos + take] = np.array(tokens[:take], dtype=np.uint16)
        pos += take
        if pos >= args.tokens:
            break

    path = os.path.join(args.out_dir, "val_000000.bin")
    buf[:pos].tofile(path)
    meta = {
        "tokenizer": "gpt2 (tiktoken)",
        "eot_token": 50256,
        "dataset": args.dataset,
        "remote_name": args.remote_name,
        "text_field": args.text_field,
        "shards": [{"path": "val_000000.bin", "tokens": int(pos)}],
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {path} ({pos:,} tokens)")
    sys.stdout.flush()
    # same HF-streaming teardown workaround as prepare_fineweb_edu.py
    os._exit(0)


if __name__ == "__main__":
    main()
