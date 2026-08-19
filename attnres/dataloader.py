"""Deterministic, stateless reader over raw uint16 token shards.

Shards are flat little-endian uint16 token streams (no header) produced by
data/prepare_fineweb_edu.py. The batch served to (micro_step, rank) is a pure
function of (micro_step, rank, world_size, batch_size, seq_len), so resuming
from a checkpoint or re-running a seed reproduces the exact data order --
provided world size and batch shape stay fixed across compared runs.
"""

import glob
import json
import os

import numpy as np
import torch


class TokenShardReader:
    def __init__(
        self,
        data_dir: str,
        split: str,
        batch_size: int,
        seq_len: int,
        rank: int = 0,
        world_size: int = 1,
    ):
        paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        assert paths, f"no {split}_*.bin shards found under {data_dir}"
        meta_path = os.path.join(data_dir, "meta.json")
        if os.path.exists(meta_path):
            # guard against stale shards from a previous prepare run silently
            # joining the token stream
            with open(meta_path) as f:
                meta = json.load(f)
            expected = sorted(
                os.path.join(data_dir, s["path"])
                for s in meta.get("shards", [])
                if s["path"].startswith(f"{split}_")
            )
            assert paths == expected, (
                f"{split} shards on disk do not match meta.json (stale shards from "
                f"a previous prepare run?): {sorted(set(paths) ^ set(expected))}"
            )
        self.mmaps = [np.memmap(p, dtype=np.uint16, mode="r") for p in paths]
        self.sizes = [len(m) for m in self.mmaps]
        self.bounds = np.cumsum([0] + self.sizes)
        self.total = int(self.bounds[-1])
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rank = rank
        self.world_size = world_size
        self.chunk = batch_size * seq_len
        assert self.total > self.chunk * world_size, (
            f"{split} split has {self.total} tokens, need more than "
            f"{self.chunk * world_size} for one step across all ranks"
        )

    def _read(self, start: int, n: int) -> np.ndarray:
        """Read n tokens starting at absolute position start, wrapping around."""
        out = np.empty(n, dtype=np.uint16)
        filled = 0
        pos = start % self.total
        while filled < n:
            i = int(np.searchsorted(self.bounds, pos, side="right")) - 1
            local = pos - int(self.bounds[i])
            take = min(n - filled, self.sizes[i] - local)
            out[filled : filled + take] = self.mmaps[i][local : local + take]
            filled += take
            pos = (pos + take) % self.total
        return out

    def batch(self, micro_step: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = (micro_step * self.world_size + self.rank) * self.chunk
        buf = self._read(start, self.chunk + 1).astype(np.int64)
        x = torch.from_numpy(buf[:-1]).view(self.batch_size, self.seq_len)
        y = torch.from_numpy(buf[1:]).view(self.batch_size, self.seq_len)
        return x, y
