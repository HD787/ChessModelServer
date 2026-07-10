from __future__ import annotations

from glob import glob
from pathlib import Path

import torch
from torch.utils.data import Dataset


class ShardedChessDataset(Dataset):
    def __init__(self, pattern: str) -> None:
        self.paths = sorted(Path(path) for path in glob(pattern))
        if not self.paths:
            raise FileNotFoundError(f"no shards matched {pattern!r}")

        self.shards = []
        self.offsets = []
        total = 0
        for path in self.paths:
            shard = torch.load(path, map_location="cpu", weights_only=False)
            count = len(shard["y"])
            self.shards.append(shard)
            self.offsets.append((total, total + count))
            total += count
        self.total = total

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int):
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)

        for shard, (start, end) in zip(self.shards, self.offsets):
            if start <= index < end:
                local = index - start
                x = torch.as_tensor(shard["x"][local], dtype=torch.float32)
                y = torch.as_tensor(shard["y"][local], dtype=torch.long)
                legal = torch.as_tensor(shard["legal"][local], dtype=torch.long)
                legal_count = torch.as_tensor(shard["legal_count"][local], dtype=torch.long)
                return x, y, legal, legal_count
        raise IndexError(index)
