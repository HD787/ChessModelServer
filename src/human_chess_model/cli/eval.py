from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from human_chess_model.checkpoint import load_model
from human_chess_model.cli.train import resolve_device
from human_chess_model.dataset import ShardedChessDataset
from human_chess_model.metrics import batch_metrics, merge_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a policy checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-glob", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device=device)
    data = ShardedChessDataset(args.data_glob)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    totals = []
    with torch.no_grad():
        for x, y, legal, legal_count in loader:
            logits = model(x.to(device))
            totals.append(
                batch_metrics(
                    logits,
                    y.to(device),
                    legal.to(device),
                    legal_count.to(device),
                )
            )
    print(merge_metrics(totals))


if __name__ == "__main__":
    main()
