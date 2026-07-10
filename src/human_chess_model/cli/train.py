from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from human_chess_model.checkpoint import save_checkpoint
from human_chess_model.dataset import ShardedChessDataset
from human_chess_model.metrics import batch_metrics, mask_illegal_logits, merge_metrics
from human_chess_model.model import build_model


@dataclass
class TrainConfig:
    train_glob: str
    val_glob: str
    out: str
    epochs: int = 5
    batch_size: int = 512
    lr: float = 3e-4
    channels: int = 128
    blocks: int = 6
    device: str = "auto"
    num_workers: int = 0


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the human chess policy model.")
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--val-glob", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    return TrainConfig(**vars(parser.parse_args()))


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_epoch(model, loader, optimizer, device: str, train: bool) -> dict:
    model.train(train)
    criterion = nn.CrossEntropyLoss()
    totals = []

    for x, y, legal, legal_count in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        legal = legal.to(device, non_blocking=True)
        legal_count = legal_count.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(x)
            masked_logits = mask_illegal_logits(logits, legal, legal_count)
            loss = criterion(masked_logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        totals.append(batch_metrics(logits.detach(), y.detach(), legal.detach(), legal_count.detach()))
    return merge_metrics(totals)


def main() -> None:
    config = parse_args()
    device = resolve_device(config.device)

    train_data = ShardedChessDataset(config.train_glob)
    val_data = ShardedChessDataset(config.val_glob)
    train_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )

    model = build_model(channels=config.channels, blocks=config.blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4)

    best_nll = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        print(f"epoch={epoch} train={train_metrics} val={val_metrics}")
        if val_metrics["nll"] < best_nll:
            best_nll = val_metrics["nll"]
            args = asdict(config)
            args["device"] = device
            save_checkpoint(config.out, model, optimizer, args=args, epoch=epoch, metrics=val_metrics)
            print(f"saved {config.out}")


if __name__ == "__main__":
    main()
