from __future__ import annotations

from pathlib import Path

import torch

from human_chess_model.model import build_policy_model


def save_checkpoint(path: str | Path, model, optimizer, args: dict, epoch: int, metrics: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "args": args,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_model(path: str | Path, device: str = "cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})
    model = build_policy_model(
        arch=args.get("arch", "cnn"),
        channels=args.get("channels", 128),
        blocks=args.get("blocks", 6),
        embed_dim=args.get("embed_dim", 192),
        layers=args.get("layers", 6),
        heads=args.get("heads", 8),
        mlp_ratio=args.get("mlp_ratio", 4.0),
        dropout=args.get("dropout", 0.05),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
