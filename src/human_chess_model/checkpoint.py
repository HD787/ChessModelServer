from __future__ import annotations

from pathlib import Path

import torch

from human_chess_model.model import build_policy_model

TORCHSCRIPT_SUFFIXES = {".ts", ".torchscript"}


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


def configure_quantized_backend() -> str | None:
    supported = list(torch.backends.quantized.supported_engines)
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    return None


def is_torchscript_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in TORCHSCRIPT_SUFFIXES


def load_torchscript_model(path: str | Path):
    configure_quantized_backend()
    model = torch.jit.load(str(path), map_location="cpu")
    model.eval()
    metadata = {
        "args": {"out": str(path), "format": "torchscript"},
        "epoch": None,
        "metrics": {},
        "inference_device": "cpu",
    }
    return model, metadata


def load_model(path: str | Path, device: str = "cpu"):
    if is_torchscript_path(path):
        return load_torchscript_model(path)

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
