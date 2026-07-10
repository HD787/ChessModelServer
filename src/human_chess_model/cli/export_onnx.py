from __future__ import annotations

import argparse

import torch

from human_chess_model.checkpoint import load_model
from human_chess_model.constants import BOARD_SHAPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a policy checkpoint to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _ = load_model(args.checkpoint, device="cpu")
    dummy = torch.zeros((1, *BOARD_SHAPE), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        args.out,
        input_names=["board"],
        output_names=["move_logits"],
        dynamic_axes={"board": {0: "batch"}, "move_logits": {0: "batch"}},
        opset_version=args.opset,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
