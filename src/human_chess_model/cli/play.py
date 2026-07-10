from __future__ import annotations

import argparse

from human_chess_model.checkpoint import load_model
from human_chess_model.cli.train import resolve_device
from human_chess_model.inference import board_from_fen, rank_moves, sample_move


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank or sample legal moves from a FEN.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fen", default="startpos")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, _ = load_model(args.checkpoint, device=device)
    board = board_from_fen(args.fen)
    sampled = sample_move(model, board, device=device, temperature=args.temperature, top_p=args.top_p)
    print(f"sampled {sampled.uci()}")
    for move, prob in rank_moves(model, board, device=device)[: args.top]:
        print(f"{move.uci()}\t{prob:.6f}")


if __name__ == "__main__":
    main()
