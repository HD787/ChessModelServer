from __future__ import annotations

import argparse
import sys

import chess

from human_chess_model.checkpoint import load_model
from human_chess_model.cli.train import resolve_device
from human_chess_model.inference import sample_move


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a human chess policy checkpoint as a UCI engine.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--name", default=None)
    return parser.parse_args()


def apply_position(command: str, board: chess.Board) -> chess.Board:
    parts = command.split()
    if len(parts) < 2 or parts[0] != "position":
        return board

    index = 1
    if parts[index] == "startpos":
        board = chess.Board()
        index += 1
    elif parts[index] == "fen":
        fen_parts = []
        index += 1
        while index < len(parts) and parts[index] != "moves":
            fen_parts.append(parts[index])
            index += 1
        board = chess.Board(" ".join(fen_parts))
    else:
        return board

    if index < len(parts) and parts[index] == "moves":
        for move_uci in parts[index + 1 :]:
            board.push(chess.Move.from_uci(move_uci))
    return board


def print_line(value: str) -> None:
    print(value, flush=True)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device=device)
    name = args.name or checkpoint.get("args", {}).get("out", args.checkpoint).split("/")[-1]
    board = chess.Board()
    temperature = args.temperature
    top_p = args.top_p

    for raw in sys.stdin:
        command = raw.strip()
        if not command:
            continue

        if command == "uci":
            print_line(f"id name HumanChess {name}")
            print_line("id author humanModel")
            print_line("option name Temperature type spin default 0 min 0 max 300")
            print_line("option name TopP type spin default 100 min 1 max 100")
            print_line("uciok")
            continue

        if command == "isready":
            print_line("readyok")
            continue

        if command == "ucinewgame":
            board = chess.Board()
            continue

        if command.startswith("setoption "):
            parts = command.split()
            if "name" in parts and "value" in parts:
                name_index = parts.index("name") + 1
                value_index = parts.index("value") + 1
                option_name = " ".join(parts[name_index : value_index - 1]).lower()
                option_value = " ".join(parts[value_index:])
                if option_name == "temperature":
                    temperature = max(0.0, float(option_value) / 100.0)
                elif option_name == "topp":
                    top_p = min(1.0, max(0.01, float(option_value) / 100.0))
            continue

        if command.startswith("position "):
            try:
                board = apply_position(command, board)
            except Exception as exc:
                print_line(f"info string invalid position: {exc}")
            continue

        if command.startswith("go"):
            try:
                if board.is_game_over():
                    print_line("bestmove 0000")
                    continue
                move = sample_move(model, board, device=device, temperature=temperature, top_p=top_p)
                print_line(f"bestmove {move.uci()}")
            except Exception as exc:
                print_line(f"info string move selection failed: {exc}")
                print_line("bestmove 0000")
            continue

        if command == "stop":
            continue

        if command == "quit":
            break


if __name__ == "__main__":
    main()
