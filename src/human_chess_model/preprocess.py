from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn
import numpy as np
import torch

from human_chess_model.features import board_to_tensor
from human_chess_model.move_encoding import encode_move, padded_legal_move_indices


def open_pgn_text(path: str | Path):
    path = Path(path)
    raw = path.open("rb")
    if path.suffix == ".zst":
        import zstandard as zstd

        stream = zstd.ZstdDecompressor().stream_reader(raw)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def is_target_game(headers, time_control: str) -> bool:
    if headers.get("Variant", "Standard") != "Standard":
        return False
    if headers.get("Rated", "").lower() != "true":
        return False
    event = headers.get("Event", "").lower()
    return time_control.lower() in event


def _elo(headers, color: chess.Color) -> int | None:
    key = "WhiteElo" if color == chess.WHITE else "BlackElo"
    try:
        return int(headers.get(key, ""))
    except ValueError:
        return None


def examples_from_game(game, min_elo: int, max_elo: int):
    board = game.board()
    for move in game.mainline_moves():
        mover_rating = _elo(game.headers, board.turn)
        if mover_rating is not None and min_elo <= mover_rating <= max_elo:
            legal, legal_count = padded_legal_move_indices(board)
            yield board_to_tensor(board), encode_move(move), legal, legal_count
        board.push(move)


def stream_examples(
    input_path: str | Path,
    min_elo: int = 650,
    max_elo: int = 750,
    time_control: str = "rapid",
) -> Iterable[tuple[np.ndarray, int, np.ndarray, int]]:
    with open_pgn_text(input_path) as handle:
        while True:
            game = chess.pgn.read_game(handle)
            if game is None:
                break
            if not is_target_game(game.headers, time_control):
                continue
            yield from examples_from_game(game, min_elo=min_elo, max_elo=max_elo)


def write_shards(
    examples: Iterable[tuple[np.ndarray, int, np.ndarray, int]],
    out_dir: str | Path,
    shard_size: int,
    prefix: str = "train",
) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    legals: list[np.ndarray] = []
    legal_counts: list[int] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index, xs, ys, legals, legal_counts
        if not xs:
            return
        path = out / f"{prefix}-{shard_index:05d}.pt"
        torch.save(
            {
                "x": np.stack(xs),
                "y": np.asarray(ys, dtype=np.int64),
                "legal": np.stack(legals),
                "legal_count": np.asarray(legal_counts, dtype=np.int64),
            },
            path,
        )
        shard_index += 1
        xs = []
        ys = []
        legals = []
        legal_counts = []

    for x, y, legal, legal_count in examples:
        xs.append(x)
        ys.append(y)
        legals.append(legal)
        legal_counts.append(legal_count)
        if len(xs) >= shard_size:
            flush()
    flush()
    return shard_index
