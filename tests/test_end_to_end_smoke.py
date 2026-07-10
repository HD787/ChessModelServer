import io

import chess.pgn
import torch
from torch.utils.data import DataLoader

from human_chess_model.cli.train import run_epoch
from human_chess_model.dataset import ShardedChessDataset
from human_chess_model.model import build_model
from human_chess_model.preprocess import examples_from_game, write_shards


PGN = """
[Event "Rated Rapid game"]
[Site "https://lichess.org/test"]
[Date "2024.01.01"]
[Round "-"]
[White "low"]
[Black "other"]
[Result "1-0"]
[WhiteElo "700"]
[BlackElo "700"]
[Variant "Standard"]
[TimeControl "600+0"]
[Rated "True"]

1. e4 e5 2. Nf3 Nc6 1-0
"""


def test_preprocess_dataset_and_one_batch_training(tmp_path):
    game = chess.pgn.read_game(io.StringIO(PGN))
    shard_count = write_shards(
        examples_from_game(game, min_elo=650, max_elo=750),
        tmp_path,
        shard_size=10,
        prefix="train",
    )
    assert shard_count == 1

    dataset = ShardedChessDataset(str(tmp_path / "train-*.pt"))
    loader = DataLoader(dataset, batch_size=2)
    model = build_model(channels=8, blocks=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_epoch(model, loader, optimizer, device="cpu", train=True)

    assert metrics["count"] == 4
    assert metrics["nll"] > 0
    assert 0 <= metrics["top1_acc"] <= 1
