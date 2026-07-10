from __future__ import annotations

import chess
import torch

from human_chess_model.features import board_to_tensor
from human_chess_model.move_encoding import decode_move, legal_move_indices


def board_from_fen(value: str) -> chess.Board:
    if value == "startpos":
        return chess.Board()
    return chess.Board(value)


@torch.no_grad()
def rank_moves(model, board: chess.Board, device: str = "cpu") -> list[tuple[chess.Move, float]]:
    legal = legal_move_indices(board)
    if not legal:
        return []

    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    logits = model(x)[0]
    legal_indices = torch.tensor(legal, device=device, dtype=torch.long)
    legal_logits = logits.index_select(0, legal_indices)
    probs = torch.softmax(legal_logits, dim=0)
    order = torch.argsort(probs, descending=True)
    ranked = []
    for item in order.tolist():
        move_index = legal_indices[item].item()
        ranked.append((decode_move(move_index), probs[item].item()))
    return ranked


@torch.no_grad()
def sample_move(
    model,
    board: chess.Board,
    device: str = "cpu",
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> chess.Move:
    legal = legal_move_indices(board)
    if not legal:
        raise ValueError("no legal moves available for this position")

    if temperature <= 0:
        return rank_moves(model, board, device=device)[0][0]

    x = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).to(device)
    logits = model(x)[0]
    legal_indices = torch.tensor(legal, device=device, dtype=torch.long)
    legal_logits = logits.index_select(0, legal_indices) / temperature
    probs = torch.softmax(legal_logits, dim=0)

    if top_p < 1.0:
        sorted_probs, sorted_order = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        keep = cumulative <= top_p
        keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered[sorted_order[keep]] = probs[sorted_order[keep]]
        probs = filtered / filtered.sum()

    sampled = torch.multinomial(probs, num_samples=1).item()
    return decode_move(legal_indices[sampled].item())
