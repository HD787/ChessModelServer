from __future__ import annotations

import chess
import numpy as np

from human_chess_model.constants import MAX_LEGAL_MOVES, MOVE_VOCAB_SIZE, PROMOTION_PIECES

_PROMOTION_TO_ID = {
    None: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
}
_ID_TO_PROMOTION = {value: key for key, value in _PROMOTION_TO_ID.items()}


def encode_move(move: chess.Move) -> int:
    """Encode a chess move as from-square, to-square, promotion-id."""
    promo_id = _PROMOTION_TO_ID[move.promotion]
    return ((move.from_square * 64) + move.to_square) * len(PROMOTION_PIECES) + promo_id


def decode_move(index: int) -> chess.Move:
    if index < 0 or index >= MOVE_VOCAB_SIZE:
        raise ValueError(f"move index out of range: {index}")
    square_pair, promo_id = divmod(index, len(PROMOTION_PIECES))
    from_square, to_square = divmod(square_pair, 64)
    return chess.Move(from_square, to_square, promotion=_ID_TO_PROMOTION[promo_id])


def legal_move_indices(board: chess.Board) -> list[int]:
    return [encode_move(move) for move in board.legal_moves]


def padded_legal_move_indices(board: chess.Board) -> tuple[np.ndarray, int]:
    indices = legal_move_indices(board)
    if len(indices) > MAX_LEGAL_MOVES:
        raise ValueError(f"too many legal moves: {len(indices)}")
    padded = np.full((MAX_LEGAL_MOVES,), -1, dtype=np.int64)
    padded[: len(indices)] = indices
    return padded, len(indices)


def legal_policy_logits(logits, board: chess.Board):
    """Return logits with illegal moves set to -inf."""
    import torch

    masked = torch.full_like(logits, float("-inf"))
    indices = torch.tensor(legal_move_indices(board), device=logits.device, dtype=torch.long)
    masked.index_copy_(0, indices, logits.index_select(0, indices))
    return masked
