from __future__ import annotations

import numpy as np
import chess

from human_chess_model.constants import BOARD_SHAPE

_PIECE_TO_PLANE = {
    (chess.WHITE, chess.PAWN): 0,
    (chess.WHITE, chess.KNIGHT): 1,
    (chess.WHITE, chess.BISHOP): 2,
    (chess.WHITE, chess.ROOK): 3,
    (chess.WHITE, chess.QUEEN): 4,
    (chess.WHITE, chess.KING): 5,
    (chess.BLACK, chess.PAWN): 6,
    (chess.BLACK, chess.KNIGHT): 7,
    (chess.BLACK, chess.BISHOP): 8,
    (chess.BLACK, chess.ROOK): 9,
    (chess.BLACK, chess.QUEEN): 10,
    (chess.BLACK, chess.KING): 11,
}


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """Encode a board as uint8 planes with white's first rank at row 7."""
    tensor = np.zeros(BOARD_SHAPE, dtype=np.uint8)

    for square, piece in board.piece_map().items():
        plane = _PIECE_TO_PLANE[(piece.color, piece.piece_type)]
        rank = chess.square_rank(square)
        file = chess.square_file(square)
        tensor[plane, 7 - rank, file] = 1

    tensor[12, :, :] = int(board.turn == chess.WHITE)
    tensor[13, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
    tensor[14, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
    tensor[15, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
    tensor[16, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))

    if board.ep_square is not None:
        rank = chess.square_rank(board.ep_square)
        file = chess.square_file(board.ep_square)
        tensor[17, 7 - rank, file] = 1

    tensor[18, :, :] = min(board.fullmove_number, 255)
    return tensor
