import chess

from human_chess_model.cli.uci import apply_position


def test_apply_startpos_moves():
    board = apply_position("position startpos moves e2e4 e7e5", chess.Board())
    assert board.fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3")
    assert board.turn == chess.WHITE


def test_apply_fen_moves():
    board = apply_position(
        "position fen 8/P7/8/8/8/8/8/k6K w - - 0 1 moves a7a8q",
        chess.Board(),
    )
    assert board.piece_at(chess.A8).piece_type == chess.QUEEN
