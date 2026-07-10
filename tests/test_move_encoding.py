import chess

from human_chess_model.move_encoding import decode_move, encode_move, legal_move_indices


def test_encode_decode_normal_move():
    move = chess.Move.from_uci("e2e4")
    assert decode_move(encode_move(move)) == move


def test_encode_decode_promotion():
    move = chess.Move.from_uci("a7a8q")
    assert decode_move(encode_move(move)) == move


def test_start_position_has_20_legal_indices():
    board = chess.Board()
    assert len(legal_move_indices(board)) == 20
