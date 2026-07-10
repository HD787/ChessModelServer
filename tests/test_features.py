import chess

from human_chess_model.constants import BOARD_SHAPE
from human_chess_model.features import board_to_tensor


def test_feature_shape_and_start_position_planes():
    x = board_to_tensor(chess.Board())
    assert x.shape == BOARD_SHAPE
    assert x[0].sum() == 8
    assert x[6].sum() == 8
    assert x[12].sum() == 64
    assert x[13].sum() == 64
    assert x[14].sum() == 64
    assert x[15].sum() == 64
    assert x[16].sum() == 64
