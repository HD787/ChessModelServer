import torch

from human_chess_model.constants import BOARD_SHAPE, MOVE_VOCAB_SIZE
from human_chess_model.model import build_model


def test_model_forward_shape():
    model = build_model(channels=16, blocks=1)
    x = torch.zeros((2, *BOARD_SHAPE))
    y = model(x)
    assert y.shape == (2, MOVE_VOCAB_SIZE)
