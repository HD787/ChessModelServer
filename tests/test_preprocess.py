import io

import chess.pgn

from human_chess_model.preprocess import examples_from_game, is_target_game


PGN = """
[Event "Rated Rapid game"]
[Site "https://lichess.org/test"]
[Date "2024.01.01"]
[Round "-"]
[White "low"]
[Black "other"]
[Result "1-0"]
[UTCDate "2024.01.01"]
[UTCTime "00:00:00"]
[WhiteElo "700"]
[BlackElo "1200"]
[WhiteRatingDiff "+1"]
[BlackRatingDiff "-1"]
[Variant "Standard"]
[TimeControl "600+0"]
[ECO "C20"]
[Opening "King's Pawn Game"]
[Termination "Normal"]
[Rated "True"]

1. e4 e5 2. Nf3 Nc6 1-0
"""


def test_target_game_filter_and_examples():
    game = chess.pgn.read_game(io.StringIO(PGN))
    assert is_target_game(game.headers, "rapid")
    examples = list(examples_from_game(game, min_elo=650, max_elo=750))
    assert len(examples) == 2
    x, y, legal, legal_count = examples[0]
    assert legal_count == 20
    assert y in set(legal[:legal_count])
