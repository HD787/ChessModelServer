import argparse

from human_chess_model.cli.download_model import DEFAULT_FILENAME, requested_filenames


def test_requested_filenames_defaults_to_published_checkpoint() -> None:
    args = argparse.Namespace(filename=[])

    assert requested_filenames(args) == [DEFAULT_FILENAME]


def test_requested_filenames_uses_explicit_values() -> None:
    args = argparse.Namespace(filename=["checkpoints/one.pt", "checkpoints/two.pt"])

    assert requested_filenames(args) == ["checkpoints/one.pt", "checkpoints/two.pt"]
