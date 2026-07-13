from human_chess_model.checkpoint import is_torchscript_path


def test_is_torchscript_path_accepts_export_suffixes() -> None:
    assert is_torchscript_path("model.ts")
    assert is_torchscript_path("model.torchscript")


def test_is_torchscript_path_rejects_training_checkpoint_suffixes() -> None:
    assert not is_torchscript_path("model.pt")
