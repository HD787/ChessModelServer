import argparse
import threading

import chess

import pytest

from human_chess_model.cli import serve_ws
from human_chess_model.cli.serve_ws import ModelRunner
from human_chess_model.cli.serve_ws import inference_options
from human_chess_model.cli.serve_ws import checkpoint_paths
from human_chess_model.cli.serve_ws import parse_model_aliases
from human_chess_model.cli.serve_ws import slugify_model_id
from human_chess_model.cli.serve_ws import unique_model_id


def test_inference_options_uses_server_defaults() -> None:
    assert inference_options({}, default_temperature=0.8, default_top_p=0.95) == (0.8, 0.95)


def test_inference_options_allows_request_argmax() -> None:
    assert inference_options(
        {"temperature": 0, "topP": 1},
        default_temperature=0.8,
        default_top_p=0.95,
    ) == (0.0, 1.0)


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ({"temperature": -0.1}, "temperature"),
        ({"topP": 0}, "topP"),
        ({"topP": 1.1}, "topP"),
    ],
)
def test_inference_options_rejects_invalid_values(message: dict[str, float], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        inference_options(message, default_temperature=1.0, default_top_p=1.0)


def test_checkpoint_paths_discovers_torchscript_artifacts(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"
    torchscript = tmp_path / "model.ts"
    ignored = tmp_path / "notes.txt"
    checkpoint.write_bytes(b"checkpoint")
    torchscript.write_bytes(b"torchscript")
    ignored.write_text("ignore me")
    args = argparse.Namespace(checkpoint=[], checkpoint_dir=[str(tmp_path)])

    assert checkpoint_paths(args) == [checkpoint, torchscript]


def test_parse_model_aliases_supports_positional_aliases(tmp_path) -> None:
    first = tmp_path / "actual-cnn-filename.pt"
    second = tmp_path / "actual-transformer-filename.ts"

    assert parse_model_aliases(["Beginner Blitz", "Intermediate Blitz"], [first, second]) == {
        first: "Beginner Blitz",
        second: "Intermediate Blitz",
    }


def test_parse_model_aliases_supports_keyed_aliases(tmp_path) -> None:
    checkpoint_dir = tmp_path / "published-models"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "private-checkpoint-name.pt"

    assert parse_model_aliases(["private-checkpoint-name=Public Name"], [checkpoint]) == {
        checkpoint: "Public Name",
    }
    assert parse_model_aliases([f"{checkpoint}=Path Matched Name"], [checkpoint]) == {
        checkpoint: "Path Matched Name",
    }
    assert parse_model_aliases(["published-models/private-checkpoint-name=Default Id Matched Name"], [checkpoint]) == {
        checkpoint: "Default Id Matched Name",
    }


def test_parse_model_aliases_rejects_unknown_key(tmp_path) -> None:
    checkpoint = tmp_path / "model.pt"

    with pytest.raises(ValueError, match="unknown --model-alias source"):
        parse_model_aliases(["missing=Public Name"], [checkpoint])


def test_alias_ids_are_public_slugs_and_unique() -> None:
    used = set()

    assert slugify_model_id("650-750 Blitz Final") == "650-750-blitz-final"
    assert unique_model_id(slugify_model_id("Same Name"), used) == "same-name"
    assert unique_model_id(slugify_model_id("Same Name"), used) == "same-name-2"


def test_runner_builds_engine_move_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_sample_move(*args, **kwargs):
        return chess.Move.from_uci("e2e4")

    monkeypatch.setattr(serve_ws, "sample_move", fake_sample_move)
    runner = ModelRunner(
        models={"beginner": object()},
        model_devices={"beginner": "cpu"},
        model_list=[{"id": "beginner", "name": "Beginner"}],
        default_model_id="beginner",
        default_temperature=0.7,
        default_top_p=0.9,
        inference_lock=threading.Lock(),
    )

    payload = runner.engine_move_payload({"requestId": "abc", "fen": "startpos"})

    assert payload["type"] == "engineMove"
    assert payload["bestmove"] == "e2e4"
    assert payload["modelId"] == "beginner"
    assert payload["requestId"] == "abc"


def test_runner_rejects_unknown_model() -> None:
    runner = ModelRunner(
        models={"beginner": object()},
        model_devices={"beginner": "cpu"},
        model_list=[{"id": "beginner", "name": "Beginner"}],
        default_model_id="beginner",
        default_temperature=1.0,
        default_top_p=1.0,
        inference_lock=threading.Lock(),
    )

    with pytest.raises(ValueError, match="unknown modelId"):
        runner.engine_move_payload({"modelId": "missing", "fen": "startpos"})
