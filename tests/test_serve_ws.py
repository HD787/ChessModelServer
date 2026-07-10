import pytest

from human_chess_model.cli.serve_ws import inference_options


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
