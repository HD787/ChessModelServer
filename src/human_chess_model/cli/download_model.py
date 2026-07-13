from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_REPO_ID = "hd787/humanchess-650-750-blitz"
DEFAULT_FILENAME = "checkpoints/v3-cnn-128x6-20epoch.pt"
DEFAULT_INT8_FILENAME = "artifacts/v3-cnn-128x6-20epoch-int8.ts"
DEFAULT_FILENAMES = [DEFAULT_FILENAME, DEFAULT_INT8_FILENAME]
DEFAULT_LOCAL_DIR = "models/humanchess-650-750-blitz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download published Human Chess model checkpoints.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="Hugging Face repo id.")
    parser.add_argument(
        "--filename",
        action="append",
        default=[],
        help="Repo file to download. Can be passed multiple times.",
    )
    parser.add_argument("--local-dir", default=DEFAULT_LOCAL_DIR, help="Directory to store downloaded model files.")
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision, branch, or commit.")
    return parser.parse_args()


def requested_filenames(args: argparse.Namespace) -> list[str]:
    return args.filename or DEFAULT_FILENAMES


def main() -> None:
    args = parse_args()
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it with `pip install 'human-chess-model[hf]'` "
            "or `pip install huggingface-hub`."
        ) from exc

    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    for filename in requested_filenames(args):
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            revision=args.revision,
            local_dir=local_dir,
        )
        print(path)


if __name__ == "__main__":
    main()
