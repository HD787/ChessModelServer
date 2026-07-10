from __future__ import annotations

import argparse

from human_chess_model.preprocess import stream_examples, write_shards


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Lichess PGN into training shards.")
    parser.add_argument("--input", required=True, help="Path to .pgn or .pgn.zst file.")
    parser.add_argument("--out", required=True, help="Output shard directory.")
    parser.add_argument("--min-elo", type=int, default=650)
    parser.add_argument("--max-elo", type=int, default=750)
    parser.add_argument("--time-control", default="rapid")
    parser.add_argument("--shard-size", type=int, default=100_000)
    parser.add_argument("--prefix", default="train")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    examples = stream_examples(
        args.input,
        min_elo=args.min_elo,
        max_elo=args.max_elo,
        time_control=args.time_control,
    )
    count = write_shards(examples, args.out, shard_size=args.shard_size, prefix=args.prefix)
    print(f"wrote {count} shards to {args.out}")


if __name__ == "__main__":
    main()
