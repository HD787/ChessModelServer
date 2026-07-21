# Human Chess Model

This project trains a supervised chess policy model to imitate human moves from
650-750 rated Lichess Rapid games. It is intentionally not an engine wrapper:
legal moves are generated with `python-chess`, the neural net scores the fixed
move vocabulary, and inference masks illegal moves before ranking or sampling.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,export]"
```

## Preprocess Lichess PGN

```bash
human-chess-preprocess \
  --input lichess_db_standard_rated_YYYY-MM.pgn.zst \
  --out data/shards \
  --min-elo 650 \
  --max-elo 750 \
  --time-control rapid \
  --shard-size 100000
```

The preprocessor streams PGN or PGN.zst files, keeps rated standard Rapid games,
and emits `.pt` shards containing:

- `x`: board tensors shaped `N x 19 x 8 x 8`
- `y`: encoded human move labels
- `legal`: padded legal move index arrays
- `legal_count`: number of valid entries in each `legal` row

Training and evaluation mask logits to the legal move set before computing loss
or top-k accuracy.

## Train

```bash
human-chess-train \
  --train-glob "data/shards/train-*.pt" \
  --val-glob "data/shards/val-*.pt" \
  --epochs 5 \
  --batch-size 512 \
  --out checkpoints/policy.pt
```

## Evaluate

```bash
human-chess-eval --checkpoint checkpoints/policy.pt --data-glob "data/shards/val-*.pt"
```

## Play From FEN

```bash
human-chess-play --checkpoint checkpoints/policy.pt --fen "startpos"
```

## Serve Moves To The Web UI

The copied Next.js chess UI lives in `web/`. Published checkpoints can be pulled
from Hugging Face into the ignored local `models/` directory:

```bash
human-chess-download-model
```

That defaults to:

```bash
huggingface-cli download hd787/humanchess-650-750-blitz \
  checkpoints/v3-cnn-128x6-20epoch.pt \
  artifacts/v3-cnn-128x6-20epoch-int8.ts \
  --local-dir ./models/humanchess-650-750-blitz
```

Then run the Python websocket server against the downloaded model directory:

```bash
human-chess-serve-ws \
  --checkpoint-dir models/humanchess-650-750-blitz/checkpoints \
  --host 127.0.0.1 \
  --port 8787
```

The same runner can serve HTTP instead of websockets:

```bash
human-chess-serve-http \
  --checkpoint artifacts/v3-cnn-128x6-20epoch-int8.ts \
  --model-alias "650-750 Blitz" \
  --host 127.0.0.1 \
  --port 8787
```

HTTP endpoints:

- `GET /health` returns `{"ok": true}`
- `GET /models` returns the available public model ids and names
- `POST /move` accepts `{"fen":"startpos","modelId":"650-750-blitz","temperature":0,"topP":1}` and returns `{"type":"engineMove","bestmove":"..."}`

Use `--model-alias` when you want the browser to see a public name instead of
the checkpoint filename. Aliases become the public model id and display name,
and checkpoint paths are not included in the websocket model list:

```bash
human-chess-serve-ws \
  --checkpoint models/humanchess-650-750-blitz/checkpoints/v3-cnn-128x6-20epoch.pt \
  --model-alias "650-750 Blitz" \
  --host 127.0.0.1 \
  --port 8787
```

For `--checkpoint-dir`, use `SOURCE=NAME`; `SOURCE` can be the filename, stem,
path, or default model id:

```bash
human-chess-serve-ws \
  --checkpoint-dir models/humanchess-650-750-blitz/checkpoints \
  --model-alias "v3-cnn-128x6-20epoch=650-750 Blitz" \
  --host 127.0.0.1 \
  --port 8787
```

In another terminal:

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3001 and enable `Play vs Human Model`.

## Export

```bash
human-chess-export-onnx --checkpoint checkpoints/policy.pt --out policy.onnx
```

## Benchmark

Use the UCI wrapper with Cute Chess for model-vs-model and model-vs-Stockfish
matches:

```bash
human-chess-uci --checkpoint checkpoints/tournament-best-models/v3-small-endgame-bestTop1-e13.pt
```

See [docs/benchmarking.md](docs/benchmarking.md).
