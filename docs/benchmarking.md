# Benchmarking

This project exposes checkpoints as UCI engines via `human-chess-uci`, so match
runners such as Cute Chess can play games against Stockfish or between model
checkpoints.

## Install Tools

On macOS:

```bash
brew install cutechess stockfish
```

On Ubuntu/Debian:

```bash
sudo apt-get install cutechess-cli stockfish
```

Confirm both are available:

```bash
cutechess-cli -help
stockfish
```

## One Model vs Stockfish Elo

This runs the v3 model against Stockfish limited to about 700 Elo:

```bash
cutechess-cli \
  -engine name=human-v3 cmd=human-chess-uci arg="--checkpoint" arg="checkpoints/tournament-best-models/v3-small-endgame-bestTop1-e13.pt" arg="--temperature" arg="0" \
  -engine name=stockfish-700 cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo=700 \
  -each proto=uci tc=40/60 \
  -rounds 100 \
  -repeat \
  -games 2 \
  -pgnout benchmark-results/human-v3-vs-stockfish-700.pgn
```

Notes:

- `--temperature 0` makes the model deterministic argmax.
- Use `--temperature 0.8` or `--temperature 1.0` to benchmark sampled play.
- `-repeat` swaps colors across paired games.
- Increase `-rounds` for less noisy results.

## Multiple Stockfish Levels

Run separate matches for Elo anchors:

```bash
for elo in 700 900 1100 1300; do
  mkdir -p benchmark-results
  cutechess-cli \
    -engine name=human-v3 cmd=human-chess-uci arg="--checkpoint" arg="checkpoints/tournament-best-models/v3-small-endgame-bestTop1-e13.pt" arg="--temperature" arg="0" \
    -engine name=stockfish-$elo cmd=stockfish option.UCI_LimitStrength=true option.UCI_Elo=$elo \
    -each proto=uci tc=40/60 \
    -rounds 100 \
    -repeat \
    -games 2 \
    -pgnout benchmark-results/human-v3-vs-stockfish-$elo.pgn
done
```

## Model vs Model

```bash
cutechess-cli \
  -engine name=v2-base cmd=human-chess-uci arg="--checkpoint" arg="checkpoints/tournament-best-models/v2-cnn128x6-base-e04.pt" arg="--temperature" arg="0" \
  -engine name=v3-endgame cmd=human-chess-uci arg="--checkpoint" arg="checkpoints/tournament-best-models/v3-small-endgame-bestTop1-e13.pt" arg="--temperature" arg="0" \
  -each proto=uci tc=40/60 \
  -rounds 100 \
  -repeat \
  -games 2 \
  -pgnout benchmark-results/v2-base-vs-v3-endgame.pgn
```

## Interpreting Results

Cute Chess reports wins/losses/draws and score percentage. Treat small runs as
directional only; chess match results are noisy. Use hundreds or thousands of
games before making architecture decisions.
