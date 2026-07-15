from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

import websockets

from human_chess_model.checkpoint import load_model
from human_chess_model.cli.train import resolve_device
from human_chess_model.inference import board_from_fen, sample_move

MODEL_SUFFIXES = {".pt", ".pth", ".ts", ".torchscript"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one or more chess policy checkpoints over websocket.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path. Can be passed multiple times.")
    parser.add_argument(
        "--checkpoint-dir",
        action="append",
        default=[],
        help="Directory to scan recursively for .pt checkpoints. Can be passed multiple times.",
    )
    parser.add_argument(
        "--model-alias",
        action="append",
        default=[],
        help=(
            "Public model name. Pass once per loaded checkpoint, or use SOURCE=NAME where SOURCE matches "
            "a checkpoint path, filename, stem, or default model id. Aliases replace the public model id."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


def response(payload: dict[str, Any], **updates: Any) -> str:
    result = dict(payload)
    result.update(updates)
    return json.dumps(result)


def inference_options(message: dict[str, Any], *, default_temperature: float, default_top_p: float) -> tuple[float, float]:
    temperature = float(message.get("temperature", default_temperature))
    top_p = float(message.get("topP", default_top_p))
    if temperature < 0:
        raise ValueError("temperature must be greater than or equal to 0")
    if not 0 < top_p <= 1:
        raise ValueError("topP must be greater than 0 and less than or equal to 1")
    return temperature, top_p


def checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(item) for item in args.checkpoint]
    for item in args.checkpoint_dir:
        paths.extend(
            sorted(path for path in Path(item).rglob("*") if path.is_file() and path.suffix.lower() in MODEL_SUFFIXES)
        )
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("provide at least one --checkpoint or --checkpoint-dir")
    return unique


def model_id(path: Path) -> str:
    parent = path.parent.name
    if parent and parent != "checkpoints":
        return f"{parent}/{path.stem}"
    return path.stem


def alias_lookup_keys(path: Path) -> set[str]:
    resolved = path.resolve()
    return {str(path), str(resolved), path.name, path.stem, model_id(path)}


def parse_model_aliases(alias_values: list[str], paths: list[Path]) -> dict[Path, str]:
    positional = []
    keyed = {}
    for value in alias_values:
        if "=" in value:
            key, alias = value.split("=", 1)
            key = key.strip()
            alias = alias.strip()
            if not key or not alias:
                raise ValueError("--model-alias mappings must use SOURCE=NAME with both sides present")
            keyed[key] = alias
            continue
        alias = value.strip()
        if not alias:
            raise ValueError("--model-alias cannot be empty")
        positional.append(alias)

    if len(positional) > len(paths):
        raise ValueError("received more positional --model-alias values than loaded checkpoints")

    aliases = {path: positional[index] for index, path in enumerate(paths[: len(positional)])}
    matched_keys = set()
    for path in paths:
        keys = alias_lookup_keys(path)
        for key, alias in keyed.items():
            if key in keys:
                aliases[path] = alias
                matched_keys.add(key)

    unknown_keys = sorted(set(keyed) - matched_keys)
    if unknown_keys:
        raise ValueError(f"unknown --model-alias source: {', '.join(unknown_keys)}")
    return aliases


def slugify_model_id(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def unique_model_id(base: str, used_ids: set[str]) -> str:
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}-{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate


async def main_async() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    models = {}
    model_devices = {}
    model_list = []
    paths = checkpoint_paths(args)
    aliases = parse_model_aliases(args.model_alias, paths)
    used_model_ids = set()
    for path in paths:
        model, checkpoint = load_model(path, device=device)
        alias = aliases.get(path)
        public_name = alias or path.stem
        public_id_base = slugify_model_id(alias) if alias else model_id(path)
        mid = unique_model_id(public_id_base, used_model_ids)
        models[mid] = model
        model_devices[mid] = checkpoint.get("inference_device", device)
        model_list.append(
            {
                "id": mid,
                "name": public_name,
                "format": checkpoint.get("args", {}).get("format", "checkpoint"),
                "epoch": checkpoint.get("epoch"),
                "metrics": checkpoint.get("metrics", {}),
            }
        )
        print(f"loaded {mid} from {path}")
    default_model_id = model_list[0]["id"]

    async def handle(websocket) -> None:
        await websocket.send(
            json.dumps(
                {
                    "type": "ready",
                    "message": "human chess model runner ready",
                    "models": model_list,
                    "defaultModelId": default_model_id,
                }
            )
        )
        async for raw in websocket:
            try:
                message = json.loads(raw)
                request_id = message.get("requestId")
                if message.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                    continue
                if message.get("type") == "models":
                    await websocket.send(
                        json.dumps({"type": "models", "models": model_list, "defaultModelId": default_model_id})
                    )
                    continue
                if message.get("type") != "engineMove":
                    await websocket.send(
                        json.dumps({"type": "error", "error": "unsupported message type", "requestId": request_id})
                    )
                    continue

                requested_model_id = message.get("modelId") or default_model_id
                model = models.get(requested_model_id)
                if model is None:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "error": f"unknown modelId: {requested_model_id}",
                                "requestId": request_id,
                            }
                        )
                    )
                    continue

                board = board_from_fen(message["fen"])
                temperature, top_p = inference_options(
                    message,
                    default_temperature=args.temperature,
                    default_top_p=args.top_p,
                )
                move = sample_move(
                    model,
                    board,
                    device=model_devices[requested_model_id],
                    temperature=temperature,
                    top_p=top_p,
                )
                await websocket.send(
                    response(
                        message,
                        type="engineMove",
                        bestmove=move.uci(),
                        modelId=requested_model_id,
                        requestId=request_id,
                    )
                )
            except Exception as exc:
                request_id = None
                try:
                    request_id = json.loads(raw).get("requestId")
                except Exception:
                    pass
                await websocket.send(json.dumps({"type": "error", "error": str(exc), "requestId": request_id}))

    print(f"serving {len(models)} model(s) on ws://{args.host}:{args.port}")
    async with websockets.serve(handle, args.host, args.port):
        await asyncio.Future()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
