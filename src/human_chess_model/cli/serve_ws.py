from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
from pathlib import Path
import threading
from typing import Any

import websockets

from human_chess_model.checkpoint import load_model
from human_chess_model.cli.train import resolve_device
from human_chess_model.inference import board_from_fen, sample_move

MODEL_SUFFIXES = {".pt", ".pth", ".ts", ".torchscript"}


@dataclass
class ModelRunner:
    models: dict[str, Any]
    model_devices: dict[str, str]
    model_list: list[dict[str, Any]]
    default_model_id: str
    default_temperature: float
    default_top_p: float
    inference_lock: threading.Lock

    def ready_payload(self) -> dict[str, Any]:
        return {
            "type": "ready",
            "message": "human chess model runner ready",
            "models": self.model_list,
            "defaultModelId": self.default_model_id,
        }

    def models_payload(self) -> dict[str, Any]:
        return {"type": "models", "models": self.model_list, "defaultModelId": self.default_model_id}

    def engine_move_payload(self, message: dict[str, Any]) -> dict[str, Any]:
        request_id = message.get("requestId")
        requested_model_id = message.get("modelId") or self.default_model_id
        model = self.models.get(requested_model_id)
        if model is None:
            raise ValueError(f"unknown modelId: {requested_model_id}")

        if "fen" not in message:
            raise ValueError("fen is required")

        board = board_from_fen(message["fen"])
        temperature, top_p = inference_options(
            message,
            default_temperature=self.default_temperature,
            default_top_p=self.default_top_p,
        )
        with self.inference_lock:
            move = sample_move(
                model,
                board,
                device=self.model_devices[requested_model_id],
                temperature=temperature,
                top_p=top_p,
            )
        return {
            **message,
            "type": "engineMove",
            "bestmove": move.uci(),
            "modelId": requested_model_id,
            "requestId": request_id,
        }


def parse_args(default_transport: str = "ws") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve one or more chess policy checkpoints.")
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
    parser.add_argument("--transport", choices=["ws", "http"], default=default_transport)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    return parser.parse_args()


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


def load_runner(args: argparse.Namespace) -> ModelRunner:
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
    return ModelRunner(
        models=models,
        model_devices=model_devices,
        model_list=model_list,
        default_model_id=default_model_id,
        default_temperature=args.temperature,
        default_top_p=args.top_p,
        inference_lock=threading.Lock(),
    )


async def serve_websocket(args: argparse.Namespace, runner: ModelRunner) -> None:

    async def handle(websocket) -> None:
        await websocket.send(json.dumps(runner.ready_payload()))
        async for raw in websocket:
            try:
                message = json.loads(raw)
                request_id = message.get("requestId")
                if message.get("type") == "ping":
                    await websocket.send(json.dumps({"type": "pong"}))
                    continue
                if message.get("type") == "models":
                    await websocket.send(json.dumps(runner.models_payload()))
                    continue
                if message.get("type") != "engineMove":
                    await websocket.send(
                        json.dumps({"type": "error", "error": "unsupported message type", "requestId": request_id})
                    )
                    continue

                await websocket.send(json.dumps(runner.engine_move_payload(message)))
            except Exception as exc:
                request_id = None
                try:
                    request_id = json.loads(raw).get("requestId")
                except Exception:
                    pass
                await websocket.send(json.dumps({"type": "error", "error": str(exc), "requestId": request_id}))

    print(f"serving {len(runner.models)} model(s) on ws://{args.host}:{args.port}")
    async with websockets.serve(handle, args.host, args.port):
        await asyncio.Future()


def make_http_handler(runner: ModelRunner):
    class HumanChessHttpHandler(BaseHTTPRequestHandler):
        server_version = "HumanChessModelHTTP/0.1"

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.address_string()} - {format % args}")

        def end_headers(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            super().end_headers()

        def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_OPTIONS(self) -> None:
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(HTTPStatus.OK, {"ok": True})
                return
            if self.path == "/models":
                self.send_json(HTTPStatus.OK, runner.models_payload())
                return
            self.send_json(HTTPStatus.NOT_FOUND, {"type": "error", "error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/move":
                self.send_json(HTTPStatus.NOT_FOUND, {"type": "error", "error": "not found"})
                return
            request_id = None
            try:
                message = self.read_json()
                request_id = message.get("requestId")
                payload = runner.engine_move_payload(message)
                self.send_json(HTTPStatus.OK, payload)
            except Exception as exc:
                self.send_json(
                    HTTPStatus.BAD_REQUEST,
                    {"type": "error", "error": str(exc), "requestId": request_id},
                )

    return HumanChessHttpHandler


def serve_http(args: argparse.Namespace, runner: ModelRunner) -> None:
    server = ThreadingHTTPServer((args.host, args.port), make_http_handler(runner))
    print(f"serving {len(runner.models)} model(s) on http://{args.host}:{args.port}")
    server.serve_forever()


async def main_async(default_transport: str = "ws") -> None:
    args = parse_args(default_transport=default_transport)
    runner = load_runner(args)
    if args.transport == "http":
        await asyncio.to_thread(serve_http, args, runner)
        return
    await serve_websocket(args, runner)


def main() -> None:
    asyncio.run(main_async())


def main_http() -> None:
    asyncio.run(main_async(default_transport="http"))


if __name__ == "__main__":
    main()
