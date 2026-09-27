from __future__ import annotations

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from model import decode_array, dumps_json, encode_array, make_weights, stage1_forward, stage2_forward


MAX_BODY_BYTES = 20 * 1024 * 1024


def make_handler(stage: int):
    weights = make_weights()
    started = time.time()

    class Handler(BaseHTTPRequestHandler):
        server_version = "SplitTransformerNode/0.1"

        def send_json(self, code: int, payload: dict[str, Any]) -> None:
            data = dumps_json(payload)
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length > MAX_BODY_BYTES:
                raise ValueError("request body too large")
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def do_GET(self) -> None:
            if self.path != "/info":
                self.send_json(404, {"ok": False, "error": "not found"})
                return
            self.send_json(
                200,
                {
                    "ok": True,
                    "stage": stage,
                    "uptime_s": round(time.time() - started, 2),
                    "parameter_bytes": weights.parameter_bytes(stage),
                },
            )

        def do_POST(self) -> None:
            try:
                if self.path != "/forward":
                    self.send_json(404, {"ok": False, "error": "not found"})
                    return
                payload = self.read_json()
                begin = time.perf_counter()
                if stage == 1:
                    hidden = stage1_forward([int(t) for t in payload["tokens"]], weights)
                    result = {"hidden": encode_array(hidden)}
                else:
                    logits = stage2_forward(decode_array(payload["hidden"]), weights)
                    result = {"logits": encode_array(logits)}
                self.send_json(
                    200,
                    {
                        "ok": True,
                        "stage": stage,
                        "elapsed_ms": round((time.perf_counter() - begin) * 1000, 3),
                        **result,
                    },
                )
            except Exception as exc:
                self.send_json(400, {"ok": False, "error": str(exc)})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} - {fmt % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one split-transformer worker stage.")
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.stage))
    print(f"stage {args.stage} listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
