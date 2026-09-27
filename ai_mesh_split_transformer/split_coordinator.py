from __future__ import annotations

import argparse
import json
import time
import urllib.request
from typing import Any

import numpy as np

from model import decode_array, encode_array, monolithic_forward, text_to_tokens, tokens_to_text


def json_request(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def forward(stage1_url: str, stage2_url: str, tokens: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    begin = time.perf_counter()
    first = json_request(f"{stage1_url.rstrip('/')}/forward", {"tokens": tokens})
    hidden = first["hidden"]
    second = json_request(f"{stage2_url.rstrip('/')}/forward", {"hidden": hidden})
    elapsed_ms = round((time.perf_counter() - begin) * 1000, 3)
    return decode_array(second["logits"]), {
        "stage1_ms": first.get("elapsed_ms"),
        "stage2_ms": second.get("elapsed_ms"),
        "total_ms": elapsed_ms,
        "hidden_bytes": len(hidden["data"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Coordinate a two-stage split transformer.")
    parser.add_argument("--stage1", required=True, help="Base URL for stage 1 worker")
    parser.add_argument("--stage2", required=True, help="Base URL for stage 2 worker")
    parser.add_argument("--text", default="AI")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    for name, url in [("stage1", args.stage1), ("stage2", args.stage2)]:
        info = json_request(f"{url.rstrip('/')}/info")
        print(f"{name}: stage={info['stage']} parameter_bytes={info['parameter_bytes']}")

    tokens = text_to_tokens(args.text)
    generated: list[int] = []
    for step in range(args.steps):
        logits, stats = forward(args.stage1, args.stage2, tokens)
        next_token = int(np.argmax(logits))
        generated.append(next_token)
        tokens.append(next_token)
        print(
            f"step={step + 1} next_token={next_token} "
            f"stage1={stats['stage1_ms']}ms stage2={stats['stage2_ms']}ms "
            f"total={stats['total_ms']}ms hidden_base64_bytes={stats['hidden_bytes']}"
        )

        if args.validate:
            expected = monolithic_forward(tokens[:-1])
            max_abs = float(np.max(np.abs(expected - logits)))
            argmax_match = int(np.argmax(expected)) == next_token
            print(f"validation max_abs={max_abs:.8f}")
            print(f"argmax_match={argmax_match}")

    print(f"generated_tokens={generated}")
    print(f"generated_text={tokens_to_text(generated)!r}")


if __name__ == "__main__":
    main()
