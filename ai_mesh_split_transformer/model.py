from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Any

import numpy as np


VOCAB_SIZE = 128
DIM = 64
HIDDEN = 128
LAYERS = 4
STAGE_1_LAYERS = 2
SEED = 20260927


def encode_array(array: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return {
        "dtype": "float32",
        "shape": list(contiguous.shape),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def decode_array(payload: dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(payload["data"].encode("ascii"))
    return np.frombuffer(raw, dtype=np.float32).reshape(payload["shape"])


def text_to_tokens(text: str) -> list[int]:
    return [ord(ch) % VOCAB_SIZE for ch in text] or [0]


def tokens_to_text(tokens: list[int]) -> str:
    return "".join(chr(32 + (token % 95)) for token in tokens)


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


@dataclass(frozen=True)
class TinyWeights:
    embedding: np.ndarray
    w1: list[np.ndarray]
    b1: list[np.ndarray]
    w2: list[np.ndarray]
    b2: list[np.ndarray]
    lm_head: np.ndarray

    def parameter_bytes(self, stage: int) -> int:
        arrays: list[np.ndarray] = []
        if stage == 1:
            arrays.append(self.embedding)
            indexes = range(0, STAGE_1_LAYERS)
        elif stage == 2:
            arrays.append(self.lm_head)
            indexes = range(STAGE_1_LAYERS, LAYERS)
        else:
            raise ValueError("stage must be 1 or 2")
        for i in indexes:
            arrays.extend([self.w1[i], self.b1[i], self.w2[i], self.b2[i]])
        return sum(arr.nbytes for arr in arrays)


def make_weights() -> TinyWeights:
    cursor = 0

    def normal(shape: tuple[int, ...], scale: float = 0.04) -> np.ndarray:
        nonlocal cursor
        size = int(np.prod(shape))
        indexes = np.arange(cursor, cursor + size, dtype=np.float64)
        cursor += size
        values = np.sin(indexes * 12.9898 + SEED) * 43758.5453
        values = values - np.floor(values)
        values = ((values * 2.0) - 1.0) * scale
        return values.reshape(shape).astype(np.float32)

    return TinyWeights(
        embedding=normal((VOCAB_SIZE, DIM)),
        w1=[normal((DIM, HIDDEN)) for _ in range(LAYERS)],
        b1=[np.zeros((HIDDEN,), dtype=np.float32) for _ in range(LAYERS)],
        w2=[normal((HIDDEN, DIM)) for _ in range(LAYERS)],
        b2=[np.zeros((DIM,), dtype=np.float32) for _ in range(LAYERS)],
        lm_head=normal((DIM, VOCAB_SIZE)),
    )


def block(x: np.ndarray, weights: TinyWeights, index: int) -> np.ndarray:
    residual = x
    x = layer_norm(x)
    x = np.tanh(x @ weights.w1[index] + weights.b1[index])
    x = x @ weights.w2[index] + weights.b2[index]
    return residual + x


def stage1_forward(tokens: list[int], weights: TinyWeights | None = None) -> np.ndarray:
    weights = weights or make_weights()
    x = weights.embedding[np.asarray(tokens, dtype=np.int64)]
    for i in range(STAGE_1_LAYERS):
        x = block(x, weights, i)
    return x.astype(np.float32)


def stage2_forward(hidden: np.ndarray, weights: TinyWeights | None = None) -> np.ndarray:
    weights = weights or make_weights()
    x = hidden.astype(np.float32, copy=False)
    for i in range(STAGE_1_LAYERS, LAYERS):
        x = block(x, weights, i)
    return (layer_norm(x[-1]) @ weights.lm_head).astype(np.float32)


def monolithic_forward(tokens: list[int], weights: TinyWeights | None = None) -> np.ndarray:
    weights = weights or make_weights()
    return stage2_forward(stage1_forward(tokens, weights), weights)


def dumps_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")
