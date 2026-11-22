#!/usr/bin/env python3
"""Direct ONNX Runtime CPU baseline (no HTTP). Same model/input as server tests."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--intra-op", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    model = ROOT / "models" / "tiny_deterministic.onnx"
    meta = json.loads((ROOT / "models" / "tiny_deterministic.json").read_text(encoding="utf-8"))
    so = ort.SessionOptions()
    so.intra_op_num_threads = args.intra_op
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(model), so, providers=["CPUExecutionProvider"])

    fixtures = np.load(ROOT / meta["fixture"])
    x = fixtures["x_b1"]

    for _ in range(args.warmup):
        sess.run(["output"], {"input": x})

    rows = []
    for _ in range(args.requests):
        t0 = time.perf_counter()
        sess.run(["output"], {"input": x})
        t1 = time.perf_counter()
        rows.append({"latency_ms": (t1 - t0) * 1000.0, "status": 200, "error": "", "server_e2e_ms": ""})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["latency_ms", "status", "error", "server_e2e_ms"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} n={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
