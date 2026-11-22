#!/usr/bin/env python3
"""Closed-loop HTTP load generator for BatchFlow.

Sends --requests POST /infer calls from --concurrency client threads (each thread sends
its next request as soon as the previous one returns) after --warmup sequential,
unrecorded requests. Every request uses the same [1,3,8,8] float32 input from
models/tiny_b1_io.bin and a 5000 ms deadline.

Writes one CSV row per measured request: client-observed latency in milliseconds
(time.perf_counter around urlopen, so it includes connection setup and JSON encode/decode
on the client), HTTP status, error, and the server-reported e2e milliseconds.

Exits non-zero if any warmup or measured request did not return HTTP 200, unless
--allow-failures is given. Percentiles are computed by analyze_results.py, not here.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import struct
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["seq", "latency_ms", "status", "error", "server_e2e_ms"]


def load_payload() -> bytes:
    raw = (ROOT / "models" / "tiny_b1_io.bin").read_bytes()
    assert raw[:4] == b"BFIO"
    batch, in_count, _out = struct.unpack_from("<iii", raw, 4)
    xin = list(struct.unpack_from(f"<{in_count}f", raw, 16))
    body = {
        "deadline_ms": 5000,
        "input": {"dtype": "float32", "shape": [batch, 3, 8, 8], "data": xin},
    }
    return json.dumps(body).encode("utf-8")


def one_request(base: str, payload: bytes, timeout: float, seq: int) -> dict:
    req = urllib.request.Request(
        base.rstrip("/") + "/infer", data=payload, method="POST",
        headers={"Content-Type": "application/json"},
    )
    status, err, server_e2e = 0, "", ""
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            data = json.loads(resp.read().decode("utf-8"))
            server_e2e = data.get("timing_ms", {}).get("e2e", "")
    except urllib.error.HTTPError as e:
        status = e.code
        err = f"http_{e.code}"
    except Exception as ex:  # noqa: BLE001
        err = type(ex).__name__
    t1 = time.perf_counter()
    return {"seq": seq, "latency_ms": (t1 - t0) * 1000.0, "status": status, "error": err,
            "server_e2e_ms": server_e2e}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18080")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--allow-failures", action="store_true",
                    help="exit 0 even if some requests did not return 200")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        ap.error("--requests and --concurrency must be positive")

    payload = load_payload()
    warm = [one_request(args.base_url, payload, args.timeout, -1) for _ in range(args.warmup)]
    warm_bad = sum(1 for r in warm if r["status"] != 200)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(one_request, args.base_url, payload, args.timeout, i)
                for i in range(args.requests)]
        rows = [f.result() for f in futs]  # in submission order

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = sum(1 for r in rows if r["status"] == 200)
    print(f"wrote {args.out.name} n={len(rows)} ok={ok} fail={len(rows) - ok} "
          f"warmup={args.warmup} warmup_fail={warm_bad}")
    if (ok != len(rows) or warm_bad) and not args.allow_failures:
        print("ERROR: not every request returned HTTP 200", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
