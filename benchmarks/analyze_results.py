#!/usr/bin/env python3
"""Summarize raw latency CSVs written by load_gen.py or direct_infer_bench.py.

Percentiles (linear interpolation between closest ranks) are computed over successful
(HTTP 200) samples only; failed samples are counted separately and never mixed into the
latency distribution.

Exit status is non-zero if any input CSV is missing or empty, if a CSV has no successful
sample, or, with --require-complete N, if any CSV does not contain exactly N samples that
all succeeded. Use --require-complete for anything that will be published.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return sorted_vals[int(k)]
    return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)


def summarize(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ok = sorted(float(r["latency_ms"]) for r in rows if str(r.get("status")) == "200")
    statuses: dict[str, int] = {}
    for r in rows:
        statuses[str(r.get("status"))] = statuses.get(str(r.get("status")), 0) + 1
    return {
        "file": path.name,
        "n": len(rows),
        "ok": len(ok),
        "non_ok": len(rows) - len(ok),
        "status_counts": statuses,
        "p50_ms": percentile(ok, 50),
        "p95_ms": percentile(ok, 95),
        "p99_ms": percentile(ok, 99),
        "mean_ms": sum(ok) / len(ok) if ok else float("nan"),
        "min_ms": ok[0] if ok else float("nan"),
        "max_ms": ok[-1] if ok else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--require-complete", type=int, metavar="N",
                    help="fail unless every CSV has exactly N samples, all HTTP 200")
    args = ap.parse_args()

    summaries, problems = [], []
    for p in args.csvs:
        if not p.exists():
            problems.append(f"{p.name}: missing")
            summaries.append({"file": p.name, "error": "MISSING"})
            continue
        s = summarize(p)
        summaries.append(s)
        if s["n"] == 0:
            problems.append(f"{p.name}: no samples")
        elif s["ok"] == 0:
            problems.append(f"{p.name}: no successful samples")
        if args.require_complete is not None and (
            s["n"] != args.require_complete or s["ok"] != args.require_complete
        ):
            problems.append(
                f"{p.name}: n={s['n']} ok={s['ok']}, required {args.require_complete} successes"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summaries": summaries, "problems": problems}, indent=2),
                        encoding="utf-8")
    print(json.dumps(summaries, indent=2))
    for msg in problems:
        print(f"ERROR: {msg}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
