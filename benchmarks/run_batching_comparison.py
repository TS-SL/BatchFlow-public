#!/usr/bin/env python3
"""Batching-disabled vs batching-enabled comparison under one identical HTTP workload.

    python benchmarks/run_batching_comparison.py --server build/batchflow_server[.exe] \
        --out-dir results/batching-comparison

For each repeat, each mode gets a fresh server (workers=1, ORT intra/inter threads=1,
--max-pending 128) on a free localhost port and the same load_gen.py workload. Modes run
in alternating order across repeats to spread slow drift over both.

Writes into --out-dir: one raw CSV per (mode, repeat), a /metrics snapshot per run,
manifest.json (revision, workload, configuration, platform, elapsed seconds per run),
summary.json and summary.md. No wall-clock timestamps or host names are recorded.

Exits non-zero, after writing what it has, if a server fails to start, a load run has any
non-200 response, or the analysis finds incomplete evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "integration"))

from server_process import ServerProcess, http  # noqa: E402

COMMON = ["--model", "models/tiny_deterministic.onnx", "--workers", "1",
          "--ort-intra-threads", "1", "--ort-inter-threads", "1", "--max-pending", "128"]
MODES = {
    "nobatch": ["--no-batching"],
    "batch8_w20": ["--max-batch", "8", "--batch-wait-ms", "20"],
}


def git_revision() -> dict:
    def run(*cmd):
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    rev = run("git", "rev-parse", "HEAD")
    dirty = run("git", "status", "--porcelain")
    return {"commit": rev or "not a git checkout",
            "uncommitted_changes": bool(dirty) if rev else None,
            "source_sha256": source_digest()}


def source_digest() -> str:
    """SHA-256 over the relative paths and bytes of every file that affects the build."""
    h = hashlib.sha256()
    roots = ["CMakeLists.txt", "vcpkg.json", "src", "include", "models", "benchmarks", "tests"]
    files = []
    for r in roots:
        p = ROOT / r
        files += [p] if p.is_file() else [f for f in p.rglob("*") if f.is_file()]
    for f in sorted(files, key=lambda f: f.relative_to(ROOT).as_posix()):
        if "__pycache__" in f.parts:
            continue
        h.update(f.relative_to(ROOT).as_posix().encode() + b"\0")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def ort_version() -> str:
    try:
        import onnxruntime  # noqa: PLC0415
        return onnxruntime.__version__
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    py = sys.executable

    manifest = {
        "description": "BatchFlow batching comparison (client-observed HTTP latency)",
        "revision": git_revision(),
        "workload": {
            "endpoint": "POST /infer",
            "input": "models/tiny_b1_io.bin x (float32 [1,3,8,8]), identical every request",
            "deadline_ms": 5000,
            "requests_per_run": args.requests,
            "concurrency": args.concurrency,
            "client_model": "closed loop: each client thread sends its next request when "
                            "the previous one returns",
            "warmup_requests_per_run": args.warmup,
            "warmup": "sequential, unrecorded, same server process",
            "repeats": args.repeats,
            "measurement": "time.perf_counter around urllib urlopen; includes TCP connect "
                           "(no keep-alive), JSON encode/decode, server queueing and inference",
        },
        "server_common_args": COMMON,
        "modes": MODES,
        "platform": {
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "logical_cpus": os.cpu_count(),
            "python": platform.python_version(),
            "onnxruntime_python": ort_version(),
        },
        "runs": [],
    }
    ok = True
    csvs = []
    for rep in range(args.repeats):
        order = list(MODES) if rep % 2 == 0 else list(reversed(MODES))
        for mode in order:
            name = f"{mode}_r{rep + 1}"
            csv_path = out / f"{name}.csv"
            run = {"name": name, "mode": mode, "repeat": rep + 1, "csv": csv_path.name}
            t0 = time.perf_counter()
            try:
                with ServerProcess(args.server, COMMON + MODES[mode]) as srv:
                    r = subprocess.run(
                        [py, str(ROOT / "benchmarks" / "load_gen.py"), "--base-url", srv.base,
                         "--requests", str(args.requests), "--concurrency", str(args.concurrency),
                         "--warmup", str(args.warmup), "--out", str(csv_path)],
                        cwd=ROOT,
                    )
                    run["load_gen_exit"] = r.returncode
                    _, metrics = http(srv.base, "GET", "/metrics")
                    (out / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2),
                                                              encoding="utf-8")
                    run["server_mean_batch_size"] = metrics.get("mean_batch_size")
                    run["server_exit"] = srv.stop()
            except Exception as ex:  # noqa: BLE001
                run["error"] = str(ex)
            run["elapsed_s"] = round(time.perf_counter() - t0, 3)
            if run.get("error") or run.get("load_gen_exit") != 0 or run.get("server_exit") != 0:
                ok = False
            manifest["runs"].append(run)
            csvs.append(csv_path)

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    r = subprocess.run(
        [py, str(ROOT / "benchmarks" / "analyze_results.py"), *map(str, csvs),
         "--out", str(out / "summary.json"), "--require-complete", str(args.requests)],
        cwd=ROOT, capture_output=True, text=True,
    )
    sys.stderr.write(r.stderr)
    ok &= r.returncode == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    by_file = {s["file"]: s for s in summary["summaries"]}
    lines = [
        f"Workload: {args.requests} requests/run, concurrency {args.concurrency}, "
        f"{args.warmup} warmups/run, {args.repeats} repeats per mode. Latency is "
        "client-observed, successful requests only, milliseconds.",
        "",
        "| run | requests | ok | mean batch | p50 | p95 | p99 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for run in manifest["runs"]:
        s = by_file.get(run["csv"], {})
        mb = run.get("server_mean_batch_size")
        fmt = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) else "n/a"  # noqa: E731
        lines.append(f"| {run['name']} | {s.get('n', 'n/a')} | {s.get('ok', 'n/a')} | "
                     f"{fmt(mb)} | {fmt(s.get('p50_ms'))} | {fmt(s.get('p95_ms'))} | "
                     f"{fmt(s.get('p99_ms'))} |")
    if not ok:
        lines += ["", "**INCOMPLETE: at least one run failed; do not publish these numbers.**"]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
