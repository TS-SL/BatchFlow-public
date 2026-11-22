#!/usr/bin/env python3
"""Mandatory live HTTP test run: starts each server configuration and enforces counts.

    python tests/integration/run_live.py --server build/batchflow_server[.exe]

Configurations:
  normal    --max-batch 8 --batch-wait-ms 5        pytest -m "not overload and not lifecycle"
  overload  --max-pending 2 --batch-wait-ms 50     pytest -m overload
  lifecycle (tests start their own servers)        pytest -m lifecycle

Exit status is non-zero if a server cannot start or become ready, if it does not exit
cleanly (code 0) on the shutdown signal, if any test fails or errors, if any test is
skipped, or if the number of passed tests differs from the number collected.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from server_process import ServerProcess  # noqa: E402

CONFIGS = [
    ("normal", ["--max-batch", "8", "--batch-wait-ms", "5"], "not overload and not lifecycle"),
    ("overload", ["--max-pending", "2", "--batch-wait-ms", "50"], "overload"),
    ("lifecycle", None, "lifecycle"),
]


def collect_count(marker: str) -> int:
    out = subprocess.run(
        [sys.executable, "-m", "pytest", str(HERE), "-m", marker, "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    ).stdout
    return sum(1 for line in out.splitlines() if "::" in line)


def run_pytest(marker: str, env: dict, junit: Path) -> dict:
    subprocess.run(
        [sys.executable, "-m", "pytest", str(HERE), "-m", marker, "-v", "-rA",
         "-p", "no:cacheprovider", f"--junitxml={junit}"],
        cwd=ROOT, env=env,
    )
    if not junit.exists():
        return {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
    suite = ET.parse(junit).getroot()
    if suite.tag == "testsuites":
        suite = suite[0]
    return {k: int(suite.get(k, 0)) for k in ("tests", "failures", "errors", "skipped")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=Path, required=True)
    ap.add_argument("--model", default="models/tiny_deterministic.onnx")
    ap.add_argument("--junit-dir", type=Path, default=ROOT / "build" / "integration")
    ap.add_argument("--only", choices=[c[0] for c in CONFIGS])
    args = ap.parse_args()
    args.junit_dir.mkdir(parents=True, exist_ok=True)
    server = args.server.resolve()

    rows, ok = [], True
    for name, server_args, marker in CONFIGS:
        if args.only and name != args.only:
            continue
        env = dict(os.environ, BATCHFLOW_REQUIRE_SERVER="1", BATCHFLOW_SERVER_BIN=str(server))
        expected = collect_count(marker)
        srv = None
        problem = ""
        try:
            if server_args is not None:
                srv = ServerProcess(server, ["--model", args.model, *server_args]).start()
                env["BATCHFLOW_BASE_URL"] = srv.base
            counts = run_pytest(marker, env, args.junit_dir / f"{name}.xml")
            if srv is not None:
                code = srv.stop()
                if code != 0:
                    problem = f"server exit code {code} after shutdown signal"
        except Exception as ex:  # noqa: BLE001
            counts = {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
            problem = str(ex)
            if srv is not None:
                srv.kill()
        passed = max(0, counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"])
        good = (not problem and expected > 0 and passed == expected and counts["skipped"] == 0
                and counts["failures"] == 0 and counts["errors"] == 0)
        ok &= good
        rows.append((name, expected, passed, counts["failures"] + counts["errors"],
                     counts["skipped"], "PASS" if good else f"FAIL {problem}".strip()))

    print("\nconfig     collected passed failed skipped result")
    for r in rows:
        print(f"{r[0]:<10} {r[1]:>9} {r[2]:>6} {r[3]:>6} {r[4]:>7} {r[5]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
