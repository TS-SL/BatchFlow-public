"""Lifecycle checks that need a dedicated server configuration.

Each test starts its own batchflow_server (BATCHFLOW_SERVER_BIN) on a free localhost port.
"""
from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path

import pytest

from server_process import http

pytestmark = pytest.mark.lifecycle

ROOT = Path(__file__).resolve().parents[2]
MODEL = "models/tiny_deterministic.onnx"


def _x1() -> list[float]:
    raw = (ROOT / "models" / "tiny_b1_io.bin").read_bytes()
    _b, in_count, _o = struct.unpack_from("<iii", raw, 4)
    return list(struct.unpack_from(f"<{in_count}f", raw, 16))


def _payload(rows: int = 1, **extra) -> bytes:
    x = _x1() * rows
    body = {"input": {"dtype": "float32", "shape": [rows, 3, 8, 8], "data": x}}
    body.update(extra)
    return json.dumps(body).encode("utf-8")


def test_isolated_deadline_expires_before_batch_window(launch):
    """10 ms deadline, 100 ms batch window, no other traffic: 408 well before 100 ms.

    The service promises the 408 at the deadline plus scheduling tolerance while the
    request is still queued. The 80 ms bound is that tolerance: client-observed HTTP
    latency on a busy CI host, which is separate from the defect being guarded against
    (an answer only after the 100 ms window, ~102 ms or more).
    """
    srv = launch("--model", MODEL, "--max-batch", "8", "--batch-wait-ms", "100")
    body = _payload(deadline_ms=10)
    http(srv.base, "POST", "/infer", body)  # untimed: first request warms the client side
    elapsed = []
    for _ in range(5):
        t0 = time.perf_counter()
        code, resp = http(srv.base, "POST", "/infer", body)
        elapsed.append((time.perf_counter() - t0) * 1000.0)
        assert code == 408, resp
        assert resp["status"] == "deadline_exceeded"
        time.sleep(0.05)
    assert max(elapsed) < 80.0, elapsed
    assert min(elapsed) >= 9.0, elapsed


def test_byte_budget_rejects_before_queueing(launch):
    """A 2-row request carries 2*192*4 = 1536 payload bytes > --max-pending-bytes 1000."""
    srv = launch("--model", MODEL, "--max-pending-bytes", "1000")
    code, resp = http(srv.base, "POST", "/infer", _payload(rows=2))
    assert code == 429
    assert resp["status"] == "overloaded"
    assert resp["error"]["code"] == "byte_budget"
    code, resp = http(srv.base, "POST", "/infer", _payload(rows=1))  # 768 bytes fits
    assert code == 200, resp


def test_multi_row_request_returns_matching_rows(launch):
    srv = launch("--model", MODEL)
    code1, one = http(srv.base, "POST", "/infer", _payload(rows=1))
    code3, three = http(srv.base, "POST", "/infer", _payload(rows=3))
    assert code1 == 200 and code3 == 200
    assert three["output"]["shape"] == [3, 4]
    row = one["output"]["data"]
    for r in range(3):
        got = three["output"]["data"][r * 4:(r + 1) * 4]
        for a, b in zip(got, row):
            assert abs(a - b) <= 1e-6 + 1e-5 * abs(b)


def test_incompatible_model_refuses_to_start(launch):
    with pytest.raises(RuntimeError) as info:
        launch("--model", "tests/fixtures/wrong_input_signature.onnx")
    assert "not compatible" in str(info.value)


def test_shutdown_answers_accepted_requests(launch):
    """Requests waiting in a 2 s batch window when shutdown starts are still answered."""
    srv = launch("--model", MODEL, "--max-batch", "64", "--batch-wait-ms", "2000")
    n = 6
    results: list[tuple[int, dict] | Exception] = [None] * n  # type: ignore[list-item]

    def worker(i: int) -> None:
        try:
            results[i] = http(srv.base, "POST", "/infer", _payload(deadline_ms=10000), timeout=10.0)
        except Exception as ex:  # noqa: BLE001
            results[i] = ex

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    # Wait (bounded) until all requests are queued in the scheduler.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        _, metrics = http(srv.base, "GET", "/metrics")
        if metrics.get("queue_depth") == n:
            break
        time.sleep(0.01)
    assert metrics.get("queue_depth") == n, metrics

    t0 = time.monotonic()
    srv.request_shutdown()
    for t in threads:
        t.join(timeout=10.0)
    exit_code = srv.wait(timeout=10.0)
    shutdown_s = time.monotonic() - t0

    assert exit_code == 0, srv.stderr
    assert shutdown_s < 5.0, shutdown_s  # did not wait out the 2 s window plus extra
    for r in results:
        assert not isinstance(r, Exception), r
        code, body = r
        assert code == 200, body
        assert body["status"] == "ok"
    assert "success=6" in srv.stdout, srv.stdout
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", srv.port), timeout=1.0).close()
