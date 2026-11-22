"""HTTP contract checks against an already running server.

Normal configuration (run_live.py starts it): --max-batch 8 --batch-wait-ms 5.
The overload case runs separately against a server started with --max-pending 2.
"""
from __future__ import annotations

import concurrent.futures
import json
import struct
from pathlib import Path

import pytest

from server_process import http

ROOT = Path(__file__).resolve().parents[2]


def _load_blob():
    raw = (ROOT / "models" / "tiny_b1_io.bin").read_bytes()
    assert raw[:4] == b"BFIO"
    batch, in_count, out_count = struct.unpack_from("<iii", raw, 4)
    offs = 16
    xin = list(struct.unpack_from(f"<{in_count}f", raw, offs))
    offs += in_count * 4
    yref = list(struct.unpack_from(f"<{out_count}f", raw, offs))
    return batch, xin, yref


def _payload(**extra) -> bytes:
    batch, xin, _ = _load_blob()
    body = {"input": {"dtype": "float32", "shape": [batch, 3, 8, 8], "data": xin}}
    body.update(extra)
    return json.dumps(body).encode("utf-8")


def test_ready(base_url):
    code, body = http(base_url, "GET", "/ready")
    assert code == 200
    assert body.get("status") == "ready"


def test_infer_matches_reference(base_url):
    _, _, yref = _load_blob()
    code, body = http(base_url, "POST", "/infer", _payload(request_id="itest-1", deadline_ms=5000))
    assert code == 200, body
    assert body["status"] == "ok"
    assert body["request_id"] == "itest-1"
    assert body["output"]["shape"] == [1, 4]
    out = body["output"]["data"]
    assert len(out) == len(yref)
    for a, b in zip(out, yref):
        assert abs(a - b) <= 1e-6 + 1e-5 * abs(b)


def test_bad_json(base_url):
    code, body = http(base_url, "POST", "/infer", b"{not-json")
    assert code == 400
    assert body.get("status") == "bad_request"


def test_shape_mismatch(base_url):
    payload = {"input": {"dtype": "float32", "shape": [1, 3, 7, 8], "data": [0.0] * (3 * 7 * 8)}}
    code, body = http(base_url, "POST", "/infer", json.dumps(payload).encode("utf-8"))
    assert code == 422
    assert body.get("status") == "shape_mismatch"


def test_deadline_above_maximum_rejected(base_url):
    code, body = http(base_url, "POST", "/infer", _payload(deadline_ms=10**15))
    assert code == 400
    assert body["error"]["code"] == "bad_deadline"


def test_oversized_body(base_url):
    # Default --max-body is 1 MiB; send ~1.2 MiB.
    huge = b"{" + (b"\"x\":" + b"1," * (600 * 1024)) + b"\"y\":0}"
    try:
        code, _ = http(base_url, "POST", "/infer", huge, timeout=10.0)
        assert code == 413
    except (ConnectionAbortedError, ConnectionResetError):
        # The server may close the stream after rejecting the body; the request still
        # did not succeed and was not buffered without bound.
        pass


def test_concurrent_infer(base_url):
    _, _, yref = _load_blob()
    payload = _payload()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda _i: http(base_url, "POST", "/infer", payload), range(16)))
    assert [code for code, _ in results] == [200] * 16
    for _, body in results:
        assert body["status"] == "ok"
        assert len(body["output"]["data"]) == len(yref)


def test_deadline_zero(base_url):
    code, body = http(base_url, "POST", "/infer", _payload(request_id="deadline-0", deadline_ms=0))
    assert code == 408
    assert body.get("status") == "deadline_exceeded"


@pytest.mark.overload
def test_overload_rejects(base_url):
    """With --max-pending 2, a burst of 32 requests from 16 threads must see 429s."""
    payload = _payload()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        results = list(
            ex.map(lambda _i: http(base_url, "POST", "/infer", payload, timeout=10.0), range(32))
        )
    codes = [c for c, _ in results]
    assert codes.count(200) + codes.count(429) == 32, codes
    assert codes.count(429) > 0
    assert codes.count(200) > 0
    for code, body in results:
        if code == 429:
            assert body["status"] == "overloaded"
            assert body["error"]["code"] == "queue_full"
