"""Shared fixtures for the HTTP integration suite.

Two execution modes:

* Local/optional (default): if no server answers at BATCHFLOW_BASE_URL, tests that need
  one are skipped, and self-launching tests are skipped without BATCHFLOW_SERVER_BIN.
* Required (BATCHFLOW_REQUIRE_SERVER=1, used by tests/integration/run_live.py and CI):
  an unreachable server or a missing binary is a test *failure*, never a skip.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from server_process import ServerProcess, http, required, server_binary  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "overload: needs a server started with --max-pending 2")
    config.addinivalue_line("markers", "lifecycle: starts its own server (BATCHFLOW_SERVER_BIN)")


def _unavailable(msg: str):
    if required():
        pytest.fail(msg, pytrace=False)
    pytest.skip(msg)


@pytest.fixture(scope="module")
def base_url():
    base = os.environ.get("BATCHFLOW_BASE_URL", "http://127.0.0.1:18080")
    try:
        code, body = http(base, "GET", "/health", timeout=2.0)
    except Exception as ex:  # noqa: BLE001
        _unavailable(f"server not reachable at {base}: {ex}")
    assert code == 200 and body.get("status") == "ok"
    return base


@pytest.fixture
def launch():
    """Factory that starts a dedicated server with the given CLI args."""
    binary = server_binary()
    if binary is None:
        _unavailable("BATCHFLOW_SERVER_BIN is not set")
    started: list[ServerProcess] = []

    def _start(*args: str) -> ServerProcess:
        srv = ServerProcess(binary, list(args))
        started.append(srv)
        return srv.start()

    yield _start
    for srv in started:
        srv.kill()
