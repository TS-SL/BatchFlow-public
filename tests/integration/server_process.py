"""Start, probe and stop a local batchflow_server process for integration tests."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def http(base: str, method: str, path: str, body: bytes | None = None, timeout: float = 5.0):
    req = urllib.request.Request(base + path, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        data = e.read().decode("utf-8")
        try:
            payload = json.loads(data) if data else {}
        except json.JSONDecodeError:
            payload = {"raw": data}
        return e.code, payload


class ServerProcess:
    """A batchflow_server child process bound to 127.0.0.1 on a free port."""

    def __init__(self, binary: Path, args: list[str], ready_timeout: float = 30.0):
        self.binary = Path(binary)
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.args = ["--host", "127.0.0.1", "--port", str(self.port), *args]
        self.ready_timeout = ready_timeout
        self.proc: subprocess.Popen | None = None
        self.stdout = ""
        self.stderr = ""

    def start(self) -> "ServerProcess":
        if not self.binary.exists():
            raise RuntimeError(f"server binary not found: {self.binary}")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.proc = subprocess.Popen(
            [str(self.binary), *self.args],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            creationflags=flags,
        )
        deadline = time.monotonic() + self.ready_timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self._collect()
                raise RuntimeError(
                    f"server exited with code {self.proc.returncode} before becoming ready: "
                    f"{self.stderr.strip()}"
                )
            try:
                code, body = http(self.base, "GET", "/ready", timeout=1.0)
                if code == 200 and body.get("status") == "ready":
                    return self
            except OSError:
                pass
            time.sleep(0.05)
        self.kill()
        raise RuntimeError(f"server not ready within {self.ready_timeout:.0f} s")

    def request_shutdown(self) -> None:
        """Send the graceful-shutdown signal (Ctrl+Break on Windows, SIGTERM elsewhere)."""
        assert self.proc is not None
        if os.name == "nt":
            os.kill(self.proc.pid, signal.CTRL_BREAK_EVENT)
        else:
            self.proc.send_signal(signal.SIGTERM)

    def wait(self, timeout: float) -> int:
        assert self.proc is not None
        try:
            self.proc.wait(timeout=timeout)
        finally:
            self._collect()
        return self.proc.returncode

    def stop(self, timeout: float = 10.0) -> int:
        if self.proc is None or self.proc.poll() is not None:
            self._collect()
            return self.proc.returncode if self.proc else 0
        self.request_shutdown()
        try:
            return self.wait(timeout)
        except subprocess.TimeoutExpired:
            self.kill()
            raise

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._collect()

    def _collect(self) -> None:
        if self.proc and self.proc.stdout and not self.proc.stdout.closed:
            self.stdout += self.proc.stdout.read()
            self.stderr += self.proc.stderr.read()

    def __enter__(self) -> "ServerProcess":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.kill()


def server_binary() -> Path | None:
    value = os.environ.get("BATCHFLOW_SERVER_BIN")
    return Path(value) if value else None


def required() -> bool:
    return os.environ.get("BATCHFLOW_REQUIRE_SERVER") == "1"


def platform_note() -> str:
    return f"{sys.platform} python {sys.version.split()[0]}"
