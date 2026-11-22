# BatchFlow

BatchFlow is a small C++20 HTTP server that runs ONNX models on the CPU with ONNX Runtime
and can group concurrent requests into dynamic batches. It is a study of what bounded
admission, deadlines and dynamic batching do to correctness and tail latency when many
clients send small inference requests at once. It is a single-process project, not a
general-purpose model server.

The server is specialised to one tensor contract: a float32 input `[N,3,8,8]` and a
float32 output `[N,4]`. The bundled model is a tiny seeded linear layer used to exercise
the serving path; it makes no accuracy claim.

## What it does

A request with one `[1,3,8,8]` tensor (shortened here; `data` has 192 numbers):

```http
POST /infer
{"request_id": "demo-1", "deadline_ms": 1000,
 "input": {"dtype": "float32", "shape": [1, 3, 8, 8], "data": [0.5, 0.5, ...]}}
```

returns the model output and server-side timings (captured from this release, started
with `--max-batch 8 --batch-wait-ms 5`):

```json
{"request_id": "demo-1", "status": "ok",
 "output": {"dtype": "float32", "shape": [1, 4],
            "data": [-0.41769838, -0.28555599, 0.42921755, 0.20073949]},
 "timing_ms": {"queue_wait": 6.86, "inference": 0.157, "e2e": 7}}
```

The request was alone, so it waited for the full 5 ms batch window before running for
0.16 ms. With a 100 ms window and `"deadline_ms": 10`, the same request gets
`408 {"status": "deadline_exceeded"}` at its deadline instead of after the window.

## How it works

```
client ──HTTP──▶ Asio/Beast event loop ──▶ admission ──▶ BatchScheduler ──▶ WorkerPool ──▶ ONNX Runtime
                 (parse, validate)         (count +      (FIFO, max size,    (own ORT
                                            byte budget)  max wait, deadlines) session per thread)
        ◀──────────── response posted back to the event loop ◀──────── per-request output rows
```

* **Event loop.** One Boost.Asio thread reads requests, validates JSON and tensor shape,
  and writes responses. It never runs inference.
* **Admission.** A request is admitted only if both the pending-request count and the
  pending input-byte budget have room; otherwise it gets `429`. The reservation is held
  until its response has been produced.
* **Batch scheduler.** One thread keeps a FIFO queue. Single-row requests are grouped up
  to `--max-batch`, waiting at most `--batch-wait-ms` after the oldest one arrived. It
  wakes early for the earliest queued deadline, so an expiring request is answered at its
  deadline rather than at the end of the window. Multi-row requests run alone.
* **Workers.** Each worker thread owns an ONNX Runtime session. Just before running a
  batch, a worker re-checks it and skips inference if every request in it has expired or
  been cancelled. After the run, the output shape is checked against `[rows, 4]` before
  rows are handed back to their requests.
* **Exactly one outcome.** Every admitted request finishes exactly once: success,
  deadline exceeded, overloaded, shutting down, cancelled or error.

[docs/architecture.md](docs/architecture.md) covers threads, ownership and the design
decisions. [docs/api-contract.md](docs/api-contract.md) gives every status code and bound.

## Build, test and run

The supported build is Windows x64 with Visual Studio 2022 (MSVC), CMake and Ninja.
Dependencies are Boost.Asio/Beast and nlohmann-json from vcpkg, plus ONNX Runtime 1.32.0
(CPU) from the official NuGet package. A Linux build is defined in the `Dockerfile` and runs in CI; see
[What has been verified](#what-has-been-verified).

From a "Developer PowerShell for VS 2022" prompt in the repository root:

```powershell
git clone --depth 1 https://github.com/microsoft/vcpkg.git tools/vcpkg
.\tools\vcpkg\bootstrap-vcpkg.bat -disableMetrics
.\scripts\fetch_onnxruntime.ps1
python -m venv .venv
.\.venv\Scripts\pip install -r requirements-dev.txt
cmake --preset default
cmake --build --preset default
```

The `default` preset fails at configure time if ONNX Runtime or the HTTP dependencies are
missing, so a successful build always contains `build/batchflow_server.exe`.

Tests:

```powershell
.\.venv\Scripts\python -m pytest tests/python -v
ctest --test-dir build --output-on-failure --no-tests=error
.\.venv\Scripts\python tests/integration/run_live.py --server build/batchflow_server.exe
```

`run_live.py` starts the server itself in three configurations (normal, low-capacity
overload, lifecycle). It fails if a server cannot start or become ready, if any test is
skipped, or if the number of passing tests differs from the number collected.
[docs/testing.md](docs/testing.md) lists what each suite covers.

Run the server (localhost only by default):

```powershell
.\build\batchflow_server.exe --port 8080 --model models/tiny_deterministic.onnx --max-batch 8 --batch-wait-ms 5
```

`--no-batching` turns batching off and `--help` lists every option. `--model` only changes
which file is loaded. The file must still match the `[N,3,8,8]` → `[N,4]` contract, or
the server refuses to start.

## Results: batching on this model makes tail latency worse

Measured locally on Windows 11 (12 logical CPUs, MSVC Release build, ONNX Runtime 1.32.0
CPU, one worker, ORT intra/inter-op threads = 1) from the source of commit `932c792`.
Later commits changed only the Linux ONNX Runtime fetch script, one test and the docs.
The load generator ran on the same machine. Each run used a fresh server and 8 closed-loop client threads sending 200
identical `[1,3,8,8]` requests after 20 unrecorded warmups, with 3 runs per mode. Latency
is client-observed: it includes a new TCP connection per request, JSON handling, queueing
and inference. Only successful requests are counted, and all 1,200 succeeded.

| Server mode | Requests (ok) | Mean batch size | p50 ms | p95 ms | p99 ms | p99 range across runs |
|---|---:|---:|---:|---:|---:|---:|
| `--no-batching` | 600 (600) | 1.00 | 2.10 | 3.15 | 3.91 | 3.71–3.92 |
| `--max-batch 8 --batch-wait-ms 20` | 600 (600) | 4.82 | 2.70 | 21.15 | 27.37 | 11.88–28.12 |

Batching did form batches of about five requests. However, the server's mean inference
time per batch was only 0.03–0.05 ms, while its mean scheduler queue wait rose from about
0.01 ms to 3.0–5.1 ms. Waiting for a batch to fill costs far more than the inference it
saves, so p95 and p99 rose sharply. Batching would need a model whose per-call cost is
large compared with the wait window before it could pay off. That case was not measured
here. Throughput was not measured.

Raw samples, server metrics and the run manifest are in
[results/batching-comparison/](results/batching-comparison/). The method and a summary of
an earlier, smaller run (retained from a revision before this release's fixes) are in
[docs/benchmarks.md](docs/benchmarks.md).

## What has been verified

| Check | Status for this release |
|---|---|
| Windows build with ORT + HTTP server target | Built from the clean release file set |
| Python reference checks (6) | Pass |
| CTest (11: smoke, ORT parity, batch mapping, 8 lifecycle cases) | Pass |
| Live HTTP: normal 8, overload 1, lifecycle 5 | 14 of 14 pass, 0 skipped |
| Batching benchmark above | Run on this release; all requests succeeded |
| GitHub Actions, Windows job | Passed on commit `76e5eda` (CI run #3): build, Python checks, CTest, live HTTP 14/14 with 0 skipped |
| GitHub Actions, Docker job (Ubuntu 24.04, GCC) | Passed on commit `76e5eda` (CI run #3): build, CTest, live HTTP (the runner fails the image build on any failure or skip), container health check |

## Limitations that affect interpretation

* **One fixed tensor contract.** The only model exercised is a tiny generated one. The
  latency results describe serving overhead around a sub-millisecond model, not
  application-level inference.
* **Single machine.** The client and server shared one machine, and runs were short
  (hundreds of requests). The numbers show how the modes compare under one workload;
  they are not capacity figures.
* **Client disconnects are not detected while a request waits.** The server does not read
  from a connection while its request is queued or running. A client that gives up does
  not cancel its work; the inference still runs, and the response write then fails.
* **Admission budgets cover only part of memory.** They count admitted requests and their
  decoded float32 input bytes. They do not bound connections, HTTP buffers (up to
  `--max-body` each, before admission), JSON trees, batch copies or outputs.
* **No preemption.** A deadline that passes during an ORT call is reported when the call
  returns, as `408` rather than a late success.

[docs/limitations.md](docs/limitations.md) has the full list.

## Repository layout

| Path | Contents |
|---|---|
| `src/`, `include/batchflow/` | Server, scheduler, worker pool, admission, ORT adapter, metrics |
| `tests/unit/` | C++ tests registered with CTest |
| `tests/integration/` | Live HTTP tests and the `run_live.py` runner |
| `tests/python/` | Independent numerical checks of the model with Python ONNX Runtime |
| `tests/fixtures/` | Negative-test ONNX models and their generator |
| `models/` | The bundled model, fixtures and provenance ([models/README.md](models/README.md)) |
| `benchmarks/` | Load generator, analyzer and the batching comparison runner |
| `results/` | Benchmark evidence ([results/README.md](results/README.md)) |
| `scripts/` | Dependency fetch, model export and a dependency-free dev build |

## License

MIT; see [LICENSE](LICENSE). Third-party dependencies and their sources are listed in
[third_party/README.md](third_party/README.md).
