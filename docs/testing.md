# Testing

Three suites, from fastest to most end-to-end. All commands run from the repository root.

## 1. Numerical reference (Python, no C++ build)

```powershell
.\.venv\Scripts\python -m pytest tests/python -v
```

Six checks run against Python ONNX Runtime, independent of the C++ code:

* Fixture parity at batch 1, 2 and 4 against `models/tiny_deterministic_fixtures.npz`
  (rtol 1e-5, atol 1e-6).
* Batched equals unbatched at batch 2 and 4.
* The model file's SHA-256 matches `models/tiny_deterministic.json`.

## 2. C++ tests (CTest)

```powershell
ctest --test-dir build --output-on-failure --no-tests=error
```

| Test | Checks |
|---|---|
| `smoke` | Library links; metrics counter works |
| `ort_parity` | C++ ORT output for `models/tiny_b1_io.bin` matches the Python reference |
| `batch_scheduler` | Four requests form a batch and complete successfully |
| `lifecycle.enqueue_refusal` | With a worker queue of capacity 0, every request gets exactly one `Rejected` completion and its admission reservation is released |
| `lifecycle.deadline_before_window` | 10 ms deadline in a 100 ms window expires in under 60 ms (and not before 9 ms) |
| `lifecycle.deadline_not_at_head` | A short-deadline request behind a long-deadline one still expires at its own deadline |
| `lifecycle.batch_output_mapping` | Nine requests (one with 2 rows) each get exactly the rows an independent single-request ORT run produces |
| `lifecycle.wrong_output_shape` | A model whose runtime output is `[2N,2]` yields `Error`, not mis-sliced success |
| `lifecycle.wrong_input_signature` | A model declaring input `[N,3,4,4]` is refused at load time |
| `lifecycle.shutdown_with_queued_work` | Shutdown during a 2 s batch window completes every queued request once, promptly |
| `lifecycle.expired_in_worker_queue` | A batch whose only request expired while waiting for a blocked worker skips inference |

The enqueue-refusal case configures the components directly. The shipping CLI sizes the
worker queue (in jobs) to `--max-pending`, and every job holds at least one admitted
request, so admission refuses before the worker queue can fill. Refusal is reachable when
the pool is stopping or when the components are embedded with a smaller worker queue.
Before the fix, that path iterated a request list that had already been moved into the
job's callback, so the requests never completed and their admission reservations leaked.

`expired_in_worker_queue` blocks the single worker deterministically. The first job's
pre-run hook waits on a gate the test opens only after the second request's deadline has
passed. Timing-based cases separate tolerance from the defect: the defect answered
after ~100 ms, and the bounds allow 50 ms of scheduling slack.

## 3. Live HTTP (mandatory runner)

```powershell
.\.venv\Scripts\python tests/integration/run_live.py --server build/batchflow_server.exe
```

The runner starts the server itself for each configuration, waits up to 30 s for
`/ready`, runs pytest with `BATCHFLOW_REQUIRE_SERVER=1`, sends the shutdown signal, and
requires exit code 0:

| Configuration | Server flags | Tests |
|---|---|---|
| normal | `--max-batch 8 --batch-wait-ms 5` | 8 contract tests: ready, reference output, bad JSON, shape mismatch, deadline above maximum, oversized body, 16 concurrent requests, deadline 0 |
| overload | `--max-pending 2 --batch-wait-ms 50` | 32 requests from 16 threads: only 200 or 429 `queue_full`, at least one of each |
| lifecycle | each test starts its own server | isolated deadline in a 100 ms window, byte budget, multi-row output rows, incompatible model refused, shutdown with 6 queued requests |

It prints a table of collected, passed, failed and skipped counts per configuration.
It exits non-zero if a server cannot start or become ready, if the server does not exit
cleanly, if anything fails or is skipped, or if the number passed differs from the number
collected.

**Required vs optional.** Running `pytest tests/integration` directly without a server is
the optional local mode, and its tests are *skipped*. With `BATCHFLOW_REQUIRE_SERVER=1` an
unreachable server is an *error*:

```text
$ BATCHFLOW_REQUIRE_SERVER=1 BATCHFLOW_BASE_URL=http://127.0.0.1:18999 python -m pytest tests/integration -m "not overload and not lifecycle" -q
6 deselected, 8 errors        (exit 1)
```

CI uses only the runner, so a missing or crashed server cannot produce a green job.

## Results for this release

Run from a clean copy of the release file set on Windows 11 x64: MSVC 19.44, CMake 4.4.4,
Ninja 1.13.2, ONNX Runtime 1.32.0 (C++ NuGet and Python wheel), Boost 1.92.0,
nlohmann-json 3.12.0, Python 3.12.10.

| Suite | Result |
|---|---|
| Numerical reference | 6 passed |
| CTest | 11 of 11 passed |
| Live HTTP | normal 8/8, overload 1/1, lifecycle 5/5; 0 skipped |
| GitHub Actions (commit `76e5eda`, run #3) | Windows job and Docker (Ubuntu 24.04, GCC) job both passed, including CTest and the live HTTP runner |

The release was configured with the same options as the `default` preset, using Boost and
nlohmann-json already installed by vcpkg (`CMAKE_PREFIX_PATH`) instead of the preset's
vcpkg toolchain file. The preset's own vcpkg install step was not re-run for this
release.
