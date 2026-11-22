# Benchmarks

## Question

On this model and machine, does dynamic batching reduce client latency compared with
running each request alone?

## Method

`benchmarks/run_batching_comparison.py` runs both modes under one identical workload:

| | |
|---|---|
| Comparator | `--no-batching` vs `--max-batch 8 --batch-wait-ms 20` |
| Common server flags | `--workers 1 --ort-intra-threads 1 --ort-inter-threads 1 --max-pending 128` |
| Model | `models/tiny_deterministic.onnx` |
| Request | the `[1,3,8,8]` float32 input in `models/tiny_b1_io.bin`, `deadline_ms` 5000, identical every time |
| Client | `benchmarks/load_gen.py`: 8 threads, closed loop (each thread sends its next request when the previous returns), 200 measured requests per run |
| Warmup | 20 sequential, unrecorded requests per run, on the same server process |
| Repeats | 3 runs per mode, a fresh server per run, mode order alternated between repeats |
| Measurement boundary | `time.perf_counter()` around `urllib.request.urlopen`: new TCP connection, request send, server parsing, admission, queueing, inference and response, and JSON decode |
| Statistics | p50, p95 and p99 by linear interpolation over **successful** samples only (`benchmarks/analyze_results.py`) |
| Validity | the run fails unless every warmup and measured request returned 200, every server started and exited cleanly, and every CSV has exactly the requested number of successes |

To reproduce (after building; see the README):

```powershell
.\.venv\Scripts\python benchmarks/run_batching_comparison.py --server build/batchflow_server.exe --out-dir results/my-run
```

The runner writes raw per-request CSVs, a `/metrics` snapshot per run, `manifest.json`
(source digest, workload, flags, OS, CPU count, Python and ORT versions; no timestamps or
host names), `summary.json` and `summary.md`. It exits non-zero and marks `summary.md`
incomplete if any run failed.

## Results for this release

Source: [results/batching-comparison/](../results/batching-comparison/). The manifest's
`source_sha256` is the source of commit `932c792`; later commits changed only the Linux
ONNX Runtime fetch script and one integration test. Measured locally (Windows 11, 12 logical CPUs, MSVC 19.44 Release, ONNX Runtime 1.32.0 CPU, Python
3.12 client on the same machine). All 1,200 measured requests and all 120 warmups
returned 200.

Pooled over the three runs of each mode (600 requests per mode), in milliseconds:

| Mode | Requests (ok) | Concurrency | Warmups | Mean batch size | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--no-batching` | 600 (600) | 8 | 3 × 20 | 1.00 | 2.10 | 3.15 | 3.91 |
| `--max-batch 8 --batch-wait-ms 20` | 600 (600) | 8 | 3 × 20 | 4.82 | 2.70 | 21.15 | 27.37 |

Per run:

| Run | ok / requests | p50 | p95 | p99 |
|---|---:|---:|---:|---:|
| nobatch r1 | 200 / 200 | 2.12 | 3.22 | 3.92 |
| nobatch r2 | 200 / 200 | 2.08 | 2.82 | 3.71 |
| nobatch r3 | 200 / 200 | 2.10 | 3.22 | 3.73 |
| batch8 w20 r1 | 200 / 200 | 2.63 | 24.92 | 27.19 |
| batch8 w20 r2 | 200 / 200 | 2.59 | 10.03 | 11.88 |
| batch8 w20 r3 | 200 / 200 | 2.90 | 24.26 | 28.12 |

Server-side `/metrics` for the same runs (these include the warmups): the mean inference
time per batch was 0.03 ms without batching and 0.04–0.05 ms with it. The mean scheduler
queue wait was about 0.01 ms without batching and 3.0–5.1 ms with it. Mean batch size per
run was 4.78–4.89.

**Conclusion.** Batching lost at p50, p95 and p99 in every run. Its p99 also varied far
more between runs (11.9–28.1 ms against 3.7–3.9 ms). With inference costing
well under a tenth of a millisecond, the wait to fill a batch dominates. This is a
negative result for batching on a tiny model. It says nothing about models whose
per-call cost is large compared with the batch window, which were not measured.

## Earlier run (retained from a revision before this release's fixes)

[results/earlier-run/](../results/earlier-run/) holds raw CSVs from a local run made with
an earlier development revision: before the deadline, enqueue-refusal and shape-check
fixes, and with the older load generator. It used one server per mode except the last
two, which shared a server. The direct-ORT row measures `session.run` in Python with no
HTTP at all, so it is a lower bound, not a comparator for the HTTP rows. The summary
below was recomputed from the retained CSVs with the current analyzer. Every sample
succeeded, so success-only and all-sample percentiles agree.

| File | Mode | Concurrency | Requests (ok) | p50 ms | p95 ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|
| `direct_ort.csv` | Python ORT `run`, no HTTP | 1 | 200 (200) | 0.008 | 0.011 | 0.012 |
| `http_nobatch_c8.csv` | `--no-batching` | 8 | 200 (200) | 2.36 | 3.21 | 4.10 |
| `http_batch8_w5_c8.csv` | `--max-batch 8 --batch-wait-ms 5` | 8 | 200 (200) | 3.04 | 17.19 | 32.94 |
| `http_batch8_w20_c8.csv` | `--max-batch 8 --batch-wait-ms 20` | 8 | 200 (200) | 2.65 | 8.08 | 12.32 |
| `http_batch8_w20_c32.csv` | `--max-batch 8 --batch-wait-ms 20` | 32 | 400 (400) | 7.08 | 10.32 | 11.75 |

It leads to the same conclusion. Note that the 5 ms window had a *worse* tail than the
20 ms window in that single run. Run-to-run variation at these sample sizes is large
compared with the differences between batching settings, so only the batching-versus-
no-batching difference should be read from it. The `*_metrics.json` files there use an
older field layout (`sum_ms`; the current server reports `sum_us`).
