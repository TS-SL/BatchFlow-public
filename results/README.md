# Results

| Directory | What it is | Reproduced on this release? |
|---|---|---|
| [batching-comparison/](batching-comparison/) | Batching off vs `--max-batch 8 --batch-wait-ms 20`, 3 × 200 requests per mode | Yes |
| [earlier-run/](earlier-run/) | Raw CSVs from a smaller local run on an earlier development revision | No, retained for comparison |

[docs/benchmarks.md](../docs/benchmarks.md) explains the workload, configuration,
measurement boundary and conclusions. Each `*.csv` has one row per measured request in
submission order. Columns: `seq` (current runner only), `latency_ms` (client-observed),
`status` (HTTP), `error`, and `server_e2e_ms` (the server's own whole-millisecond e2e).
`*_metrics.json` is the server's `/metrics` snapshot at the end of a run, warmups
included.
