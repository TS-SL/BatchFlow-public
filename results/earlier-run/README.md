# Earlier run (redacted copy)

This is a redacted copy of raw evidence from a local run on an earlier development
revision, made before this release's scheduler and tooling fixes. The CSV and metrics
contents are unchanged. Only these edits were made for publication:

* Directory and file names no longer contain the run's wall-clock date and time
  (`[date redacted]`). `http_batch8_metrics.json` was renamed
  `http_batch8_w5_metrics.json` to say which configuration it belongs to.
* The original host-inventory file, which held only an OS string, CPU count and a
  timestamp, was removed. The host was Windows (NT 10.0.26200) x64 with 12 logical CPUs.
* The original processed summary contained absolute local paths and was replaced by
  `summary.json`, recomputed from these CSVs with the current `analyze_results.py`.

Run configuration, from the runner script of that revision: one worker, ORT intra-op 1,
`--max-pending 128`, 200 measured requests (400 for the concurrency-32 file) after 20
warmups, closed-loop clients. `direct_ort.csv` times Python `InferenceSession.run` alone,
with no HTTP. The two `http_batch8_w20_*` files were collected one after the other on the
same server process; the other HTTP files each had a fresh server. Metrics snapshots
exist only for the no-batching and 5 ms-window servers.
