Workload: 200 requests/run, concurrency 8, 20 warmups/run, 3 repeats per mode. Latency is client-observed, successful requests only, milliseconds.

| run | requests | ok | mean batch | p50 | p95 | p99 |
|---|---:|---:|---:|---:|---:|---:|
| nobatch_r1 | 200 | 200 | 1.00 | 2.12 | 3.22 | 3.92 |
| batch8_w20_r1 | 200 | 200 | 4.78 | 2.63 | 24.92 | 27.19 |
| batch8_w20_r2 | 200 | 200 | 4.89 | 2.59 | 10.03 | 11.88 |
| nobatch_r2 | 200 | 200 | 1.00 | 2.08 | 2.82 | 3.71 |
| nobatch_r3 | 200 | 200 | 1.00 | 2.10 | 3.22 | 3.73 |
| batch8_w20_r3 | 200 | 200 | 4.78 | 2.90 | 24.26 | 28.12 |
