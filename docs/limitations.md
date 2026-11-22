# Limitations

## Scope

* Single process, single machine. There is no replication, distribution or
  high-availability story.
* CPU execution provider only.
* One tensor contract (`[N,3,8,8]` float32 → `[N,4]` float32), with tensor names fixed as
  `input` and `output`. Other models cannot be served without code changes.
* The bundled model is a generated linear layer with seeded weights. It makes no accuracy
  claim and is far cheaper per call than a real network.

## Behaviour

* **Client disconnects are not observed while a request is pending.** The session does
  not read from the socket while its request is queued or in inference. Abandoned work
  still runs, `/metrics` counts it as a success, and the response write fails. The 499
  `cancelled` outcome exists in the state machine but is effectively unreachable over
  HTTP.
* **No preemption of inference.** ORT calls run to completion. A deadline that passes
  during a call yields 408 after the call, not at the deadline.
* **Expiry is reported at the deadline plus scheduling latency.** On Windows this was
  11–33 ms for a 10 ms deadline in local tests. Timer and thread-scheduling resolution
  depend on the OS and load.
* **One request per connection at a time.** HTTP pipelining is not used. The benchmark
  client also opens a new connection per request, so measured latency includes TCP
  connection setup.
* **Worker failures after startup are not recovered.** If a worker thread exits because
  of an exception other than one ORT reports through `Run`, its queued jobs are not
  completed. In this code base only model-load failure ends a worker, and the server then
  refuses to start.
* **Model paths on Windows must be ASCII.** The path is widened byte by byte.
* `e2e` in responses is truncated to whole milliseconds. The server-side histograms use
  power-of-two millisecond buckets starting at 1 ms, so sub-millisecond detail is only in
  the client CSVs.
* `/metrics` counters include benchmark warmup requests.

## Resource bounds

* The admission byte budget counts decoded float32 input bytes only. Process memory also
  includes every open connection's buffers (bodies up to `--max-body` each, before
  admission), parsed JSON documents, the batch input copy, ORT allocations and response
  JSON. The number of concurrent connections is not limited.
* Requests rejected with 413 because their body is too large close the connection.

## Evidence

* All measurements are short local runs (hundreds of requests per configuration) with
  the client on the same machine as the server. They compare modes; they are not
  capacity numbers.
* Throughput was not measured.
* The GitHub Actions workflow and the Linux/Docker build are defined but have not been run
  for this release.
* `vcpkg.json` has no `builtin-baseline`, so CI resolves Boost and nlohmann-json from the
  vcpkg revision it clones. The local build used Boost 1.92.0.
* Sanitizer options exist in CMake (`BATCHFLOW_ENABLE_ASAN/UBSAN/TSAN`), but no sanitizer
  run is part of this release's evidence.
