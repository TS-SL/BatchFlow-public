# HTTP API contract

Default bind: `127.0.0.1:8080`. All bodies are JSON (`Content-Type: application/json`).

## Tensor contract

The server serves exactly one tensor signature:

| | Name | dtype | Shape |
|---|---|---|---|
| Input | `input` | float32 | `[N, 3, 8, 8]`, N ≥ 1 |
| Output | `output` | float32 | `[N, 4]` |

`--model PATH` selects the ONNX file and nothing else. At load time the server requires
one input and one output with the names, dtype and fixed dimensions above. Symbolic or
unknown dimensions are accepted. A model that does not match makes the server exit with
code 2 and a message such as `model input shape [-1,3,4,4] is not compatible with
[-1,3,8,8]`. After every batch, the actual output shape must equal `[rows, 4]`, or every
request in the batch gets 500. A model whose output only has the right element count
(for example `[2N, 2]`) is therefore rejected rather than mis-sliced.

## POST /infer

```json
{
  "request_id": "optional-client-id",
  "deadline_ms": 5000,
  "input": {"dtype": "float32", "shape": [1, 3, 8, 8], "data": [0.0, ...]}
}
```

* `request_id` is echoed back; the server generates one if it is absent.
* `data` is the row-major flattened tensor; its length must equal the product of `shape`.
* `deadline_ms` defaults to 5000. It must be an integer and at most 3,600,000 (1 hour).
  Zero or a negative value is answered immediately with 408.

### Deadline semantics

* The deadline clock (monotonic) starts **after** the body has been fully read, parsed and
  validated and the request has been admitted. Time spent receiving the body, which the
  30 s socket timeout bounds, is not charged to the deadline.
* While a request is queued in the scheduler, the server answers 408 at its deadline. The
  scheduler wakes for the earliest queued deadline, so the response comes at the deadline
  plus thread-scheduling and event-loop latency, not at the end of the batch window. On
  the development machine (Windows), 20 isolated requests with a 10 ms deadline and a
  100 ms window got their 408 after 11–33 ms client-observed, apart from the first
  request on a new client at 51 ms. Before the fix the same requests got 408 after
  102–142 ms, because the scheduler slept until the batch window closed.
* A request whose batch is waiting for a busy worker is re-checked right before
  inference. If every request in that batch has expired, inference is skipped.
* ORT calls are not interrupted. If the deadline passes during inference, the request
  gets 408 when inference returns, never a late 200.
* The server does not promise a response *before* the deadline. A request can succeed
  only if its batch completes before its deadline.

### Success: 200

```json
{"request_id": "demo-1", "status": "ok",
 "output": {"dtype": "float32", "shape": [1, 4], "data": [-0.4177, -0.2856, 0.4292, 0.2007]},
 "timing_ms": {"queue_wait": 6.86, "inference": 0.157, "e2e": 7}}
```

`queue_wait` runs from scheduler submission to batch dispatch; `inference` is the ORT call
for the whole batch; `e2e` runs from admission to completion, truncated to whole
milliseconds.

### Errors

Error bodies have the form
`{"request_id": "...", "status": "...", "error": {"code": "...", "message": "..."}}`.

| HTTP | `status` | `error.code` | Raised when |
|---|---|---|---|
| 400 | `bad_request` | `malformed_json`, `missing_input`, `bad_dtype`, `bad_shape`, `bad_data`, `length_mismatch`, `bad_deadline` | Body is not valid JSON or fields are malformed |
| 413 | `payload_too_large` | `body_limit` | Body exceeds `--max-body` (the connection is then closed) |
| 422 | `shape_mismatch` | `shape_mismatch` | Shape is not `[N,3,8,8]` with N > 0 |
| 429 | `overloaded` | `queue_full` | Pending-request limit (`--max-pending`) reached at admission |
| 429 | `overloaded` | `byte_budget` | Pending input-byte budget (`--max-pending-bytes`) would be exceeded |
| 429 | `overloaded` | `scheduler_queue_full`, `worker_queue_full` | An internal queue refused admitted work (not expected with CLI defaults) |
| 408 | `deadline_exceeded` | `deadline` | Deadline passed before a successful result |
| 499 | `cancelled` | `cancelled` | Request was flagged cancelled (see below) |
| 503 | `not_ready` | `not_ready` | Model not loaded, or server shutting down, when the request arrived |
| 503 | `shutting_down` | `shutting_down` | Admitted request was still queued after shutdown |
| 500 | `inference_error` | `infer_failed` | ORT failed or the output shape was wrong; no internal details are returned |

**Overload, deadline and cancellation are distinct.** Overload (429) means the request
was refused *before* it was queued. Deadline (408) means it was admitted and did not
finish in time. Cancellation (499) exists in the state machine, but the HTTP layer only
sets the cancellation flag when it closes a connection. It does not read from a
connection while that connection's request is pending, so a client disconnect during
queueing or inference is not observed until the response write fails. In practice, an
abandoned request runs to completion and is counted as a success in `/metrics`.

## GET /health

`200 {"status":"ok"}` while the process is serving HTTP.

## GET /ready

`200 {"status":"ready"}` once the model has loaded and the server is accepting work. 503
`{"status":"not_ready"}` before that and during shutdown.

## GET /metrics

JSON with request counters (`requests_total`, `requests_success`, `requests_rejected`,
`requests_error`, `requests_expired`, `requests_cancelled`), batch counters
(`batches_formed`, `batch_size_sum`, `mean_batch_size`, `jobs_skipped`), `queue_depth`,
`active_workers`, and `e2e`, `queue_wait` and `inference` latency histograms with bucket
upper edges of 1, 2, 4, …, 1024 ms and +Inf. `requests_success` counts responses
produced, not responses delivered to the client.

## Bounds

| Bound | Flag (default) | Applies | Covers |
|---|---|---|---|
| Request body size | `--max-body` (1 MiB) | Before admission, per connection | Bytes the HTTP parser will buffer for one body |
| Header size | Beast default (8 KiB) | Before admission | Request line and headers |
| Socket read timeout | fixed 30 s | Before admission | Time to receive one request |
| Pending requests | `--max-pending` (64) | Admission → response produced | Count of admitted requests (queued, batched or in inference) |
| Pending input bytes | `--max-pending-bytes` (8 MiB) | Admission → response produced | Decoded float32 input payload: 4 bytes × elements |
| Scheduler queue | `--max-pending` | After admission | Requests waiting for batching (cannot exceed admitted count) |
| Worker job queue | `--max-pending` jobs | After admission | Batches waiting for a worker |
| Batch size / wait | `--max-batch` (8), `--batch-wait-ms` (5) | Scheduler | Rows per batch, wait after the oldest request |
| Deadline | `deadline_ms` ≤ 3,600,000 | After admission | Per request |

The byte budget is **not** a bound on process memory. It excludes concurrent connections
(their number is not limited), header and body buffers before admission, the parsed JSON
document (several times larger than the payload), the batch input copy, ORT's own
allocations, output tensors and response JSON.

## Shutdown

See [architecture.md](architecture.md#shutdown). In short: readiness drops to 503, the
listener closes, already admitted work runs to a terminal response, and the process exits
0 after a 1 s write grace period.
