# Architecture

BatchFlow is one process with three kinds of threads and one shared request object.

```
Clients
   │ HTTP/1.1 (Boost.Beast over Asio)          event-loop thread (1)
   ▼
 read body (≤ --max-body) → parse JSON → validate shape → admit (count + bytes)
   ▼
 BatchScheduler: FIFO queue, max batch size, max wait, deadline sweep   scheduler thread (1)
   ▼
 WorkerPool: bounded job queue → pre-run check → ORT Run → shape check  worker threads (--workers)
   ▼
 split output rows per request → net::post completion → HTTP response on event loop
```

## Components

| Component | File | Responsibility |
|---|---|---|
| `HttpServer` / `Session` | `src/transport/http_server.cpp` | Accept, read with a body limit, parse and validate, admit, submit, write responses |
| `AdmissionController` | `src/scheduler/admission.cpp` | Pending-request count and pending input-byte budget; readiness flag |
| `BatchScheduler` | `src/scheduler/batch_scheduler.cpp` | FIFO queue, batch assembly, deadline and cancellation sweep, output demultiplexing |
| `WorkerPool` | `src/scheduler/worker_pool.cpp` | Inference threads, each with its own `OnnxSession`; bounded job queue |
| `OnnxSession` | `src/inference/onnx_session.cpp` | ORT session, load-time signature check, one `Run` per batch |
| `Metrics` | `src/metrics/metrics.cpp` | Atomic counters and fixed-bucket latency histograms served at `/metrics` |

## Ownership and synchronisation

| Mutable state | Owner | Synchronisation |
|---|---|---|
| Socket, parser, buffers | `Session`, on its Asio strand | Asio executor |
| Admission counters | `AdmissionController` | mutex |
| Scheduler queue | `BatchScheduler` | mutex + condition variable |
| Worker job queue | `WorkerPool` | mutex + condition variable |
| ORT session | one per worker thread | not shared |
| Request terminal state | `ClientRequest::try_finish()` | atomic compare-exchange |
| Metrics | process-wide | relaxed atomics |

A `ClientRequest` is shared by the HTTP session, the scheduler queue and, while a batch is
in flight, the batch's completion callback. Whoever first wins `try_finish()` delivers the
single terminal outcome. Every other path checks the flag and does nothing. The completion
callback posts back to the event loop, which writes the response and releases the
admission reservation.

## Request lifecycle

```
received → validated → admitted → queued → batched → dispatched → [inference] → terminal
```

Terminal outcomes and where they arise:

| Outcome | Raised by | HTTP |
|---|---|---|
| Success | worker completion, after the shape check | 200 |
| Expired | scheduler sweep (queued), dispatch, worker pre-run check, or after inference | 408 |
| Cancelled | cancellation flag seen at the same points | 499 (see limitations) |
| Rejected | worker queue refused the batch | 429 `worker_queue_full` |
| ShuttingDown | request left in the scheduler after shutdown | 503 `shutting_down` |
| Error | ORT failure or output shape mismatch | 500 |

Requests refused before admission (malformed, bad shape, over budget, not ready, deadline
of 0) are answered directly by the session and never enter the scheduler.

## Scheduling

* The queue is FIFO. A batch takes consecutive single-row requests from the head, up to
  `--max-batch`. A multi-row request (N > 1) is always dispatched alone, so rows from
  different requests never need reordering.
* The scheduler waits until the earliest of: the batch is full, the head's batch window
  (`enqueued_at + --batch-wait-ms`) closes, or the earliest deadline anywhere in the queue
  passes. On every wake it sweeps the whole queue, not just the head, and fails cancelled
  and expired requests immediately.
* Batches can queue behind each other in the worker pool. Right before `Run`, the worker
  calls the batch's pre-run check. If no request in the batch can still succeed (all
  cancelled, expired or already finished), the ORT call is skipped and counted in
  `jobs_skipped`.
* After `Run`, the scheduler's completion callback requires the output shape to be exactly
  `[total_rows, 4]`, not merely the right element count, before slicing rows back to
  requests in order.

## Threading defaults

One event-loop thread, one scheduler thread, `--workers` worker threads (default 1), and
ORT intra-op and inter-op thread counts of 1 each (`--ort-intra-threads`,
`--ort-inter-threads`). These defaults keep ORT's own thread pools from competing with the
worker threads on a small machine.

## Shutdown

On SIGINT, SIGTERM or (on Windows) Ctrl+Break:

1. Readiness is cleared: `/ready` and new `/infer` requests on open connections get 503.
2. The listening socket is closed.
3. The scheduler stops accepting submissions and dispatches everything already queued
   without waiting for the batch window.
4. The worker pool drains its queue, so in-flight and queued batches finish inference.
5. The event loop gets a 1 s grace period to write those responses, then stops.

Accepted requests therefore get a real response: their result, or 408 if their deadline
passed. Only a request still in the scheduler queue after step 3, which requires a
dispatch failure, is failed with 503 `shutting_down`.

## Design decisions

* **Inference never runs on the event loop.** Even a fast model would stall every
  connection while it ran. Workers run ORT, and completions are posted back with
  `net::post`.
* **One ORT session per worker.** This avoids any question of concurrent `Run` calls on
  one session, at the cost of one copy of the model per worker.
* **FIFO with a size cap and a wait cap, nothing more.** There is no priority or
  shape-aware scheduling; the measurements give no reason to add it.
* **Batch only single-row requests together.** Multi-row requests run alone, which keeps
  output demultiplexing a contiguous slice per request.
* **At-most-once completion through an atomic flag.** The flag stops success,
  cancellation, expiry and rejection paths from double-completing a request or
  double-releasing its admission reservation.
* **Localhost by default.** A non-loopback bind requires `--allow-non-local-bind`. The
  container uses it and is meant to be published only to `127.0.0.1` on the host.
* **A tiny generated model first.** A seeded two-node graph with committed reference
  fixtures makes numerical parity and batch equivalence checkable before any serving
  claim, without downloading a third-party model.
* **ONNX Runtime from NuGet instead of vcpkg.** The vcpkg ORT port failed to fetch its
  source archive for 1.32.0. The official CPU package of the same version as the Python
  wheel is used instead (see `third_party/README.md`).
