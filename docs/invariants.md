# Correctness invariants

What the service is meant to guarantee, and the test or code that checks each one.
"Design" means the property follows from the code structure and no test exercises it
directly.

| # | Invariant | Checked by |
|---|---|---|
| 1 | An admitted request receives exactly one terminal outcome | `ClientRequest::try_finish`; `lifecycle.enqueue_refusal`, `lifecycle.shutdown_with_queued_work`, `lifecycle.batch_output_mapping` (each asserts one completion per request) |
| 2 | Every admission reservation is released exactly once | `lifecycle.enqueue_refusal` (count and bytes return to 0); release happens once in the event-loop completion handler |
| 3 | A successful response carries the rows computed for that request's input | `lifecycle.batch_output_mapping`, HTTP `test_multi_row_request_returns_matching_rows` |
| 4 | Success agrees with an independent reference within rtol 1e-5 / atol 1e-6 | `tests/python`, `ort_parity`, HTTP `test_infer_matches_reference` |
| 5 | Batched and unbatched inference agree within the same tolerance | `tests/python/test_batched_equals_unbatched`, `lifecycle.batch_output_mapping` |
| 6 | Output is returned only if its shape is exactly `[rows, 4]` | `lifecycle.wrong_output_shape` |
| 7 | A model with an incompatible signature is never served | `lifecycle.wrong_input_signature`, HTTP `test_incompatible_model_refuses_to_start` |
| 8 | Admitted requests ≤ `--max-pending`; admitted input bytes ≤ `--max-pending-bytes` | `AdmissionController`; HTTP `test_overload_rejects`, `test_byte_budget_rejects_before_queueing` |
| 9 | An expired request is never returned as a success | Deadline checked at sweep, dispatch, pre-run and after inference; `lifecycle.deadline_*`, HTTP `test_isolated_deadline_expires_before_batch_window` |
| 10 | A queued request that expires is answered at its deadline, not at the end of the batch window | `lifecycle.deadline_before_window`, `lifecycle.deadline_not_at_head`, HTTP isolated-deadline test |
| 11 | Inference is skipped for a batch with no live request | `lifecycle.expired_in_worker_queue` |
| 12 | Inference never runs on the event-loop thread | Design: only `WorkerPool` threads call `OnnxSession::run` |
| 13 | Shutdown answers every accepted request and exits with code 0 | `lifecycle.shutdown_with_queued_work`, HTTP `test_shutdown_answers_accepted_requests`, and the runner's exit-code check for every configuration |
| 14 | Error responses do not expose internal paths or exception text for inference failures | Design: `inference_error` returns a fixed message |
| 15 | Non-loopback binds require an explicit flag | `main.cpp` refuses the bind otherwise (not covered by a test) |

## Known gaps

* Cancelling on client disconnect is not implemented for pending requests (see
  [limitations.md](limitations.md)). No invariant above depends on it.
* Process memory is not bounded by the admission budget (see
  [api-contract.md](api-contract.md#bounds)).
