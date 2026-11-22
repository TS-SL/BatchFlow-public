#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace batchflow {

struct ServerConfig {
  std::string bind_host{"127.0.0.1"};
  std::uint16_t bind_port{8080};
  bool allow_non_local_bind{false};  // containers must pass --allow-non-local-bind
  std::string model_path{"models/tiny_deterministic.onnx"};

  // Pre-admission bound: HTTP body size accepted by the parser (per connection).
  std::size_t max_request_bytes{1 * 1024 * 1024};  // 1 MiB
  // Admission bounds, held from admission until the response has been produced.
  // max_pending_input_bytes counts decoded float32 input payload only (4 bytes/element),
  // not HTTP buffers, JSON trees, batch copies or outputs.
  std::size_t max_pending_requests{64};
  std::size_t max_pending_input_bytes{8 * 1024 * 1024};

  // Worker pool size (inference threads). Keep small; ORT intra-op is separate.
  std::size_t worker_threads{1};
  int ort_intra_op_threads{1};
  int ort_inter_op_threads{1};

  // Default client deadline if omitted; larger requested deadlines are rejected.
  std::int64_t default_deadline_ms{5000};
  std::int64_t max_deadline_ms{60 * 60 * 1000};

  // Dynamic batching: FIFO + max size + max wait.
  bool enable_batching{true};
  std::size_t max_batch_size{8};
  std::int64_t max_batch_wait_ms{5};
};

}  // namespace batchflow
