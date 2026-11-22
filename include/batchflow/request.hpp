#pragma once

#include "batchflow/onnx_session.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace batchflow {

enum class TerminalKind {
  Success,
  Error,
  Expired,
  Cancelled,
  Rejected,      // capacity refused after admission (worker queue full)
  ShuttingDown,  // server stopped before the request could be dispatched
};

// Shared across HTTP session + scheduler. At most one terminal completion fires.
struct ClientRequest : std::enable_shared_from_this<ClientRequest> {
  std::string request_id;
  std::vector<float> input;
  std::int64_t rows{1};  // input batch dimension (must be 1 to join dynamic batches)
  std::size_t input_bytes{0};  // float32 payload bytes charged to the admission budget
  std::chrono::steady_clock::time_point enqueued_at{};
  // Absolute monotonic deadline. The HTTP layer starts the clock after the body has been
  // read, parsed and admitted (see docs/api-contract.md).
  std::chrono::steady_clock::time_point deadline{};
  std::function<void(InferResult, TerminalKind)> completion;

  std::atomic<bool> cancelled{false};
  std::atomic<bool> finished{false};

  // Returns true if this caller won the right to deliver the terminal outcome.
  bool try_finish() {
    bool expected = false;
    return finished.compare_exchange_strong(expected, true, std::memory_order_acq_rel);
  }
};

}  // namespace batchflow
