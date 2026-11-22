#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace batchflow {

// Fixed-bucket latency histogram (milliseconds). Thread-safe observe via atomics.
struct LatencyHistogram {
  static constexpr int kBuckets = 12;
  // Upper edges (ms): 1,2,4,8,16,32,64,128,256,512,1024,+Inf
  std::atomic<std::uint64_t> buckets[kBuckets]{};
  std::atomic<std::uint64_t> sum_ms{0};
  std::atomic<std::uint64_t> count{0};

  void observe_ms(double ms) noexcept;
  [[nodiscard]] std::string to_json() const;
};

struct Metrics {
  std::atomic<std::uint64_t> requests_total{0};
  std::atomic<std::uint64_t> requests_success{0};
  std::atomic<std::uint64_t> requests_rejected{0};
  std::atomic<std::uint64_t> requests_error{0};
  std::atomic<std::uint64_t> requests_expired{0};
  std::atomic<std::uint64_t> requests_cancelled{0};
  std::atomic<std::uint64_t> batches_formed{0};
  std::atomic<std::uint64_t> batch_size_sum{0};
  std::atomic<std::uint64_t> jobs_skipped{0};  // batches dropped before ORT (no live request)
  std::atomic<std::uint64_t> active_admitted{0};  // approximate; admission is source of truth

  LatencyHistogram e2e_ms;
  LatencyHistogram queue_wait_ms;
  LatencyHistogram infer_ms;

  [[nodiscard]] std::string to_json(std::uint64_t queue_depth, std::uint64_t active_workers) const;
};

Metrics& global_metrics();

}  // namespace batchflow
