#include "batchflow/metrics.hpp"

#include <cmath>
#include <sstream>

namespace batchflow {
namespace {

int bucket_index(double ms) {
  if (ms < 0) ms = 0;
  double edge = 1.0;
  for (int i = 0; i < LatencyHistogram::kBuckets - 1; ++i) {
    if (ms <= edge) return i;
    edge *= 2.0;
  }
  return LatencyHistogram::kBuckets - 1;
}

}  // namespace

void LatencyHistogram::observe_ms(double ms) noexcept {
  const int idx = bucket_index(ms);
  buckets[idx].fetch_add(1, std::memory_order_relaxed);
  // Store sum in microseconds to avoid truncating sub-millisecond work to zero.
  const double clamped = ms < 0 ? 0.0 : ms;
  sum_ms.fetch_add(static_cast<std::uint64_t>(clamped * 1000.0), std::memory_order_relaxed);
  count.fetch_add(1, std::memory_order_relaxed);
}

std::string LatencyHistogram::to_json() const {
  std::ostringstream os;
  const auto sum_us = sum_ms.load(std::memory_order_relaxed);
  const auto n = count.load(std::memory_order_relaxed);
  os << "{\"count\":" << n
     << ",\"sum_us\":" << sum_us
     << ",\"mean_ms\":" << (n == 0 ? 0.0 : (static_cast<double>(sum_us) / 1000.0) / static_cast<double>(n))
     << ",\"buckets\":[";
  for (int i = 0; i < kBuckets; ++i) {
    if (i) os << ',';
    os << buckets[i].load(std::memory_order_relaxed);
  }
  os << "],\"bucket_edges_ms\":[1,2,4,8,16,32,64,128,256,512,1024,\"Inf\"]}";
  return os.str();
}

std::string Metrics::to_json(std::uint64_t queue_depth, std::uint64_t active_workers) const {
  const auto batches = batches_formed.load(std::memory_order_relaxed);
  const auto bsum = batch_size_sum.load(std::memory_order_relaxed);
  const double mean_batch =
      batches == 0 ? 0.0 : static_cast<double>(bsum) / static_cast<double>(batches);

  std::ostringstream os;
  os << '{'
     << "\"requests_total\":" << requests_total.load(std::memory_order_relaxed) << ','
     << "\"requests_success\":" << requests_success.load(std::memory_order_relaxed) << ','
     << "\"requests_rejected\":" << requests_rejected.load(std::memory_order_relaxed) << ','
     << "\"requests_error\":" << requests_error.load(std::memory_order_relaxed) << ','
     << "\"requests_expired\":" << requests_expired.load(std::memory_order_relaxed) << ','
     << "\"requests_cancelled\":" << requests_cancelled.load(std::memory_order_relaxed) << ','
     << "\"batches_formed\":" << batches << ','
     << "\"batch_size_sum\":" << bsum << ','
     << "\"jobs_skipped\":" << jobs_skipped.load(std::memory_order_relaxed) << ','
     << "\"mean_batch_size\":" << mean_batch << ','
     << "\"queue_depth\":" << queue_depth << ','
     << "\"active_workers\":" << active_workers << ','
     << "\"latency_ms\":{"
     << "\"e2e\":" << e2e_ms.to_json() << ','
     << "\"queue_wait\":" << queue_wait_ms.to_json() << ','
     << "\"inference\":" << infer_ms.to_json()
     << "}}";
  return os.str();
}

Metrics& global_metrics() {
  static Metrics m;
  return m;
}

}  // namespace batchflow
