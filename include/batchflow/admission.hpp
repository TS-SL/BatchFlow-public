#pragma once

#include "batchflow/config.hpp"
#include "batchflow/metrics.hpp"

#include <cstddef>
#include <mutex>

namespace batchflow {

enum class AdmitResult {
  Ok,
  NotReady,
  PendingLimit,
  ByteLimit,
};

// Counts work from successful admit until release (queued or inflight).
class AdmissionController {
 public:
  explicit AdmissionController(ServerConfig cfg, Metrics& metrics);

  [[nodiscard]] AdmitResult try_admit(std::size_t input_bytes);
  void release(std::size_t input_bytes);

  [[nodiscard]] bool ready() const;
  void set_ready(bool ready);

  [[nodiscard]] std::size_t active_count() const;
  [[nodiscard]] std::size_t active_bytes() const;

 private:
  ServerConfig cfg_;
  Metrics& metrics_;
  mutable std::mutex mu_;
  std::size_t active_{0};
  std::size_t active_bytes_{0};
  bool ready_{false};
};

}  // namespace batchflow
