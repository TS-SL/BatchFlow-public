#include "batchflow/admission.hpp"

namespace batchflow {

AdmissionController::AdmissionController(ServerConfig cfg, Metrics& metrics)
    : cfg_(std::move(cfg)), metrics_(metrics) {}

AdmitResult AdmissionController::try_admit(std::size_t input_bytes) {
  std::lock_guard lock(mu_);
  if (!ready_) return AdmitResult::NotReady;
  if (active_ >= cfg_.max_pending_requests) {
    metrics_.requests_rejected.fetch_add(1, std::memory_order_relaxed);
    return AdmitResult::PendingLimit;
  }
  if (active_bytes_ + input_bytes > cfg_.max_pending_input_bytes) {
    metrics_.requests_rejected.fetch_add(1, std::memory_order_relaxed);
    return AdmitResult::ByteLimit;
  }
  ++active_;
  active_bytes_ += input_bytes;
  metrics_.requests_total.fetch_add(1, std::memory_order_relaxed);
  return AdmitResult::Ok;
}

void AdmissionController::release(std::size_t input_bytes) {
  std::lock_guard lock(mu_);
  if (active_ > 0) --active_;
  if (active_bytes_ >= input_bytes) active_bytes_ -= input_bytes;
  else active_bytes_ = 0;
}

bool AdmissionController::ready() const {
  std::lock_guard lock(mu_);
  return ready_;
}

void AdmissionController::set_ready(bool ready) {
  std::lock_guard lock(mu_);
  ready_ = ready;
}

std::size_t AdmissionController::active_count() const {
  std::lock_guard lock(mu_);
  return active_;
}

std::size_t AdmissionController::active_bytes() const {
  std::lock_guard lock(mu_);
  return active_bytes_;
}

}  // namespace batchflow
