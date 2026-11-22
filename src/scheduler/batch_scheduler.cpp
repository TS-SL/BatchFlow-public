#include "batchflow/batch_scheduler.hpp"

#include <algorithm>
#include <utility>

namespace batchflow {
namespace {

using Clock = std::chrono::steady_clock;

// Served tensor contract: input [N,3,8,8] -> output [N,4] (see onnx_session.hpp).
constexpr std::size_t kElemsPerRow = 3 * 8 * 8;
constexpr std::int64_t kOutPerRow = 4;

void count_terminal(Metrics& metrics, TerminalKind kind) {
  switch (kind) {
    case TerminalKind::Expired:
      metrics.requests_expired.fetch_add(1, std::memory_order_relaxed);
      break;
    case TerminalKind::Cancelled:
      metrics.requests_cancelled.fetch_add(1, std::memory_order_relaxed);
      break;
    case TerminalKind::Rejected:
    case TerminalKind::ShuttingDown:
      metrics.requests_rejected.fetch_add(1, std::memory_order_relaxed);
      break;
    case TerminalKind::Error:
      metrics.requests_error.fetch_add(1, std::memory_order_relaxed);
      break;
    case TerminalKind::Success:
      break;  // counted by the transport when the response is produced
  }
}

// Delivers a failure outcome if this caller wins the request's single terminal slot.
void finish_failed(Metrics& metrics, const std::shared_ptr<ClientRequest>& req,
                   TerminalKind kind, std::string message) {
  if (!req || !req->try_finish()) return;
  count_terminal(metrics, kind);
  InferResult r;
  r.ok = false;
  r.error = std::move(message);
  if (req->completion) req->completion(std::move(r), kind);
}

bool is_live(const ClientRequest& req, Clock::time_point now) {
  return !req.finished.load(std::memory_order_acquire) &&
         !req.cancelled.load(std::memory_order_acquire) && now < req.deadline;
}

}  // namespace

BatchScheduler::BatchScheduler(ServerConfig cfg, WorkerPool& workers, Metrics& metrics)
    : cfg_(std::move(cfg)), workers_(workers), metrics_(metrics) {
  thread_ = std::thread([this] { run(); });
}

BatchScheduler::~BatchScheduler() { shutdown(); }

std::size_t BatchScheduler::queued() const {
  std::lock_guard lock(mu_);
  return queue_.size();
}

bool BatchScheduler::submit(std::shared_ptr<ClientRequest> req) {
  {
    std::lock_guard lock(mu_);
    if (stopping_) return false;
    if (queue_.size() >= cfg_.max_pending_requests) return false;
    req->enqueued_at = Clock::now();
    queue_.push_back(std::move(req));
  }
  cv_.notify_one();
  return true;
}

// Stops accepting new work, then lets run() dispatch everything already queued without
// waiting for the batch window. Dispatched work is completed by WorkerPool::shutdown(),
// which drains its queue before joining. Anything still queued after the scheduler
// thread exits (only possible if dispatch failed) is failed with ShuttingDown.
void BatchScheduler::shutdown() {
  {
    std::lock_guard lock(mu_);
    if (stopping_) return;
    stopping_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();

  std::deque<std::shared_ptr<ClientRequest>> left;
  {
    std::lock_guard lock(mu_);
    left.swap(queue_);
  }
  for (auto& req : left) {
    fail_request(req, TerminalKind::ShuttingDown, "server shutting down");
  }
}

void BatchScheduler::fail_request(const std::shared_ptr<ClientRequest>& req, TerminalKind kind,
                                  std::string message) {
  finish_failed(metrics_, req, kind, std::move(message));
}

void BatchScheduler::dispatch_batch(std::vector<std::shared_ptr<ClientRequest>> batch) {
  if (batch.empty()) return;

  // Drop cancelled/expired requests that are still in the assembled batch.
  std::vector<std::shared_ptr<ClientRequest>> live;
  live.reserve(batch.size());
  const auto now = Clock::now();
  for (auto& req : batch) {
    if (req->cancelled.load(std::memory_order_acquire)) {
      fail_request(req, TerminalKind::Cancelled, "cancelled before dispatch");
      continue;
    }
    if (now >= req->deadline) {
      fail_request(req, TerminalKind::Expired, "deadline exceeded before dispatch");
      continue;
    }
    live.push_back(req);
  }
  if (live.empty()) return;

  InferJob job;
  std::int64_t total_rows = 0;
  for (auto& req : live) total_rows += req->rows;
  job.batch_size = total_rows;
  job.input.reserve(static_cast<std::size_t>(total_rows) * kElemsPerRow);
  for (auto& req : live) {
    job.input.insert(job.input.end(), req->input.begin(), req->input.end());
  }

  metrics_.batches_formed.fetch_add(1, std::memory_order_relaxed);
  metrics_.batch_size_sum.fetch_add(static_cast<std::uint64_t>(live.size()),
                                    std::memory_order_relaxed);

  const auto dispatch_at = Clock::now();
  auto* metrics = &metrics_;

  // Runs on the worker thread right before ORT. A batch may wait in the worker queue
  // behind earlier batches, so re-check that at least one request can still succeed.
  job.still_needed = [live, metrics] {
    const auto t = Clock::now();
    for (const auto& req : live) {
      if (is_live(*req, t)) return true;
    }
    metrics->jobs_skipped.fetch_add(1, std::memory_order_relaxed);
    return false;
  };

  // The completion keeps its own copy of the request list: `live` is still needed below
  // if the worker pool refuses the job.
  job.completion = [live, metrics, dispatch_at](InferResult result) {
    std::int64_t expected_rows = 0;
    for (const auto& req : live) expected_rows += req->rows;
    const std::vector<std::int64_t> expected_shape = {expected_rows, kOutPerRow};
    const bool shape_ok = result.ok && result.output_shape == expected_shape &&
                          result.output.size() ==
                              static_cast<std::size_t>(expected_rows * kOutPerRow);

    const auto now = Clock::now();
    std::size_t offset = 0;
    for (const auto& req : live) {
      const std::size_t n = static_cast<std::size_t>(req->rows * kOutPerRow);
      const std::size_t begin = offset;
      offset += n;

      // Lifecycle outcomes take precedence over the batch result: a request that was
      // cancelled or expired is reported as such whether or not ORT ran.
      if (req->cancelled.load(std::memory_order_acquire)) {
        finish_failed(*metrics, req, TerminalKind::Cancelled,
                      result.skipped ? "cancelled before inference" : "cancelled during inference");
        continue;
      }
      if (now >= req->deadline) {
        finish_failed(*metrics, req, TerminalKind::Expired,
                      result.skipped ? "deadline exceeded before inference"
                                     : "deadline exceeded during inference");
        continue;
      }
      if (!result.ok) {
        finish_failed(*metrics, req, TerminalKind::Error,
                      result.skipped ? "inference skipped" : result.error);
        continue;
      }
      if (!shape_ok) {
        finish_failed(*metrics, req, TerminalKind::Error, "batch output shape mismatch");
        continue;
      }
      if (!req->try_finish()) continue;

      InferResult one;
      one.ok = true;
      one.output.assign(result.output.begin() + static_cast<std::ptrdiff_t>(begin),
                        result.output.begin() + static_cast<std::ptrdiff_t>(begin + n));
      one.output_shape = {req->rows, kOutPerRow};
      one.infer_ms = result.infer_ms;
      one.queue_wait_ms =
          std::chrono::duration<double, std::milli>(dispatch_at - req->enqueued_at).count();
      if (req->completion) req->completion(std::move(one), TerminalKind::Success);
    }
  };

  if (!workers_.enqueue(std::move(job))) {
    for (auto& req : live) {
      fail_request(req, TerminalKind::Rejected, "worker queue full");
    }
  }
}

void BatchScheduler::run() {
  for (;;) {
    std::vector<std::shared_ptr<ClientRequest>> batch;
    {
      std::unique_lock lock(mu_);
      for (;;) {
        if (stopping_ && queue_.empty()) return;

        // Fail every cancelled/expired request anywhere in the queue, not only at the
        // head, and find the earliest remaining deadline.
        const auto now = Clock::now();
        std::vector<std::pair<std::shared_ptr<ClientRequest>, TerminalKind>> dead;
        auto earliest_deadline = Clock::time_point::max();
        for (auto it = queue_.begin(); it != queue_.end();) {
          const auto& req = *it;
          if (req->cancelled.load(std::memory_order_acquire)) {
            dead.emplace_back(std::move(*it), TerminalKind::Cancelled);
            it = queue_.erase(it);
          } else if (now >= req->deadline) {
            dead.emplace_back(std::move(*it), TerminalKind::Expired);
            it = queue_.erase(it);
          } else {
            earliest_deadline = std::min(earliest_deadline, req->deadline);
            ++it;
          }
        }
        if (!dead.empty()) {
          lock.unlock();
          for (auto& [req, kind] : dead) {
            fail_request(req, kind,
                         kind == TerminalKind::Cancelled ? "cancelled while queued"
                                                         : "deadline exceeded while queued");
          }
          lock.lock();
          continue;
        }

        if (queue_.empty()) {
          cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
          continue;
        }

        const bool batching = cfg_.enable_batching;
        const std::size_t max_b = std::max<std::size_t>(1, cfg_.max_batch_size);
        auto& head = queue_.front();

        // Multi-row requests are never mixed; dispatch alone.
        if (!batching || head->rows != 1 || max_b == 1) {
          batch.push_back(std::move(queue_.front()));
          queue_.pop_front();
          break;
        }

        const auto window_end =
            head->enqueued_at + std::chrono::milliseconds(cfg_.max_batch_wait_ms);
        const bool full = queue_.size() >= max_b;

        if (!full && now < window_end && !stopping_) {
          // Wake for whichever comes first: the batch window closing, or the earliest
          // queued deadline (so an expired request is answered at its deadline rather
          // than at the end of the window). Spurious or early wakeups simply re-run the
          // loop above.
          const auto wake_at = std::min(window_end, earliest_deadline);
          const auto size_before = queue_.size();
          cv_.wait_until(lock, wake_at, [&] {
            return stopping_ || queue_.size() >= max_b || queue_.size() != size_before;
          });
          continue;
        }

        // Assemble compatible N=1 requests up to max_b.
        while (!queue_.empty() && batch.size() < max_b) {
          if (queue_.front()->rows != 1) break;
          batch.push_back(std::move(queue_.front()));
          queue_.pop_front();
        }
        break;
      }
    }

    if (!batch.empty()) dispatch_batch(std::move(batch));
  }
}

}  // namespace batchflow
