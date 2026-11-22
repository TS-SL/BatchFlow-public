#include "batchflow/worker_pool.hpp"

#include <chrono>
#include <utility>

namespace batchflow {

WorkerPool::WorkerPool(std::string model_path, std::size_t threads, std::size_t max_queue,
                       SessionOptions ort_opts)
    : model_path_(std::move(model_path)),
      ort_opts_(ort_opts),
      max_queue_(max_queue),
      worker_count_(threads == 0 ? 1 : threads) {
  workers_.reserve(worker_count_);
  for (std::size_t i = 0; i < worker_count_; ++i) {
    workers_.emplace_back([this] { worker_main(); });
  }
}

WorkerPool::~WorkerPool() { shutdown(); }

bool WorkerPool::model_ready() const {
  std::lock_guard lock(mu_);
  return model_ready_;
}

std::string WorkerPool::load_error() const {
  std::lock_guard lock(mu_);
  return load_error_;
}

std::size_t WorkerPool::queued_jobs() const {
  std::lock_guard lock(mu_);
  return queue_.size();
}

bool WorkerPool::enqueue(InferJob job) {
  {
    std::lock_guard lock(mu_);
    if (stopping_) return false;
    if (queue_.size() >= max_queue_) return false;
    queue_.push(std::move(job));
  }
  cv_.notify_one();
  return true;
}

void WorkerPool::shutdown() {
  {
    std::lock_guard lock(mu_);
    if (stopping_) return;
    stopping_ = true;
  }
  cv_.notify_all();
  for (auto& t : workers_) {
    if (t.joinable()) t.join();
  }
  workers_.clear();
}

void WorkerPool::worker_main() {
  try {
    OnnxSession session(model_path_, ort_opts_);
    {
      std::lock_guard lock(mu_);
      model_ready_ = session.available();
      if (!session.available()) load_error_ = session.load_error();
    }
    if (!session.available()) {
      return;
    }

    for (;;) {
      InferJob job;
      {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [&] { return stopping_ || !queue_.empty(); });
        if (stopping_ && queue_.empty()) return;
        job = std::move(queue_.front());
        queue_.pop();
      }
      InferResult result;
      if (job.still_needed && !job.still_needed()) {
        result.ok = false;
        result.skipped = true;
        result.error = "inference skipped: no live request in batch";
      } else {
        const auto t0 = std::chrono::steady_clock::now();
        result = session.run(job.input, job.batch_size);
        const auto t1 = std::chrono::steady_clock::now();
        result.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      }
      if (job.completion) {
        job.completion(std::move(result));
      }
    }
  } catch (const std::exception& ex) {
    std::lock_guard lock(mu_);
    model_ready_ = false;
    load_error_ = ex.what();
  } catch (...) {
    std::lock_guard lock(mu_);
    model_ready_ = false;
    load_error_ = "unknown error in worker";
  }
}

}  // namespace batchflow
