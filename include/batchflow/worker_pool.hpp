#pragma once

#include "batchflow/onnx_session.hpp"

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace batchflow {

struct InferJob {
  std::vector<float> input;
  std::int64_t batch_size{1};
  // Optional pre-run check, evaluated on the worker thread immediately before inference.
  // Returning false skips the ORT call; completion then receives a result with
  // ok=false and skipped=true.
  std::function<bool()> still_needed;
  std::function<void(InferResult)> completion;  // invoked on worker thread
};

// Bounded worker pool. Inference runs here — never on the Asio event loop.
class WorkerPool {
 public:
  WorkerPool(std::string model_path, std::size_t threads, std::size_t max_queue,
             SessionOptions ort_opts = {});
  ~WorkerPool();

  WorkerPool(const WorkerPool&) = delete;
  WorkerPool& operator=(const WorkerPool&) = delete;

  [[nodiscard]] bool model_ready() const;
  // Reason the model failed to load (empty if it loaded or is still loading).
  [[nodiscard]] std::string load_error() const;
  [[nodiscard]] std::size_t queued_jobs() const;
  [[nodiscard]] std::size_t worker_count() const { return worker_count_; }
  // Returns false if the queue is full or the pool is stopping. The job and its
  // completion are then destroyed without being invoked, so the caller must fail the work.
  bool enqueue(InferJob job);
  void shutdown();

 private:
  void worker_main();

  std::string model_path_;
  SessionOptions ort_opts_;
  std::size_t max_queue_;
  std::size_t worker_count_{0};
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::queue<InferJob> queue_;
  bool stopping_{false};
  bool model_ready_{false};
  std::string load_error_;
  std::vector<std::thread> workers_;
};

}  // namespace batchflow
