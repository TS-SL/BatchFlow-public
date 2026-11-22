#pragma once

#include "batchflow/config.hpp"
#include "batchflow/metrics.hpp"
#include "batchflow/request.hpp"
#include "batchflow/worker_pool.hpp"

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <thread>

namespace batchflow {

// FIFO queue + dynamic batching (max batch size / max wait). Owns batch assembly only;
// ORT execution stays in WorkerPool.
class BatchScheduler {
 public:
  BatchScheduler(ServerConfig cfg, WorkerPool& workers, Metrics& metrics);
  ~BatchScheduler();

  BatchScheduler(const BatchScheduler&) = delete;
  BatchScheduler& operator=(const BatchScheduler&) = delete;

  // Returns false if the bounded queue is full.
  bool submit(std::shared_ptr<ClientRequest> req);
  void shutdown();

  [[nodiscard]] std::size_t queued() const;

 private:
  void run();
  void dispatch_batch(std::vector<std::shared_ptr<ClientRequest>> batch);
  void fail_request(const std::shared_ptr<ClientRequest>& req, TerminalKind kind,
                    std::string message);

  ServerConfig cfg_;
  WorkerPool& workers_;
  Metrics& metrics_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<std::shared_ptr<ClientRequest>> queue_;
  bool stopping_{false};
  std::thread thread_;
};

}  // namespace batchflow
