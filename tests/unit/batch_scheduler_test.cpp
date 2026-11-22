#include "batchflow/batch_scheduler.hpp"
#include "batchflow/config.hpp"
#include "batchflow/metrics.hpp"
#include "batchflow/worker_pool.hpp"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

int main() {
  using namespace batchflow;
  ServerConfig cfg;
  cfg.model_path = "models/tiny_deterministic.onnx";
  cfg.enable_batching = true;
  cfg.max_batch_size = 4;
  cfg.max_batch_wait_ms = 50;
  cfg.max_pending_requests = 32;
  cfg.worker_threads = 1;

  Metrics metrics;
  SessionOptions ort_opts;
  WorkerPool workers(cfg.model_path, cfg.worker_threads, cfg.max_pending_requests, ort_opts);
  for (int i = 0; i < 500 && !workers.model_ready(); ++i) {  // up to 10 s
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!workers.model_ready()) {
    std::cerr << "FAILED: model not ready\n";
    return 2;
  }

  BatchScheduler scheduler(cfg, workers, metrics);

  constexpr int kN = 4;
  std::mutex mu;
  std::condition_variable cv;
  int done = 0;
  std::vector<InferResult> results(kN);
  std::vector<TerminalKind> kinds(kN);

  // Distinct inputs so mapping mistakes are detectable.
  for (int i = 0; i < kN; ++i) {
    auto req = std::make_shared<ClientRequest>();
    req->request_id = "r" + std::to_string(i);
    req->rows = 1;
    req->input.assign(3 * 8 * 8, static_cast<float>(i + 1));
    req->input_bytes = req->input.size() * sizeof(float);
    req->deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    req->completion = [&, i](InferResult r, TerminalKind k) {
      {
        std::lock_guard lock(mu);
        results[i] = std::move(r);
        kinds[i] = k;
        ++done;
      }
      cv.notify_one();
    };
    if (!scheduler.submit(req)) {
      std::cerr << "FAILED: submit\n";
      return 1;
    }
  }

  {
    std::unique_lock lock(mu);
    if (!cv.wait_for(lock, std::chrono::seconds(5), [&] { return done == kN; })) {
      std::cerr << "FAILED: timeout waiting for completions done=" << done << "\n";
      return 1;
    }
  }

  for (int i = 0; i < kN; ++i) {
    if (kinds[i] != TerminalKind::Success || !results[i].ok) {
      std::cerr << "FAILED: request " << i << " kind/error\n";
      return 1;
    }
    if (results[i].output.size() != 4) {
      std::cerr << "FAILED: bad output size for " << i << "\n";
      return 1;
    }
  }

  // Outputs for constant-filled inputs should differ across i (linear model).
  if (results[0].output[0] == results[1].output[0] && results[1].output[0] == results[2].output[0]) {
    std::cerr << "FAILED: outputs look identical — possible mapping bug\n";
    return 1;
  }

  if (metrics.batches_formed.load() < 1) {
    std::cerr << "FAILED: expected at least one batch\n";
    return 1;
  }

  scheduler.shutdown();
  workers.shutdown();
  std::cout << "batch_scheduler_ok batches=" << metrics.batches_formed.load()
            << " mean_batch="
            << (static_cast<double>(metrics.batch_size_sum.load()) /
                static_cast<double>(metrics.batches_formed.load()))
            << "\n";
  return 0;
}
