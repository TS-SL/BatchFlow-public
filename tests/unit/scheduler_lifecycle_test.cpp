// Request-lifecycle regression tests for BatchScheduler + WorkerPool.
//
// Each case is selected by name on the command line so CTest reports them separately:
//   batchflow_lifecycle_tests <case>
// Run from the repository root (models/ and tests/fixtures/ are opened by relative path).

#include "batchflow/admission.hpp"
#include "batchflow/batch_scheduler.hpp"
#include "batchflow/config.hpp"
#include "batchflow/metrics.hpp"
#include "batchflow/onnx_session.hpp"
#include "batchflow/worker_pool.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace batchflow;
using Clock = std::chrono::steady_clock;

constexpr std::size_t kElems = 3 * 8 * 8;
const char* kModel = "models/tiny_deterministic.onnx";

int g_failures = 0;
#define CHECK(cond, msg)                                                          \
  do {                                                                            \
    if (!(cond)) {                                                                \
      std::cerr << "CHECK FAILED line " << __LINE__ << ": " << msg \
                << "\n";                                                          \
      ++g_failures;                                                               \
    }                                                                             \
  } while (0)

// Records every completion so tests can assert "exactly one terminal outcome".
struct Recorder {
  std::mutex mu;
  std::condition_variable cv;
  struct Entry {
    int calls = 0;
    TerminalKind kind = TerminalKind::Error;
    InferResult result;
    Clock::time_point at{};
  };
  std::map<std::string, Entry> entries;
  int total_calls = 0;

  std::function<void(InferResult, TerminalKind)> make(const std::string& id,
                                                      std::function<void()> on_done = {}) {
    return [this, id, on_done](InferResult r, TerminalKind k) {
      {
        std::lock_guard lock(mu);
        auto& e = entries[id];
        ++e.calls;
        e.kind = k;
        e.result = std::move(r);
        e.at = Clock::now();
        ++total_calls;
      }
      if (on_done) on_done();
      cv.notify_all();
    };
  }

  bool wait_for_calls(int n, std::chrono::milliseconds timeout) {
    std::unique_lock lock(mu);
    return cv.wait_for(lock, timeout, [&] { return total_calls >= n; });
  }
};

std::shared_ptr<ClientRequest> make_request(const std::string& id, std::vector<float> input,
                                            std::int64_t rows,
                                            std::chrono::milliseconds deadline) {
  auto req = std::make_shared<ClientRequest>();
  req->request_id = id;
  req->rows = rows;
  req->input = std::move(input);
  req->input_bytes = req->input.size() * sizeof(float);
  req->deadline = Clock::now() + deadline;
  return req;
}

std::vector<float> random_input(std::mt19937& rng, std::int64_t rows) {
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> v(static_cast<std::size_t>(rows) * kElems);
  for (auto& x : v) x = dist(rng);
  return v;
}

bool wait_ready(WorkerPool& pool) {
  for (int i = 0; i < 500 && !pool.model_ready(); ++i) {  // up to 10 s
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  return pool.model_ready();
}

const char* kind_name(TerminalKind k) {
  switch (k) {
    case TerminalKind::Success: return "Success";
    case TerminalKind::Error: return "Error";
    case TerminalKind::Expired: return "Expired";
    case TerminalKind::Cancelled: return "Cancelled";
    case TerminalKind::Rejected: return "Rejected";
    case TerminalKind::ShuttingDown: return "ShuttingDown";
  }
  return "?";
}

// --- Cases -----------------------------------------------------------------------------

// A worker queue of capacity 0 refuses every job. Each request handed to the scheduler
// must still receive exactly one terminal completion (Rejected), and the admission
// reservation taken for it must be released exactly once.
//
// The shipping CLI sizes the worker queue (in jobs) equal to the admission cap (in
// requests), so admission normally refuses first; this case configures the components
// directly to force the refusal branch.
void case_enqueue_refusal() {
  ServerConfig cfg;
  cfg.enable_batching = true;
  cfg.max_batch_size = 4;
  cfg.max_batch_wait_ms = 1;
  cfg.max_pending_requests = 16;
  Metrics metrics;
  AdmissionController admission(cfg, metrics);
  admission.set_ready(true);

  WorkerPool workers(kModel, 1, /*max_queue=*/0);
  CHECK(wait_ready(workers), "model not ready");
  BatchScheduler scheduler(cfg, workers, metrics);

  Recorder rec;
  std::mt19937 rng(7);
  constexpr int kN = 3;
  for (int i = 0; i < kN; ++i) {
    auto req = make_request("r" + std::to_string(i), random_input(rng, 1), 1,
                            std::chrono::seconds(5));
    const auto bytes = req->input_bytes;
    CHECK(admission.try_admit(bytes) == AdmitResult::Ok, "admit");
    req->completion = rec.make(req->request_id, [&admission, bytes] { admission.release(bytes); });
    CHECK(scheduler.submit(req), "submit");
  }

  const bool all = rec.wait_for_calls(kN, std::chrono::seconds(2));
  CHECK(all, "expected " << kN << " completions, got " << rec.total_calls);
  // Give a duplicate completion a chance to show up before asserting exactly-once.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  {
    std::lock_guard lock(rec.mu);
    CHECK(rec.total_calls == kN, "total completions " << rec.total_calls);
    for (auto& [id, e] : rec.entries) {
      CHECK(e.calls == 1, id << " completed " << e.calls << " times");
      CHECK(e.kind == TerminalKind::Rejected, id << " kind " << kind_name(e.kind));
    }
  }
  CHECK(admission.active_count() == 0, "admission still holds " << admission.active_count());
  CHECK(admission.active_bytes() == 0, "admission still holds bytes " << admission.active_bytes());
  CHECK(metrics.requests_rejected.load() == kN, "rejected metric " << metrics.requests_rejected.load());
  scheduler.shutdown();
  workers.shutdown();
}

// An isolated request with a 10 ms deadline inside a 100 ms batch window must expire at
// its deadline, not at the end of the window. The 60 ms bound leaves 50 ms of
// operating-system scheduling tolerance while still failing the old behaviour (~100 ms).
void case_deadline_before_window() {
  ServerConfig cfg;
  cfg.enable_batching = true;
  cfg.max_batch_size = 8;
  cfg.max_batch_wait_ms = 100;
  Metrics metrics;
  WorkerPool workers(kModel, 1, 16);
  CHECK(wait_ready(workers), "model not ready");
  BatchScheduler scheduler(cfg, workers, metrics);

  Recorder rec;
  std::mt19937 rng(11);
  auto req = make_request("iso", random_input(rng, 1), 1, std::chrono::milliseconds(10));
  const auto t0 = Clock::now();
  req->completion = rec.make("iso");
  CHECK(scheduler.submit(req), "submit");
  CHECK(rec.wait_for_calls(1, std::chrono::seconds(2)), "no completion");
  std::lock_guard lock(rec.mu);
  const auto& e = rec.entries["iso"];
  const double ms = std::chrono::duration<double, std::milli>(e.at - t0).count();
  std::cout << "deadline_before_window: kind=" << kind_name(e.kind) << " after " << ms << " ms\n";
  CHECK(e.kind == TerminalKind::Expired, "kind " << kind_name(e.kind));
  CHECK(ms >= 9.0, "expired before its deadline: " << ms << " ms");
  CHECK(ms < 60.0, "expiry waited for the batch window: " << ms << " ms");
}

// A request with a short deadline queued behind one with a long deadline must still be
// expired at its own deadline (the earliest deadline in the queue, not only the head).
void case_deadline_not_at_head() {
  ServerConfig cfg;
  cfg.enable_batching = true;
  cfg.max_batch_size = 8;
  cfg.max_batch_wait_ms = 100;
  Metrics metrics;
  WorkerPool workers(kModel, 1, 16);
  CHECK(wait_ready(workers), "model not ready");
  BatchScheduler scheduler(cfg, workers, metrics);

  Recorder rec;
  std::mt19937 rng(13);
  auto head = make_request("head", random_input(rng, 1), 1, std::chrono::seconds(5));
  auto tail = make_request("tail", random_input(rng, 1), 1, std::chrono::milliseconds(10));
  head->completion = rec.make("head");
  tail->completion = rec.make("tail");
  const auto t0 = Clock::now();
  CHECK(scheduler.submit(head), "submit head");
  CHECK(scheduler.submit(tail), "submit tail");
  CHECK(rec.wait_for_calls(2, std::chrono::seconds(2)), "missing completions");
  std::lock_guard lock(rec.mu);
  const auto& t = rec.entries["tail"];
  const double ms = std::chrono::duration<double, std::milli>(t.at - t0).count();
  std::cout << "deadline_not_at_head: tail kind=" << kind_name(t.kind) << " after " << ms << " ms\n";
  CHECK(t.kind == TerminalKind::Expired, "tail kind " << kind_name(t.kind));
  CHECK(ms < 60.0, "tail expiry waited for the batch window: " << ms << " ms");
  CHECK(rec.entries["head"].kind == TerminalKind::Success, "head kind "
                                                               << kind_name(rec.entries["head"].kind));
}

// Each request in a batch receives exactly the rows computed for its own input. The
// reference is an independent single-request run of the same ONNX session. A 2-row
// request is mixed into the stream to cover the "multi-row dispatches alone" path.
void case_batch_output_mapping() {
  ServerConfig cfg;
  cfg.enable_batching = true;
  cfg.max_batch_size = 8;
  cfg.max_batch_wait_ms = 50;
  Metrics metrics;

  OnnxSession ref_session(kModel);
  CHECK(ref_session.available(), "reference session unavailable");

  WorkerPool workers(kModel, 1, 16);
  CHECK(wait_ready(workers), "model not ready");
  BatchScheduler scheduler(cfg, workers, metrics);

  Recorder rec;
  std::mt19937 rng(42);
  struct Expect {
    std::string id;
    std::int64_t rows;
    std::vector<float> out;
  };
  std::vector<Expect> expects;
  std::vector<std::shared_ptr<ClientRequest>> reqs;
  const std::vector<std::int64_t> rows = {1, 1, 1, 2, 1, 1, 1, 1, 1};
  for (std::size_t i = 0; i < rows.size(); ++i) {
    auto in = random_input(rng, rows[i]);
    auto ref = ref_session.run(in, rows[i]);
    CHECK(ref.ok, "reference run failed: " << ref.error);
    const std::string id = "m" + std::to_string(i);
    expects.push_back({id, rows[i], ref.output});
    auto req = make_request(id, std::move(in), rows[i], std::chrono::seconds(5));
    req->completion = rec.make(id);
    reqs.push_back(req);
  }
  for (auto& r : reqs) CHECK(scheduler.submit(r), "submit");
  CHECK(rec.wait_for_calls(static_cast<int>(reqs.size()), std::chrono::seconds(5)),
        "missing completions");

  std::lock_guard lock(rec.mu);
  for (const auto& ex : expects) {
    const auto& e = rec.entries[ex.id];
    CHECK(e.calls == 1, ex.id << " calls " << e.calls);
    CHECK(e.kind == TerminalKind::Success, ex.id << " kind " << kind_name(e.kind));
    const std::vector<std::int64_t> want_shape = {ex.rows, 4};
    CHECK(e.result.output_shape == want_shape, ex.id << " output shape");
    CHECK(e.result.output.size() == ex.out.size(), ex.id << " output size");
    for (std::size_t j = 0; j < ex.out.size() && j < e.result.output.size(); ++j) {
      const float tol = 1e-6f + 1e-5f * std::fabs(ex.out[j]);
      CHECK(std::fabs(e.result.output[j] - ex.out[j]) <= tol, ex.id << " element " << j);
    }
  }
  CHECK(metrics.batches_formed.load() < rows.size(), "no multi-request batch was formed");
}

// A model whose runtime output is [2N,2] has the right element count but the wrong
// shape. Requests must fail with Error instead of receiving mis-sliced rows.
void case_wrong_output_shape() {
  ServerConfig cfg;
  cfg.enable_batching = true;
  cfg.max_batch_size = 4;
  cfg.max_batch_wait_ms = 20;
  Metrics metrics;
  WorkerPool workers("tests/fixtures/wrong_output_shape.onnx", 1, 16);
  CHECK(wait_ready(workers), "fixture model not ready");
  BatchScheduler scheduler(cfg, workers, metrics);

  Recorder rec;
  std::mt19937 rng(5);
  for (int i = 0; i < 3; ++i) {
    auto req = make_request("w" + std::to_string(i), random_input(rng, 1), 1,
                            std::chrono::seconds(5));
    req->completion = rec.make(req->request_id);
    CHECK(scheduler.submit(req), "submit");
  }
  CHECK(rec.wait_for_calls(3, std::chrono::seconds(5)), "missing completions");
  std::lock_guard lock(rec.mu);
  for (auto& [id, e] : rec.entries) {
    CHECK(e.kind == TerminalKind::Error, id << " kind " << kind_name(e.kind));
  }
}

// The load-time signature check refuses a model whose declared input is not [N,3,8,8].
void case_wrong_input_signature() {
  OnnxSession s("tests/fixtures/wrong_input_signature.onnx");
  CHECK(!s.available(), "session accepted a model with input [N,3,4,4]");
  CHECK(!s.load_error().empty(), "no load error reported");
  std::cout << "wrong_input_signature: " << s.load_error() << "\n";
  OnnxSession good(kModel);
  CHECK(good.available(), "reference model rejected: " << good.load_error());
}

// Shutdown with work still inside the batch window: every accepted request gets exactly
// one terminal completion. Requests already queued are dispatched (not dropped), so they
// complete with Success.
void case_shutdown_with_queued_work() {
  ServerConfig cfg;
  cfg.enable_batching = true;
  cfg.max_batch_size = 8;
  cfg.max_batch_wait_ms = 2000;  // far longer than the test; shutdown must not wait for it
  Metrics metrics;
  WorkerPool workers(kModel, 1, 16);
  CHECK(wait_ready(workers), "model not ready");
  BatchScheduler scheduler(cfg, workers, metrics);

  Recorder rec;
  std::mt19937 rng(3);
  constexpr int kN = 5;
  for (int i = 0; i < kN; ++i) {
    auto req = make_request("s" + std::to_string(i), random_input(rng, 1), 1,
                            std::chrono::seconds(10));
    req->completion = rec.make(req->request_id);
    CHECK(scheduler.submit(req), "submit");
  }
  const auto t0 = Clock::now();
  scheduler.shutdown();
  workers.shutdown();
  const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  std::cout << "shutdown_with_queued_work: shutdown took " << ms << " ms\n";
  CHECK(ms < 1000.0, "shutdown waited for the batch window");
  std::lock_guard lock(rec.mu);
  CHECK(rec.total_calls == kN, "completions " << rec.total_calls);
  for (auto& [id, e] : rec.entries) {
    CHECK(e.calls == 1, id << " calls " << e.calls);
    CHECK(e.kind == TerminalKind::Success, id << " kind " << kind_name(e.kind));
  }
  auto late = make_request("late", random_input(rng, 1), 1, std::chrono::seconds(1));
  CHECK(!scheduler.submit(late), "submit accepted after shutdown");
}

// A job whose requests all expired while it waited in the worker queue is not run.
// The first job's pre-run hook blocks the single worker until the second job's request
// has expired, which makes the ordering deterministic without sleeps in the worker.
void case_expired_in_worker_queue_skips_inference() {
  ServerConfig cfg;
  cfg.enable_batching = false;
  Metrics metrics;
  WorkerPool workers(kModel, 1, 16);
  CHECK(wait_ready(workers), "model not ready");

  std::mutex gate_mu;
  std::condition_variable gate_cv;
  bool gate_open = false;
  bool blocker_started = false;

  InferJob blocker;
  blocker.input.assign(kElems, 0.0f);
  blocker.batch_size = 1;
  blocker.still_needed = [&] {
    std::unique_lock lock(gate_mu);
    blocker_started = true;
    gate_cv.notify_all();
    gate_cv.wait(lock, [&] { return gate_open; });
    return true;
  };
  std::mutex done_mu;
  std::condition_variable done_cv;
  int done = 0;
  blocker.completion = [&](InferResult) {
    std::lock_guard lock(done_mu);
    ++done;
    done_cv.notify_all();
  };
  CHECK(workers.enqueue(std::move(blocker)), "enqueue blocker");
  {
    std::unique_lock lock(gate_mu);
    gate_cv.wait(lock, [&] { return blocker_started; });
  }

  BatchScheduler scheduler(cfg, workers, metrics);
  Recorder rec;
  std::mt19937 rng(9);
  auto req = make_request("q", random_input(rng, 1), 1, std::chrono::milliseconds(20));
  req->completion = rec.make("q");
  CHECK(scheduler.submit(req), "submit");
  // Wait until the scheduler has handed the job to the (blocked) worker queue.
  for (int i = 0; i < 200 && workers.queued_jobs() == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  CHECK(workers.queued_jobs() == 1, "job not in worker queue");
  std::this_thread::sleep_for(std::chrono::milliseconds(40));  // past the 20 ms deadline
  {
    std::lock_guard lock(gate_mu);
    gate_open = true;
  }
  gate_cv.notify_all();

  CHECK(rec.wait_for_calls(1, std::chrono::seconds(2)), "no completion");
  std::lock_guard lock(rec.mu);
  const auto& e = rec.entries["q"];
  CHECK(e.calls == 1, "calls " << e.calls);
  CHECK(e.kind == TerminalKind::Expired, "kind " << kind_name(e.kind));
  CHECK(e.result.error == "deadline exceeded before inference", "message: " << e.result.error);
  CHECK(metrics.jobs_skipped.load() == 1, "jobs_skipped " << metrics.jobs_skipped.load());
}

}  // namespace

int main(int argc, char** argv) {
  const std::map<std::string, void (*)()> cases = {
      {"enqueue_refusal", case_enqueue_refusal},
      {"deadline_before_window", case_deadline_before_window},
      {"deadline_not_at_head", case_deadline_not_at_head},
      {"batch_output_mapping", case_batch_output_mapping},
      {"wrong_output_shape", case_wrong_output_shape},
      {"wrong_input_signature", case_wrong_input_signature},
      {"shutdown_with_queued_work", case_shutdown_with_queued_work},
      {"expired_in_worker_queue", case_expired_in_worker_queue_skips_inference},
  };
  if (argc != 2 || !cases.count(argv[1])) {
    std::cerr << "usage: batchflow_lifecycle_tests <case>\ncases:";
    for (auto& [name, fn] : cases) std::cerr << " " << name;
    std::cerr << "\n";
    return 2;
  }
  cases.at(argv[1])();
  if (g_failures) {
    std::cerr << argv[1] << ": " << g_failures << " check(s) failed\n";
    return 1;
  }
  std::cout << argv[1] << ": ok\n";
  return 0;
}
