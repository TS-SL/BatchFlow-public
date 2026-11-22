#include "batchflow/admission.hpp"
#include "batchflow/batch_scheduler.hpp"
#include "batchflow/config.hpp"
#include "batchflow/http_server.hpp"
#include "batchflow/metrics.hpp"
#include "batchflow/version.hpp"
#include "batchflow/worker_pool.hpp"

#include <boost/asio/io_context.hpp>
#include <boost/asio/signal_set.hpp>
#include <boost/asio/steady_timer.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

batchflow::ServerConfig parse_args(int argc, char** argv) {
  batchflow::ServerConfig cfg;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&](std::string& out) {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      out = argv[++i];
    };
    auto next_u16 = [&](std::uint16_t& out) {
      std::string s;
      next(s);
      out = static_cast<std::uint16_t>(std::stoi(s));
    };
    auto next_sz = [&](std::size_t& out) {
      std::string s;
      next(s);
      out = static_cast<std::size_t>(std::stoull(s));
    };
    auto next_i64 = [&](std::int64_t& out) {
      std::string s;
      next(s);
      out = static_cast<std::int64_t>(std::stoll(s));
    };
    if (a == "--host") next(cfg.bind_host);
    else if (a == "--port") next_u16(cfg.bind_port);
    else if (a == "--model") next(cfg.model_path);
    else if (a == "--max-body") next_sz(cfg.max_request_bytes);
    else if (a == "--max-pending") next_sz(cfg.max_pending_requests);
    else if (a == "--max-pending-bytes") next_sz(cfg.max_pending_input_bytes);
    else if (a == "--workers") next_sz(cfg.worker_threads);
    else if (a == "--ort-intra-threads") {
      std::string s;
      next(s);
      cfg.ort_intra_op_threads = std::stoi(s);
    } else if (a == "--ort-inter-threads") {
      std::string s;
      next(s);
      cfg.ort_inter_op_threads = std::stoi(s);
    } else if (a == "--max-batch") next_sz(cfg.max_batch_size);
    else if (a == "--batch-wait-ms") next_i64(cfg.max_batch_wait_ms);
    else if (a == "--no-batching") cfg.enable_batching = false;
    else if (a == "--allow-non-local-bind") cfg.allow_non_local_bind = true;
    else if (a == "--help") {
      std::cout << "BatchFlow server " << batchflow::kVersionString << "\n"
                << "  --host 127.0.0.1 --port 8080 --model PATH\n"
                << "  --max-body N --max-pending N --max-pending-bytes N --workers N\n"
                << "  --ort-intra-threads N --ort-inter-threads N\n"
                << "  --max-batch N --batch-wait-ms N --no-batching\n"
                << "  --allow-non-local-bind   (required for non-localhost binds)\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg: " + a);
    }
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto cfg = parse_args(argc, argv);
    if (!cfg.allow_non_local_bind && cfg.bind_host != "127.0.0.1" &&
        cfg.bind_host != "localhost" && cfg.bind_host != "::1") {
      std::cerr << "Refusing non-local bind host '" << cfg.bind_host
                << "'. Pass --allow-non-local-bind for containers (and publish to 127.0.0.1 only).\n";
      return 2;
    }

    std::cout << "BatchFlow " << batchflow::kVersionString << " listening on http://"
              << cfg.bind_host << ":" << cfg.bind_port << " model=" << cfg.model_path
              << " batching=" << (cfg.enable_batching ? "on" : "off")
              << " max_batch=" << cfg.max_batch_size
              << " batch_wait_ms=" << cfg.max_batch_wait_ms << "\n";

    boost::asio::io_context ioc{1};
    auto& metrics = batchflow::global_metrics();
    batchflow::AdmissionController admission(cfg, metrics);
    batchflow::SessionOptions ort_opts;
    ort_opts.intra_op_threads = cfg.ort_intra_op_threads;
    ort_opts.inter_op_threads = cfg.ort_inter_op_threads;
    batchflow::WorkerPool workers(cfg.model_path, cfg.worker_threads, cfg.max_pending_requests,
                                  ort_opts);
    batchflow::BatchScheduler scheduler(cfg, workers, metrics);

    // Allow up to 30 s for session creation on a slow or busy host; a load failure
    // (missing file, incompatible signature) ends the wait immediately.
    for (int i = 0; i < 1500 && !workers.model_ready() && workers.load_error().empty(); ++i) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    admission.set_ready(workers.model_ready());
    if (!workers.model_ready()) {
      const auto why = workers.load_error();
      std::cerr << "Model not ready: " << (why.empty() ? "timed out loading model" : why)
                << "\n";
      scheduler.shutdown();
      return 2;
    }

    batchflow::HttpServer server(cfg, ioc, admission, workers, scheduler);
    server.start();

    boost::asio::signal_set signals(ioc, SIGINT);
#ifdef SIGTERM
    signals.add(SIGTERM);
#endif
#ifdef SIGBREAK
    signals.add(SIGBREAK);  // Windows Ctrl+Break, deliverable to a separate process group
#endif
    boost::asio::steady_timer grace(ioc);
    // Shutdown contract: stop accepting connections and mark the server not ready (new
    // /infer requests on open connections get 503), dispatch already-queued work without
    // waiting for the batch window, finish in-flight inference, then give the event loop
    // a short grace period to write those responses before exiting.
    signals.async_wait([&](boost::system::error_code ec, int) {
      if (ec) return;
      std::cout << "Shutting down..." << std::endl;
      admission.set_ready(false);
      server.stop();
      scheduler.shutdown();
      workers.shutdown();  // drains queued jobs; their completions are posted to ioc
      grace.expires_after(std::chrono::seconds(1));
      grace.async_wait([&](boost::system::error_code) { ioc.stop(); });
    });

    ioc.run();
    scheduler.shutdown();
    workers.shutdown();
    std::cout << "Bye. success=" << metrics.requests_success.load()
              << " rejected=" << metrics.requests_rejected.load()
              << " errors=" << metrics.requests_error.load()
              << " expired=" << metrics.requests_expired.load()
              << " cancelled=" << metrics.requests_cancelled.load()
              << " batches=" << metrics.batches_formed.load() << "\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "fatal: " << ex.what() << "\n";
    return 1;
  }
}
