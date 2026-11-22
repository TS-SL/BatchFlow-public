#pragma once

#include "batchflow/admission.hpp"
#include "batchflow/batch_scheduler.hpp"
#include "batchflow/config.hpp"
#include "batchflow/worker_pool.hpp"

#include <boost/asio/io_context.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <memory>
#include <string>

namespace batchflow {

class HttpServer {
 public:
  HttpServer(ServerConfig cfg, boost::asio::io_context& ioc, AdmissionController& admission,
             WorkerPool& workers, BatchScheduler& scheduler);
  void start();
  void stop();

 private:
  void do_accept();

  ServerConfig cfg_;
  boost::asio::io_context& ioc_;
  boost::asio::ip::tcp::acceptor acceptor_;
  AdmissionController& admission_;
  WorkerPool& workers_;
  BatchScheduler& scheduler_;
  bool stopping_{false};
};

}  // namespace batchflow
