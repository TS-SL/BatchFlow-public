#include "batchflow/http_server.hpp"

#include "batchflow/metrics.hpp"
#include "batchflow/request.hpp"
#include "batchflow/version.hpp"

#include <boost/asio/dispatch.hpp>
#include <boost/asio/strand.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>

#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;
using json = nlohmann::json;

namespace batchflow {
namespace {

std::string make_request_id() {
  using clock = std::chrono::steady_clock;
  static std::atomic<std::uint64_t> seq{0};
  const auto now = clock::now().time_since_epoch().count();
  std::ostringstream os;
  os << "bf-" << now << "-" << seq.fetch_add(1, std::memory_order_relaxed);
  return os.str();
}

http::response<http::string_body> make_json_response(http::status status, const json& body,
                                                     unsigned version, bool keep_alive) {
  http::response<http::string_body> res{status, version};
  res.set(http::field::server, std::string("BatchFlow/") + kVersionString);
  res.set(http::field::content_type, "application/json");
  res.keep_alive(keep_alive);
  res.body() = body.dump();
  res.prepare_payload();
  return res;
}

struct ParsedInfer {
  std::string request_id;
  std::int64_t deadline_ms{5000};
  std::vector<std::int64_t> shape;
  std::vector<float> data;
  std::string error;
  std::string error_code;
  http::status status{http::status::bad_request};
};

ParsedInfer parse_infer_body(std::string_view body, const ServerConfig& cfg) {
  ParsedInfer out;
  json j;
  try {
    j = json::parse(body);
  } catch (const json::exception&) {
    out.error_code = "malformed_json";
    out.error = "request body is not valid JSON";
    out.status = http::status::bad_request;
    return out;
  }

  if (j.contains("request_id") && j["request_id"].is_string()) {
    out.request_id = j["request_id"].get<std::string>();
  } else {
    out.request_id = make_request_id();
  }

  if (j.contains("deadline_ms")) {
    if (!j["deadline_ms"].is_number_integer()) {
      out.error_code = "bad_deadline";
      out.error = "deadline_ms must be an integer";
      return out;
    }
    out.deadline_ms = j["deadline_ms"].get<std::int64_t>();
    if (out.deadline_ms > cfg.max_deadline_ms) {
      out.error_code = "bad_deadline";
      out.error = "deadline_ms exceeds the server maximum";
      return out;
    }
  } else {
    out.deadline_ms = cfg.default_deadline_ms;
  }

  if (!j.contains("input") || !j["input"].is_object()) {
    out.error_code = "missing_input";
    out.error = "input object is required";
    return out;
  }
  const auto& input = j["input"];
  if (!input.contains("dtype") || !input["dtype"].is_string() ||
      input["dtype"].get<std::string>() != "float32") {
    out.error_code = "bad_dtype";
    out.error = "only float32 dtype is supported";
    return out;
  }
  if (!input.contains("shape") || !input["shape"].is_array()) {
    out.error_code = "bad_shape";
    out.error = "input.shape must be an array";
    return out;
  }
  for (const auto& d : input["shape"]) {
    if (!d.is_number_integer()) {
      out.error_code = "bad_shape";
      out.error = "shape dimensions must be integers";
      return out;
    }
    out.shape.push_back(d.get<std::int64_t>());
  }
  if (out.shape.size() != 4 || out.shape[1] != 3 || out.shape[2] != 8 || out.shape[3] != 8 ||
      out.shape[0] <= 0) {
    out.error_code = "shape_mismatch";
    out.error = "expected shape [N,3,8,8] with N>0";
    out.status = http::status::unprocessable_entity;
    return out;
  }

  if (!input.contains("data") || !input["data"].is_array()) {
    out.error_code = "bad_data";
    out.error = "input.data must be an array";
    return out;
  }
  std::int64_t expected = 1;
  for (auto d : out.shape) expected *= d;
  if (static_cast<std::int64_t>(input["data"].size()) != expected) {
    out.error_code = "length_mismatch";
    out.error = "data length must equal product(shape)";
    return out;
  }
  out.data.reserve(static_cast<std::size_t>(expected));
  for (const auto& v : input["data"]) {
    if (!v.is_number()) {
      out.error_code = "bad_data";
      out.error = "data elements must be numbers";
      return out;
    }
    out.data.push_back(v.get<float>());
  }
  return out;
}

class Session : public std::enable_shared_from_this<Session> {
 public:
  Session(tcp::socket socket, ServerConfig cfg, AdmissionController& admission, WorkerPool& workers,
          BatchScheduler& scheduler, net::io_context& ioc)
      : stream_(std::move(socket)),
        cfg_(std::move(cfg)),
        admission_(admission),
        workers_(workers),
        scheduler_(scheduler),
        ioc_(ioc) {}

  void run() {
    net::dispatch(stream_.get_executor(),
                  beast::bind_front_handler(&Session::do_read, shared_from_this()));
  }

 private:
  void do_read() {
    request_ = {};
    parser_.emplace();
    parser_->body_limit(cfg_.max_request_bytes);
    stream_.expires_after(std::chrono::seconds(30));
    http::async_read(stream_, buffer_, *parser_,
                     beast::bind_front_handler(&Session::on_read, shared_from_this()));
  }

  void on_read(beast::error_code ec, std::size_t) {
    if (ec == http::error::end_of_stream) return do_close();
    if (ec == beast::http::error::body_limit) {
      auto res = make_json_response(
          http::status::payload_too_large,
          json{{"request_id", make_request_id()},
               {"status", "payload_too_large"},
               {"error", {{"code", "body_limit"}, {"message", "request body exceeds limit"}}}},
          11, false);
      return write_response(std::move(res), false);
    }
    if (ec) return do_close();

    request_ = parser_->release();
    handle_request();
  }

  void handle_request() {
    const auto method = request_.method();
    const auto target = std::string(request_.target());
    const unsigned ver = request_.version();
    const bool keep_alive = request_.keep_alive();

    if (method == http::verb::get && (target == "/health" || target.rfind("/health?", 0) == 0)) {
      return write_response(make_json_response(http::status::ok, json{{"status", "ok"}}, ver, keep_alive),
                            keep_alive);
    }

    if (method == http::verb::get && (target == "/ready" || target.rfind("/ready?", 0) == 0)) {
      const bool ok = admission_.ready() && workers_.model_ready();
      auto status = ok ? http::status::ok : http::status::service_unavailable;
      return write_response(
          make_json_response(status, json{{"status", ok ? "ready" : "not_ready"}}, ver, keep_alive),
          keep_alive);
    }

    if (method == http::verb::get && (target == "/metrics" || target.rfind("/metrics?", 0) == 0)) {
      const auto body = global_metrics().to_json(scheduler_.queued(), workers_.worker_count());
      http::response<http::string_body> res{http::status::ok, ver};
      res.set(http::field::server, std::string("BatchFlow/") + kVersionString);
      res.set(http::field::content_type, "application/json");
      res.keep_alive(keep_alive);
      res.body() = body;
      res.prepare_payload();
      return write_response(std::move(res), keep_alive);
    }

    if (method == http::verb::post && (target == "/infer" || target.rfind("/infer?", 0) == 0)) {
      return handle_infer(ver, keep_alive);
    }

    write_response(make_json_response(
                       http::status::not_found,
                       json{{"request_id", make_request_id()},
                            {"status", "not_found"},
                            {"error", {{"code", "not_found"}, {"message", "unknown path"}}}},
                       ver, keep_alive),
                   keep_alive);
  }

  void handle_infer(unsigned ver, bool keep_alive) {
    if (!admission_.ready() || !workers_.model_ready()) {
      global_metrics().requests_rejected.fetch_add(1, std::memory_order_relaxed);
      return write_response(
          make_json_response(
              http::status::service_unavailable,
              json{{"request_id", make_request_id()},
                   {"status", "not_ready"},
                   {"error", {{"code", "not_ready"}, {"message", "server not ready"}}}},
              ver, keep_alive),
          keep_alive);
    }

    auto parsed = parse_infer_body(request_.body(), cfg_);
    if (!parsed.error.empty()) {
      global_metrics().requests_error.fetch_add(1, std::memory_order_relaxed);
      return write_response(
          make_json_response(parsed.status,
                             json{{"request_id", parsed.request_id},
                                  {"status", parsed.status == http::status::unprocessable_entity
                                                 ? "shape_mismatch"
                                                 : "bad_request"},
                                  {"error", {{"code", parsed.error_code}, {"message", parsed.error}}}},
                             ver, keep_alive),
          keep_alive);
    }

    const std::size_t bytes = parsed.data.size() * sizeof(float);
    const auto admit = admission_.try_admit(bytes);
    if (admit != AdmitResult::Ok) {
      const char* code = admit == AdmitResult::ByteLimit ? "byte_budget" : "queue_full";
      return write_response(
          make_json_response(
              http::status::too_many_requests,
              json{{"request_id", parsed.request_id},
                   {"status", "overloaded"},
                   {"error", {{"code", code}, {"message", "admission limits exceeded"}}}},
              ver, keep_alive),
          keep_alive);
    }

    // Deadline accounting starts here: after the body has been read and parsed and the
    // request admitted. Time spent receiving and parsing the body is not charged.
    const auto start = std::chrono::steady_clock::now();
    if (parsed.deadline_ms <= 0) {
      admission_.release(bytes);
      global_metrics().requests_expired.fetch_add(1, std::memory_order_relaxed);
      return write_response(
          make_json_response(
              http::status::request_timeout,
              json{{"request_id", parsed.request_id},
                   {"status", "deadline_exceeded"},
                   {"error", {{"code", "deadline"}, {"message", "deadline already expired"}}}},
              ver, keep_alive),
          keep_alive);
    }

    auto req = std::make_shared<ClientRequest>();
    req->request_id = parsed.request_id;
    req->input = std::move(parsed.data);
    req->rows = parsed.shape[0];
    req->input_bytes = bytes;
    req->deadline = start + std::chrono::milliseconds(parsed.deadline_ms);
    active_request_ = req;

    auto self = shared_from_this();
    const std::string request_id = req->request_id;
    req->completion = [self, bytes, start, ver, keep_alive, request_id](InferResult result,
                                                                        TerminalKind kind) {
      net::post(self->ioc_, [self, bytes, start, ver, keep_alive, request_id,
                             result = std::move(result), kind]() mutable {
        self->on_infer_done(request_id, bytes, start, ver, keep_alive, std::move(result), kind);
      });
    };

    if (!scheduler_.submit(req)) {
      admission_.release(bytes);
      active_request_.reset();
      req->try_finish();
      global_metrics().requests_rejected.fetch_add(1, std::memory_order_relaxed);
      return write_response(
          make_json_response(
              http::status::too_many_requests,
              json{{"request_id", request_id},
                   {"status", "overloaded"},
                   {"error", {{"code", "scheduler_queue_full"}, {"message", "scheduler queue full"}}}},
              ver, keep_alive),
          keep_alive);
    }
  }

  void on_infer_done(std::string request_id, std::size_t bytes,
                     std::chrono::steady_clock::time_point start, unsigned ver, bool keep_alive,
                     InferResult result, TerminalKind kind) {
    active_request_.reset();
    struct ReleaseGuard {
      AdmissionController& adm;
      std::size_t bytes;
      ~ReleaseGuard() { adm.release(bytes); }
    } release_guard{admission_, bytes};

    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - start)
                             .count();

    if (kind == TerminalKind::Expired) {
      return write_response(
          make_json_response(
              http::status::request_timeout,
              json{{"request_id", request_id},
                   {"status", "deadline_exceeded"},
                   {"error", {{"code", "deadline"}, {"message", result.error}}}},
              ver, keep_alive),
          keep_alive);
    }
    if (kind == TerminalKind::Cancelled) {
      return write_response(
          make_json_response(
              static_cast<http::status>(499),
              json{{"request_id", request_id},
                   {"status", "cancelled"},
                   {"error", {{"code", "cancelled"}, {"message", result.error}}}},
              ver, keep_alive),
          keep_alive);
    }
    if (kind == TerminalKind::Rejected) {
      return write_response(
          make_json_response(
              http::status::too_many_requests,
              json{{"request_id", request_id},
                   {"status", "overloaded"},
                   {"error", {{"code", "worker_queue_full"}, {"message", result.error}}}},
              ver, keep_alive),
          keep_alive);
    }
    if (kind == TerminalKind::ShuttingDown) {
      return write_response(
          make_json_response(
              http::status::service_unavailable,
              json{{"request_id", request_id},
                   {"status", "shutting_down"},
                   {"error", {{"code", "shutting_down"}, {"message", result.error}}}},
              ver, false),
          false);
    }
    if (kind != TerminalKind::Success || !result.ok) {
      return write_response(
          make_json_response(
              http::status::internal_server_error,
              json{{"request_id", request_id},
                   {"status", "inference_error"},
                   {"error", {{"code", "infer_failed"}, {"message", "inference failed"}}}},
              ver, keep_alive),
          keep_alive);
    }

    global_metrics().requests_success.fetch_add(1, std::memory_order_relaxed);
    global_metrics().e2e_ms.observe_ms(static_cast<double>(elapsed));
    if (result.queue_wait_ms >= 0) global_metrics().queue_wait_ms.observe_ms(result.queue_wait_ms);
    if (result.infer_ms >= 0) global_metrics().infer_ms.observe_ms(result.infer_ms);

    json out_shape = json::array();
    for (auto d : result.output_shape) out_shape.push_back(d);
    json out_data = json::array();
    for (float v : result.output) out_data.push_back(v);
    json timing = {{"e2e", elapsed}};
    if (result.queue_wait_ms >= 0) timing["queue_wait"] = result.queue_wait_ms;
    if (result.infer_ms >= 0) timing["inference"] = result.infer_ms;
    write_response(make_json_response(
                       http::status::ok,
                       json{{"request_id", request_id},
                            {"status", "ok"},
                            {"output",
                             {{"dtype", "float32"}, {"shape", out_shape}, {"data", out_data}}},
                            {"timing_ms", timing}},
                       ver, keep_alive),
                   keep_alive);
  }

  void write_response(http::response<http::string_body> res, bool keep_alive) {
    auto sp = std::make_shared<http::response<http::string_body>>(std::move(res));
    http::async_write(stream_, *sp,
                      [self = shared_from_this(), sp, keep_alive](beast::error_code ec, std::size_t) {
                        if (ec) return self->do_close();
                        if (!keep_alive) return self->do_close();
                        self->do_read();
                      });
  }

  void do_close() {
    if (active_request_) {
      active_request_->cancelled.store(true, std::memory_order_release);
    }
    beast::error_code ec;
    stream_.socket().shutdown(tcp::socket::shutdown_send, ec);
  }

  beast::tcp_stream stream_;
  beast::flat_buffer buffer_;
  std::optional<http::request_parser<http::string_body>> parser_;
  http::request<http::string_body> request_;
  ServerConfig cfg_;
  AdmissionController& admission_;
  WorkerPool& workers_;
  BatchScheduler& scheduler_;
  net::io_context& ioc_;
  std::shared_ptr<ClientRequest> active_request_;
};

}  // namespace

HttpServer::HttpServer(ServerConfig cfg, net::io_context& ioc, AdmissionController& admission,
                       WorkerPool& workers, BatchScheduler& scheduler)
    : cfg_(std::move(cfg)),
      ioc_(ioc),
      acceptor_(net::make_strand(ioc)),
      admission_(admission),
      workers_(workers),
      scheduler_(scheduler) {}

void HttpServer::start() {
  const auto address = net::ip::make_address(cfg_.bind_host);
  tcp::endpoint ep{address, cfg_.bind_port};
  acceptor_.open(ep.protocol());
  acceptor_.set_option(net::socket_base::reuse_address(true));
  acceptor_.bind(ep);
  acceptor_.listen(net::socket_base::max_listen_connections);
  do_accept();
}

void HttpServer::stop() {
  stopping_ = true;
  beast::error_code ec;
  acceptor_.close(ec);
}

void HttpServer::do_accept() {
  acceptor_.async_accept(net::make_strand(ioc_),
                         [this](beast::error_code ec, tcp::socket socket) {
                           if (!ec) {
                             std::make_shared<Session>(std::move(socket), cfg_, admission_, workers_,
                                                       scheduler_, ioc_)
                                 ->run();
                           }
                           if (!stopping_) do_accept();
                         });
}

}  // namespace batchflow
