#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace batchflow {

// Fixed tensor contract served by BatchFlow. Any model passed via --model must expose
// exactly one float32 input named "input" with shape [N,3,8,8] and one float32 output
// named "output" with shape [N,4]. Symbolic (-1) dimensions in the model are accepted at
// load time; the actual output shape is re-checked after every batch.
struct TensorSpec {
  std::string name;
  std::vector<std::int64_t> shape;  // dynamic batch dim uses -1 when applicable
  std::string dtype;                // e.g. "float32"
};

struct InferResult {
  bool ok = false;
  std::string error;
  std::vector<float> output;
  std::vector<std::int64_t> output_shape;
  double infer_ms{-1.0};
  double queue_wait_ms{-1.0};
  bool skipped = false;  // inference was not run (see InferJob::still_needed)
};

struct SessionOptions {
  int intra_op_threads{1};
  int inter_op_threads{1};
};

class OnnxSession {
 public:
  explicit OnnxSession(std::string model_path, SessionOptions opts = {});
  ~OnnxSession();

  OnnxSession(const OnnxSession&) = delete;
  OnnxSession& operator=(const OnnxSession&) = delete;

  // True only if ORT is linked and the model loaded with a compatible signature.
  [[nodiscard]] bool available() const;
  [[nodiscard]] const std::string& load_error() const;
  [[nodiscard]] std::string_view backend() const;
  [[nodiscard]] const TensorSpec& input_spec() const;
  [[nodiscard]] const TensorSpec& output_spec() const;

  // Runs one batch. input length must equal product(shape) with batch filled in.
  InferResult run(std::span<const float> input, std::int64_t batch_size);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace batchflow
