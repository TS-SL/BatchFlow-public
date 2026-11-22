#include "batchflow/onnx_session.hpp"

#include <array>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

#if defined(BATCHFLOW_HAS_ONNXRUNTIME)
#include <onnxruntime_cxx_api.h>
#endif

namespace batchflow {

struct OnnxSession::Impl {
  std::string model_path;
  bool ort_linked = false;
  bool loaded = false;
  std::string load_error;
  TensorSpec input{"input", {-1, 3, 8, 8}, "float32"};
  TensorSpec output{"output", {-1, 4}, "float32"};
#if defined(BATCHFLOW_HAS_ONNXRUNTIME)
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "batchflow"};
  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> session;
  Ort::AllocatorWithDefaultOptions allocator;
#endif
};

#if defined(BATCHFLOW_HAS_ONNXRUNTIME)
namespace {

std::string shape_str(const std::vector<std::int64_t>& s) {
  std::ostringstream os;
  os << '[';
  for (std::size_t i = 0; i < s.size(); ++i) os << (i ? "," : "") << s[i];
  os << ']';
  return os.str();
}

// A declared dimension matches if it equals the expected value or is symbolic/unknown.
bool dims_compatible(const std::vector<std::int64_t>& got, const std::vector<std::int64_t>& want) {
  if (got.size() != want.size()) return false;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (want[i] < 0 || got[i] < 0) continue;
    if (got[i] != want[i]) return false;
  }
  return true;
}

// Returns an empty string if the session's declared I/O matches the served contract.
std::string check_signature(Ort::Session& s, Ort::AllocatorWithDefaultOptions& alloc,
                            const TensorSpec& in, const TensorSpec& out) {
  if (s.GetInputCount() != 1 || s.GetOutputCount() != 1) {
    return "model must have exactly one input and one output";
  }
  const auto in_name = s.GetInputNameAllocated(0, alloc);
  const auto out_name = s.GetOutputNameAllocated(0, alloc);
  if (in.name != in_name.get() || out.name != out_name.get()) {
    return std::string("model tensors must be named '") + in.name + "' and '" + out.name +
           "', got '" + in_name.get() + "' and '" + out_name.get() + "'";
  }
  // The shape-info views borrow from their TypeInfo, which must outlive them.
  const Ort::TypeInfo in_type = s.GetInputTypeInfo(0);
  const Ort::TypeInfo out_type = s.GetOutputTypeInfo(0);
  const auto in_info = in_type.GetTensorTypeAndShapeInfo();
  const auto out_info = out_type.GetTensorTypeAndShapeInfo();
  if (in_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      out_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    return "model input and output must be float32";
  }
  const auto in_shape = in_info.GetShape();
  const auto out_shape = out_info.GetShape();
  if (!dims_compatible(in_shape, in.shape)) {
    return "model input shape " + shape_str(in_shape) + " is not compatible with " +
           shape_str(in.shape);
  }
  if (!dims_compatible(out_shape, out.shape)) {
    return "model output shape " + shape_str(out_shape) + " is not compatible with " +
           shape_str(out.shape);
  }
  return {};
}

}  // namespace
#endif

OnnxSession::OnnxSession(std::string model_path, SessionOptions opts) : impl_(new Impl) {
  impl_->model_path = std::move(model_path);
#if defined(BATCHFLOW_HAS_ONNXRUNTIME)
  impl_->ort_linked = true;
  try {
  const int intra = opts.intra_op_threads > 0 ? opts.intra_op_threads : 1;
  const int inter = opts.inter_op_threads > 0 ? opts.inter_op_threads : 1;
  impl_->session_options.SetIntraOpNumThreads(intra);
  impl_->session_options.SetInterOpNumThreads(inter);
  impl_->session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
#ifdef _WIN32
  // Byte-wise widening: only ASCII model paths are supported on Windows.
  const std::wstring wpath(impl_->model_path.begin(), impl_->model_path.end());
  impl_->session = std::make_unique<Ort::Session>(impl_->env, wpath.c_str(), impl_->session_options);
#else
  impl_->session = std::make_unique<Ort::Session>(impl_->env, impl_->model_path.c_str(),
                                                    impl_->session_options);
#endif
    impl_->load_error =
        check_signature(*impl_->session, impl_->allocator, impl_->input, impl_->output);
    impl_->loaded = impl_->load_error.empty();
  } catch (const std::exception& ex) {
    impl_->load_error = std::string("failed to load model: ") + ex.what();
    impl_->loaded = false;
  }
  if (!impl_->loaded) impl_->session.reset();
#else
  (void)opts;
  impl_->ort_linked = false;
  impl_->load_error = "ONNX Runtime not linked in this build";
#endif
}

OnnxSession::~OnnxSession() { delete impl_; }

bool OnnxSession::available() const { return impl_->ort_linked && impl_->loaded; }

const std::string& OnnxSession::load_error() const { return impl_->load_error; }

std::string_view OnnxSession::backend() const {
  return impl_->ort_linked ? "onnxruntime-cpu" : "stub-unlinked";
}

const TensorSpec& OnnxSession::input_spec() const { return impl_->input; }

const TensorSpec& OnnxSession::output_spec() const { return impl_->output; }

InferResult OnnxSession::run(std::span<const float> input, std::int64_t batch_size) {
  InferResult r;
  if (!available()) {
    r.ok = false;
    r.error = impl_->load_error;
    return r;
  }
  if (batch_size <= 0) {
    r.ok = false;
    r.error = "batch_size must be > 0";
    return r;
  }

#if defined(BATCHFLOW_HAS_ONNXRUNTIME)
  try {
    const std::int64_t elems_per = 3 * 8 * 8;
    const std::size_t expected = static_cast<std::size_t>(batch_size * elems_per);
    if (input.size() != expected) {
      r.ok = false;
      r.error = "input element count mismatch";
      return r;
    }

    const std::array<std::int64_t, 4> shape{batch_size, 3, 8, 8};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        mem, const_cast<float*>(input.data()), input.size(), shape.data(), shape.size());

    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    auto outputs = impl_->session->Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1,
                                       output_names, 1);
    float* out = outputs[0].GetTensorMutableData<float>();
    const auto out_info = outputs[0].GetTensorTypeAndShapeInfo();
    const auto out_shape = out_info.GetShape();
    const std::size_t out_count = out_info.GetElementCount();
    r.output.assign(out, out + out_count);
    r.output_shape.assign(out_shape.begin(), out_shape.end());
    r.ok = true;
    return r;
  } catch (const Ort::Exception& ex) {
    r.ok = false;
    r.error = ex.what();
    return r;
  } catch (const std::exception& ex) {
    r.ok = false;
    r.error = ex.what();
    return r;
  }
#else
  (void)input;
  (void)batch_size;
  r.ok = false;
  r.error = "ONNX Runtime not linked in this build";
  return r;
#endif
}

}  // namespace batchflow
