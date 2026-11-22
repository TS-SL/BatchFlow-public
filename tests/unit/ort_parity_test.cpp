#include "batchflow/onnx_session.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// Reading NPZ from C++ would need a zip reader, so this test compares against a compact
// binary blob written by scripts/export_tiny_model.py: models/tiny_b1_io.bin
// Format: int32 magic 'BFIO', int32 batch, int32 in_count, int32 out_count, floats...

namespace {

bool load_blob(const std::string& path, std::int64_t& batch, std::vector<float>& in,
               std::vector<float>& out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) return false;
  char magic[4];
  f.read(magic, 4);
  if (std::string(magic, 4) != "BFIO") return false;
  std::int32_t b = 0, in_count = 0, out_count = 0;
  f.read(reinterpret_cast<char*>(&b), 4);
  f.read(reinterpret_cast<char*>(&in_count), 4);
  f.read(reinterpret_cast<char*>(&out_count), 4);
  batch = b;
  in.resize(static_cast<std::size_t>(in_count));
  out.resize(static_cast<std::size_t>(out_count));
  f.read(reinterpret_cast<char*>(in.data()), in_count * 4);
  f.read(reinterpret_cast<char*>(out.data()), out_count * 4);
  return static_cast<bool>(f) || f.eof();
}

bool close(const std::vector<float>& a, const std::vector<float>& b, float rtol, float atol) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float diff = std::fabs(a[i] - b[i]);
    const float tol = atol + rtol * std::fabs(b[i]);
    if (diff > tol) return false;
  }
  return true;
}

}  // namespace

int main() {
  const std::string model = "models/tiny_deterministic.onnx";
  const std::string blob = "models/tiny_b1_io.bin";
  std::int64_t batch = 0;
  std::vector<float> in, ref;
  if (!load_blob(blob, batch, in, ref)) {
    std::cerr << "FAILED: cannot load " << blob << " (run scripts/export_tiny_model.py)\n";
    return 2;
  }

  batchflow::OnnxSession session(model);
  if (!session.available()) {
    std::cerr << "FAILED: ORT not linked\n";
    return 2;
  }
  auto result = session.run(in, batch);
  if (!result.ok) {
    std::cerr << "FAILED: run error: " << result.error << "\n";
    return 1;
  }
  if (!close(result.output, ref, 1e-5f, 1e-6f)) {
    std::cerr << "FAILED: numerical mismatch vs reference blob\n";
    return 1;
  }
  std::cout << "ort_parity_ok batch=" << batch << " out_elems=" << result.output.size() << "\n";
  return 0;
}
