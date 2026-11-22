#include "batchflow/onnx_session.hpp"
#include "batchflow/version.hpp"

#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

static bool load_blob(const std::string& path, std::int64_t& batch, std::vector<float>& in,
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
  return true;
}

int main(int argc, char** argv) {
  const char* model = argc > 1 ? argv[1] : "models/tiny_deterministic.onnx";
  const char* blob = argc > 2 ? argv[2] : "models/tiny_b1_io.bin";
  std::cout << "BatchFlow infer_ref " << batchflow::kVersionString << "\n";
  batchflow::OnnxSession session(model);
  std::cout << "backend=" << session.backend()
            << " available=" << (session.available() ? "yes" : "no") << "\n";
  if (!session.available()) {
    std::cerr << "BLOCKED: ONNX Runtime not linked\n";
    return 2;
  }

  std::int64_t batch = 0;
  std::vector<float> in, ref;
  if (!load_blob(blob, batch, in, ref)) {
    std::cerr << "FAILED: cannot read blob " << blob << "\n";
    return 2;
  }
  auto result = session.run(in, batch);
  if (!result.ok) {
    std::cerr << "FAILED: " << result.error << "\n";
    return 1;
  }
  std::cout << "ok batch=" << batch << " out=[";
  for (std::size_t i = 0; i < result.output.size(); ++i) {
    if (i) std::cout << ", ";
    std::cout << result.output[i];
  }
  std::cout << "]\n";
  return 0;
}
