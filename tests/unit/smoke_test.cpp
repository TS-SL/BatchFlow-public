#include "batchflow/metrics.hpp"
#include "batchflow/version.hpp"

#include <cstdlib>
#include <iostream>

int main() {
  if (batchflow::kVersionMajor != 0) {
    std::cerr << "unexpected major version\n";
    return 1;
  }
  auto& m = batchflow::global_metrics();
  m.requests_total.fetch_add(1);
  if (m.requests_total.load() != 1) {
    std::cerr << "metrics counter failed\n";
    return 1;
  }
  std::cout << "smoke_ok version=" << batchflow::kVersionString << "\n";
  return 0;
}
