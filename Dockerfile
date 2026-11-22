# BatchFlow CPU-only inference server (Linux container).
# Binds to 0.0.0.0 inside the container; publish only to 127.0.0.1 on the host.
# Example: docker run --rm -p 127.0.0.1:8080:8080 batchflow:local

FROM ubuntu:24.04 AS build
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl ca-certificates zip unzip tar pkg-config \
    python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN git clone --depth 1 https://github.com/microsoft/vcpkg.git /opt/vcpkg \
    && /opt/vcpkg/bootstrap-vcpkg.sh -disableMetrics

RUN bash scripts/fetch_onnxruntime.sh 1.32.0

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements-dev.txt \
    && /opt/venv/bin/python scripts/export_tiny_model.py \
    && /opt/venv/bin/python tests/fixtures/make_wrong_shape_model.py

# Adapt CMake for Linux ORT shared library naming.
# BATCHFLOW_REQUIRE_FULL_BUILD fails configuration if ORT or the HTTP deps are missing.
RUN cmake -S . -B build -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DVCPKG_TARGET_TRIPLET=x64-linux \
      -DBATCHFLOW_ORT_ROOT=/src/third_party/onnxruntime \
      -DBATCHFLOW_REQUIRE_FULL_BUILD=ON \
    && cmake --build build \
    && ctest --test-dir build --output-on-failure --no-tests=error \
    && /opt/venv/bin/python tests/integration/run_live.py --server build/batchflow_server

FROM ubuntu:24.04 AS runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=build /src/build/batchflow_server /app/batchflow_server
COPY --from=build /src/models/tiny_deterministic.onnx /app/models/tiny_deterministic.onnx
COPY --from=build /src/third_party/onnxruntime/lib/ /app/lib/
ENV LD_LIBRARY_PATH=/app/lib

EXPOSE 8080
# Container listens on all interfaces; host publish should stay localhost-only.
ENTRYPOINT ["/app/batchflow_server"]
CMD ["--host","0.0.0.0","--port","8080","--allow-non-local-bind","--model","models/tiny_deterministic.onnx","--max-batch","8","--batch-wait-ms","5"]
