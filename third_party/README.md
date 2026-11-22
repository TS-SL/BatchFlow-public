# Third-party dependencies

Nothing in this directory except this file is committed. Dependencies are fetched at
build time.

| Dependency | Version used for this release | Source | License |
|---|---|---|---|
| ONNX Runtime (CPU) | 1.32.0 | NuGet `Microsoft.ML.OnnxRuntime`, fetched by `scripts/fetch_onnxruntime.ps1` (Windows) or `scripts/fetch_onnxruntime.sh` (Linux) into `third_party/onnxruntime/` | MIT |
| Boost (Asio, Beast, System) | 1.92.0 | vcpkg manifest (`vcpkg.json`) | Boost Software License 1.0 |
| nlohmann-json | 3.12.0 | vcpkg manifest (`vcpkg.json`) | MIT |
| Python: numpy, onnx, onnxruntime, pytest | see `requirements-dev.txt` | PyPI | BSD-3-Clause, Apache-2.0, MIT, MIT |

`vcpkg.json` does not pin a `builtin-baseline`, so Boost and nlohmann-json resolve to
whatever versions the cloned vcpkg revision provides.

## Why ONNX Runtime comes from NuGet

The vcpkg `onnxruntime` port failed for 1.32.0 because its source archive URL
(`https://github.com/microsoft/onnxruntime/archive/v1.32.0.tar.gz`) returned HTTP 404,
while Boost built normally. The official prebuilt CPU package of the same version as the
Python wheel is used instead. The fetch scripts record name, version, source URL and
platform in `third_party/onnxruntime/VERSION.txt`.

The ONNX Runtime package ships its own license and third-party notices inside the
NuGet archive. Anyone redistributing a build that includes `onnxruntime.dll` or
`libonnxruntime.so` (for example the Docker image) must include them.
