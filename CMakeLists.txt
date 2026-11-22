cmake_minimum_required(VERSION 3.20)
project(batchflow VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WIN32)
  add_compile_definitions(_WIN32_WINNT=0x0A00)
endif()

option(BATCHFLOW_BUILD_TESTS "Build unit/integration tests" ON)
option(BATCHFLOW_BUILD_SERVER "Build HTTP server (requires Boost + nlohmann_json)" ON)
option(BATCHFLOW_ENABLE_ASAN "Enable AddressSanitizer (supported toolchains)" OFF)
option(BATCHFLOW_ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" OFF)
option(BATCHFLOW_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
option(BATCHFLOW_REQUIRE_FULL_BUILD
  "Fail configuration unless ONNX Runtime and the HTTP server dependencies are found" OFF)

find_package(Threads REQUIRED)

set(BATCHFLOW_ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root")
set(BATCHFLOW_HAS_ORT FALSE)
set(BATCHFLOW_ORT_LIB "")
if(EXISTS "${BATCHFLOW_ORT_ROOT}/include/onnxruntime_cxx_api.h")
  if(WIN32 AND EXISTS "${BATCHFLOW_ORT_ROOT}/lib/onnxruntime.lib")
    set(BATCHFLOW_HAS_ORT TRUE)
    set(BATCHFLOW_ORT_LIB "${BATCHFLOW_ORT_ROOT}/lib/onnxruntime.lib")
  elseif(NOT WIN32)
    file(GLOB BATCHFLOW_ORT_SO_CANDIDATES
      "${BATCHFLOW_ORT_ROOT}/lib/libonnxruntime.so"
      "${BATCHFLOW_ORT_ROOT}/lib/libonnxruntime.so.*"
    )
    if(BATCHFLOW_ORT_SO_CANDIDATES)
      list(GET BATCHFLOW_ORT_SO_CANDIDATES 0 BATCHFLOW_ORT_LIB)
      set(BATCHFLOW_HAS_ORT TRUE)
    endif()
  endif()
endif()

add_library(batchflow_core STATIC
  src/inference/onnx_session.cpp
  src/metrics/metrics.cpp
  src/scheduler/admission.cpp
  src/scheduler/worker_pool.cpp
  src/scheduler/batch_scheduler.cpp
)
target_include_directories(batchflow_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(batchflow_core
  PUBLIC
    Threads::Threads
)

if(BATCHFLOW_HAS_ORT)
  target_compile_definitions(batchflow_core PUBLIC BATCHFLOW_HAS_ONNXRUNTIME=1)
  target_include_directories(batchflow_core PUBLIC "${BATCHFLOW_ORT_ROOT}/include")
  target_link_libraries(batchflow_core PUBLIC "${BATCHFLOW_ORT_LIB}")
  if(UNIX AND NOT APPLE)
    target_link_options(batchflow_core PUBLIC "-Wl,-rpath,${BATCHFLOW_ORT_ROOT}/lib")
  endif()
  message(STATUS "ONNX Runtime: FOUND (${BATCHFLOW_ORT_LIB})")
else()
  message(STATUS "ONNX Runtime: NOT FOUND. Run scripts/fetch_onnxruntime.ps1 or .sh")
endif()

if(BATCHFLOW_ENABLE_ASAN)
  target_compile_options(batchflow_core PUBLIC -fsanitize=address)
  target_link_options(batchflow_core PUBLIC -fsanitize=address)
endif()
if(BATCHFLOW_ENABLE_UBSAN)
  target_compile_options(batchflow_core PUBLIC -fsanitize=undefined)
  target_link_options(batchflow_core PUBLIC -fsanitize=undefined)
endif()
if(BATCHFLOW_ENABLE_TSAN)
  target_compile_options(batchflow_core PUBLIC -fsanitize=thread)
  target_link_options(batchflow_core PUBLIC -fsanitize=thread)
endif()

function(batchflow_copy_ort_dlls target_name)
  if(BATCHFLOW_HAS_ORT AND WIN32)
    add_custom_command(TARGET ${target_name} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${BATCHFLOW_ORT_ROOT}/bin/onnxruntime.dll"
        $<TARGET_FILE_DIR:${target_name}>
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${BATCHFLOW_ORT_ROOT}/lib/onnxruntime_providers_shared.dll"
        $<TARGET_FILE_DIR:${target_name}>
      VERBATIM
    )
  endif()
endfunction()

add_executable(batchflow_infer_ref src/tools/infer_ref.cpp)
target_link_libraries(batchflow_infer_ref PRIVATE batchflow_core)
batchflow_copy_ort_dlls(batchflow_infer_ref)

set(BATCHFLOW_HAS_HTTP FALSE)
if(BATCHFLOW_BUILD_SERVER)
  find_package(nlohmann_json CONFIG QUIET)
  find_package(Boost QUIET COMPONENTS system)
  if(nlohmann_json_FOUND AND Boost_FOUND)
    set(BATCHFLOW_HAS_HTTP TRUE)
    add_executable(batchflow_server
      src/main.cpp
      src/transport/http_server.cpp
    )
    target_link_libraries(batchflow_server PRIVATE
      batchflow_core
      Boost::system
      nlohmann_json::nlohmann_json
    )
    if(WIN32)
      target_link_libraries(batchflow_server PRIVATE ws2_32 mswsock)
    endif()
    batchflow_copy_ort_dlls(batchflow_server)
    message(STATUS "HTTP server: ENABLED")
  else()
    message(STATUS "HTTP server: DISABLED (Boost/nlohmann_json not found — use cmake --preset default)")
  endif()
endif()

if(BATCHFLOW_BUILD_TESTS)
  enable_testing()
  add_executable(batchflow_smoke_tests tests/unit/smoke_test.cpp)
  target_link_libraries(batchflow_smoke_tests PRIVATE batchflow_core)
  batchflow_copy_ort_dlls(batchflow_smoke_tests)
  add_test(NAME smoke COMMAND batchflow_smoke_tests)

  if(BATCHFLOW_HAS_ORT)
    add_executable(batchflow_ort_parity_tests tests/unit/ort_parity_test.cpp)
    target_link_libraries(batchflow_ort_parity_tests PRIVATE batchflow_core)
    batchflow_copy_ort_dlls(batchflow_ort_parity_tests)
    add_test(NAME ort_parity COMMAND batchflow_ort_parity_tests
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(batchflow_batch_scheduler_tests tests/unit/batch_scheduler_test.cpp)
    target_link_libraries(batchflow_batch_scheduler_tests PRIVATE batchflow_core)
    batchflow_copy_ort_dlls(batchflow_batch_scheduler_tests)
    add_test(NAME batch_scheduler COMMAND batchflow_batch_scheduler_tests
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

    add_executable(batchflow_lifecycle_tests tests/unit/scheduler_lifecycle_test.cpp)
    target_link_libraries(batchflow_lifecycle_tests PRIVATE batchflow_core)
    batchflow_copy_ort_dlls(batchflow_lifecycle_tests)
    foreach(case IN ITEMS
        enqueue_refusal deadline_before_window deadline_not_at_head batch_output_mapping
        wrong_output_shape wrong_input_signature shutdown_with_queued_work
        expired_in_worker_queue)
      add_test(NAME lifecycle.${case} COMMAND batchflow_lifecycle_tests ${case}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
      set_tests_properties(lifecycle.${case} PROPERTIES TIMEOUT 30)
    endforeach()
  endif()
endif()

if(BATCHFLOW_REQUIRE_FULL_BUILD AND NOT (BATCHFLOW_HAS_ORT AND BATCHFLOW_HAS_HTTP))
  message(FATAL_ERROR "BATCHFLOW_REQUIRE_FULL_BUILD=ON but ORT linked=${BATCHFLOW_HAS_ORT}, "
                      "HTTP linked=${BATCHFLOW_HAS_HTTP}")
endif()

message(STATUS "BatchFlow ${PROJECT_VERSION} — C++${CMAKE_CXX_STANDARD}")
message(STATUS "  Tests: ${BATCHFLOW_BUILD_TESTS}")
message(STATUS "  ORT linked: ${BATCHFLOW_HAS_ORT}")
message(STATUS "  HTTP linked: ${BATCHFLOW_HAS_HTTP}")
