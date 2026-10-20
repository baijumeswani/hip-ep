/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "execution_status.h"
#include "hip_custom_kernels.h"
#include "runtime_state_internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

enum class Failure { None, Allocation, Drain, Free, Launch };

struct FakeDevice {
  Failure failure{Failure::None};
  std::map<void *, size_t> allocations;
  std::vector<std::string> calls;
  bool pending{};
  int failures{};
} device;

#define CHECK(condition)                                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);     \
      ++device.failures;                                                       \
    }                                                                          \
  } while (0)

RuntimeState make_state() {
  RuntimeState state{};
  state.stream = &device;
  return state;
}

const void *dequantize(RuntimeState &state, int n = 8) {
  static const unsigned char weights[128]{};
  static const uint16_t scales[8]{};
  return hipdnn_ep_dequant_matmul_nbits_fp16(&state, state.stream, weights,
                                             scales, nullptr, n, 32, 32);
}

void release(RuntimeState &state) {
  device.failure = Failure::None;
  CHECK(hipStreamSynchronize(state.stream) == hipSuccess);
  if (state.workspace)
    CHECK(hipFree(state.workspace) == hipSuccess);
  if (state.matmul_nbits_fp16_scratch)
    CHECK(hipFree(state.matmul_nbits_fp16_scratch) == hipSuccess);
  state = {};
  CHECK(device.allocations.empty());
  device.calls.clear();
}

void test_workspace_growth_and_failure() {
  for (const auto failure :
       {Failure::None, Failure::Allocation, Failure::Drain, Failure::Free}) {
    auto state = make_state();
    CHECK(hipdnn_ep_state_ensure_workspace(&state, 128) == 0);
    void *previous = state.workspace;
    device.calls.clear();
    device.pending = true;
    device.failure = failure;
    const int result = hipdnn_ep_state_ensure_workspace(&state, 129);
    CHECK(state.host_error_flag == (failure == Failure::None ? 0 : -1));
    if (failure == Failure::Drain || failure == Failure::Free) {
      CHECK(result != 0);
      CHECK(state.workspace == previous && state.workspace_size == 128);
      CHECK(device.calls == (failure == Failure::Drain
                                 ? std::vector<std::string>{"drain"}
                                 : std::vector<std::string>{"drain", "free"}));
      device.failure = Failure::None;
      CHECK(hipdnn_ep_state_ensure_workspace(&state, 129) == 0);
      CHECK(state.workspace_size >= 129);
    } else {
      CHECK(device.calls ==
            (std::vector<std::string>{"drain", "free", "allocate"}));
      if (failure == Failure::Allocation) {
        CHECK(result != 0);
        CHECK(state.workspace == nullptr && state.workspace_size == 0);
        device.failure = Failure::None;
        CHECK(hipdnn_ep_state_ensure_workspace(&state, 129) == 0);
      } else {
        CHECK(result == 0);
        CHECK(state.workspace_size == 192);
      }
    }
    release(state);
  }

  auto state = make_state();
  device.failure = Failure::Allocation;
  CHECK(hipdnn_ep_state_ensure_workspace(&state, 128) != 0);
  CHECK(state.workspace == nullptr && state.workspace_size == 0);
  device.failure = Failure::None;
  CHECK(hipdnn_ep_state_ensure_workspace(&state, 128) == 0);
  device.calls.clear();
  CHECK(hipdnn_ep_state_ensure_workspace(&state, 127) == 0);
  CHECK(device.calls.empty());
  release(state);
}

void test_dequantization_growth_and_failure() {
  for (const auto failure :
       {Failure::None, Failure::Allocation, Failure::Drain, Failure::Free}) {
    auto state = make_state();
    CHECK(dequantize(state) != nullptr);
    void *previous = state.matmul_nbits_fp16_scratch;
    device.calls.clear();
    device.failure = failure;
    const void *output = dequantize(state, 16);
    CHECK(state.host_error_flag == (failure == Failure::None ? 0 : -1));
    if (failure == Failure::Drain || failure == Failure::Free) {
      CHECK(output == nullptr);
      CHECK(state.matmul_nbits_fp16_scratch == previous);
      CHECK(state.matmul_nbits_fp16_scratch_size == 512);
      CHECK(device.calls == (failure == Failure::Drain
                                 ? std::vector<std::string>{"drain"}
                                 : std::vector<std::string>{"drain", "free"}));
      device.failure = Failure::None;
      CHECK(dequantize(state, 16) != nullptr);
      CHECK(state.matmul_nbits_fp16_scratch_size == 1024);
    } else if (failure == Failure::Allocation) {
      CHECK(output == nullptr);
      CHECK(state.matmul_nbits_fp16_scratch == nullptr);
      CHECK(state.matmul_nbits_fp16_scratch_size == 0);
      CHECK(device.calls ==
            (std::vector<std::string>{"drain", "free", "allocate"}));
      device.failure = Failure::None;
      CHECK(dequantize(state, 16) != nullptr);
    } else {
      CHECK(output == state.matmul_nbits_fp16_scratch);
      CHECK(state.matmul_nbits_fp16_scratch_size == 1024);
      CHECK(device.calls ==
            (std::vector<std::string>{"drain", "free", "allocate", "launch",
                                      "status"}));
    }
    release(state);
  }
}

void test_launch_failure_and_independent_buffers() {
  auto state = make_state();
  CHECK(hipdnn_ep_state_ensure_workspace(&state, 128) == 0);
  void *workspace = state.workspace;
  device.failure = Failure::Launch;
  CHECK(dequantize(state) == nullptr);
  CHECK(state.host_error_flag == -1);
  CHECK(state.matmul_nbits_fp16_scratch != nullptr);
  CHECK(state.workspace == workspace && state.workspace_size == 128);
  device.failure = Failure::None;
  device.calls.clear();
  CHECK(dequantize(state) == state.matmul_nbits_fp16_scratch);
  CHECK(device.calls == (std::vector<std::string>{"launch", "status"}));
  CHECK(state.workspace != state.matmul_nbits_fp16_scratch);
  release(state);
}

void test_execution_failure_and_retry() {
  auto state = make_state();
  int error = 0;
  state.device_error_flag = &error;
  auto run = [&] {
    ExecutionStatusGuard status(&state);
    if (hipdnn_ep_state_ensure_workspace(&state, 128) != 0)
      return -1;
    return status.Finish(0);
  };
  device.failure = Failure::Allocation;
  CHECK(run() == -1);
  CHECK(error != 0);
  device.failure = Failure::None;
  CHECK(run() == 0);
  CHECK(error != 0);
  error = 0;
  device.calls.clear();
  CHECK(run() == 0);
  CHECK(error == 0);
  CHECK(device.calls.empty());
  {
    ExecutionStatusGuard status(&state);
    CHECK(status.Finish(-7) == -7);
  }
  CHECK(error != 0);
  release(state);
}

void test_fp32_tiles_share_weight_scratch_not_workspace() {
  auto state = make_state();
  CHECK(hipdnn_ep_state_ensure_workspace(&state, 128) == 0);
  void *workspace = state.workspace;
  CHECK(dequantize(state) != nullptr);
  device.calls.clear();
  void *tile =
      hipdnn_ep_state_ensure_matmul_nbits_scratch(&state, state.stream, 1024);
  CHECK(tile != nullptr && tile != workspace);
  CHECK(state.matmul_nbits_fp16_scratch_size == 1024);
  CHECK(device.calls ==
        (std::vector<std::string>{"drain", "free", "allocate"}));
  device.calls.clear();
  CHECK(dequantize(state) == tile);
  CHECK(state.workspace == workspace && state.workspace_size == 128);
  CHECK(device.calls == (std::vector<std::string>{"launch", "status"}));
  release(state);
}

} // namespace

extern "C" void *hipdnn_ep_state_get_stream(RuntimeState *state) {
  return state->stream;
}

extern "C" void *
hipdnn_ep_state_get_error_flag_device_ptr(RuntimeState *state) {
  return state->device_error_flag;
}

extern "C" hipError_t hipMemsetAsync(void *output, int value, size_t size,
                                     hipStream_t stream) {
  CHECK(stream == &device);
  CHECK(size == sizeof(int));
  device.calls.emplace_back("record_error");
  std::memset(output, value, size);
  return hipSuccess;
}

extern "C" hipError_t hipMalloc(void **pointer, size_t bytes) {
  device.calls.emplace_back("allocate");
  if (device.failure == Failure::Allocation)
    return 2;
  *pointer = std::malloc(bytes);
  if (!*pointer)
    return 2;
  device.allocations.emplace(*pointer, bytes);
  return hipSuccess;
}

extern "C" hipError_t hipFree(void *pointer) {
  device.calls.emplace_back("free");
  CHECK(!device.pending);
  if (device.failure == Failure::Free)
    return 999;
  CHECK(device.allocations.erase(pointer) == 1);
  std::free(pointer);
  return hipSuccess;
}

extern "C" hipError_t hipStreamSynchronize(hipStream_t stream) {
  device.calls.emplace_back("drain");
  CHECK(stream == &device);
  if (device.failure == Failure::Drain)
    return 999;
  device.pending = false;
  return hipSuccess;
}

extern "C" const char *hipGetErrorString(hipError_t error) {
  return error == 2 ? "injected allocation failure" : "injected device failure";
}

extern "C" hipError_t hipGetLastError() {
  device.calls.emplace_back("status");
  return device.failure == Failure::Launch ? 999 : hipSuccess;
}

extern "C" void hip_matmul_nbits_dequant_b_fp16(void *stream, const void *,
                                                const void *, const void *,
                                                void *output, int n, int k,
                                                int group_size) {
  device.calls.emplace_back("launch");
  CHECK(stream == &device);
  CHECK(device.allocations.at(output) >=
        static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(uint16_t));
  CHECK(group_size == 32);
  if (device.failure != Failure::Launch)
    device.pending = true;
}

int main() {
  test_workspace_growth_and_failure();
  test_dequantization_growth_and_failure();
  test_launch_failure_and_independent_buffers();
  test_execution_failure_and_retry();
  test_fp32_tiles_share_weight_scratch_not_workspace();
  if (device.failures) {
    fprintf(stderr, "device scratch: %d failures\n", device.failures);
    return 1;
  }
  puts("device scratch allocation/drain/free/launch failures and retry: PASS");
  return 0;
}
