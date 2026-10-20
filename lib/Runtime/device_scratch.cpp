/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "debug_log.h"
#include "hipdnn_ep_runtime.h"
#include "runtime_state_internal.h"

#if defined(HIPDNN_EP_REAL_RUNTIME)
#include "hip_custom_kernels.h"
#endif

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

int grow_device_buffer(hipStream_t stream, void *&buffer, size_t &capacity,
                       size_t bytes, const char *name) {
  if (capacity >= bytes)
    return 0;

  if (buffer) {
    if (stream) {
      const hipError_t status = hipStreamSynchronize(stream);
      if (status != hipSuccess) {
        fprintf(stderr, "%s: draining old buffer failed: %s\n", name,
                hipGetErrorString(status));
        return -1;
      }
    }
    const hipError_t status = hipFree(buffer);
    if (status != hipSuccess) {
      fprintf(stderr, "%s: freeing old buffer failed: %s\n", name,
              hipGetErrorString(status));
      return -1;
    }
    buffer = nullptr;
    capacity = 0;
  }

  void *allocation = nullptr;
  const hipError_t status = hipMalloc(&allocation, bytes);
  if (status != hipSuccess) {
    fprintf(stderr, "%s: hipMalloc(%zu) failed: %s\n", name, bytes,
            hipGetErrorString(status));
    return -1;
  }
  buffer = allocation;
  capacity = bytes;
  return 0;
}

} // namespace

void *hipdnn_ep_state_get_workspace(RuntimeState *state) {
  return state ? state->workspace : nullptr;
}

size_t hipdnn_ep_state_get_workspace_size(RuntimeState *state) {
  return state ? state->workspace_size : 0;
}

int hipdnn_ep_state_ensure_workspace(RuntimeState *state, size_t needed_size) {
  if (!state) {
    fprintf(stderr, "hipdnn_ep_state_ensure_workspace: null state\n");
    return -1;
  }
  if (state->workspace_size >= needed_size)
    return 0;

  // Preserve amortized growth for operators whose workspace increases per
  // token.
  size_t bytes = needed_size;
  const size_t increment = state->workspace_size / 2;
  if (state->workspace_size <= std::numeric_limits<size_t>::max() - increment) {
    const size_t grown = state->workspace_size + increment;
    if (grown > bytes)
      bytes = grown;
  }
  if (grow_device_buffer(state->stream, state->workspace, state->workspace_size,
                         bytes, "hipdnn_ep_state_ensure_workspace") != 0) {
    state->host_error_flag = -1;
    return -1;
  }
  RUNTIME_DEBUG_LOG(
      "[workspace] Allocated shared workspace: %zu bytes (requested %zu)\n",
      bytes, needed_size);
  return 0;
}

#if defined(HIPDNN_EP_REAL_RUNTIME)
const void *hipdnn_ep_dequant_matmul_nbits_fp16(
    RuntimeState *state, void *stream, const void *weights, const void *scales,
    const void *zeros, int n, int k, int group_size) {
  if (!state || !stream || !weights || !scales || n <= 0 || k <= 0 ||
      group_size <= 0 ||
      static_cast<size_t>(n) > std::numeric_limits<size_t>::max() /
                                   sizeof(uint16_t) / static_cast<size_t>(k)) {
    fprintf(stderr, "matmul_nbits: invalid fp16 scratch arguments\n");
    if (state)
      state->host_error_flag = -1;
    return nullptr;
  }
  const size_t bytes =
      static_cast<size_t>(n) * static_cast<size_t>(k) * sizeof(uint16_t);
  const size_t previous_size = state->matmul_nbits_fp16_scratch_size;
  if (grow_device_buffer(static_cast<hipStream_t>(stream),
                         state->matmul_nbits_fp16_scratch,
                         state->matmul_nbits_fp16_scratch_size, bytes,
                         "matmul_nbits fp16 scratch") != 0) {
    state->host_error_flag = -1;
    return nullptr;
  }
  if (previous_size < bytes)
    RUNTIME_DEBUG_LOG("[matmul_nbits] Allocated shared fp16 weight scratch: "
                      "%zu bytes\n",
                      bytes);

  void *output = state->matmul_nbits_fp16_scratch;
  hip_matmul_nbits_dequant_b_fp16(stream, weights, scales, zeros, output, n, k,
                                  group_size);
  const hipError_t status = hipGetLastError();
  if (status != hipSuccess) {
    fprintf(stderr, "matmul_nbits: dequantizing fp16 scratch failed: %s\n",
            hipGetErrorString(status));
    state->host_error_flag = -1;
    return nullptr;
  }
  return output;
}
#endif
