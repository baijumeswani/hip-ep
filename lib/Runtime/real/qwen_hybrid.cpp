/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <limits>
#include <utility>

namespace {

int finish(RuntimeState *state, void *stream, void *error, int status,
           const char *operation) {
  if (status == 0)
    return 0;
  hipdnn_ep_state_record_execution_status(state, status);
  std::fprintf(stderr, "HIP %s failed: %d\n", operation, status);
  if (hipMemsetAsync(error, 0xff, sizeof(int),
                     static_cast<hipStream_t>(stream)) != hipSuccess) {
    std::fprintf(stderr, "HIP %s cannot record execution failure\n", operation);
    std::abort();
  }
  return -1;
}

bool disjointCapture(
    void *updates, uint64_t update_bytes,
    std::initializer_list<std::pair<const void *, uint64_t>> buffers) {
  const auto start = reinterpret_cast<uintptr_t>(updates);
  const auto maximum = std::numeric_limits<uintptr_t>::max();
  if (!updates || !update_bytes || update_bytes > maximum - start)
    return false;
  for (auto [pointer, bytes] : buffers) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    if (!pointer || bytes > maximum - address ||
        (start < address + bytes && address < start + update_bytes))
      return false;
  }
  return true;
}

} // namespace

int wrap_varlen_causal_conv(RuntimeState *state, void *input, void *weight,
                            void *cumulative, void *bias, void *past,
                            void *output, void *present, int64_t tokens,
                            int64_t output_tokens, int64_t cumulative_count,
                            int64_t batch, int64_t present_batch,
                            int64_t channels, int64_t input_dtype) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  int status = -1;
  if (batch > 0 && batch < INT64_MAX && present_batch == batch &&
      cumulative_count == batch + 1 && output_tokens == tokens)
    status = hip_varlen_causal_conv(
        stream, input, weight, static_cast<const int32_t *>(cumulative), bias,
        past, output, present, tokens, batch, channels, error, input_dtype);
  return finish(state, stream, error, status, "VarlenCausalConvWithState");
}

int wrap_gated_delta_net(RuntimeState *state, void *query, void *key,
                         void *value, void *cumulative, void *decay, void *beta,
                         void *past, void *a_log, void *dt_bias, void *output,
                         void *present, int64_t tokens, int64_t key_tokens,
                         int64_t value_tokens, int64_t decay_tokens,
                         int64_t beta_tokens, int64_t cumulative_count,
                         int64_t batch, int64_t present_batch,
                         int64_t output_tokens, int64_t key_heads,
                         int64_t value_heads, float scale,
                         int64_t input_dtype) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  int status = -1;
  if (batch > 0 && batch < INT64_MAX && batch == present_batch &&
      cumulative_count == batch + 1 && key_tokens == tokens &&
      value_tokens == tokens && decay_tokens == tokens &&
      beta_tokens == tokens && output_tokens == tokens)
    status = hip_gated_delta_net(
        stream, query, key, value, static_cast<const int32_t *>(cumulative),
        static_cast<const float *>(decay), static_cast<const float *>(beta),
        static_cast<const float *>(past), static_cast<const float *>(a_log),
        static_cast<const float *>(dt_bias), output,
        static_cast<float *>(present), tokens, batch, key_heads, value_heads,
        scale, error, input_dtype);
  return finish(state, stream, error, status, "GatedDeltaNet");
}

int wrap_varlen_causal_conv_updates(
    RuntimeState *state, void *input, void *weight, void *cumulative,
    void *bias, void *past, void *capture_count, void *output, void *present,
    void *updates, int64_t tokens, int64_t output_tokens,
    int64_t cumulative_count, int64_t batch, int64_t present_batch,
    int64_t channels, int64_t capture_batch, int64_t update_batch,
    int64_t update_capacity, int64_t update_channels, int64_t capacity,
    int64_t input_dtype) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  int status = -1;
  if (tokens > 0 && tokens <= INT_MAX && batch > 0 && batch <= 65535 &&
      channels > 0 && channels <= INT_MAX / 4 && present_batch == batch &&
      cumulative_count == batch + 1 && output_tokens == tokens &&
      capture_batch == batch && update_batch == batch && capacity > 0 &&
      capacity <= 7 && update_capacity == capacity &&
      update_channels == channels &&
      (input_dtype == HIP_DTYPE_FLOAT16 || input_dtype == HIP_DTYPE_FLOAT32)) {
    const uint64_t bytes = input_dtype == HIP_DTYPE_FLOAT32 ? 4 : 2;
    const uint64_t input_bytes = tokens * channels * bytes;
    const uint64_t state_bytes = batch * channels * 3 * bytes;
    const uint64_t update_bytes = batch * capacity * channels * bytes;
    if (disjointCapture(updates, update_bytes,
                        {{input, input_bytes},
                         {weight, channels * 4 * bytes},
                         {cumulative, (batch + 1) * sizeof(int32_t)},
                         {bias, channels * bytes},
                         {past, state_bytes},
                         {capture_count, batch * sizeof(int32_t)},
                         {output, input_bytes},
                         {present, state_bytes}}))
      status = hip_varlen_causal_conv_updates(
          stream, input, weight, static_cast<const int32_t *>(cumulative), bias,
          past, static_cast<const int32_t *>(capture_count), output, present,
          updates, tokens, batch, channels, capacity, error, input_dtype);
  }
  return finish(state, stream, error, status,
                "VarlenCausalConvWithState compact capture");
}

int wrap_gated_delta_net_updates(
    RuntimeState *state, void *query, void *key, void *value, void *cumulative,
    void *decay, void *beta, void *past, void *a_log, void *dt_bias,
    void *capture_count, void *active, void *output, void *present,
    void *updates, int64_t tokens, int64_t key_tokens, int64_t value_tokens,
    int64_t decay_tokens, int64_t beta_tokens, int64_t cumulative_count,
    int64_t batch, int64_t present_batch, int64_t output_tokens,
    int64_t capture_batch, int64_t active_count, int64_t update_batch,
    int64_t update_width, int64_t key_heads, int64_t value_heads,
    int64_t capacity, float scale, int64_t input_dtype) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  int status = -1;
  if (tokens > 0 && tokens <= INT_MAX && batch > 0 && batch <= 65535 &&
      batch == present_batch && cumulative_count == batch + 1 &&
      key_tokens == tokens && value_tokens == tokens &&
      decay_tokens == tokens && beta_tokens == tokens &&
      output_tokens == tokens && capture_batch == batch && active_count == 1 &&
      update_batch == batch && capacity > 0 && capacity <= 7 && key_heads > 0 &&
      value_heads >= key_heads && value_heads <= 65535 &&
      value_heads % key_heads == 0 &&
      update_width ==
          capacity * (value_heads + key_heads * 128 + value_heads * 128) &&
      (input_dtype == HIP_DTYPE_FLOAT16 || input_dtype == HIP_DTYPE_FLOAT32)) {
    const uint64_t bytes = input_dtype == HIP_DTYPE_FLOAT32 ? 4 : 2;
    const uint64_t query_bytes = tokens * key_heads * 128 * bytes;
    const uint64_t value_bytes = tokens * value_heads * 128 * bytes;
    const uint64_t state_bytes =
        batch * value_heads * 128 * 128 * sizeof(float);
    const uint64_t gate_bytes = tokens * value_heads * sizeof(float);
    if (disjointCapture(updates, batch * update_width * sizeof(float),
                        {{query, query_bytes},
                         {key, query_bytes},
                         {value, value_bytes},
                         {cumulative, (batch + 1) * sizeof(int32_t)},
                         {decay, gate_bytes},
                         {beta, gate_bytes},
                         {past, state_bytes},
                         {a_log, value_heads * sizeof(float)},
                         {dt_bias, value_heads * sizeof(float)},
                         {capture_count, batch * sizeof(int32_t)},
                         {active, sizeof(int32_t)},
                         {output, value_bytes},
                         {present, state_bytes}}))
      status = hip_gated_delta_net_updates(
          stream, query, key, value, static_cast<const int32_t *>(cumulative),
          static_cast<const float *>(decay), static_cast<const float *>(beta),
          static_cast<const float *>(past), static_cast<const float *>(a_log),
          static_cast<const float *>(dt_bias),
          static_cast<const int32_t *>(capture_count),
          static_cast<const int32_t *>(active), output,
          static_cast<float *>(present), static_cast<float *>(updates), tokens,
          batch, key_heads, value_heads, capacity, scale, error, input_dtype);
  }
  return finish(state, stream, error, status, "GatedDeltaNet compact capture");
}

int wrap_gated_rms_norm(RuntimeState *state, void *input, void *weight,
                        void *gate, void *output, int64_t rows,
                        int64_t gate_rows, int64_t output_rows, int64_t width,
                        float epsilon, int64_t input_dtype) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  int status = -1;
  if (rows == gate_rows && rows == output_rows)
    status = hip_gated_rms_norm(stream, input, weight, gate, output, rows,
                                width, epsilon, error, input_dtype);
  return finish(state, stream, error, status, "GatedRMSNorm");
}

int wrap_mrotary_embedding(RuntimeState *state, void *input, void *positions,
                           void *cos_cache, void *sin_cache, void *output,
                           int64_t tokens, int64_t position_tokens,
                           int64_t cos_length, int64_t sin_length,
                           int64_t output_tokens, int64_t heads) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  int status = -1;
  if (tokens == position_tokens && tokens == output_tokens &&
      cos_length == sin_length)
    status = hip_mrotary_embedding_fp32(
        stream, static_cast<const float *>(input),
        static_cast<const int64_t *>(positions),
        static_cast<const float *>(cos_cache),
        static_cast<const float *>(sin_cache), static_cast<float *>(output),
        tokens, cos_length, heads, error);
  return finish(state, stream, error, status, "MRotaryEmbedding");
}
