/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <cstdio>
#include <cstdlib>

namespace {

int finish(void *stream, void *error, int status, const char *operation) {
  if (status == 0)
    return 0;
  std::fprintf(stderr, "HIP %s failed: %d\n", operation, status);
  if (hipMemsetAsync(error, 0xff, sizeof(int),
                     static_cast<hipStream_t>(stream)) != hipSuccess) {
    std::fprintf(stderr, "HIP %s cannot record execution failure\n", operation);
    std::abort();
  }
  return -1;
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
  return finish(stream, error, status, "VarlenCausalConvWithState");
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
  return finish(stream, error, status, "GatedDeltaNet");
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
  return finish(stream, error, status, "GatedRMSNorm");
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
  return finish(stream, error, status, "MRotaryEmbedding");
}
