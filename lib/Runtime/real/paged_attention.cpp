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

int RecordFailure(void *stream, void *error_flag) {
  const auto status = hipMemsetAsync(error_flag, 0xff, sizeof(int),
                                     static_cast<hipStream_t>(stream));
  if (status != hipSuccess) {
    std::fprintf(stderr,
                 "HIP PagedAttention cannot record execution failure: %s\n",
                 hipGetErrorString(status));
    std::abort();
  }
  return -1;
}

} // namespace

int wrap_paged_attention(RuntimeState *state, void *query, void *past_key,
                         void *past_value, void *cumulative_lengths,
                         void *past_lengths, void *block_table, void *cos_cache,
                         void *sin_cache, void *output, void *present_key,
                         void *present_value, int64_t tokens, int64_t batch,
                         int64_t table_columns, int64_t num_blocks,
                         int64_t block_size, int64_t num_heads,
                         int64_t kv_num_heads, int64_t rotary_length,
                         int64_t cumulative_count, int64_t table_rows,
                         int64_t value_blocks, int64_t value_block_size,
                         int64_t sin_length, float scale) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error_flag = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error_flag)
    return -1;
  if (batch <= 0 || batch >= INT64_MAX || cumulative_count != batch + 1 ||
      table_rows != batch || value_blocks != num_blocks ||
      value_block_size != block_size || sin_length != rotary_length ||
      past_key != present_key || past_value != present_value) {
    std::fprintf(stderr,
                 "HIP PagedAttention requires matching metadata/cache extents "
                 "and caller-bound past/present cache aliases\n");
    return RecordFailure(stream, error_flag);
  }
  const int status = hip_paged_attention_fp16(
      stream, query, past_key, past_value,
      static_cast<const int32_t *>(cumulative_lengths),
      static_cast<const int32_t *>(past_lengths),
      static_cast<const int32_t *>(block_table), cos_cache, sin_cache, output,
      tokens, batch, table_columns, num_blocks, block_size, num_heads,
      kv_num_heads, rotary_length, scale, error_flag);
  if (status != 0) {
    std::fprintf(stderr, "HIP FP16 PagedAttention failed: %d\n", status);
    return RecordFailure(stream, error_flag);
  }
  return status;
}
