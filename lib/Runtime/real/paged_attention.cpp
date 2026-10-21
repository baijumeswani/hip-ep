/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../hipdnn_ep_runtime.h"
#include "hip_custom_kernels.h"
#include "runtime_types.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

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

int wrap_paged_attention_int8(
    RuntimeState *state, int op_state_slot, void *query, void *key, void *value,
    void *past_key, void *past_value, void *cumulative, void *past_lengths,
    void *table, void *key_scale, void *value_scale, void *output,
    void *present_key, void *present_value, int64_t tokens, int64_t key_tokens,
    int64_t value_tokens, int64_t output_tokens, int64_t batch,
    int64_t cumulative_count, int64_t table_rows, int64_t table_columns,
    int64_t key_blocks, int64_t key_page_size, int64_t value_blocks,
    int64_t value_page_size, int64_t present_key_blocks,
    int64_t present_key_page_size, int64_t present_value_blocks,
    int64_t present_value_page_size, int64_t heads, int64_t kv_heads,
    float scale, int64_t residual_bits) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto *error =
      static_cast<int *>(hipdnn_ep_state_get_error_flag_device_ptr(state));
  if (!stream || !error)
    return -1;
  const auto fail = [&](const char *message) {
    std::fprintf(stderr, "HIP INT8 PagedAttention: %s\n", message);
    return RecordFailure(stream, error);
  };
  const auto execute = [state](int status) {
    hipdnn_ep_state_record_execution_status(state, status);
    return status;
  };
  if (!query || !key || !value || !past_key || !past_value || !cumulative ||
      !past_lengths || !table || !key_scale || !value_scale || !output ||
      tokens <= 0 || tokens > INT_MAX || batch <= 0 || batch > tokens ||
      batch > 65535 || cumulative_count != batch + 1 || table_rows != batch ||
      key_tokens != tokens || value_tokens != tokens ||
      output_tokens != tokens || table_columns <= 0 ||
      table_columns > INT_MAX || key_blocks <= 0 || key_blocks > INT_MAX ||
      key_page_size < 16 || key_page_size > 4096 ||
      (key_page_size & (key_page_size - 1)) != 0 || heads <= 0 ||
      heads > 65535 || kv_heads <= 0 || kv_heads > heads ||
      heads % kv_heads != 0 ||
      key_blocks > INT64_MAX / (key_page_size * kv_heads * 256) ||
      (residual_bits != 0 && residual_bits != 8) || !std::isfinite(scale) ||
      scale < 0 || value_blocks != key_blocks ||
      value_page_size != key_page_size || present_key_blocks != key_blocks ||
      present_value_blocks != key_blocks ||
      present_key_page_size != key_page_size ||
      present_value_page_size != key_page_size || past_key != present_key ||
      past_value != present_value)
    return fail("invalid geometry, metadata extents or caller-bound aliases");
  const int cache_components = residual_bits ? 2 : 1;
  if (key_blocks >
      INT64_MAX / (key_page_size * kv_heads * 256 * cache_components))
    return fail("cache byte extent overflows int64");

  const bool decode = tokens == batch;
  // Long KV tiles amortize short-query launches; prefill tiles stay within
  // the same fixed workspace ceiling.
  int query_tile =
      static_cast<int>(std::min<int64_t>(tokens, decode ? 1 : 512));
  int key_tile = tokens <= 32 ? 16384 : 3072;
  const auto workspace_elements = [&](int queries, int keys) {
    return 3 * static_cast<size_t>(heads) * queries * 256 +
           2 * static_cast<size_t>(kv_heads) * keys * 256 +
           static_cast<size_t>(heads) * queries * keys +
           3 * static_cast<size_t>(heads) * queries;
  };
  constexpr size_t workspace_limit = 256 * 1024 * 1024;
  while (query_tile > 64 && workspace_elements(query_tile, key_tile) >
                                workspace_limit / sizeof(float))
    query_tile /= 2;
  while (key_tile > 4096 && workspace_elements(query_tile, key_tile) >
                                workspace_limit / sizeof(float))
    key_tile /= 2;
  const size_t query_elements = static_cast<size_t>(heads) * query_tile * 256;
  const size_t cache_elements = static_cast<size_t>(kv_heads) * key_tile * 256;
  const size_t score_elements =
      static_cast<size_t>(heads) * query_tile * key_tile;
  const size_t elements = workspace_elements(query_tile, key_tile);
  if (elements > workspace_limit / sizeof(float))
    return fail("attention geometry exceeds the bounded workspace");
  auto *scratch =
      static_cast<float *>(hipdnn_ep_state_ensure_paged_attention_scratch(
          state, elements * sizeof(float)));
  if (!scratch)
    return fail("cannot reserve bounded attention workspace");
  float *packed_query = scratch;
  float *packed_key = packed_query + query_elements;
  float *packed_value = packed_key + cache_elements;
  float *scores = packed_value + cache_elements;
  float *partial = scores + score_elements;
  float *accumulator = partial + query_elements;
  float *statistics = accumulator + query_elements;
  if (execute(hip_paged_int8_append(
          stream, static_cast<const float *>(key),
          static_cast<const float *>(value), static_cast<int8_t *>(past_key),
          static_cast<int8_t *>(past_value),
          static_cast<const int32_t *>(cumulative),
          static_cast<const int32_t *>(past_lengths),
          static_cast<const int32_t *>(table),
          static_cast<const float *>(key_scale),
          static_cast<const float *>(value_scale), static_cast<int>(tokens),
          static_cast<int>(batch), static_cast<int>(table_columns),
          static_cast<int>(key_blocks), static_cast<int>(key_page_size),
          static_cast<int>(kv_heads), error,
          static_cast<int>(residual_bits))) != 0)
    return fail("metadata/scale validation or cache append launch failed");

  // Only tiny control vectors cross to the host. Cache and attention tensors
  // stay on device.
  std::vector<int32_t> lengths(static_cast<size_t>(2 * batch + 1));
  int execution_error = 0;
  const auto hip_stream = static_cast<hipStream_t>(stream);
  const auto cumulative_bytes =
      static_cast<size_t>(batch + 1) * sizeof(int32_t);
  const auto past_bytes = static_cast<size_t>(batch) * sizeof(int32_t);
  const auto cumulative_status =
      hipMemcpyAsync(lengths.data(), cumulative, cumulative_bytes,
                     hipMemcpyDeviceToHost, hip_stream);
  const auto past_status =
      hipMemcpyAsync(lengths.data() + batch + 1, past_lengths, past_bytes,
                     hipMemcpyDeviceToHost, hip_stream);
  const auto error_status =
      hipMemcpyAsync(&execution_error, error, sizeof(execution_error),
                     hipMemcpyDeviceToHost, hip_stream);
  if (hipStreamSynchronize(hip_stream) != hipSuccess) {
    std::fprintf(stderr,
                 "HIP INT8 PagedAttention cannot drain host readback\n");
    std::abort();
  }
  if (cumulative_status != hipSuccess || past_status != hipSuccess ||
      error_status != hipSuccess) {
    hipdnn_ep_state_record_execution_status(
        state, HIPDNN_EP_ERR_DEVICE_EXECUTION_FAILED);
    return fail("metadata readback failed");
  }
  if (execution_error)
    return fail("invalid page metadata or nonpositive/nonfinite cache scales");

  const int groups = static_cast<int>(heads / kv_heads);
  for (int request = 0; request < batch; ++request) {
    const int start = lengths[request];
    const int end = lengths[request + 1];
    const int prefix = lengths[static_cast<size_t>(batch + 1 + request)];
    for (int begin = start; begin < end;) {
      const int queries = std::min(query_tile, end - begin);
      const int rows = queries * groups;
      const int total = prefix + (begin - start) + queries;
      const auto *query_begin = static_cast<const float *>(query) +
                                static_cast<int64_t>(begin) * heads * 256;
      auto *output_begin = static_cast<float *>(output) +
                           static_cast<int64_t>(begin) * heads * 256;
      if (execute(hip_paged_int8_pack_query(
              stream, query_begin, packed_query, queries,
              static_cast<int>(heads), static_cast<int>(kv_heads), error)) != 0)
        return fail("query packing failed");
      for (int key_begin = 0; key_begin < total; key_begin += key_tile) {
        const int keys = std::min(key_tile, total - key_begin);
        if (execute(hip_paged_int8_pack_cache(
                stream, static_cast<const int8_t *>(past_key),
                static_cast<const int8_t *>(past_value),
                static_cast<const int32_t *>(table) +
                    static_cast<int64_t>(request) * table_columns,
                static_cast<const float *>(key_scale),
                static_cast<const float *>(value_scale), packed_key,
                packed_value, key_begin, keys, static_cast<int>(key_page_size),
                static_cast<int>(kv_heads), error,
                static_cast<int>(residual_bits))) != 0 ||
            wrap_hipblasLtMatmul(state, op_state_slot, packed_query, packed_key,
                                 scores, rows, keys, 256, kv_heads,
                                 HIPDNN_EP_DATATYPE_FLOAT,
                                 static_cast<int64_t>(keys) * 256, 0, 1) != 0 ||
            execute(hip_paged_int8_softmax(
                stream, scores, statistics, queries, keys,
                static_cast<int>(heads), static_cast<int>(kv_heads),
                prefix + (begin - start), key_begin,
                scale == 0 ? 0.0625f : scale, error)) != 0 ||
            wrap_hipblasLtMatmul(state, op_state_slot, scores, packed_value,
                                 partial, rows, 256, keys, kv_heads,
                                 HIPDNN_EP_DATATYPE_FLOAT,
                                 static_cast<int64_t>(keys) * 256, 0, 0) != 0 ||
            execute(hip_paged_int8_merge(
                stream, partial, accumulator, statistics, output_begin, queries,
                static_cast<int>(heads), static_cast<int>(kv_heads),
                key_begin == 0, key_begin + keys == total, error)) != 0)
          return fail("tiled attention execution failed");
      }
      begin += queries;
    }
  }
  return 0;
}

static int RunPagedAttentionBf16(
    RuntimeState *state, void *query, void *key, void *value, void *past_key,
    void *past_value, void *cumulative, void *past_lengths, void *table,
    void *cos_cache, void *sin_cache, void *q_norm_weight, void *k_norm_weight,
    void *output, void *present_key, void *present_value, int64_t tokens,
    int64_t key_tokens, int64_t value_tokens, int64_t output_tokens,
    int64_t batch, int64_t cumulative_count, int64_t table_rows,
    int64_t table_columns, int64_t key_blocks, int64_t key_page_size,
    int64_t value_blocks, int64_t value_page_size, int64_t present_key_blocks,
    int64_t present_key_page_size, int64_t present_value_blocks,
    int64_t present_value_page_size, int64_t rotary_length, int64_t sin_length,
    int64_t heads, int64_t kv_heads, int64_t local_window_size, float scale,
    float qk_norm_epsilon, bool separate) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  const auto fail = [&](const char *message) {
    std::fprintf(stderr, "HIP BF16 PagedAttention: %s\n", message);
    return RecordFailure(stream, error);
  };
  if (!query ||
      (separate &&
       (!key || !value || key_tokens != tokens || value_tokens != tokens)) ||
      tokens <= 0 || tokens > INT_MAX || output_tokens != tokens ||
      batch <= 0 || batch > tokens || batch > 65535 ||
      cumulative_count != batch + 1 || table_rows != batch ||
      table_columns <= 0 || table_columns > INT_MAX || key_blocks <= 0 ||
      key_blocks > INT_MAX || key_page_size < 16 || key_page_size > 4096 ||
      (key_page_size & (key_page_size - 1)) != 0 ||
      value_blocks != key_blocks || value_page_size != key_page_size ||
      present_key_blocks != key_blocks || present_value_blocks != key_blocks ||
      present_key_page_size != key_page_size ||
      present_value_page_size != key_page_size || past_key != present_key ||
      past_value != present_value || rotary_length <= 0 ||
      rotary_length > INT_MAX || sin_length != rotary_length || heads <= 0 ||
      heads > 65535 || kv_heads <= 0 || kv_heads > heads ||
      heads + kv_heads > 65535 || heads % kv_heads != 0 ||
      local_window_size <= 0 || local_window_size > 4096 ||
      !std::isfinite(scale) || scale < 0 || !std::isfinite(qk_norm_epsilon) ||
      qk_norm_epsilon <= 0 ||
      key_blocks > INT64_MAX / (key_page_size * kv_heads * 128 * 2) ||
      static_cast<uint64_t>(tokens) >
          std::numeric_limits<size_t>::max() / sizeof(uint16_t) / heads / 128)
    return fail("invalid geometry, metadata extents or caller-bound aliases");
  auto *query_scratch = hipdnn_ep_state_ensure_paged_attention_scratch(
      state, static_cast<size_t>(tokens) * heads * 128 * sizeof(uint16_t));
  if (!query_scratch)
    return fail("cannot reserve packed BF16 query workspace");
  const auto *packed = static_cast<const uint16_t *>(query);
  const int query_stride =
      static_cast<int>(separate ? heads : heads + 2 * kv_heads) * 128;
  const int kv_stride =
      separate ? static_cast<int>(kv_heads) * 128 : query_stride;
  const auto status = hip_paged_attention_bf16_strided(
      stream, query, separate ? key : packed + heads * 128,
      separate ? value : packed + (heads + kv_heads) * 128, past_key,
      past_value, static_cast<const int32_t *>(cumulative),
      static_cast<const int32_t *>(past_lengths),
      static_cast<const int32_t *>(table), cos_cache, sin_cache, q_norm_weight,
      k_norm_weight, query_scratch, output, static_cast<int>(tokens),
      static_cast<int>(batch), static_cast<int>(table_columns),
      static_cast<int>(key_blocks), static_cast<int>(key_page_size),
      static_cast<int>(heads), static_cast<int>(kv_heads), query_stride,
      kv_stride, static_cast<int>(rotary_length),
      static_cast<int>(local_window_size), scale, qk_norm_epsilon, error);
  hipdnn_ep_state_record_execution_status(state, status);
  if (status != 0)
    return fail("metadata validation, QK normalization/RoPE or attention "
                "launch failed");
  return 0;
}

int wrap_paged_attention_bf16(
    RuntimeState *state, void *query, void *past_key, void *past_value,
    void *cumulative, void *past_lengths, void *table, void *cos_cache,
    void *sin_cache, void *q_norm_weight, void *k_norm_weight, void *output,
    void *present_key, void *present_value, int64_t tokens,
    int64_t output_tokens, int64_t batch, int64_t cumulative_count,
    int64_t table_rows, int64_t table_columns, int64_t key_blocks,
    int64_t key_page_size, int64_t value_blocks, int64_t value_page_size,
    int64_t present_key_blocks, int64_t present_key_page_size,
    int64_t present_value_blocks, int64_t present_value_page_size,
    int64_t rotary_length, int64_t sin_length, int64_t heads, int64_t kv_heads,
    int64_t local_window_size, float scale, float qk_norm_epsilon) {
  return RunPagedAttentionBf16(
      state, query, nullptr, nullptr, past_key, past_value, cumulative,
      past_lengths, table, cos_cache, sin_cache, q_norm_weight, k_norm_weight,
      output, present_key, present_value, tokens, tokens, tokens, output_tokens,
      batch, cumulative_count, table_rows, table_columns, key_blocks,
      key_page_size, value_blocks, value_page_size, present_key_blocks,
      present_key_page_size, present_value_blocks, present_value_page_size,
      rotary_length, sin_length, heads, kv_heads, local_window_size, scale,
      qk_norm_epsilon, false);
}

int wrap_paged_attention_bf16_separate(
    RuntimeState *state, void *query, void *key, void *value, void *past_key,
    void *past_value, void *cumulative, void *past_lengths, void *table,
    void *cos_cache, void *sin_cache, void *q_norm_weight, void *k_norm_weight,
    void *output, void *present_key, void *present_value, int64_t tokens,
    int64_t key_tokens, int64_t value_tokens, int64_t output_tokens,
    int64_t batch, int64_t cumulative_count, int64_t table_rows,
    int64_t table_columns, int64_t key_blocks, int64_t key_page_size,
    int64_t value_blocks, int64_t value_page_size, int64_t present_key_blocks,
    int64_t present_key_page_size, int64_t present_value_blocks,
    int64_t present_value_page_size, int64_t rotary_length, int64_t sin_length,
    int64_t heads, int64_t kv_heads, int64_t local_window_size, float scale,
    float qk_norm_epsilon) {
  return RunPagedAttentionBf16(
      state, query, key, value, past_key, past_value, cumulative, past_lengths,
      table, cos_cache, sin_cache, q_norm_weight, k_norm_weight, output,
      present_key, present_value, tokens, key_tokens, value_tokens,
      output_tokens, batch, cumulative_count, table_rows, table_columns,
      key_blocks, key_page_size, value_blocks, value_page_size,
      present_key_blocks, present_key_page_size, present_value_blocks,
      present_value_page_size, rotary_length, sin_length, heads, kv_heads,
      local_window_size, scale, qk_norm_epsilon, true);
}

int wrap_paged_attention_separate(
    RuntimeState *state, void *query, void *key, void *value, void *past_key,
    void *past_value, void *cumulative, void *past_lengths, void *table,
    void *output, void *present_key, void *present_value, int64_t tokens,
    int64_t key_tokens, int64_t value_tokens, int64_t output_tokens,
    int64_t batch, int64_t cumulative_count, int64_t table_rows,
    int64_t table_columns, int64_t key_blocks, int64_t key_page_size,
    int64_t value_blocks, int64_t value_page_size, int64_t present_key_blocks,
    int64_t present_key_page_size, int64_t present_value_blocks,
    int64_t present_value_page_size, int64_t heads, int64_t kv_heads,
    float scale, int64_t input_dtype) {
  if (!state)
    return -1;
  auto stream = hipdnn_ep_state_get_stream(state);
  auto error = hipdnn_ep_state_get_error_flag_device_ptr(state);
  if (!stream || !error)
    return -1;
  if (batch <= 0 || batch >= INT64_MAX || cumulative_count != batch + 1 ||
      table_rows != batch || key_tokens != tokens || value_tokens != tokens ||
      output_tokens != tokens || value_blocks != key_blocks ||
      value_page_size != key_page_size || present_key_blocks != key_blocks ||
      present_value_blocks != key_blocks ||
      present_key_page_size != key_page_size ||
      present_value_page_size != key_page_size || past_key != present_key ||
      past_value != present_value) {
    std::fprintf(stderr,
                 "HIP separate PagedAttention requires matching metadata, "
                 "Q/K/V and caller-bound cache aliases\n");
    return RecordFailure(stream, error);
  }
  int status = hip_paged_attention_separate(
      stream, query, key, value, past_key, past_value,
      static_cast<const int32_t *>(cumulative),
      static_cast<const int32_t *>(past_lengths),
      static_cast<const int32_t *>(table), output, tokens, batch, table_columns,
      key_blocks, key_page_size, heads, kv_heads, scale, error, input_dtype);
  hipdnn_ep_state_record_execution_status(state, status);
  if (status != 0) {
    std::fprintf(stderr, "HIP separate PagedAttention failed: %d\n", status);
    return RecordFailure(stream, error);
  }
  return 0;
}

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
  hipdnn_ep_state_record_execution_status(state, status);
  if (status != 0) {
    std::fprintf(stderr, "HIP FP16 PagedAttention failed: %d\n", status);
    return RecordFailure(stream, error_flag);
  }
  return status;
}
