/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>

namespace {

constexpr int kHeadSize = 64;
constexpr int kMaximumContext = 4096;
constexpr int kThreads = 128;

__global__ void Validate(const int32_t *cumulative, const int32_t *past,
                         const int32_t *table, int tokens, int batch,
                         int columns, int blocks, int page_size,
                         int rotary_length, int *error) {
  const int request = blockIdx.x;
  const int start = cumulative[request];
  const int end = cumulative[request + 1];
  const int prefix = past[request];
  if (cumulative[0] != 0 || cumulative[batch] != tokens || start < 0 ||
      end <= start || end > tokens || prefix < 0 ||
      prefix > kMaximumContext || end - start > kMaximumContext - prefix) {
    if (threadIdx.x == 0)
      atomicExch(error, -1);
    return;
  }
  const int length = prefix + end - start;
  if (length > rotary_length ||
      static_cast<int64_t>(length) > static_cast<int64_t>(columns) * page_size) {
    if (threadIdx.x == 0)
      atomicExch(error, -1);
    return;
  }
  const int pages = (length - 1) / page_size + 1;
  for (int page = threadIdx.x; page < pages; page += blockDim.x) {
    const int physical = table[static_cast<int64_t>(request) * columns + page];
    if (physical < 0 || physical >= blocks)
      atomicExch(error, -1);
    for (int previous = 0; previous < page; ++previous) {
      if (physical == table[static_cast<int64_t>(request) * columns + previous])
        atomicExch(error, -1);
    }
    const int write_begin = max(0, prefix - page * page_size);
    const int write_end = min(page_size, length - page * page_size);
    if (write_begin >= write_end)
      continue;
    for (int other = 0; other < batch; ++other) {
      if (other == request)
        continue;
      const int other_start = cumulative[other];
      const int other_end = cumulative[other + 1];
      const int other_prefix = past[other];
      if (other_start < 0 || other_end <= other_start || other_end > tokens ||
          other_prefix < 0 || other_prefix > kMaximumContext ||
          other_end - other_start > kMaximumContext - other_prefix)
        continue;
      const int other_length = other_prefix + other_end - other_start;
      const int other_pages = (other_length - 1) / page_size + 1;
      if (other_pages > columns)
        continue;
      for (int other_page = 0; other_page < other_pages; ++other_page) {
        if (physical == table[static_cast<int64_t>(other) * columns + other_page] &&
            write_begin < min(page_size, other_length - other_page * page_size))
          atomicExch(error, -1);
      }
    }
  }
}

__device__ int RequestForToken(const int32_t *cumulative, int token, int batch) {
  int request = 0;
  while (request + 1 < batch && token >= cumulative[request + 1])
    ++request;
  return request;
}

__device__ float Rotate(const __half *input, const __half *cos_cache,
                        const __half *sin_cache, int position, int dimension) {
  const int partner = dimension < 32 ? dimension + 32 : dimension - 32;
  const int cache_offset = position * 32 + dimension % 32;
  const float first = __half2float(input[dimension]) * __half2float(cos_cache[cache_offset]);
  const float second = __half2float(input[partner]) * __half2float(sin_cache[cache_offset]);
  return __half2float(__float2half(first + (dimension < 32 ? -second : second)));
}

__global__ void Append(const __half *query, __half *key_cache,
                       __half *value_cache, const int32_t *cumulative,
                       const int32_t *past, const int32_t *table,
                       const __half *cos_cache, const __half *sin_cache,
                       int tokens, int batch, int columns, int page_size,
                       int heads, int kv_heads, const int *error) {
  if (*error)
    return;
  const int token = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int dimension = threadIdx.x;
  const int request = RequestForToken(cumulative, token, batch);
  const int position = past[request] + token - cumulative[request];
  const int physical = table[static_cast<int64_t>(request) * columns + position / page_size];
  const int64_t slot = (static_cast<int64_t>(physical) * page_size + position % page_size) * kv_heads;
  const int64_t row = static_cast<int64_t>(token) * (heads + 2 * kv_heads) * kHeadSize;
  const __half *key = query + row + (heads + kv_head) * kHeadSize;
  const __half *value = query + row + (heads + kv_heads + kv_head) * kHeadSize;
  key_cache[(slot + kv_head) * kHeadSize + dimension] =
      __float2half(Rotate(key, cos_cache, sin_cache, position, dimension));
  value_cache[(slot + kv_head) * kHeadSize + dimension] = value[dimension];
}

__global__ void Attend(
    const __half *query, const __half *key_cache, const __half *value_cache,
    const int32_t *cumulative, const int32_t *past, const int32_t *table,
    const __half *cos_cache, const __half *sin_cache, __half *output,
    int batch, int columns, int page_size, int heads, int kv_heads,
    float scale, const int *error) {
  if (*error)
    return;
  __shared__ float scores[kMaximumContext];
  __shared__ float q[kHeadSize];
  __shared__ float reduction[kThreads];
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int thread = threadIdx.x;
  const int request = RequestForToken(cumulative, token, batch);
  const int position = past[request] + token - cumulative[request];
  const int length = position + 1;
  const int kv_head = head / (heads / kv_heads);
  const int64_t row = static_cast<int64_t>(token) * (heads + 2 * kv_heads) * kHeadSize;
  if (thread < kHeadSize)
    q[thread] = Rotate(query + row + head * kHeadSize, cos_cache, sin_cache, position, thread);
  __syncthreads();

  float maximum = -INFINITY;
  for (int key = thread; key < length; key += kThreads) {
    const int physical = table[static_cast<int64_t>(request) * columns + key / page_size];
    const int64_t slot = (static_cast<int64_t>(physical) * page_size + key % page_size) * kv_heads + kv_head;
    float dot = 0;
    for (int dimension = 0; dimension < kHeadSize; ++dimension)
      dot += q[dimension] * __half2float(key_cache[slot * kHeadSize + dimension]);
    scores[key] = dot * scale;
    maximum = fmaxf(maximum, scores[key]);
  }
  reduction[thread] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width /= 2) {
    if (thread < width)
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + width]);
    __syncthreads();
  }
  maximum = reduction[0];
  __syncthreads();
  float sum = 0;
  for (int key = thread; key < length; key += kThreads) {
    scores[key] = expf(scores[key] - maximum);
    sum += scores[key];
  }
  reduction[thread] = sum;
  __syncthreads();
  for (int width = kThreads / 2; width > 0; width /= 2) {
    if (thread < width)
      reduction[thread] += reduction[thread + width];
    __syncthreads();
  }
  if (thread < kHeadSize) {
    float value = 0;
    for (int key = 0; key < length; ++key) {
      const int physical = table[static_cast<int64_t>(request) * columns + key / page_size];
      const int64_t slot = (static_cast<int64_t>(physical) * page_size + key % page_size) * kv_heads + kv_head;
      value += scores[key] * __half2float(value_cache[slot * kHeadSize + thread]);
    }
    output[(static_cast<int64_t>(token) * heads + head) * kHeadSize + thread] =
        __float2half(value / reduction[0]);
  }
}

} // namespace

extern "C" HIP_KERNEL_API int hip_paged_attention_fp16(
    void *stream, const void *query, void *key_cache, void *value_cache,
    const int32_t *cumulative_lengths, const int32_t *past_lengths,
    const int32_t *block_table, const void *cos_cache, const void *sin_cache,
    void *output, int64_t tokens, int64_t batch, int64_t table_columns,
    int64_t num_blocks, int64_t block_size, int64_t num_heads,
    int64_t kv_num_heads, int64_t rotary_length, float scale, void *error_flag) {
  if (!stream || !query || !key_cache || !value_cache || !cumulative_lengths ||
      !past_lengths || !block_table || !cos_cache || !sin_cache || !output ||
      !error_flag || tokens <= 0 || tokens > INT_MAX || batch <= 0 ||
      batch > tokens || table_columns <= 0 || table_columns > INT_MAX ||
      num_blocks <= 0 || num_blocks > INT_MAX || block_size < 16 ||
      block_size > kMaximumContext || (block_size & (block_size - 1)) != 0 ||
      num_heads <= 0 || num_heads > 65535 || kv_num_heads <= 0 ||
      kv_num_heads > num_heads || num_heads % kv_num_heads != 0 ||
      num_blocks > INT64_MAX / (block_size * kv_num_heads * kHeadSize * 2) ||
      rotary_length <= 0 || rotary_length > INT_MAX / 32 ||
      !std::isfinite(scale) || scale < 0)
    return hipErrorInvalidValue;
  auto hip_stream = static_cast<hipStream_t>(stream);
  auto error = static_cast<int *>(error_flag);
  (void)hipGetLastError();
  hipLaunchKernelGGL(Validate, dim3(batch), dim3(kThreads), 0, hip_stream,
                     cumulative_lengths, past_lengths, block_table,
                     static_cast<int>(tokens), static_cast<int>(batch),
                     static_cast<int>(table_columns), static_cast<int>(num_blocks),
                     static_cast<int>(block_size), static_cast<int>(rotary_length), error);
  auto status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(Append, dim3(tokens, kv_num_heads), dim3(kHeadSize), 0, hip_stream,
                     static_cast<const __half *>(query),
                     static_cast<__half *>(key_cache), static_cast<__half *>(value_cache),
                     cumulative_lengths, past_lengths, block_table,
                     static_cast<const __half *>(cos_cache), static_cast<const __half *>(sin_cache),
                     static_cast<int>(tokens), static_cast<int>(batch),
                     static_cast<int>(table_columns), static_cast<int>(block_size),
                     static_cast<int>(num_heads), static_cast<int>(kv_num_heads), error);
  status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(Attend, dim3(tokens, num_heads), dim3(kThreads), 0, hip_stream,
                     static_cast<const __half *>(query),
                     static_cast<const __half *>(key_cache), static_cast<const __half *>(value_cache),
                     cumulative_lengths, past_lengths, block_table,
                     static_cast<const __half *>(cos_cache), static_cast<const __half *>(sin_cache),
                     static_cast<__half *>(output),
                     static_cast<int>(batch), static_cast<int>(table_columns),
                     static_cast<int>(block_size), static_cast<int>(num_heads),
                     static_cast<int>(kv_num_heads), scale == 0 ? 0.125f : scale, error);
  return hipGetLastError();
}
