/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <type_traits>

namespace {

constexpr int kHeadSize = 64;
constexpr int kMaximumContext = 4096;
constexpr int kThreads = 128;

template <int MaximumContext = kMaximumContext>
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
      prefix > MaximumContext || end - start > MaximumContext - prefix) {
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
  for (int page = blockIdx.y * blockDim.x + threadIdx.x; page < pages;
       page += blockDim.x * gridDim.y) {
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
          other_prefix < 0 || other_prefix > MaximumContext ||
          other_end - other_start > MaximumContext - other_prefix)
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

__global__ void ValidateScales(const float *key, const float *value,
                               int channels, int *error, int residual_bits) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel < channels &&
      (!isfinite(key[channel]) || key[channel] <= 0 ||
       !isfinite(value[channel]) || value[channel] <= 0 ||
       (residual_bits &&
        (key[channel] / 256.0f <= 0 || value[channel] / 256.0f <= 0))))
    atomicExch(error, -1);
}

__device__ void StoreInt8Cache(float value, float scale, int8_t *cache,
                               int residual_bits) {
  if (!residual_bits) {
    cache[0] = static_cast<int8_t>(
        __float2int_rn(fminf(127.0f, fmaxf(-128.0f, value / scale))));
    return;
  }
  const int code = __float2int_rn(
      fminf(32767.0f, fmaxf(-32768.0f, value / (scale / 256.0f))));
  const int high = (code + 32768) / 256 - 128;
  cache[0] = static_cast<int8_t>(high);
  cache[256] = static_cast<int8_t>(code - 256 * high - 128);
}

__device__ float ReadInt8Cache(const int8_t *cache, float scale,
                               int residual_bits) {
  if (!residual_bits)
    return static_cast<float>(cache[0]) * scale;
  const int code = 256 * static_cast<int>(cache[0]) +
                   static_cast<int>(cache[256]) + 128;
  return static_cast<float>(code) * (scale / 256.0f);
}

__device__ int RequestForToken(const int32_t *cumulative, int token, int batch) {
  int request = 0;
  while (request + 1 < batch && token >= cumulative[request + 1])
    ++request;
  return request;
}

__global__ void AppendInt8(
    const float *key, const float *value, int8_t *key_cache,
    int8_t *value_cache, const int32_t *cumulative, const int32_t *past,
    const int32_t *table, const float *key_scale, const float *value_scale,
    int batch, int columns, int page_size, int kv_heads, const int *error,
    int residual_bits) {
  if (*error)
    return;
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int channel = head * 256 + threadIdx.x;
  const int request = RequestForToken(cumulative, token, batch);
  const int position = past[request] + (token - cumulative[request]);
  const int page = table[static_cast<int64_t>(request) * columns + position / page_size];
  const int64_t source = static_cast<int64_t>(token) * kv_heads * 256 + channel;
  const int width = residual_bits ? 512 : 256;
  const int64_t destination =
      ((static_cast<int64_t>(page) * page_size + position % page_size) * kv_heads + head) *
          width + threadIdx.x;
  StoreInt8Cache(key[source], key_scale[channel], key_cache + destination,
                  residual_bits);
  StoreInt8Cache(value[source], value_scale[channel], value_cache + destination,
                  residual_bits);
}

__global__ void PackQuery(const float *query, float *packed, int queries,
                           int heads, int kv_heads, const int *error) {
  if (*error)
    return;
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int groups = heads / kv_heads;
  const int row = (head / groups) * queries * groups + token * groups + head % groups;
  packed[static_cast<int64_t>(row) * 256 + threadIdx.x] =
      query[(static_cast<int64_t>(token) * heads + head) * 256 + threadIdx.x];
}

__global__ void PackInt8Cache(
    const int8_t *key_cache, const int8_t *value_cache, const int32_t *table,
    const float *key_scale, const float *value_scale, float *key, float *value,
    int begin, int count, int page_size, int kv_heads, const int *error,
    int residual_bits) {
  if (*error)
    return;
  const int position = begin + blockIdx.x;
  const int head = blockIdx.y;
  const int dimension = threadIdx.x;
  const int channel = head * 256 + dimension;
  const int page = table[position / page_size];
  const int width = residual_bits ? 512 : 256;
  const int64_t source =
      ((static_cast<int64_t>(page) * page_size + position % page_size) * kv_heads + head) *
          width + dimension;
  const int64_t destination = (static_cast<int64_t>(head) * count + blockIdx.x) * 256 + dimension;
  key[destination] = ReadInt8Cache(key_cache + source, key_scale[channel],
                                    residual_bits);
  value[destination] = ReadInt8Cache(value_cache + source, value_scale[channel],
                                      residual_bits);
}

__global__ void OnlineSoftmax(
    float *scores, float *statistics, int queries, int keys, int heads,
    int kv_heads, int query_position, int key_position, float scale,
    const int *error) {
  if (*error)
    return;
  __shared__ float reduction[kThreads];
  const int row = blockIdx.x;
  const int thread = threadIdx.x;
  const int rows = queries * heads;
  const int query = (row % (queries * (heads / kv_heads))) / (heads / kv_heads);
  float *values = scores + static_cast<int64_t>(row) * keys;
  float maximum = -INFINITY;
  for (int column = thread; column < keys; column += kThreads) {
    const float score = key_position + column <= query_position + query
                            ? values[column] * scale
                            : -INFINITY;
    values[column] = score;
    maximum = fmaxf(maximum, score);
  }
  reduction[thread] = maximum;
  __syncthreads();
  for (int width = kThreads / 2; width; width /= 2) {
    if (thread < width)
      reduction[thread] = fmaxf(reduction[thread], reduction[thread + width]);
    __syncthreads();
  }
  const float previous_maximum = key_position == 0 ? -INFINITY : statistics[row];
  const float previous_sum = key_position == 0 ? 0 : statistics[rows + row];
  maximum = fmaxf(reduction[0], previous_maximum);
  __syncthreads();
  const float correction = expf(previous_maximum - maximum);
  float sum = 0;
  for (int column = thread; column < keys; column += kThreads) {
    values[column] = expf(values[column] - maximum);
    sum += values[column];
  }
  reduction[thread] = sum;
  __syncthreads();
  for (int width = kThreads / 2; width; width /= 2) {
    if (thread < width)
      reduction[thread] += reduction[thread + width];
    __syncthreads();
  }
  if (thread == 0) {
    statistics[row] = maximum;
    statistics[rows + row] = correction * previous_sum + reduction[0];
    statistics[2 * rows + row] = correction;
  }
}

__global__ void MergeAttention(
    const float *partial, float *accumulator, const float *statistics,
    float *output, int queries, int heads, int kv_heads, int first, int last,
    const int *error) {
  if (*error)
    return;
  const int row = blockIdx.x;
  const int dimension = threadIdx.x;
  const int rows = queries * heads;
  const int64_t index = static_cast<int64_t>(row) * 256 + dimension;
  const float result = partial[index] +
      (first ? 0.0f : accumulator[index] * statistics[2 * rows + row]);
  if (last) {
    const int groups = heads / kv_heads;
    const int query = (row % (queries * groups)) / groups;
    const int head = (row / (queries * groups)) * groups + row % groups;
    output[(static_cast<int64_t>(query) * heads + head) * 256 + dimension] =
        result / statistics[rows + row];
  } else {
    accumulator[index] = result;
  }
}

__device__ float Rotate(const __half *input, const __half *cos_cache,
                        const __half *sin_cache, int position, int dimension) {
  const int partner = dimension < 32 ? dimension + 32 : dimension - 32;
  const int cache_offset = position * 32 + dimension % 32;
  const float first = __half2float(input[dimension]) * __half2float(cos_cache[cache_offset]);
  const float second = __half2float(input[partner]) * __half2float(sin_cache[cache_offset]);
  return __half2float(__float2half(first + (dimension < 32 ? -second : second)));
}

template <int HeadSize, bool Separate, typename T = __half>
__global__ void Append(const T *query, const T *key_input,
                       const T *value_input, __half *key_cache,
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
  const int64_t row =
      static_cast<int64_t>(token) * (heads + 2 * kv_heads) * HeadSize;
  const T *key = Separate
      ? key_input + (static_cast<int64_t>(token) * kv_heads + kv_head) * HeadSize
      : query + row + (heads + kv_head) * HeadSize;
  const T *value = Separate
      ? value_input + (static_cast<int64_t>(token) * kv_heads + kv_head) * HeadSize
      : query + row + (heads + kv_heads + kv_head) * HeadSize;
  if constexpr (Separate)
    key_cache[(slot + kv_head) * HeadSize + dimension] = static_cast<__half>(key[dimension]);
  else
    key_cache[(slot + kv_head) * HeadSize + dimension] =
        __float2half(Rotate(key, cos_cache, sin_cache, position, dimension));
  value_cache[(slot + kv_head) * HeadSize + dimension] = static_cast<__half>(value[dimension]);
}

template <int HeadSize, bool Separate, typename T = __half>
__global__ void Attend(
    const T *query, const __half *key_cache, const __half *value_cache,
    const int32_t *cumulative, const int32_t *past, const int32_t *table,
    const __half *cos_cache, const __half *sin_cache, T *output,
    int batch, int columns, int page_size, int heads, int kv_heads,
    float scale, const int *error) {
  if (*error)
    return;
  __shared__ float scores[kMaximumContext];
  __shared__ float q[HeadSize];
  __shared__ float reduction[kThreads];
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int thread = threadIdx.x;
  const int request = RequestForToken(cumulative, token, batch);
  const int position = past[request] + token - cumulative[request];
  const int length = position + 1;
  const int kv_head = head / (heads / kv_heads);
  const int64_t row =
      static_cast<int64_t>(token) *
      (Separate ? heads : heads + 2 * kv_heads) * HeadSize;
  for (int d = thread; d < HeadSize; d += kThreads) {
    if constexpr (Separate)
      q[d] = static_cast<float>(query[row + head * HeadSize + d]);
    else
      q[d] = Rotate(query + row + head * HeadSize, cos_cache, sin_cache,
                     position, d);
  }
  __syncthreads();

  float maximum = -INFINITY;
  for (int key = thread; key < length; key += kThreads) {
    const int physical = table[static_cast<int64_t>(request) * columns + key / page_size];
    const int64_t slot = (static_cast<int64_t>(physical) * page_size + key % page_size) * kv_heads + kv_head;
    float dot = 0;
    for (int dimension = 0; dimension < HeadSize; ++dimension)
      dot += q[dimension] * __half2float(key_cache[slot * HeadSize + dimension]);
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
  for (int d = thread; d < HeadSize; d += kThreads) {
    float value = 0;
    for (int key = 0; key < length; ++key) {
      const int physical = table[static_cast<int64_t>(request) * columns + key / page_size];
      const int64_t slot = (static_cast<int64_t>(physical) * page_size + key % page_size) * kv_heads + kv_head;
      value += scores[key] * __half2float(value_cache[slot * HeadSize + d]);
    }
    output[(static_cast<int64_t>(token) * heads + head) * HeadSize + d] =
        static_cast<T>(value / reduction[0]);
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
  hipLaunchKernelGGL((Validate<>), dim3(batch), dim3(kThreads), 0, hip_stream,
                     cumulative_lengths, past_lengths, block_table,
                     static_cast<int>(tokens), static_cast<int>(batch),
                     static_cast<int>(table_columns), static_cast<int>(num_blocks),
                     static_cast<int>(block_size), static_cast<int>(rotary_length), error);
  auto status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL((Append<64, false, __half>), dim3(tokens, kv_num_heads), dim3(kHeadSize), 0, hip_stream,
                     static_cast<const __half *>(query), nullptr, nullptr,
                     static_cast<__half *>(key_cache), static_cast<__half *>(value_cache),
                     cumulative_lengths, past_lengths, block_table,
                     static_cast<const __half *>(cos_cache), static_cast<const __half *>(sin_cache),
                     static_cast<int>(tokens), static_cast<int>(batch),
                     static_cast<int>(table_columns), static_cast<int>(block_size),
                     static_cast<int>(num_heads), static_cast<int>(kv_num_heads), error);
  status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL((Attend<64, false, __half>), dim3(tokens, num_heads), dim3(kThreads), 0, hip_stream,
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

extern "C" HIP_KERNEL_API int hip_paged_attention_separate(
    void *stream, const void *query, const void *key, const void *value,
    void *key_cache, void *value_cache, const int32_t *cumulative,
    const int32_t *past_lengths, const int32_t *table, void *output,
    int64_t tokens, int64_t batch, int64_t table_columns, int64_t blocks,
    int64_t page_size, int64_t heads, int64_t kv_heads, float scale,
    void *error_flag, int64_t input_dtype) {
  if (!stream || !query || !key || !value || !key_cache || !value_cache ||
      !cumulative || !past_lengths || !table || !output || !error_flag ||
      tokens <= 0 || tokens > INT_MAX || batch <= 0 || batch > tokens ||
      table_columns <= 0 || table_columns > INT_MAX ||
      blocks <= 0 || blocks > INT_MAX || page_size < 16 ||
      page_size > kMaximumContext || (page_size & (page_size - 1)) != 0 ||
      heads <= 0 || heads > 65535 || kv_heads <= 0 || kv_heads > heads ||
      heads % kv_heads != 0 ||
      blocks > INT64_MAX / (page_size * kv_heads * 256 * 2) ||
      !std::isfinite(scale) || scale < 0 ||
      (input_dtype != HIP_DTYPE_FLOAT16 && input_dtype != HIP_DTYPE_FLOAT32))
    return hipErrorInvalidValue;
  auto hip_stream = static_cast<hipStream_t>(stream);
  auto error = static_cast<int *>(error_flag);
  (void)hipGetLastError();
  hipLaunchKernelGGL((Validate<>), dim3(batch), dim3(kThreads), 0, hip_stream,
                     cumulative, past_lengths, table, static_cast<int>(tokens),
                     static_cast<int>(batch), static_cast<int>(table_columns),
                     static_cast<int>(blocks), static_cast<int>(page_size),
                     kMaximumContext, error);
  auto status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  const auto launch = [&](auto *type) {
    using T = std::remove_pointer_t<decltype(type)>;
    hipLaunchKernelGGL(
      (Append<256, true, T>), dim3(tokens, kv_heads), dim3(256), 0, hip_stream,
      static_cast<const T *>(query), static_cast<const T *>(key),
      static_cast<const T *>(value), static_cast<__half *>(key_cache),
      static_cast<__half *>(value_cache), cumulative, past_lengths, table,
      nullptr, nullptr, static_cast<int>(tokens), static_cast<int>(batch),
      static_cast<int>(table_columns), static_cast<int>(page_size),
      static_cast<int>(heads), static_cast<int>(kv_heads), error);
    auto launch_status = hipGetLastError();
    if (launch_status != hipSuccess)
      return launch_status;
    hipLaunchKernelGGL(
      (Attend<256, true, T>), dim3(tokens, heads), dim3(kThreads), 0, hip_stream,
      static_cast<const T *>(query), static_cast<const __half *>(key_cache),
      static_cast<const __half *>(value_cache), cumulative, past_lengths, table,
      nullptr, nullptr, static_cast<T *>(output), static_cast<int>(batch),
      static_cast<int>(table_columns), static_cast<int>(page_size),
      static_cast<int>(heads), static_cast<int>(kv_heads),
      scale == 0 ? 0.0625f : scale, error);
    return hipGetLastError();
  };
  if (input_dtype == HIP_DTYPE_FLOAT32)
    return launch(static_cast<float *>(nullptr));
  return launch(static_cast<__half *>(nullptr));
}

extern "C" HIP_KERNEL_API int hip_paged_int8_append(
    void *stream, const float *key, const float *value, int8_t *key_cache,
    int8_t *value_cache, const int32_t *cumulative, const int32_t *past,
    const int32_t *table, const float *key_scale, const float *value_scale,
    int tokens, int batch, int columns, int blocks, int page_size, int kv_heads,
    int *error, int residual_bits) {
  if (residual_bits != 0 && residual_bits != 8)
    return hipErrorInvalidValue;
  const auto hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  const int validation_blocks =
      (std::min(columns, 262144 / page_size) + kThreads - 1) / kThreads;
  hipLaunchKernelGGL((Validate<262144>), dim3(batch, validation_blocks),
                    dim3(kThreads), 0,
                    hip_stream, cumulative, past, table, tokens, batch,
                    columns, blocks, page_size, 262144, error);
  auto status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(ValidateScales, dim3(kv_heads), dim3(256), 0, hip_stream,
                    key_scale, value_scale, kv_heads * 256, error,
                    residual_bits);
  status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(AppendInt8, dim3(tokens, kv_heads), dim3(256), 0,
                    hip_stream, key, value, key_cache, value_cache,
                    cumulative, past, table, key_scale, value_scale, batch,
                    columns, page_size, kv_heads, error, residual_bits);
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_paged_int8_pack_query(
    void *stream, const float *query, float *packed, int queries, int heads,
    int kv_heads, int *error) {
  (void)hipGetLastError();
  hipLaunchKernelGGL(PackQuery, dim3(queries, heads), dim3(256), 0,
                    static_cast<hipStream_t>(stream), query, packed,
                    queries, heads, kv_heads, error);
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_paged_int8_pack_cache(
    void *stream, const int8_t *key_cache, const int8_t *value_cache,
    const int32_t *table, const float *key_scale, const float *value_scale,
    float *key, float *value, int begin, int count, int page_size, int kv_heads,
    int *error, int residual_bits) {
  if (residual_bits != 0 && residual_bits != 8)
    return hipErrorInvalidValue;
  (void)hipGetLastError();
  hipLaunchKernelGGL(PackInt8Cache, dim3(count, kv_heads), dim3(256), 0,
                    static_cast<hipStream_t>(stream), key_cache, value_cache,
                    table, key_scale, value_scale, key, value, begin, count,
                    page_size, kv_heads, error, residual_bits);
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_paged_int8_softmax(
    void *stream, float *scores, float *statistics, int queries, int keys,
    int heads, int kv_heads, int query_position, int key_position, float scale,
    int *error) {
  (void)hipGetLastError();
  hipLaunchKernelGGL(OnlineSoftmax, dim3(queries * heads), dim3(kThreads), 0,
                    static_cast<hipStream_t>(stream), scores, statistics,
                    queries, keys, heads, kv_heads, query_position,
                    key_position, scale, error);
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_paged_int8_merge(
    void *stream, const float *partial, float *accumulator,
    const float *statistics, float *output, int queries, int heads,
    int kv_heads, int first, int last, int *error) {
  (void)hipGetLastError();
  hipLaunchKernelGGL(MergeAttention, dim3(queries * heads), dim3(256), 0,
                    static_cast<hipStream_t>(stream), partial, accumulator,
                    statistics, output, queries, heads, kv_heads, first,
                    last, error);
  return hipGetLastError();
}
