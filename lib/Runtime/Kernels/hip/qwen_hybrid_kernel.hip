/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <climits>
#include <cmath>
#include <type_traits>

namespace {

constexpr int kThreads = 256;

__global__ void ValidateLengths(const int32_t *cu, int tokens, int batch,
                                int *error) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b < batch &&
      (cu[0] != 0 || cu[batch] != tokens || cu[b] < 0 ||
       cu[b + 1] < cu[b] || cu[b + 1] > tokens))
    atomicExch(error, -1);
}

template <typename T>
__global__ void CausalConv(
    const T *input, const T *weight, const int32_t *cu,
    const T *bias, const T *past, T *output, T *present,
    int channels, const int *error) {
  if (*error)
    return;
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels)
    return;
  int b = blockIdx.y;
  int64_t offset = (static_cast<int64_t>(b) * channels + c) * 3;
  T history[3] = {past[offset], past[offset + 1], past[offset + 2]};
  for (int t = cu[b]; t < cu[b + 1]; ++t) {
    int64_t index = static_cast<int64_t>(t) * channels + c;
    T current = input[index];
    float sum = static_cast<float>(bias[c]);
    for (int j = 0; j < 3; ++j)
      sum += static_cast<float>(history[j]) * static_cast<float>(weight[c * 4 + j]);
    sum += static_cast<float>(current) * static_cast<float>(weight[c * 4 + 3]);
    output[index] = static_cast<T>(sum / (1.0f + expf(-sum)));
    history[0] = history[1];
    history[1] = history[2];
    history[2] = current;
  }
  for (int j = 0; j < 3; ++j)
    present[offset + j] = history[j];
}

__device__ float WaveSum(float value) {
  for (int delta = warpSize / 2; delta > 0; delta /= 2)
    value += __shfl_down(value, delta);
  return __shfl(value, 0);
}

template <typename T>
__global__ void GatedDelta(
    const T *query, const T *key, const T *value,
    const int32_t *cu, const float *decay, const float *beta,
    const float *past, const float *a_log, const float *dt_bias,
    T *output, float *present, int key_heads, int value_heads,
    float scale, const int *error) {
  if (*error)
    return;
  const int lane = threadIdx.x % warpSize;
  const int row = blockIdx.x * (kThreads / warpSize) + threadIdx.x / warpSize;
  if (row >= 128)
    return;
  const int h = blockIdx.y, b = blockIdx.z;
  const int kh = h / (value_heads / key_heads);
  const int64_t state_base =
      ((static_cast<int64_t>(b) * value_heads + h) * 128 + row) * 128;
  // Each wave owns one V-major state row, including across aliased updates.
  float state[4];
  for (int d = lane, i = 0; d < 128; d += warpSize, ++i)
    state[i] = past[state_base + d];
  for (int t = cu[b]; t < cu[b + 1]; ++t) {
    const int64_t qbase = (static_cast<int64_t>(t) * key_heads + kh) * 128;
    const int64_t vbase = (static_cast<int64_t>(t) * value_heads + h) * 128;
    float q[4], k[4], qnorm = 0, knorm = 0;
    for (int d = lane, i = 0; d < 128; d += warpSize, ++i) {
      q[i] = static_cast<float>(query[qbase + d]);
      k[i] = static_cast<float>(key[qbase + d]);
      qnorm += q[i] * q[i];
      knorm += k[i] * k[i];
    }
    qnorm = rsqrtf(WaveSum(qnorm) + 1e-12f);
    knorm = rsqrtf(WaveSum(knorm) + 1e-12f);
    const int64_t gate_index = static_cast<int64_t>(t) * value_heads + h;
    const float z = decay[gate_index] + dt_bias[h];
    const float softplus = fmaxf(z, 0.0f) + log1pf(expf(-fabsf(z)));
    const float factor = expf(-expf(a_log[h]) * softplus);
    float retrieved = 0;
    for (int d = lane, i = 0; d < 128; d += warpSize, ++i) {
      q[i] *= qnorm;
      k[i] *= knorm;
      state[i] *= factor;
      retrieved += state[i] * k[i];
    }
    retrieved = WaveSum(retrieved);
    const float delta = (static_cast<float>(value[vbase + row]) - retrieved) /
                        (1.0f + expf(-beta[gate_index]));
    float result = 0;
    for (int d = lane, i = 0; d < 128; d += warpSize, ++i) {
      state[i] += k[i] * delta;
      result += state[i] * q[i];
    }
    result = WaveSum(result);
    if (lane == 0)
      output[vbase + row] = static_cast<T>(result * scale);
  }
  for (int d = lane, i = 0; d < 128; d += warpSize, ++i)
    present[state_base + d] = state[i];
}

template <typename T>
__global__ void GatedNorm(const T *input, const T *weight,
                          const T *gate, T *output, float epsilon,
                          const int *error) {
  if (*error)
    return;
  __shared__ float sums[128];
  const int d = threadIdx.x;
  const int64_t index = static_cast<int64_t>(blockIdx.x) * 128 + d;
  const float x = static_cast<float>(input[index]);
  sums[d] = x * x;
  __syncthreads();
  for (int stride = 64; stride; stride /= 2) {
    if (d < stride)
      sums[d] += sums[d + stride];
    __syncthreads();
  }
  const float z = static_cast<float>(gate[index]);
  output[index] = static_cast<T>(x * rsqrtf(sums[0] / 128 + epsilon) *
                              static_cast<float>(weight[d]) *
                              (z / (1.0f + expf(-z))));
}

__global__ void ValidatePositions(const int64_t *positions, int64_t count,
                                  int64_t length, int *error) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count && (positions[index] < 0 || positions[index] >= length))
    atomicExch(error, -1);
}

__global__ void MRotary(const float *input, const int64_t *positions,
                        const float *cos_cache, const float *sin_cache,
                        float *output, int tokens, int heads,
                        const int *error) {
  if (*error)
    return;
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<int64_t>(tokens) * heads * 256)
    return;
  const int d = index % 256;
  if (d >= 64) {
    output[index] = input[index];
    return;
  }
  const int token = index / (heads * 256);
  const int col = d % 32;
  const int axis = col % 3 == 1 && col < 33 ? 1
                   : col % 3 == 2 && col < 30 ? 2
                                             : 0;
  const int64_t position = positions[static_cast<int64_t>(axis) * tokens + token];
  const int64_t cache_index = position * 32 + col;
  const float partner = input[index + (d < 32 ? 32 : -32)];
  output[index] = input[index] * cos_cache[cache_index] +
                  (d < 32 ? -partner : partner) * sin_cache[cache_index];
}

int ValidatePacked(void *stream, const int32_t *cu, int64_t tokens,
                   int64_t batch, void *error) {
  if (!stream || !cu || !error || tokens <= 0 || tokens > INT_MAX ||
      batch <= 0 || batch > 65535)
    return hipErrorInvalidValue;
  (void)hipGetLastError();
  hipLaunchKernelGGL(ValidateLengths, dim3((batch + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, static_cast<hipStream_t>(stream), cu,
                     static_cast<int>(tokens), static_cast<int>(batch),
                     static_cast<int *>(error));
  return hipGetLastError();
}

} // namespace

extern "C" HIP_KERNEL_API int hip_varlen_causal_conv(
    void *stream, const void *input, const void *weight,
    const int32_t *cumulative, const void *bias, const void *past,
    void *output, void *present, int64_t tokens, int64_t batch,
    int64_t channels, void *error_flag, int64_t input_dtype) {
  if (!input || !weight || !bias || !past || !output || !present ||
      channels <= 0 || channels > INT_MAX / 4 ||
      (input_dtype != HIP_DTYPE_FLOAT16 && input_dtype != HIP_DTYPE_FLOAT32))
    return hipErrorInvalidValue;
  auto status = ValidatePacked(stream, cumulative, tokens, batch, error_flag);
  if (status != hipSuccess)
    return status;
  const auto launch = [&](auto *type) {
    using T = std::remove_pointer_t<decltype(type)>;
    hipLaunchKernelGGL(CausalConv<T>,
                     dim3((channels + kThreads - 1) / kThreads, batch),
                     dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                     static_cast<const T *>(input),
                     static_cast<const T *>(weight), cumulative,
                     static_cast<const T *>(bias),
                     static_cast<const T *>(past),
                     static_cast<T *>(output),
                     static_cast<T *>(present), static_cast<int>(channels),
                     static_cast<const int *>(error_flag));
  };
  if (input_dtype == HIP_DTYPE_FLOAT32)
    launch(static_cast<float *>(nullptr));
  else
    launch(static_cast<__half *>(nullptr));
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_gated_delta_net(
    void *stream, const void *query, const void *key, const void *value,
    const int32_t *cumulative, const float *decay, const float *beta,
    const float *past, const float *a_log, const float *dt_bias,
    void *output, float *present, int64_t tokens, int64_t batch,
    int64_t key_heads, int64_t value_heads, float scale, void *error_flag,
    int64_t input_dtype) {
  if (!query || !key || !value || !decay || !beta || !past || !a_log ||
      !dt_bias || !output || !present || key_heads <= 0 ||
      value_heads < key_heads || value_heads > 65535 ||
      value_heads % key_heads != 0 || !std::isfinite(scale) || scale <= 0 ||
      (input_dtype != HIP_DTYPE_FLOAT16 && input_dtype != HIP_DTYPE_FLOAT32))
    return hipErrorInvalidValue;
  auto status = ValidatePacked(stream, cumulative, tokens, batch, error_flag);
  if (status != hipSuccess)
    return status;
  int device = 0;
  status = hipGetDevice(&device);
  if (status != hipSuccess)
    return status;
  int wave_size = 0;
  status = hipDeviceGetAttribute(&wave_size, hipDeviceAttributeWarpSize, device);
  if (status != hipSuccess)
    return status;
  if (wave_size != 32 && wave_size != 64)
    return hipErrorInvalidValue;
  const auto launch = [&](auto *type) {
    using T = std::remove_pointer_t<decltype(type)>;
    hipLaunchKernelGGL(GatedDelta<T>,
                     dim3(128 / (kThreads / wave_size), value_heads, batch),
                     dim3(kThreads), 0, static_cast<hipStream_t>(stream),
                     static_cast<const T *>(query),
                     static_cast<const T *>(key),
                     static_cast<const T *>(value), cumulative, decay,
                     beta, past, a_log, dt_bias,
                     static_cast<T *>(output), present,
                     static_cast<int>(key_heads), static_cast<int>(value_heads),
                     scale, static_cast<const int *>(error_flag));
  };
  if (input_dtype == HIP_DTYPE_FLOAT32)
    launch(static_cast<float *>(nullptr));
  else
    launch(static_cast<__half *>(nullptr));
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_gated_rms_norm(
    void *stream, const void *input, const void *weight, const void *gate,
    void *output, int64_t rows, int64_t width, float epsilon, void *error_flag,
    int64_t input_dtype) {
  if (!stream || !input || !weight || !gate || !output || !error_flag ||
      rows <= 0 || width <= 0 || width % 128 || width > INT_MAX ||
      rows > INT_MAX / (width / 128) || !std::isfinite(epsilon) || epsilon <= 0 ||
      (input_dtype != HIP_DTYPE_FLOAT16 && input_dtype != HIP_DTYPE_FLOAT32))
    return hipErrorInvalidValue;
  (void)hipGetLastError();
  const auto launch = [&](auto *type) {
    using T = std::remove_pointer_t<decltype(type)>;
    hipLaunchKernelGGL(GatedNorm<T>, dim3(rows * (width / 128)), dim3(128), 0,
                     static_cast<hipStream_t>(stream),
                     static_cast<const T *>(input),
                     static_cast<const T *>(weight),
                     static_cast<const T *>(gate),
                     static_cast<T *>(output), epsilon,
                     static_cast<const int *>(error_flag));
  };
  if (input_dtype == HIP_DTYPE_FLOAT32)
    launch(static_cast<float *>(nullptr));
  else
    launch(static_cast<__half *>(nullptr));
  return hipGetLastError();
}

extern "C" HIP_KERNEL_API int hip_mrotary_embedding_fp32(
    void *stream, const float *input, const int64_t *positions,
    const float *cos_cache, const float *sin_cache, float *output,
    int64_t tokens, int64_t cache_length, int64_t heads, void *error_flag) {
  if (!stream || !input || !positions || !cos_cache || !sin_cache || !output ||
      !error_flag || input == output || tokens <= 0 || tokens > INT_MAX / 3 ||
      heads <= 0 || heads > 65535 || cache_length <= 0 ||
      cache_length > INT64_MAX / 32 ||
      tokens > static_cast<int64_t>(INT_MAX) * kThreads / (heads * 256))
    return hipErrorInvalidValue;
  auto hip_stream = static_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(ValidatePositions,
                     dim3((tokens * 3 + kThreads - 1) / kThreads),
                     dim3(kThreads), 0, hip_stream, positions, tokens * 3,
                     cache_length, static_cast<int *>(error_flag));
  auto status = hipGetLastError();
  if (status != hipSuccess)
    return status;
  hipLaunchKernelGGL(
      MRotary, dim3((tokens * heads * 256 + kThreads - 1) / kThreads),
      dim3(kThreads), 0, hip_stream, input, positions, cos_cache, sin_cache,
      output, static_cast<int>(tokens), static_cast<int>(heads),
      static_cast<const int *>(error_flag));
  return hipGetLastError();
}
