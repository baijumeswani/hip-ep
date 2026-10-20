// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: sed 's/cache_dtype = ""/cache_dtype = "int8"/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" --assign-op-state-slots --convert-hip-to-llvm | FileCheck %s --check-prefix=LLVM
// RUN: sed 's/"PER_CHANNEL"/"PER_TENSOR"/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=QUANT
// RUN: sed 's/4x1x256xf32/4x256xf32/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=SCALES
// RUN: sed 's/256xi8/256xui8/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=UNSIGNED
// RUN: sed 's/cache_dtype = ""/cache_dtype = "float16"/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=DTYPE
// RUN: sed 's/is_causal = 1/is_causal = 0/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=CAUSAL
// RUN: sed 's/kv_cache_layout = "SEPARATE"/kv_cache_layout = "LATENT"/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=LAYOUT
// RUN: sed 's/kv_num_heads = 4/kv_cache_residual_bits = 8 : si64, kv_num_heads = 4/g;s/4x256xi8/4x512xi8/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" --assign-op-state-slots --convert-hip-to-llvm | FileCheck %s --check-prefix=LLVM
// RUN: sed 's/kv_num_heads = 4/kv_cache_residual_bits = 8 : si64, kv_num_heads = 4/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=WIDTH
// RUN: sed 's/kv_num_heads = 4/kv_cache_residual_bits = 4 : si64, kv_num_heads = 4/g' %s | hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip 2>&1 | FileCheck %s --check-prefix=RESIDUAL

// QUANT: INT8 caches require static PER_CHANNEL scales
// SCALES: INT8 requires separate FP32 Q/K/V and FP32 [kv_heads,1,256] K/V scales
// UNSIGNED: quantized caches must be signed INT8, not UINT8
// DTYPE: cache dtype attribute must match FP16, BF16 or signed INT8 storage
// CAUSAL: requires grouped heads, supported causal/window policy and supported RoPE layout
// LAYOUT: requires separate K/V cache storage
// WIDTH: matching static cache geometry
// RESIDUAL: residual bits must be 0, or 8 with signed INT8 caches

module {
  func.func @main_graph(
      %query: tensor<?x6144xf32>, %key: tensor<?x1024xf32>,
      %value: tensor<?x1024xf32>,
      %past_key: tensor<?x256x4x256xi8>, %past_value: tensor<?x256x4x256xi8>,
      %cumulative: tensor<?xi32>, %past: tensor<?xi32>,
      %table: tensor<?x?xi32>, %key_scale: tensor<4x1x256xf32>,
      %value_scale: tensor<4x1x256xf32>)
      -> (tensor<?x6144xf32>, tensor<?x256x4x256xi8>, tensor<?x256x4x256xi8>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %result:3 = "onnx.Custom"(
        %query, %key, %value, %past_key, %past_value, %cumulative, %past, %table,
        %none, %none, %none, %none, %none, %none, %key_scale, %value_scale)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft", num_heads = 24 : si64,
         kv_num_heads = 4 : si64, do_rotary = 0 : si64,
         is_causal = 1 : si64, kv_cache_layout = "SEPARATE",
         k_cache_dtype = "", v_cache_dtype = "",
         k_quant_type = "PER_CHANNEL", v_quant_type = "PER_CHANNEL"}
        : (tensor<?x6144xf32>, tensor<?x1024xf32>, tensor<?x1024xf32>,
           tensor<?x256x4x256xi8>, tensor<?x256x4x256xi8>,
           tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>,
           none, none, none, none, none, none,
           tensor<4x1x256xf32>, tensor<4x1x256xf32>)
        -> (tensor<?x6144xf32>, tensor<?x256x4x256xi8>, tensor<?x256x4x256xi8>)
    // CHECK-LABEL: func.func @main_graph
    // CHECK: hip.paged_attention_int8
    // CHECK-SAME: kv_num_heads = 4
    // CHECK-SAME: num_heads = 24
    // CHECK-SAME: scale = 6.250000e-02
    return %result#0, %result#1, %result#2
        : tensor<?x6144xf32>, tensor<?x256x4x256xi8>, tensor<?x256x4x256xi8>
  }
}

// LLVM-LABEL: llvm.func private @main_graph
// LLVM: llvm.call @wrap_paged_attention_int8({{.*}}) : (!llvm.ptr, i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32, i64) -> i32
