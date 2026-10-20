// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" --convert-hip-to-llvm | FileCheck %s --check-prefix=LLVM

module {
  func.func @main_graph(
      %query: tensor<?x6144xbf16>,
      %key: tensor<20x256x8x128xbf16>, %value: tensor<20x256x8x128xbf16>,
      %cumulative: tensor<?xi32>, %past: tensor<?xi32>,
      %table: tensor<?x?xi32>,
      %cos: tensor<262152x64xbf16>, %sin: tensor<262152x64xbf16>,
      %q_norm: tensor<128xbf16>, %k_norm: tensor<128xbf16>)
      -> (tensor<?x4096xbf16>, tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %result:3 = "onnx.Custom"(%query, %none, %none, %key, %value,
        %cumulative, %past, %table, %cos, %sin, %none, %none, %q_norm, %k_norm)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft", num_heads = 32 : si64,
         kv_num_heads = 8 : si64, do_rotary = 1 : si64, is_causal = 0 : si64,
         local_window_size = 2048 : si64, qk_norm_epsilon = 1.0e-6 : f32}
        : (tensor<?x6144xbf16>, none, none,
           tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>,
           tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>,
           tensor<262152x64xbf16>, tensor<262152x64xbf16>,
           none, none, tensor<128xbf16>, tensor<128xbf16>)
        -> (tensor<?x4096xbf16>, tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>)
    return %result#0, %result#1, %result#2 :
        tensor<?x4096xbf16>, tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>
  }
}

// CHECK-LABEL: func.func @main_graph
// CHECK: hip.paged_attention_bf16
// CHECK-SAME: kv_num_heads = 8
// CHECK-SAME: local_window_size = 2048
// CHECK-SAME: num_heads = 32
// CHECK-SAME: qk_norm_epsilon =
// LLVM-LABEL: llvm.func private @main_graph
// LLVM: llvm.call @wrap_paged_attention_bf16
