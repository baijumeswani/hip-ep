// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" --convert-hip-to-llvm | FileCheck %s --check-prefix=LLVM

module {
  func.func @main_graph(
      %query: tensor<?x1152xf16>,
      %key: tensor<32x256x2x64xf16>,
      %value: tensor<32x256x2x64xf16>,
      %cumulative: tensor<?xi32>, %past: tensor<?xi32>,
      %table: tensor<?x16xi32>,
      %cos: tensor<4096x32xf16>, %sin: tensor<4096x32xf16>)
      -> (tensor<?x896xf16>, tensor<32x256x2x64xf16>, tensor<32x256x2x64xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %result:3 = "onnx.Custom"(%query, %none, %none, %key, %value,
                             %cumulative, %past, %table, %cos, %sin)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft", num_heads = 14 : si64,
         kv_num_heads = 2 : si64, do_rotary = 1 : si64}
        : (tensor<?x1152xf16>, none, none,
           tensor<32x256x2x64xf16>, tensor<32x256x2x64xf16>,
           tensor<?xi32>, tensor<?xi32>, tensor<?x16xi32>,
           tensor<4096x32xf16>, tensor<4096x32xf16>)
        -> (tensor<?x896xf16>, tensor<32x256x2x64xf16>, tensor<32x256x2x64xf16>)
    // CHECK-LABEL: func.func @main_graph
    // CHECK: tensor.dim
    // CHECK: tensor.empty({{.*}}) : tensor<?x896xf16>
    // CHECK: hip.paged_attention
    // CHECK-SAME: kv_num_heads = 2
    // CHECK-SAME: num_heads = 14
    // CHECK-SAME: scale = 1.250000e-01
    return %result#0, %result#1, %result#2 : tensor<?x896xf16>, tensor<32x256x2x64xf16>, tensor<32x256x2x64xf16>
  }
}

// LLVM-LABEL: llvm.func private @main_graph
// LLVM: llvm.call @wrap_paged_attention({{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, f32) -> i32
