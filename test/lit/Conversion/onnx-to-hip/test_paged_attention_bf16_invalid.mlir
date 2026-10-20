// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @main_graph(
      %query: tensor<?x6144xbf16>,
      %key: tensor<20x256x8x128xbf16>, %value: tensor<20x256x8x128xbf16>,
      %cumulative: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x?xi32>,
      %cos: tensor<4096x64xbf16>, %sin: tensor<4096x64xbf16>)
      -> (tensor<?x4096xbf16>, tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    // expected-error @+1 {{BF16 head-128 attention requires BF16 Q/K norm weights}}
    %result:3 = "onnx.Custom"(%query, %none, %none, %key, %value,
        %cumulative, %past, %table, %cos, %sin)
        <{function_name = "PagedAttention"}>
        {domain_name = "com.microsoft", num_heads = 32 : si64,
         kv_num_heads = 8 : si64, do_rotary = 1 : si64, is_causal = 0 : si64,
         local_window_size = 2048 : si64}
        : (tensor<?x6144xbf16>, none, none,
           tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>,
           tensor<?xi32>, tensor<?xi32>, tensor<?x?xi32>,
           tensor<4096x64xbf16>, tensor<4096x64xbf16>)
        -> (tensor<?x4096xbf16>, tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>)
    return %result#0, %result#1, %result#2 :
        tensor<?x4096xbf16>, tensor<20x256x8x128xbf16>, tensor<20x256x8x128xbf16>
  }
}
