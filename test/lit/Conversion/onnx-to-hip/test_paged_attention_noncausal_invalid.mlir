// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @main_graph(%query: tensor<2x192xf16>,
                      %cache: tensor<1x256x1x64xf16>,
                      %cu: tensor<2xi32>, %past: tensor<1xi32>,
                      %table: tensor<1x1xi32>, %rope: tensor<256x32xf16>)
      -> (tensor<2x64xf16>, tensor<1x256x1x64xf16>, tensor<1x256x1x64xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    // expected-error @+1 {{requires grouped heads, full causal attention and non-interleaved fused RoPE}}
    %result:3 = "onnx.Custom"(%query, %none, %none, %cache, %cache, %cu, %past, %table, %rope, %rope)
        {function_name = "PagedAttention", domain_name = "com.microsoft",
         num_heads = 1 : si64, kv_num_heads = 1 : si64,
         do_rotary = 1 : si64, is_causal = 0 : si64}
        : (tensor<2x192xf16>, none, none, tensor<1x256x1x64xf16>, tensor<1x256x1x64xf16>,
           tensor<2xi32>, tensor<1xi32>, tensor<1x1xi32>, tensor<256x32xf16>, tensor<256x32xf16>)
        -> (tensor<2x64xf16>, tensor<1x256x1x64xf16>, tensor<1x256x1x64xf16>)
    return %result#0, %result#1, %result#2 : tensor<2x64xf16>, tensor<1x256x1x64xf16>, tensor<1x256x1x64xf16>
  }
}
