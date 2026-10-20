// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics %s

module {
  func.func @main_graph(%A: tensor<1x896xf16>, %B: tensor<896x28x16xui8>, %S: tensor<896x28xf16>) -> tensor<1x896xf16> {
    // expected-error @+1 {{MatMulNBits: CUDA-prepacked weights are unsupported; expected raw blockwise layout}}
    %Y = "onnx.Custom"(%A, %B, %S) {
      function_name = "MatMulNBits", domain_name = "com.microsoft",
      K = 896 : si64, N = 896 : si64, bits = 4 : si64,
      block_size = 32 : si64, weight_prepacked = 1 : si64
    } : (tensor<1x896xf16>, tensor<896x28x16xui8>, tensor<896x28xf16>) -> tensor<1x896xf16>
    return %Y : tensor<1x896xf16>
  }
}
