// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @main_graph(%x: tensor<1x2xbf16>, %k: tensor<1xi64>) -> (tensor<1x1xbf16>, tensor<1x1xi64>) {
    // expected-error @+1 {{BF16 TopK is not supported}}
    %values, %indices = "onnx.TopK"(%x, %k) {axis = -1 : si64} : (tensor<1x2xbf16>, tensor<1xi64>) -> (tensor<1x1xbf16>, tensor<1x1xi64>)
    return %values, %indices : tensor<1x1xbf16>, tensor<1x1xi64>
  }
}
