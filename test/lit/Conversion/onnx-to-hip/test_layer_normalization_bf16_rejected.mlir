// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @main_graph(%x: tensor<1x3xbf16>, %scale: tensor<3xbf16>) -> tensor<1x3xbf16> {
    // expected-error @+1 {{BF16 LayerNormalization is not supported}}
    %y = "onnx.LayerNormalization"(%x, %scale) {axis = -1 : si64} : (tensor<1x3xbf16>, tensor<3xbf16>) -> tensor<1x3xbf16>
    return %y : tensor<1x3xbf16>
  }
}
