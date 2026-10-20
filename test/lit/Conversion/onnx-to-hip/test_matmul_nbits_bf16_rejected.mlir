// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: split-file %s %t
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics %t/bits.mlir
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics %t/block.mlir
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics %t/zp.mlir
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics %t/group.mlir
// RUN: hip-mlir-opt --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics %t/packed.mlir

//--- bits.mlir
module {
  func.func @main_graph(%a: tensor<1x128xbf16>, %b: tensor<64x4x32xui8>,
                        %scales: tensor<64x4xbf16>) -> tensor<1x64xbf16> {
    // expected-error @+1 {{BF16 MatMulNBits requires raw symmetric INT4/block-32 without zero_points or g_idx}}
    %y = "onnx.Custom"(%a, %b, %scales) {
      function_name = "MatMulNBits", domain_name = "com.microsoft",
      K = 128 : si64, N = 64 : si64, bits = 8 : si64, block_size = 32 : si64
    } : (tensor<1x128xbf16>, tensor<64x4x32xui8>, tensor<64x4xbf16>) -> tensor<1x64xbf16>
    return %y : tensor<1x64xbf16>
  }
}

//--- block.mlir
module {
  func.func @main_graph(%a: tensor<1x128xbf16>, %b: tensor<64x1x64xui8>,
                        %scales: tensor<64x1xbf16>) -> tensor<1x64xbf16> {
    // expected-error @+1 {{BF16 MatMulNBits requires raw symmetric INT4/block-32 without zero_points or g_idx}}
    %y = "onnx.Custom"(%a, %b, %scales) {
      function_name = "MatMulNBits", domain_name = "com.microsoft",
      K = 128 : si64, N = 64 : si64, bits = 4 : si64, block_size = 128 : si64
    } : (tensor<1x128xbf16>, tensor<64x1x64xui8>, tensor<64x1xbf16>) -> tensor<1x64xbf16>
    return %y : tensor<1x64xbf16>
  }
}

//--- zp.mlir
module {
  func.func @main_graph(%a: tensor<1x128xbf16>, %b: tensor<64x4x16xui8>,
                        %scales: tensor<64x4xbf16>, %zp: tensor<64x2xui8>) -> tensor<1x64xbf16> {
    // expected-error @+1 {{BF16 MatMulNBits requires raw symmetric INT4/block-32 without zero_points or g_idx}}
    %y = "onnx.Custom"(%a, %b, %scales, %zp) {
      function_name = "MatMulNBits", domain_name = "com.microsoft",
      K = 128 : si64, N = 64 : si64, bits = 4 : si64, block_size = 32 : si64
    } : (tensor<1x128xbf16>, tensor<64x4x16xui8>, tensor<64x4xbf16>, tensor<64x2xui8>) -> tensor<1x64xbf16>
    return %y : tensor<1x64xbf16>
  }
}

//--- group.mlir
module {
  func.func @main_graph(%a: tensor<1x128xbf16>, %b: tensor<64x4x16xui8>,
                        %scales: tensor<64x4xbf16>, %group: tensor<128xi32>) -> tensor<1x64xbf16> {
    %none = "onnx.None"() : () -> none
    // expected-error @+1 {{BF16 MatMulNBits requires raw symmetric INT4/block-32 without zero_points or g_idx}}
    %y = "onnx.Custom"(%a, %b, %scales, %none, %group) {
      function_name = "MatMulNBits", domain_name = "com.microsoft",
      K = 128 : si64, N = 64 : si64, bits = 4 : si64, block_size = 32 : si64
    } : (tensor<1x128xbf16>, tensor<64x4x16xui8>, tensor<64x4xbf16>, none, tensor<128xi32>) -> tensor<1x64xbf16>
    return %y : tensor<1x64xbf16>
  }
}

//--- packed.mlir
module {
  func.func @main_graph(%a: tensor<1x128xbf16>, %b: tensor<64x4x16xui8>,
                        %scales: tensor<64x4xbf16>) -> tensor<1x64xbf16> {
    // expected-error @+1 {{MatMulNBits: CUDA-prepacked weights are unsupported}}
    %y = "onnx.Custom"(%a, %b, %scales) {
      function_name = "MatMulNBits", domain_name = "com.microsoft",
      K = 128 : si64, N = 64 : si64, bits = 4 : si64, block_size = 32 : si64,
      weight_prepacked = 1 : si64
    } : (tensor<1x128xbf16>, tensor<64x4x16xui8>, tensor<64x4xbf16>) -> tensor<1x64xbf16>
    return %y : tensor<1x64xbf16>
  }
}
