// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// RUN: not hip-mlir-opt %s --convert-hip-to-llvm 2>&1 | FileCheck %s
// CHECK: BF16 standalone Softmax is not supported

module {
  func.func @unsupported(%ctx: !hip.context, %x: memref<1x3xbf16, 1>, %y: memref<1x3xbf16, 1>) {
    hip.miopen.softmax(%ctx) ins(%x : memref<1x3xbf16, 1>) outs(%y : memref<1x3xbf16, 1>)
    return
  }
}
