// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
// RUN: hip-mlir-opt %s --convert-hip-to-llvm | FileCheck %s

module {
  func.func @host_bounds(%ctx: !hip.context,
      %data: memref<?x6xbf16, 1>, %starts: memref<1xi32, 1>,
      %ends: memref<1xi32, 1>, %axes: memref<1xi32, 1>,
      %steps: memref<1xi32, 1>, %output: memref<?x6xbf16, 1>) {
    hip.slice(%ctx)
        ins(%data, %starts, %ends :
            memref<?x6xbf16, 1>, memref<1xi32, 1>, memref<1xi32, 1>)
        axes(%axes : memref<1xi32, 1>)
        steps(%steps : memref<1xi32, 1>)
        outs(%output : memref<?x6xbf16, 1>)
        {hipdnn.constant_starts = array<i64: -1>,
         hipdnn.constant_ends = array<i64: -2147483648>,
         hipdnn.constant_axes = array<i64: 0>,
         hipdnn.constant_steps = array<i64: -1>}
    return
  }
}

// CHECK-LABEL: llvm.func @host_bounds
// CHECK: llvm.alloca
// CHECK: llvm.store
// CHECK: llvm.call @wrap_slice_host(
