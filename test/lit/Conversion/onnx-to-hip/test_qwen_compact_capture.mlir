// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" --convert-hip-to-llvm | FileCheck %s --check-prefix=LLVM

module {
  func.func @main_graph(
      %x: tensor<?x10240xf32>, %w: tensor<10240x1x4xf32>,
      %cu: tensor<?xi32>, %bias: tensor<10240xf32>,
      %past: tensor<?x10240x3xf32>, %counts: tensor<?xi32>)
      -> (tensor<?x10240xf32>, tensor<?x10240x3xf32>, tensor<?x7x10240xf32>) {
    %y:3 = "onnx.Custom"(%x, %w, %cu, %bias, %past, %counts) {
      function_name = "VarlenCausalConvWithState", domain_name = "com.microsoft",
      activation = "silu", state_update_capacity = 7 : si64
    } : (tensor<?x10240xf32>, tensor<10240x1x4xf32>, tensor<?xi32>,
         tensor<10240xf32>, tensor<?x10240x3xf32>, tensor<?xi32>)
        -> (tensor<?x10240xf32>, tensor<?x10240x3xf32>, tensor<?x7x10240xf32>)
    // CHECK-LABEL: func.func @main_graph
    // CHECK: hip.varlen_causal_conv_updates
    // CHECK-SAME: state_update_capacity = 7
    // LLVM-LABEL: llvm.func private @main_graph
    // LLVM: llvm.call @wrap_varlen_causal_conv_updates
    return %y#0, %y#1, %y#2 : tensor<?x10240xf32>, tensor<?x10240x3xf32>, tensor<?x7x10240xf32>
  }

  func.func @gdn(
      %q: tensor<?x16x128xf32>, %k: tensor<?x16x128xf32>,
      %v: tensor<?x48x128xf32>, %cu: tensor<?xi32>,
      %decay: tensor<?x48xf32>, %beta: tensor<?x48xf32>,
      %past: tensor<?x48x128x128xf32>, %a: tensor<48xf32>, %dt: tensor<48xf32>,
      %counts: tensor<?xi32>, %active: tensor<?xi32>)
      -> (tensor<?x48x128xf32>, tensor<?x48x128x128xf32>, tensor<?x57680xf32>) {
    %y:3 = "onnx.Custom"(%q, %k, %v, %cu, %decay, %beta, %past, %a, %dt, %counts, %active) {
      function_name = "GatedDeltaNet", domain_name = "com.microsoft",
      update_rule = "gated_delta", gate_activation = "qwen",
      beta_activation = "sigmoid", qk_l2_norm = 1 : si64,
      chunk_size = 64 : si64, scale = 0.0 : f32, state_update_capacity = 7 : si64
    } : (tensor<?x16x128xf32>, tensor<?x16x128xf32>, tensor<?x48x128xf32>,
         tensor<?xi32>, tensor<?x48xf32>, tensor<?x48xf32>,
         tensor<?x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>,
         tensor<?xi32>, tensor<?xi32>)
        -> (tensor<?x48x128xf32>, tensor<?x48x128x128xf32>, tensor<?x57680xf32>)
    // CHECK-LABEL: func.func @gdn
    // CHECK: hip.gated_delta_net_updates
    // CHECK-SAME: state_update_capacity = 7
    // LLVM-LABEL: llvm.func @gdn
    // LLVM: llvm.call @wrap_gated_delta_net_updates
    return %y#0, %y#1, %y#2 : tensor<?x48x128xf32>, tensor<?x48x128x128xf32>, tensor<?x57680xf32>
  }
}
