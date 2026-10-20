// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --split-input-file --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @main_graph(
      %x: tensor<?x10240xf32>, %w: tensor<10240x1x4xf32>,
      %cu: tensor<?xi32>, %bias: tensor<10240xf32>,
      %past: tensor<?x10240x3xf32>, %counts: tensor<?xi32>)
      -> (tensor<?x10240xf32>, tensor<?x10240x3xf32>, tensor<?x8x10240xf32>) {
    // expected-error @+1 {{compact state_update_capacity must be in [0,7]}}
    %y:3 = "onnx.Custom"(%x, %w, %cu, %bias, %past, %counts) {
      function_name = "VarlenCausalConvWithState", domain_name = "com.microsoft",
      activation = "silu", state_update_capacity = 8 : si64
    } : (tensor<?x10240xf32>, tensor<10240x1x4xf32>, tensor<?xi32>,
         tensor<10240xf32>, tensor<?x10240x3xf32>, tensor<?xi32>)
        -> (tensor<?x10240xf32>, tensor<?x10240x3xf32>, tensor<?x8x10240xf32>)
    return %y#0, %y#1, %y#2 : tensor<?x10240xf32>, tensor<?x10240x3xf32>, tensor<?x8x10240xf32>
  }
}

// -----

module {
  func.func @main_graph(
      %q: tensor<?x16x128xf32>, %k: tensor<?x16x128xf32>,
      %v: tensor<?x48x128xf32>, %cu: tensor<?xi32>,
      %decay: tensor<?x48xf32>, %beta: tensor<?x48xf32>,
      %past: tensor<?x48x128x128xf32>, %a: tensor<48xf32>, %dt: tensor<48xf32>,
      %counts: tensor<?xi32>, %active: tensor<2xi32>)
      -> (tensor<?x48x128xf32>, tensor<?x48x128x128xf32>, tensor<?x57680xf32>) {
    // expected-error @+1 {{compact recurrence requires int32 [batch] counts, int32 [1] activity and FP32 [batch,capsule_width] output}}
    %y:3 = "onnx.Custom"(%q, %k, %v, %cu, %decay, %beta, %past, %a, %dt, %counts, %active) {
      function_name = "GatedDeltaNet", domain_name = "com.microsoft",
      update_rule = "gated_delta", gate_activation = "qwen",
      beta_activation = "sigmoid", qk_l2_norm = 1 : si64,
      chunk_size = 64 : si64, scale = 0.0 : f32, state_update_capacity = 7 : si64
    } : (tensor<?x16x128xf32>, tensor<?x16x128xf32>, tensor<?x48x128xf32>,
         tensor<?xi32>, tensor<?x48xf32>, tensor<?x48xf32>,
         tensor<?x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>,
         tensor<?xi32>, tensor<2xi32>)
        -> (tensor<?x48x128xf32>, tensor<?x48x128x128xf32>, tensor<?x57680xf32>)
    return %y#0, %y#1, %y#2 : tensor<?x48x128xf32>, tensor<?x48x128x128xf32>, tensor<?x57680xf32>
  }
}
