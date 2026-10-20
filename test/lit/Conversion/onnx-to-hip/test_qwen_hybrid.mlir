// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip | FileCheck %s
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --one-shot-bufferize="bufferize-function-boundaries" --convert-hip-to-llvm | FileCheck %s --check-prefix=LLVM

module {
  func.func @main_graph(%x: tensor<?x10240xf16>, %w: tensor<10240x1x4xf16>,
                  %cu: tensor<?xi32>, %bias: tensor<10240xf16>,
                  %past: tensor<?x10240x3xf16>)
      -> (tensor<?x10240xf16>, tensor<?x10240x3xf16>) {
    %y:2 = "onnx.Custom"(%x, %w, %cu, %bias, %past) {
      function_name = "VarlenCausalConvWithState",
      domain_name = "com.microsoft", activation = "silu"
    } : (tensor<?x10240xf16>, tensor<10240x1x4xf16>, tensor<?xi32>,
         tensor<10240xf16>, tensor<?x10240x3xf16>)
        -> (tensor<?x10240xf16>, tensor<?x10240x3xf16>)
    // CHECK-LABEL: func.func @main_graph
    // CHECK: hip.varlen_causal_conv
    // LLVM-LABEL: llvm.func private @main_graph
    // LLVM: llvm.call @wrap_varlen_causal_conv
    return %y#0, %y#1 : tensor<?x10240xf16>, tensor<?x10240x3xf16>
  }

  func.func @gdn(%q: tensor<?x16x128xf16>, %k: tensor<?x16x128xf16>,
                 %v: tensor<?x48x128xf16>, %cu: tensor<?xi32>,
                 %decay: tensor<?x48xf32>, %beta: tensor<?x48xf32>,
                 %past: tensor<?x48x128x128xf32>,
                 %a: tensor<48xf32>, %dt: tensor<48xf32>)
      -> (tensor<?x48x128xf16>, tensor<?x48x128x128xf32>) {
    %y:2 = "onnx.Custom"(%q, %k, %v, %cu, %decay, %beta, %past, %a, %dt) {
      function_name = "GatedDeltaNet", domain_name = "com.microsoft",
      update_rule = "gated_delta", gate_activation = "qwen",
      beta_activation = "sigmoid", qk_l2_norm = 1 : si64,
      chunk_size = 64 : si64, scale = 0.0 : f32
    } : (tensor<?x16x128xf16>, tensor<?x16x128xf16>, tensor<?x48x128xf16>,
         tensor<?xi32>, tensor<?x48xf32>, tensor<?x48xf32>,
         tensor<?x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
        -> (tensor<?x48x128xf16>, tensor<?x48x128x128xf32>)
    // CHECK-LABEL: func.func @gdn
    // CHECK: hip.gated_delta_net
    // CHECK-SAME: key_heads = 16
    // CHECK-SAME: value_heads = 48
    // LLVM-LABEL: llvm.func @gdn
    // LLVM: llvm.call @wrap_gated_delta_net
    return %y#0, %y#1 : tensor<?x48x128xf16>, tensor<?x48x128x128xf32>
  }

  func.func @norm(%x: tensor<?x6144xf16>, %w: tensor<128xf16>,
                  %gate: tensor<?x6144xf16>) -> tensor<?x6144xf16> {
    %y = "onnx.Custom"(%x, %w, %gate) {
      function_name = "GatedRMSNorm", domain_name = "com.microsoft",
      epsilon = 1.0e-6 : f32
    } : (tensor<?x6144xf16>, tensor<128xf16>, tensor<?x6144xf16>)
        -> tensor<?x6144xf16>
    // CHECK-LABEL: func.func @norm
    // CHECK: hip.gated_rms_norm
    // LLVM-LABEL: llvm.func @norm
    // LLVM: llvm.call @wrap_gated_rms_norm
    return %y : tensor<?x6144xf16>
  }

  func.func @norm_fp32(%x: tensor<?x6144xf32>, %w: tensor<128xf32>,
                       %gate: tensor<?x6144xf32>) -> tensor<?x6144xf32> {
    %y = "onnx.Custom"(%x, %w, %gate) {
      function_name = "GatedRMSNorm", domain_name = "com.microsoft",
      epsilon = 1.0e-6 : f32
    } : (tensor<?x6144xf32>, tensor<128xf32>, tensor<?x6144xf32>)
        -> tensor<?x6144xf32>
    // CHECK-LABEL: func.func @norm_fp32
    // CHECK: hip.gated_rms_norm
    // LLVM-LABEL: llvm.func @norm_fp32
    // LLVM: llvm.mlir.constant({{.*}} : f32) : f32
    // LLVM-NEXT: %[[NORM_DTYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
    // LLVM: llvm.call @wrap_gated_rms_norm({{.*}}, %[[NORM_DTYPE]])
    return %y : tensor<?x6144xf32>
  }

  func.func @mrope(%x: tensor<1x?x6144xf32>, %pos: tensor<3x1x?xi64>,
                   %cos: tensor<262144x32xf32>, %sin: tensor<262144x32xf32>)
      -> tensor<1x?x6144xf32> {
    %y = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding", domain_name = "com.microsoft",
      num_heads = 24 : si64, rotary_embedding_dim = 64 : si64,
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      mrope_layout = 1 : si64, is_packed_batching = 1 : si64
    } : (tensor<1x?x6144xf32>, tensor<3x1x?xi64>,
         tensor<262144x32xf32>, tensor<262144x32xf32>) -> tensor<1x?x6144xf32>
    // CHECK-LABEL: func.func @mrope
    // CHECK: hip.mrotary_embedding
    // LLVM-LABEL: llvm.func @mrope
    // LLVM: llvm.call @wrap_mrotary_embedding
    return %y : tensor<1x?x6144xf32>
  }

  func.func @separate_paged_fp32(
      %q: tensor<?x6144xf32>, %k: tensor<?x1024xf32>, %v: tensor<?x1024xf32>,
      %past_k: tensor<64x256x4x256xf16>, %past_v: tensor<64x256x4x256xf16>,
      %cu: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x16xi32>)
      -> (tensor<?x6144xf32>, tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %y:3 = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %cu, %past, %table, %none, %none) {
      function_name = "PagedAttention", domain_name = "com.microsoft",
      num_heads = 24 : si64, kv_num_heads = 4 : si64, do_rotary = 0 : si64
    } : (tensor<?x6144xf32>, tensor<?x1024xf32>, tensor<?x1024xf32>,
         tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>,
         tensor<?xi32>, tensor<?xi32>, tensor<?x16xi32>, none, none)
        -> (tensor<?x6144xf32>, tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>)
    // CHECK-LABEL: func.func @separate_paged_fp32
    // CHECK: hip.paged_attention_separate
    // LLVM-LABEL: llvm.func @separate_paged_fp32
    // LLVM: llvm.mlir.constant({{.*}} : f32) : f32
    // LLVM-NEXT: %[[PAGED_DTYPE:.*]] = llvm.mlir.constant(0 : i64) : i64
    // LLVM: llvm.call @wrap_paged_attention_separate({{.*}}, %[[PAGED_DTYPE]])
    return %y#0, %y#1, %y#2 : tensor<?x6144xf32>, tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>
  }

  func.func @separate_paged(
      %q: tensor<?x6144xf16>, %k: tensor<?x1024xf16>, %v: tensor<?x1024xf16>,
      %past_k: tensor<64x256x4x256xf16>, %past_v: tensor<64x256x4x256xf16>,
      %cu: tensor<?xi32>, %past: tensor<?xi32>, %table: tensor<?x16xi32>,
      %hint: tensor<3xi32>)
      -> (tensor<?x6144xf16>, tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>) {
    %none = "onnx.NoValue"() {value} : () -> none
    %y:3 = "onnx.Custom"(%q, %k, %v, %past_k, %past_v, %cu, %past, %table,
                        %none, %none, %none, %none, %none, %none, %none, %none, %hint) {
      function_name = "PagedAttention", domain_name = "com.microsoft",
      num_heads = 24 : si64, kv_num_heads = 4 : si64, do_rotary = 0 : si64
    } : (tensor<?x6144xf16>, tensor<?x1024xf16>, tensor<?x1024xf16>,
         tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>,
         tensor<?xi32>, tensor<?xi32>, tensor<?x16xi32>, none, none,
         none, none, none, none, none, none,
         tensor<3xi32>)
        -> (tensor<?x6144xf16>, tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>)
    // CHECK-LABEL: func.func @separate_paged
    // CHECK: hip.paged_attention_separate
    // LLVM-LABEL: llvm.func @separate_paged
    // LLVM: llvm.call @wrap_paged_attention_separate
    return %y#0, %y#1, %y#2 : tensor<?x6144xf16>, tensor<64x256x4x256xf16>, tensor<64x256x4x256xf16>
  }
}
