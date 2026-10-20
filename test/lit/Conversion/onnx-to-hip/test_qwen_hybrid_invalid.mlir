// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
//
// RUN: hip-mlir-opt %s --hip-add-context-arg --convert-onnx-to-hip --verify-diagnostics

module {
  func.func @norm_sigmoid(%x: tensor<?x128xf32>, %w: tensor<128xf32>,
                          %gate: tensor<?x128xf32>) -> tensor<?x128xf32> {
    // expected-error @+1 {{requires SiLU gating}}
    %y = "onnx.Custom"(%x, %w, %gate) {
      function_name = "GatedRMSNorm", domain_name = "com.microsoft",
      activation = "sigmoid"
    } : (tensor<?x128xf32>, tensor<128xf32>, tensor<?x128xf32>)
        -> tensor<?x128xf32>
    return %y : tensor<?x128xf32>
  }

  func.func @norm_dynamic_gate_width(%x: tensor<?x128xf32>, %w: tensor<128xf32>,
                                     %gate: tensor<?x?xf32>) -> tensor<?x128xf32> {
    // expected-error @+1 {{requires matching FP16/FP32 packed input/gate}}
    %y = "onnx.Custom"(%x, %w, %gate) {
      function_name = "GatedRMSNorm", domain_name = "com.microsoft"
    } : (tensor<?x128xf32>, tensor<128xf32>, tensor<?x?xf32>)
        -> tensor<?x128xf32>
    return %y : tensor<?x128xf32>
  }

  func.func @mrope_scale(%x: tensor<1x?x256xf32>, %pos: tensor<3x1x?xi64>,
                         %cos: tensor<?x32xf32>, %sin: tensor<?x32xf32>)
      -> tensor<1x?x256xf32> {
    // expected-error @+1 {{requires unit rotary scale}}
    %y = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding", domain_name = "com.microsoft",
      num_heads = 1 : si64, rotary_embedding_dim = 64 : si64,
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      mrope_layout = 1 : si64, is_packed_batching = 1 : si64, scale = 2.0 : f32
    } : (tensor<1x?x256xf32>, tensor<3x1x?xi64>,
         tensor<?x32xf32>, tensor<?x32xf32>) -> tensor<1x?x256xf32>
    return %y : tensor<1x?x256xf32>
  }

  func.func @mrope_dynamic_sine_width(%x: tensor<1x?x256xf32>, %pos: tensor<3x1x?xi64>,
                                      %cos: tensor<?x32xf32>, %sin: tensor<?x?xf32>)
      -> tensor<1x?x256xf32> {
    // expected-error @+1 {{requires packed FP32 head-256 partial MRoPE}}
    %y = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding", domain_name = "com.microsoft",
      num_heads = 1 : si64, rotary_embedding_dim = 64 : si64,
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      mrope_layout = 1 : si64, is_packed_batching = 1 : si64
    } : (tensor<1x?x256xf32>, tensor<3x1x?xi64>,
         tensor<?x32xf32>, tensor<?x?xf32>) -> tensor<1x?x256xf32>
    return %y : tensor<1x?x256xf32>
  }

  func.func @main_graph(%x: tensor<?x10240xf16>, %w: tensor<10240x1x3xf16>,
                         %cu: tensor<?xi32>, %bias: tensor<10240xf16>,
                         %past: tensor<?x10240x3xf16>)
      -> (tensor<?x10240xf16>, tensor<?x10240x3xf16>) {
    // expected-error @+1 {{requires matching FP16/FP32 packed kernel-4 convolution, SiLU, int32 cumulative lengths and three-element carry}}
    %y:2 = "onnx.Custom"(%x, %w, %cu, %bias, %past) {
      function_name = "VarlenCausalConvWithState",
      domain_name = "com.microsoft", activation = "silu"
    } : (tensor<?x10240xf16>, tensor<10240x1x3xf16>, tensor<?xi32>,
         tensor<10240xf16>, tensor<?x10240x3xf16>)
        -> (tensor<?x10240xf16>, tensor<?x10240x3xf16>)
    return %y#0, %y#1 : tensor<?x10240xf16>, tensor<?x10240x3xf16>
  }

  func.func @gdn_bf16(%q: tensor<?x16x128xbf16>, %k: tensor<?x16x128xbf16>,
                      %v: tensor<?x48x128xbf16>, %cu: tensor<?xi32>,
                      %decay: tensor<?x48xf32>, %beta: tensor<?x48xf32>,
                      %past: tensor<?x48x128x128xf32>,
                      %a: tensor<48xf32>, %dt: tensor<48xf32>)
      -> (tensor<?x48x128xbf16>, tensor<?x48x128x128xf32>) {
    // expected-error @+1 {{requires FP16 or FP32 activation tensors}}
    %y:2 = "onnx.Custom"(%q, %k, %v, %cu, %decay, %beta, %past, %a, %dt) {
      function_name = "GatedDeltaNet", domain_name = "com.microsoft",
      update_rule = "gated_delta", gate_activation = "qwen",
      beta_activation = "sigmoid", qk_l2_norm = 1 : si64
    } : (tensor<?x16x128xbf16>, tensor<?x16x128xbf16>, tensor<?x48x128xbf16>,
         tensor<?xi32>, tensor<?x48xf32>, tensor<?x48xf32>,
         tensor<?x48x128x128xf32>, tensor<48xf32>, tensor<48xf32>)
        -> (tensor<?x48x128xbf16>, tensor<?x48x128x128xf32>)
    return %y#0, %y#1 : tensor<?x48x128xbf16>, tensor<?x48x128x128xf32>
  }

  func.func @norm_group(%x: tensor<?x6144xf16>, %w: tensor<256xf16>,
                        %gate: tensor<?x6144xf16>) -> tensor<?x6144xf16> {
    // expected-error @+1 {{requires matching FP16/FP32 packed input/gate, group width 128 and positive finite epsilon}}
    %y = "onnx.Custom"(%x, %w, %gate) {
      function_name = "GatedRMSNorm", domain_name = "com.microsoft"
    } : (tensor<?x6144xf16>, tensor<256xf16>, tensor<?x6144xf16>)
        -> tensor<?x6144xf16>
    return %y : tensor<?x6144xf16>
  }

  func.func @norm_mixed(%x: tensor<?x6144xf32>, %w: tensor<128xf16>,
                        %gate: tensor<?x6144xf32>) -> tensor<?x6144xf32> {
    // expected-error @+1 {{requires matching FP16/FP32 packed input/gate}}
    %y = "onnx.Custom"(%x, %w, %gate) {
      function_name = "GatedRMSNorm", domain_name = "com.microsoft"
    } : (tensor<?x6144xf32>, tensor<128xf16>, tensor<?x6144xf32>)
        -> tensor<?x6144xf32>
    return %y : tensor<?x6144xf32>
  }

  func.func @mrope_layout(%x: tensor<1x?x6144xf32>, %pos: tensor<3x1x?xi64>,
                          %cos: tensor<262144x32xf32>, %sin: tensor<262144x32xf32>)
      -> tensor<1x?x6144xf32> {
    // expected-error @+1 {{requires packed FP32 head-256 partial MRoPE with layout 1 and sections [11,11,10]}}
    %y = "onnx.Custom"(%x, %pos, %cos, %sin) {
      function_name = "MRotaryEmbedding", domain_name = "com.microsoft",
      num_heads = 24 : si64, rotary_embedding_dim = 64 : si64,
      mrope_section = [11 : si64, 11 : si64, 10 : si64],
      mrope_layout = 0 : si64, is_packed_batching = 1 : si64
    } : (tensor<1x?x6144xf32>, tensor<3x1x?xi64>,
         tensor<262144x32xf32>, tensor<262144x32xf32>) -> tensor<1x?x6144xf32>
    return %y : tensor<1x?x6144xf32>
  }
}
