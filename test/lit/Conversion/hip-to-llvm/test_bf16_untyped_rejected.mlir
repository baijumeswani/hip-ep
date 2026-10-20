// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

// Exercise the existing valid FP16 ABI fixtures with native BF16 storage.
// These arithmetic ABIs encode byte widths, not explicit floating-point types.
// RUN: sed 's/f16/bf16/g' %S/test_layer_normalization.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=NORM
// RUN: sed 's/f16/bf16/g' %S/test_rope.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=ROPE
// RUN: sed 's/f16/bf16/g' %S/test_gqa.mlir | not hip-mlir-opt --assign-op-state-slots --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=GQA
// RUN: sed 's/f16/bf16/g' %S/test_multi_head_attention.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=MHA
// RUN: sed 's/f16/bf16/g' %S/test_causal_conv_with_state.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=CONV
// RUN: sed 's/f16/bf16/g' %S/test_qmoe.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=MOE
// RUN: sed 's/f16/bf16/g' %S/test_qmoe_amd.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=AMD
// RUN: sed -e 's/f32/bf16/g' -e 's/reduction = "none"/reduction = "add"/' %S/test_scatter_elements.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=SCATTER
// RUN: sed 's/f16/bf16/g' %S/test_topk.mlir | not hip-mlir-opt --convert-hip-to-llvm 2>&1 | FileCheck %s --check-prefix=TOPK

// NORM: BF16 LayerNormalization is not supported
// ROPE: BF16 standalone RoPE is not supported
// GQA: BF16 GQA is not supported
// MHA: BF16 MultiHeadAttention is not supported
// CONV: BF16 CausalConvWithState is not supported
// MOE: BF16 QMoE is not supported
// AMD: BF16 QMoEAmd is not supported
// SCATTER: BF16 ScatterElements reductions are not supported
// TOPK: BF16 TopK is not supported
