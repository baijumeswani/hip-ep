/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include <type_traits>

namespace mlir::hip {
namespace {

template <typename Op> struct QwenHybridLowering : ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;
  using OpAdaptor = typename Op::Adaptor;

  LogicalResult
  matchAndRewrite(Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Type> signature(op->getNumOperands(), this->getPtrType());
    SmallVector<Value> arguments{adaptor.getCtx()};
    for (Value operand : adaptor.getOperands().drop_front())
      arguments.push_back(extractContiguousMemRefPtr(operand, rewriter, loc));
    const auto dim = [&](unsigned operand, unsigned index) {
      arguments.push_back(getMemRefDimSize(
          cast<MemRefType>(op->getOperand(operand).getType()), index,
          adaptor.getOperands()[operand], rewriter, loc));
      signature.push_back(rewriter.getI64Type());
    };
    const auto integer = [&](int64_t value) {
      arguments.push_back(
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                   rewriter.getI64IntegerAttr(value)));
      signature.push_back(rewriter.getI64Type());
    };
    const auto floating = [&](FloatAttr value) {
      arguments.push_back(LLVM::ConstantOp::create(
          rewriter, loc, rewriter.getF32Type(), value));
      signature.push_back(rewriter.getF32Type());
    };
    StringRef functionName;
    if constexpr (std::is_same_v<Op, VarlenCausalConvOp> ||
                  std::is_same_v<Op, VarlenCausalConvUpdatesOp>) {
      constexpr bool capture = std::is_same_v<Op, VarlenCausalConvUpdatesOp>;
      functionName = capture ? "wrap_varlen_causal_conv_updates"
                             : "wrap_varlen_causal_conv";
      dim(1, 0);
      dim(capture ? 7 : 6, 0);
      dim(3, 0);
      dim(5, 0);
      dim(capture ? 8 : 7, 0);
      dim(1, 1);
      if constexpr (capture) {
        dim(6, 0);
        for (unsigned index : {0u, 1u, 2u})
          dim(9, index);
        integer(op.getStateUpdateCapacity());
      }
    } else if constexpr (std::is_same_v<Op, GatedDeltaNetOp> ||
                         std::is_same_v<Op, GatedDeltaNetUpdatesOp>) {
      constexpr bool capture = std::is_same_v<Op, GatedDeltaNetUpdatesOp>;
      functionName =
          capture ? "wrap_gated_delta_net_updates" : "wrap_gated_delta_net";
      for (unsigned operand : {1u, 2u, 3u, 5u, 6u, 4u, 7u, capture ? 13u : 11u,
                               capture ? 12u : 10u})
        dim(operand, 0);
      if constexpr (capture) {
        dim(10, 0);
        dim(11, 0);
        dim(14, 0);
        dim(14, 1);
      }
      integer(op.getKeyHeads());
      integer(op.getValueHeads());
      if constexpr (capture)
        integer(op.getStateUpdateCapacity());
      floating(op.getScaleAttr());
    } else if constexpr (std::is_same_v<Op, GatedRmsNormOp>) {
      functionName = "wrap_gated_rms_norm";
      for (unsigned operand : {1u, 3u, 4u})
        dim(operand, 0);
      dim(1, 1);
      floating(op.getEpsilonAttr());
    } else {
      functionName = "wrap_mrotary_embedding";
      dim(1, 1);
      dim(2, 2);
      dim(3, 0);
      dim(4, 0);
      dim(5, 1);
      integer(op.getNumHeads());
    }
    if constexpr (!std::is_same_v<Op, MRotaryEmbeddingOp>)
      integer(getHipdnnDataType(
          cast<MemRefType>(op->getOperand(1).getType()).getElementType()));
    auto function = LLVM::lookupOrCreateFn(
        rewriter, op->template getParentOfType<ModuleOp>(), functionName,
        signature, rewriter.getI32Type());
    if (failed(function))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *function, arguments);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQwenHybridLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns.add<QwenHybridLowering<VarlenCausalConvOp>,
               QwenHybridLowering<GatedDeltaNetOp>,
               QwenHybridLowering<VarlenCausalConvUpdatesOp>,
               QwenHybridLowering<GatedDeltaNetUpdatesOp>,
               QwenHybridLowering<GatedRmsNormOp>,
               QwenHybridLowering<MRotaryEmbeddingOp>>(converter);
}

} // namespace mlir::hip
