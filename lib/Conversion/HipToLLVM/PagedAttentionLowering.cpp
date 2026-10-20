/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir::hip {
namespace {

struct PagedAttentionInt8Lowering
    : ConvertOpToLLVMPattern<PagedAttentionInt8Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PagedAttentionInt8Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto loc = op.getLoc();
    SmallVector<Type> signature{getPtrType(), rewriter.getI32Type()};
    signature.append(13, getPtrType());
    SmallVector<Value> arguments{adaptor.getCtx(),
                                 getOpStateSlotValue(op, rewriter, loc)};
    for (Value operand : adaptor.getOperands().drop_front())
      arguments.push_back(extractContiguousMemRefPtr(operand, rewriter, loc));
    const auto dimension = [&](unsigned operand, unsigned index) {
      signature.push_back(rewriter.getI64Type());
      arguments.push_back(getMemRefDimSize(
          cast<MemRefType>(op->getOperand(operand).getType()), index,
          adaptor.getOperands()[operand], rewriter, loc));
    };
    for (unsigned operand : {1u, 2u, 3u, 11u, 7u, 6u, 8u})
      dimension(operand, 0);
    dimension(8, 1);
    for (unsigned operand : {4u, 5u, 12u, 13u}) {
      dimension(operand, 0);
      dimension(operand, 1);
    }
    for (int64_t value : {op.getNumHeads(), op.getKvNumHeads()}) {
      signature.push_back(rewriter.getI64Type());
      arguments.push_back(
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                   rewriter.getI64IntegerAttr(value)));
    }
    signature.push_back(rewriter.getF32Type());
    arguments.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getF32Type(), op.getScaleAttr()));
    signature.push_back(rewriter.getI64Type());
    arguments.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(), op.getResidualBitsAttr()));
    auto function = LLVM::lookupOrCreateFn(
        rewriter, op->getParentOfType<ModuleOp>(), "wrap_paged_attention_int8",
        signature, rewriter.getI32Type());
    if (failed(function))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *function, arguments);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PagedAttentionSeparateLowering
    : ConvertOpToLLVMPattern<PagedAttentionSeparateOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PagedAttentionSeparateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto loc = op.getLoc();
    SmallVector<Type> signature(12, getPtrType());
    SmallVector<Value> arguments{adaptor.getCtx()};
    for (Value operand : adaptor.getOperands().drop_front())
      arguments.push_back(extractContiguousMemRefPtr(operand, rewriter, loc));
    const auto dimension = [&](unsigned operand, unsigned index) {
      signature.push_back(rewriter.getI64Type());
      arguments.push_back(getMemRefDimSize(
          cast<MemRefType>(op->getOperand(operand).getType()), index,
          adaptor.getOperands()[operand], rewriter, loc));
    };
    for (unsigned operand : {1u, 2u, 3u, 9u, 7u, 6u, 8u})
      dimension(operand, 0);
    dimension(8, 1);
    for (unsigned operand : {4u, 5u, 10u, 11u}) {
      dimension(operand, 0);
      dimension(operand, 1);
    }
    for (int64_t value : {op.getNumHeads(), op.getKvNumHeads()}) {
      signature.push_back(rewriter.getI64Type());
      arguments.push_back(
          LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                   rewriter.getI64IntegerAttr(value)));
    }
    signature.push_back(rewriter.getF32Type());
    arguments.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getF32Type(), op.getScaleAttr()));
    signature.push_back(rewriter.getI64Type());
    arguments.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(getHipdnnDataType(
            cast<MemRefType>(op.getQuery().getType()).getElementType()))));
    auto function = LLVM::lookupOrCreateFn(
        rewriter, op->getParentOfType<ModuleOp>(),
        "wrap_paged_attention_separate", signature, rewriter.getI32Type());
    if (failed(function))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *function, arguments);
    rewriter.eraseOp(op);
    return success();
  }
};

struct PagedAttentionLowering : ConvertOpToLLVMPattern<PagedAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PagedAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const auto loc = op.getLoc();
    const auto pointer = getPtrType();
    const auto i32 = rewriter.getI32Type();
    const auto i64 = rewriter.getI64Type();
    SmallVector<Type> signature(12, pointer);
    signature.append(13, i64);
    signature.push_back(rewriter.getF32Type());
    SmallVector<Value> arguments{adaptor.getCtx()};
    for (Value operand : adaptor.getOperands().drop_front())
      arguments.push_back(extractContiguousMemRefPtr(operand, rewriter, loc));

    const auto dimension = [&](Value tensor, Value descriptor, unsigned index) {
      return getMemRefDimSize(cast<MemRefType>(tensor.getType()), index,
                              descriptor, rewriter, loc);
    };
    arguments.push_back(dimension(op.getQuery(), adaptor.getQuery(), 0));
    arguments.push_back(
        dimension(op.getPastLengths(), adaptor.getPastLengths(), 0));
    arguments.push_back(
        dimension(op.getBlockTable(), adaptor.getBlockTable(), 1));
    arguments.push_back(dimension(op.getPastKey(), adaptor.getPastKey(), 0));
    arguments.push_back(dimension(op.getPastKey(), adaptor.getPastKey(), 1));
    for (int64_t value : {op.getNumHeads(), op.getKvNumHeads()})
      arguments.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i64, rewriter.getI64IntegerAttr(value)));
    arguments.push_back(dimension(op.getCosCache(), adaptor.getCosCache(), 0));
    arguments.push_back(dimension(op.getCumulativeLengths(),
                                  adaptor.getCumulativeLengths(), 0));
    arguments.push_back(
        dimension(op.getBlockTable(), adaptor.getBlockTable(), 0));
    arguments.push_back(
        dimension(op.getPastValue(), adaptor.getPastValue(), 0));
    arguments.push_back(
        dimension(op.getPastValue(), adaptor.getPastValue(), 1));
    arguments.push_back(dimension(op.getSinCache(), adaptor.getSinCache(), 0));
    arguments.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getF32Type(), op.getScaleAttr()));
    auto function =
        LLVM::lookupOrCreateFn(rewriter, op->getParentOfType<ModuleOp>(),
                               "wrap_paged_attention", signature, i32);
    if (failed(function))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *function, arguments);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePagedAttentionLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns) {
  patterns.add<PagedAttentionLowering, PagedAttentionSeparateLowering,
               PagedAttentionInt8Lowering>(converter);
}

} // namespace mlir::hip
